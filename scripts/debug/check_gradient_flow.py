"""Gradient-flow probe for the embedded.yaml recipe.

Builds the ASRModel exactly as configs/experiments/embedded.yaml does
(GLM-ASR-Nano-2512 encoder, Qwen3-0.6B decoder, MLP projector, full-decoder
fine-tune i.e. freeze_language_model=False), runs one synthetic forward +
backward, and reports:

  - which submodules have requires_grad=True/False
  - per-module gradient norms (encoder must be None; projector + LM must be
    finite, non-zero)
  - whether the frozen encoder accidentally received gradient
  - the projector vs LM gradient-norm ratio (sanity check for the split LR)
  - whether the <audio> embed_tokens / lm_head row sees any gradient (it
    shouldn't on the input side since masked_scatter replaces it; on the
    output side only if labels contain <audio>, which they shouldn't)
  - any NaN/Inf in grads or activations

Usage:
    poetry run python scripts/debug/check_gradient_flow.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Any, NamedTuple, TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch
import typer
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence

from scripts.train import optimizer_param_groups
from scripts.utils import get_project_root
from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.asr_types import AudioFeatureExtractor


class ParamGroup(TypedDict):
    """One optimizer param group as ASRTrainer would build it, plus audit fields."""

    label: str
    params: list[torch.nn.Parameter]
    param_names: list[str]
    lr: float | None
    wd: float | None


def load_embedded_training_knobs() -> dict[str, float | None]:
    """Read training LR/WD knobs from configs/experiments/embedded.yaml.

    Returns a dict with keys: learning_rate, decoder_learning_rate,
    weight_decay, projector_weight_decay. Missing keys map to None;
    the caller falls back to printing actuals without the mismatch check.
    """
    repo_root = get_project_root()
    yaml_path = repo_root / "configs" / "experiments" / "embedded.yaml"
    if not yaml_path.exists():
        return {
            "learning_rate": None,
            "decoder_learning_rate": None,
            "weight_decay": None,
            "projector_weight_decay": None,
        }
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path))
    root = cfg if isinstance(cfg, dict) else {}
    training: dict[str, Any] = root.get("training") or {}

    def _to_float(v: Any) -> float | None:
        return float(v) if v is not None else None

    return {
        "learning_rate": _to_float(training.get("learning_rate")),
        "decoder_learning_rate": _to_float(training.get("decoder_learning_rate")),
        "weight_decay": _to_float(training.get("weight_decay")),
        "projector_weight_decay": _to_float(training.get("projector_weight_decay")),
    }


_GROUP_LABELS = {"other": "projector", "decoder": "decoder", "encoder": "encoder"}


def build_param_groups(
    model: ASRModel,
    knobs: dict[str, float | None],
) -> list[ParamGroup]:
    """ASRTrainer.create_optimizer's component x decay split, labelled for the audit.

    Built by scripts/train.py's optimizer_param_groups, the same routing the
    trainer applies; each group also carries `param_names` and the configured
    `lr` / `wd` under embedded.yaml's knobs (encoder knobs fall back to base).
    """
    return [
        {
            "label": f"{_GROUP_LABELS[g.component]:9s} / {'decay' if g.decay else 'no-decay'}",
            "params": g.params,
            "param_names": g.names,
            "lr": g.lr,
            "wd": g.weight_decay,
        }
        for g in optimizer_param_groups(
            model,
            lr=knobs["learning_rate"],
            weight_decay=knobs["weight_decay"],
            decoder_lr=knobs["decoder_learning_rate"],
            projector_wd=knobs["projector_weight_decay"],
        )
    ]


def effective_update_norms(groups: list[ParamGroup]) -> dict[str, float]:
    """SGD-style lr x ||grad|| approximation aggregated to projector/decoder.

    NOT Adam's true update: Adam normalizes per-parameter by sqrt(v) + eps,
    which we don't have at step 0. This is a back-of-envelope sanity check
    on whether the LR split is producing roughly the per-step motion ratio
    it was tuned for.
    """
    proj_total = 0.0
    dec_total = 0.0
    for g in groups:
        lr = g["lr"]
        if lr is None or not g["params"]:
            continue
        gn = grad_norm(g["params"])
        contribution = lr * gn
        if g["label"].startswith("projector"):
            proj_total += contribution
        elif g["label"].startswith("decoder"):
            dec_total += contribution
    ratio = (proj_total / dec_total) if dec_total > 0 else float("inf")
    return {
        "projector": proj_total,
        "decoder": dec_total,
        "ratio": ratio,
    }


def build_model(dtype: torch.dtype, device: str, model_id: str | None = None) -> ASRModel:
    """Build the model used by configs/experiments/embedded.yaml.

    If `model_id` is provided, load the trained checkpoint from the Hub or a
    local path; otherwise build a fresh model with base-LM weights and a
    randomly-initialized projector. Gradient flow (which params get grads)
    is independent of weights, but absolute gradient magnitudes — and the
    projector/decoder ratio — depend on the trained state, so checkpoint
    loading matters for the "is the projector dominating?" diagnostic.
    """
    if model_id is None:
        cfg = ASRConfig(
            audio_model_id="zai-org/GLM-ASR-Nano-2512",
            text_model_id="Qwen/Qwen3-0.6B",
            projector_type="mlp",
            projector_pool_stride=4,
            projector_hidden_dim=2048,
            freeze_language_model=False,
            # Use eager so we don't depend on flash-attn being importable on the
            # local box; gradient flow is independent of attention impl.
            attn_implementation="eager",
        )
        model = ASRModel(cfg)
    else:
        model = ASRModel.from_pretrained(
            model_id,
            attn_implementation="eager",
        )
        # from_pretrained may have left freeze_language_model=True from the
        # saved config; force it off so the LM gets gradient as in training.
        for p in model.language_model.parameters():
            p.requires_grad_(True)
    # PreTrainedModel.to is decorated with functools.wraps(nn.Module.to), which
    # type checkers read as the unbound function (missing `self`). Calling it
    # through the nn.Module type still dispatches to the same override.
    cast(torch.nn.Module, model).to(device=device, dtype=dtype)
    return model


def parameter_summary(model: ASRModel) -> dict[str, tuple[int, int]]:
    """Return {top_module: (trainable_params, total_params)}."""
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name, p in model.named_parameters():
        top = name.split(".", 1)[0]
        buckets[top][1] += p.numel()
        if p.requires_grad:
            buckets[top][0] += p.numel()
    return {k: (v[0], v[1]) for k, v in buckets.items()}


def synthetic_batch(
    model: ASRModel,
    batch_size: int = 2,
    audio_seconds: float = 4.0,
    response: str = "hello world this is a gradient flow test",
) -> dict[str, torch.Tensor]:
    """Build a batch shaped exactly like train_collator.DataCollator output."""
    sr = model.feature_extractor.sampling_rate
    n_samples = int(audio_seconds * sr)
    audio_arrays: list[npt.NDArray[np.float32]] = [
        torch.randn(n_samples).numpy() for _ in range(batch_size)
    ]
    feature_extractor = cast(AudioFeatureExtractor, model.feature_extractor)
    audio_out = feature_extractor(
        audio_arrays,
        sampling_rate=sr,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
    )

    audio_attention_mask: torch.Tensor = audio_out["attention_mask"]
    input_features: torch.Tensor = audio_out["input_features"]
    # A probe of ASRModel internals: it reproduces what forward() computes.
    encoder_lengths = model._compute_encoder_output_lengths  # pyright: ignore[reportPrivateUsage]
    enc_lengths = encoder_lengths(audio_attention_mask)
    token_counts = model.projector.get_output_length(enc_lengths).to(torch.long)

    tok = model.tokenizer
    samples_input_ids: list[list[int]] = []
    samples_labels: list[list[int]] = []
    for i in range(batch_size):
        n_audio = int(token_counts[i].item())
        user = ("<audio>" * n_audio) + " " + ASRModel.TRANSCRIBE_PROMPT
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
        ]
        full_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        prompt_text = tok.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(full_text, str)
        assert isinstance(prompt_text, str)
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        sample_labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
        sample_labels = sample_labels[: len(full_ids)]
        samples_input_ids.append(list(full_ids))
        samples_labels.append(sample_labels)

    pad_id = tok.pad_token_id
    assert isinstance(pad_id, int)
    ids_t = [torch.tensor(ids, dtype=torch.long) for ids in samples_input_ids]
    input_ids = pad_sequence(ids_t, batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence([torch.ones_like(t) for t in ids_t], batch_first=True)
    labels = pad_sequence(
        [torch.tensor(lab, dtype=torch.long) for lab in samples_labels],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_features": input_features,
        "audio_attention_mask": audio_attention_mask,
        "audio_token_counts": token_counts,
    }


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def projector_submodule_norms(model: ASRModel) -> dict[str, float]:
    """Per-named-submodule grad norms for the MLP projector."""
    norms: dict[str, float] = {}
    for name, param in model.projector.named_parameters():
        if param.grad is None:
            continue
        norms[name] = float(param.grad.detach().float().pow(2).sum().sqrt().item())
    return norms


class _Component(NamedTuple):
    """.grad coverage and gradient norm of one top-level component."""

    with_grad: int
    total: int
    norm: float


def _component_grads(module: torch.nn.Module) -> _Component:
    """Count a module's params that got .grad, and their combined grad norm."""
    params = list(module.parameters())
    with_grad = sum(1 for p in params if p.grad is not None)
    return _Component(with_grad, len(params), grad_norm(params))


class _Routing(NamedTuple):
    """Optimizer param groups plus the routing-audit counts from section [9]."""

    knobs: dict[str, float | None]
    groups: list[ParamGroup]
    total_routed: int
    expected_trainable: int
    encoder_in_groups: int


def _print_parameter_summary(model: ASRModel) -> None:
    """[1] Trainable / total parameter counts per top-level submodule."""
    print("[1] Parameter summary (trainable / total):")
    for top, (tr, tot) in sorted(parameter_summary(model).items()):
        pct = 100 * tr / tot if tot else 0.0
        print(f"    {top:18s} {tr:>14,d} / {tot:>14,d}  ({pct:5.1f}%)")
    print()


def _forward_pass(model: ASRModel, dtype: torch.dtype, device: str) -> torch.Tensor:
    """[2] Run one synthetic labelled forward and print loss/logit finiteness."""
    batch = synthetic_batch(model)
    batch = {k: v.to(device) for k, v in batch.items()}
    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(dtype)

    print("[2] Forward pass...")
    model.train()
    # Ask for the logits back. A labelled forward normally skips the lm_head
    # projection entirely (see ASRModel.forward), which is right for training
    # -- the tensor is vocab-sized -- but this probe exists to find NaNs, and
    # the logits are where a bad projector output shows up first. The kwarg
    # only exists on liger's patched forward, so it is gated on the same flag
    # ASRModel uses.
    forward_kwargs: dict[str, bool] = {}
    if model._lm_accepts_skip_logits:  # pyright: ignore[reportPrivateUsage]
        forward_kwargs["skip_logits"] = False
    outputs = model(**batch, **forward_kwargs)
    loss = outputs.loss
    print(f"    loss = {loss.item():.4f}  finite={torch.isfinite(loss).item()}")
    if outputs.logits is None:
        print("    logits: skipped (fused cross-entropy, no lm_head projection)")
    else:
        print(
            f"    logits: shape={tuple(outputs.logits.shape)}  "
            f"finite={torch.isfinite(outputs.logits).all().item()}"
        )
    print()
    return loss


def _print_grad_coverage(enc: _Component, proj: _Component, lm: _Component) -> None:
    """[3] (after backward) How many params of each component received .grad."""
    print(f"    encoder   params with .grad: {enc.with_grad}/{enc.total} (expected 0 — frozen)")
    print(f"    projector params with .grad: {proj.with_grad}/{proj.total} (expected all)")
    print(f"    decoder   params with .grad: {lm.with_grad}/{lm.total} (expected all)")
    print()


def _print_grad_norms(enc: _Component, proj: _Component, lm: _Component) -> None:
    """[4] Per-component gradient norms and the projector/decoder ratio."""
    print("[4] Gradient norms:")
    print(f"    ||grad_encoder||   = {enc.norm:.6e}")
    print(f"    ||grad_projector|| = {proj.norm:.6e}")
    print(f"    ||grad_decoder||   = {lm.norm:.6e}")
    if lm.norm > 0:
        print(f"    projector / decoder norm ratio = {proj.norm / lm.norm:.3f}")
    print()


def _print_projector_submodules(model: ASRModel) -> dict[str, float]:
    """[4b] Projector submodule gradient norms, relative to the largest."""
    sub_norms = projector_submodule_norms(model)
    print("[4b] Projector submodule gradient norms:")
    max_norm = max(sub_norms.values())
    for name, n in sub_norms.items():
        ratio = (n / max_norm) if max_norm > 0 else 0.0
        print(f"    {name:18s} ||grad|| = {n:.6e}  ({ratio:5.2f}x of max)")
    print()
    return sub_norms


def _decoder_group_key(name: str) -> str:
    """Bucket a decoder parameter name into its breakdown group."""
    if "embed_tokens" in name:
        return "embed_tokens"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if ".self_attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    if "norm" in name:
        return "norm"
    return "other"


def _print_decoder_breakdown(model: ASRModel) -> None:
    """[5] Decoder gradient norms grouped by embed/attn/mlp/norm/lm_head."""
    print("[5] Per-submodule gradient norms (decoder breakdown):")
    decoder_groups: defaultdict[str, list[torch.nn.Parameter]] = defaultdict(list)
    for name, p in model.language_model.named_parameters():
        if p.grad is not None:
            decoder_groups[_decoder_group_key(name)].append(p)
    for key in ("embed_tokens", "attn", "mlp", "norm", "lm_head", "other"):
        if key in decoder_groups:
            print(f"    {key:14s} ||grad|| = {grad_norm(decoder_groups[key]):.6e}")
    print()


def _print_audio_token_rows(model: ASRModel) -> None:
    """[6] Gradient on the <audio> row of the input embedding and lm_head."""
    print("[6] <audio>-token row gradient (sanity check):")
    audio_id = model.audio_token_id
    # The input embedding is an nn.Embedding and the output head an nn.Linear;
    # both hold their weight as a Parameter (a Tensor), never a submodule.
    embed_weight = model.language_model.get_input_embeddings().weight
    assert isinstance(embed_weight, torch.Tensor)
    if embed_weight.grad is not None:
        row_grad = embed_weight.grad[audio_id].detach().float()
        print(f"    embed_tokens[<audio>] ||grad|| = {row_grad.norm().item():.6e}")
        print("    (should be zero if labels mask user prompt and no assistant token is <audio>)")
    out_emb = model.get_output_embeddings()
    out_weight = out_emb.weight if out_emb is not None else None
    assert out_weight is None or isinstance(out_weight, torch.Tensor)
    tied = out_weight is not None and out_weight.data_ptr() == embed_weight.data_ptr()
    if out_weight is not None and out_weight.grad is not None and not tied:
        # Untied head — separate gradient meaningful.
        row_grad = out_weight.grad[audio_id].detach().float()
        print(f"    lm_head[<audio>]      ||grad|| = {row_grad.norm().item():.6e}")
    else:
        print(f"    lm_head tied to embed_tokens: {tied}")
    print()


def _scan_nonfinite_grads(model: ASRModel) -> int:
    """[7] Print every param with a NaN/Inf grad; return how many there were."""
    print("[7] NaN / Inf scan over all grads:")
    bad = 0
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad += 1
            print(f"    !! non-finite grad in {name}")
    if bad == 0:
        print("    all grads finite")
    print()
    return bad


def _print_optimizer_routing(model: ASRModel) -> _Routing:
    """[9] Build ASRTrainer-style param groups and audit their coverage."""
    print("[9] Optimizer param-group routing (mirrors scripts/train.py ASRTrainer):")
    knobs = load_embedded_training_knobs()
    groups = build_param_groups(model, knobs)
    print(
        f"    {'group':22s} {'params':>6s} {'numel':>14s}  "
        f"{'||grad||':>12s}  {'lr':>8s}  {'wd':>6s}"
    )
    total_routed = 0
    for g in groups:
        numel = sum(p.numel() for p in g["params"])
        gn = grad_norm(g["params"])
        lr_str = f"{g['lr']:.1e}" if g["lr"] is not None else "?"
        wd_str = f"{g['wd']}" if g["wd"] is not None else "?"
        print(
            f"    {g['label']:22s} {len(g['params']):>6d} {numel:>14,d}  "
            f"{gn:>12.2e}  {lr_str:>8s}  {wd_str:>6s}"
        )
        total_routed += len(g["params"])

    expected_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(
        f"    coverage: {total_routed} trainable params placed in {expected_trainable} group slots "
        f"({'no orphans' if total_routed == expected_trainable else 'ORPHANS PRESENT'})"
    )

    encoder_param_ptrs = {p.data_ptr() for p in model.audio_tower.parameters()}
    encoder_in_groups = sum(
        1 for g in groups for p in g["params"] if p.data_ptr() in encoder_param_ptrs
    )
    if encoder_in_groups:
        print(f"    !! encoder params in optimizer groups: {encoder_in_groups} (should be 0)")
    print()
    return _Routing(knobs, groups, total_routed, expected_trainable, encoder_in_groups)


def _print_effective_update(groups: list[ParamGroup]) -> None:
    """[10] lr x ||grad|| per side, a sanity check on the LR split."""
    eff = effective_update_norms(groups)
    print("[10] Effective per-step update estimate (lr \u00d7 ||grad||, SGD-style approximation):")
    print(f"    projector contribution: {eff['projector']:.2e}")
    print(f"    decoder   contribution: {eff['decoder']:.2e}")
    ratio_str = "inf (decoder=0)" if math.isinf(eff["ratio"]) else f"{eff['ratio']:.3f}"
    print(f"    ratio (projector / decoder): {ratio_str}")
    print("    note: Adam's per-parameter normalization will modify these; this is a")
    print("          sanity check on the LR split, not a literal step magnitude.")
    print()


def _verdict_issues(
    enc: _Component, proj: _Component, lm: _Component, bad: int, routing: _Routing
) -> list[str]:
    """Hard failures: wrong grad flow, non-finite grads, or misrouted params."""
    issues: list[str] = []
    if enc.with_grad != 0 or enc.norm != 0.0:
        issues.append("encoder is receiving gradient (should be frozen)")
    if proj.with_grad != proj.total or proj.norm == 0.0:
        issues.append("projector grads incomplete or zero")
    if lm.with_grad != lm.total or lm.norm == 0.0:
        issues.append("decoder grads incomplete or zero")
    if bad:
        issues.append(f"{bad} param(s) have non-finite grads")
    # Optimizer-routing audit (relies on the counts from [9])
    if routing.total_routed != routing.expected_trainable:
        issues.append(
            f"optimizer group routing: {routing.total_routed} routed "
            f"vs {routing.expected_trainable} trainable (orphans)"
        )
    if routing.encoder_in_groups:
        issues.append(
            f"optimizer group routing: {routing.encoder_in_groups} frozen "
            "encoder param(s) ended up in an optimizer group"
        )
    return issues


def _verdict_warnings(sub_norms: dict[str, float], knobs: dict[str, float | None]) -> list[str]:
    """Soft findings: a starved projector linear_1, or LRs off the designed values."""
    warnings: list[str] = []
    # Projector linear_1 starvation check (relies on [4b])
    if sub_norms:
        max_sub = max(sub_norms.values())
        linear_1_norm = sub_norms.get("linear_1.weight", 0.0)
        if max_sub > 0 and linear_1_norm < 0.01 * max_sub:
            warnings.append(
                "projector linear_1 < 1% of max submodule grad — RMSNorm-init "
                "claim at projectors.py:30 may be wrong; verify before relying on it"
            )
    # LR/WD mismatch vs embedded.yaml. Only check when knobs were readable.
    if knobs["learning_rate"] is not None and knobs["learning_rate"] != 1e-3:
        warnings.append(
            f"embedded.yaml learning_rate={knobs['learning_rate']} "
            "differs from the 1e-3 this probe was designed against"
        )
    if knobs["decoder_learning_rate"] is not None and knobs["decoder_learning_rate"] != 1e-4:
        warnings.append(
            f"embedded.yaml decoder_learning_rate={knobs['decoder_learning_rate']} "
            "differs from the 1e-4 this probe was designed against"
        )
    return warnings


def _print_verdict(issues: list[str], warnings: list[str]) -> None:
    """[11] FAIL/WARN lines, or the all-clear summary when nothing failed."""
    print("[11] Verdict:")
    for s in issues:
        print(f"    [FAIL] {s}")
    for w in warnings:
        print(f"    [WARN] {w}")
    if not issues:
        print("    [OK] gradient flow matches embedded.yaml's intent:")
        print("         - encoder frozen (no grad)")
        print("         - projector + decoder fully trainable, all params got grad")
        print("         - all grads finite")
        print("         - optimizer groups route every trainable param exactly once")


def report(model: ASRModel, dtype: torch.dtype, device: str) -> None:
    """Run one forward + backward and print every audit section, then a verdict."""
    print(f"== embedded.yaml gradient flow probe ({dtype}, {device}) ==\n")
    _print_parameter_summary(model)
    loss = _forward_pass(model, dtype, device)

    print("[3] Backward pass...")
    loss.backward()
    enc = _component_grads(model.audio_tower)
    proj = _component_grads(model.projector)
    lm = _component_grads(model.language_model)
    _print_grad_coverage(enc, proj, lm)
    _print_grad_norms(enc, proj, lm)

    sub_norms = _print_projector_submodules(model)
    _print_decoder_breakdown(model)
    _print_audio_token_rows(model)
    bad = _scan_nonfinite_grads(model)
    routing = _print_optimizer_routing(model)
    _print_effective_update(routing.groups)

    _print_verdict(
        _verdict_issues(enc, proj, lm, bad, routing),
        _verdict_warnings(sub_norms, routing.knobs),
    )


class Dtype(StrEnum):
    """Torch dtypes the probe can run in."""

    float32 = "float32"
    bfloat16 = "bfloat16"
    float16 = "float16"


def main(
    model: Annotated[
        str | None,
        typer.Argument(
            help="HuggingFace model ID (or local path) of a trained checkpoint; "
            "omit to build a fresh model from base-LM weights + a random projector "
            "(verifies plumbing, not training state)"
        ),
    ] = None,
    dtype: Annotated[Dtype, typer.Option("--dtype", help="Torch dtype to run in")] = Dtype.float32,
    device: Annotated[str, typer.Option("--device", help="cpu / cuda / mps")] = "cpu",
) -> None:
    """Probe gradient flow on a checkpoint (per-component grad norms)."""
    torch_dtype = {
        Dtype.float32: torch.float32,
        Dtype.bfloat16: torch.bfloat16,
        Dtype.float16: torch.float16,
    }[Dtype(dtype)]
    torch.manual_seed(0)
    built = build_model(torch_dtype, device, model_id=model)
    report(built, torch_dtype, device)


if __name__ == "__main__":
    typer.run(main)
