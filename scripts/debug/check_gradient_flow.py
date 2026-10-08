"""Gradient-flow probe for a training experiment config.

Builds the ASRModel exactly as an experiment config does -- by default
configs/experiments/stage_1.yaml (GLM-ASR-Nano-2512 encoder, Qwen3-0.6B
decoder, MLP projector, full-decoder fine-tune i.e. freeze_language_model=False)
-- runs one synthetic forward + backward through the real training
DataCollator, and reports:

  - which submodules have requires_grad=True/False
  - per-module gradient norms (every trainable param must get a finite,
    non-zero grad; frozen ones none)
  - whether a frozen component (e.g. the encoder) accidentally received gradient
  - the projector vs LM gradient-norm ratio (sanity check for the split LR)
  - whether the <audio> embed_tokens / lm_head row sees any gradient (it
    shouldn't on the input side since masked_scatter replaces it; on the
    output side only if labels contain <audio>, which they shouldn't)
  - any NaN/Inf in grads or activations

Usage:
    poetry run python scripts/debug/check_gradient_flow.py [--experiment stage_1]
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Any, NamedTuple, TypedDict, cast

import torch
import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from transformers import set_seed

from scripts.train import (
    build_asr_config,
    disable_chat_template_thinking,
    optimizer_param_groups,
)
from scripts.train_collator import DataCollator
from scripts.train_config import register_configs
from scripts.utils import get_project_root
from tiny_audio.asr_modeling import ASRModel


class ParamGroup(TypedDict):
    """One optimizer param group as ASRTrainer would build it, plus audit fields."""

    label: str
    params: list[torch.nn.Parameter]
    param_names: list[str]
    lr: float | None
    wd: float | None


def load_experiment_config(experiment: str) -> DictConfig:
    """Compose configs/config.yaml + `+experiments=<experiment>` as training does.

    Composed rather than read from the experiment file alone: experiments
    inherit every key they do not set from config.yaml and its defaults
    (training/production.yaml), so the file by itself is missing knobs the
    trainer uses. Attention is forced to eager so the probe does not depend
    on flash-attn being importable locally; gradient flow is independent of
    the attention implementation.
    """
    register_configs()
    configs = get_project_root() / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        return compose(
            config_name="config",
            overrides=[f"+experiments={experiment}", "training.attn_implementation=eager"],
        )


def training_knobs(cfg: DictConfig) -> dict[str, float | None]:
    """The LR / WD knobs ASRTrainer.create_optimizer reads from `training:`.

    Unset keys map to None, which optimizer_param_groups resolves the same way
    the trainer does (component overrides fall back to the base values).
    """
    training = cfg.training

    def _to_float(v: Any) -> float | None:
        return float(v) if v is not None else None

    return {
        key: _to_float(training.get(key))
        for key in (
            "learning_rate",
            "decoder_learning_rate",
            "encoder_learning_rate",
            "weight_decay",
            "projector_weight_decay",
            "encoder_weight_decay",
        )
    }


_GROUP_LABELS = {"other": "projector", "decoder": "decoder", "encoder": "encoder"}


def build_param_groups(
    model: ASRModel,
    knobs: dict[str, float | None],
) -> list[ParamGroup]:
    """ASRTrainer.create_optimizer's component x decay split, labelled for the audit.

    Built by scripts/train.py's optimizer_param_groups, the same routing the
    trainer applies; each group also carries `param_names` and the configured
    `lr` / `wd` under the experiment's knobs.
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
            encoder_lr=knobs["encoder_learning_rate"],
            encoder_wd=knobs["encoder_weight_decay"],
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


def build_model(
    dtype: torch.dtype, device: str, cfg: DictConfig, model_id: str | None = None
) -> ASRModel:
    """Build the model the experiment config `cfg` trains.

    If `model_id` is provided, load the trained checkpoint from the Hub or a
    local path; otherwise build a fresh model through scripts/train.py's own
    build_asr_config (base-LM weights, a randomly-initialized projector, the
    experiment's freeze settings). Gradient flow (which params get grads)
    is independent of weights, but absolute gradient magnitudes — and the
    projector/decoder ratio — depend on the trained state, so checkpoint
    loading matters for the "is the projector dominating?" diagnostic.
    """
    if model_id is None:
        model = ASRModel(build_asr_config(cfg))
    else:
        model = ASRModel.from_pretrained(
            model_id,
            attn_implementation="eager",
        )
        # from_pretrained may have left freeze_language_model=True from the
        # saved config; force it off so the LM gets gradient as in training.
        for p in model.language_model.parameters():
            p.requires_grad_(True)
    # Training applies this before building its DataCollator; the probe's
    # batch goes through the same collator, so it needs the same template.
    disable_chat_template_thinking(model)
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
    audio_seconds: tuple[float, ...] = (4.0, 3.0),
    response: str = "hello world this is a gradient flow test",
) -> dict[str, torch.Tensor]:
    """Collate random-noise rows through the real training DataCollator.

    Built with scripts/train_collator.DataCollator, configured as
    scripts/train.py configures it, rather than by hand: the collator pads
    text on the left (trl's DataCollatorForChatML), emits `model.audio_token`
    rather than a literal "<audio>", and masks labels its own way, and a
    probe that diverges on any of those audits a batch training never sees.
    The clip lengths differ so the batch actually carries padding.
    """
    sr = model.feature_extractor.sampling_rate
    collator = DataCollator(
        tokenizer=model.tokenizer,
        feature_extractor=model.feature_extractor,
        sample_rate=sr,
        projector=model.projector,
        encoder_conv_layers=model.config.encoder_conv_layers,
        audio_token=model.audio_token,
    )
    features: list[dict[str, Any]] = [
        {
            "audio": {"array": torch.randn(int(seconds * sr)).numpy(), "sampling_rate": sr},
            "text": response,
        }
        for seconds in audio_seconds
    ]
    batch = collator(features)
    if len(batch["input_ids"]) != len(features):
        msg = f"DataCollator dropped {len(features) - len(batch['input_ids'])} probe row(s)"
        raise RuntimeError(msg)
    return batch


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm over every present .grad, as one flattened vector (0.0 if none)."""
    grads = [p.grad.detach().float() for p in params if p.grad is not None]
    return float(torch.nn.utils.get_total_norm(grads))


def projector_submodule_norms(model: ASRModel) -> dict[str, float]:
    """Per-named-submodule grad norms for the MLP projector."""
    return {
        name: float(param.grad.detach().float().norm())
        for name, param in model.projector.named_parameters()
        if param.grad is not None
    }


class _Component(NamedTuple):
    """.grad coverage and gradient norm of one top-level component."""

    with_grad: int
    trainable: int
    total: int
    norm: float


def _component_grads(module: torch.nn.Module) -> _Component:
    """Count a module's params that got .grad / are trainable, and their grad norm."""
    params = list(module.parameters())
    with_grad = sum(1 for p in params if p.grad is not None)
    trainable = sum(1 for p in params if p.requires_grad)
    return _Component(with_grad, trainable, len(params), grad_norm(params))


class _Routing(NamedTuple):
    """Optimizer param groups plus the routing-audit counts from section [9]."""

    groups: list[ParamGroup]
    total_routed: int
    expected_trainable: int
    frozen_in_groups: int


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
    """[3] (after backward) How many params of each component received .grad.

    Expected = the component's trainable (requires_grad) params, which the
    experiment's freeze settings decide -- 0 for a frozen encoder, all of the
    decoder for a full fine-tune, all but embed_tokens under
    freeze_text_embed_tokens.
    """
    for label, comp in (("encoder  ", enc), ("projector", proj), ("decoder  ", lm)):
        print(
            f"    {label} params with .grad: {comp.with_grad}/{comp.total} "
            f"(expected {comp.trainable} trainable)"
        )
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


def _print_optimizer_routing(model: ASRModel, knobs: dict[str, float | None]) -> _Routing:
    """[9] Build ASRTrainer-style param groups and audit their coverage."""
    print("[9] Optimizer param-group routing (mirrors scripts/train.py ASRTrainer):")
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

    frozen_in_groups = sum(1 for g in groups for p in g["params"] if not p.requires_grad)
    if frozen_in_groups:
        print(f"    !! frozen params in optimizer groups: {frozen_in_groups} (should be 0)")
    print()
    return _Routing(groups, total_routed, expected_trainable, frozen_in_groups)


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


def _coverage_issue(label: str, comp: _Component) -> str | None:
    """Why `comp`'s gradients disagree with its freeze settings, or None."""
    if comp.trainable == 0:
        if comp.with_grad or comp.norm != 0.0:
            return f"{label} is receiving gradient (should be frozen)"
        return None
    if comp.with_grad != comp.trainable or comp.norm == 0.0:
        return f"{label} grads incomplete or zero ({comp.with_grad}/{comp.trainable} trainable)"
    return None


def _verdict_issues(
    enc: _Component, proj: _Component, lm: _Component, bad: int, routing: _Routing
) -> list[str]:
    """Hard failures: wrong grad flow, non-finite grads, or misrouted params."""
    issues = [
        issue
        for label, comp in (("encoder", enc), ("projector", proj), ("decoder", lm))
        if (issue := _coverage_issue(label, comp)) is not None
    ]
    if bad:
        issues.append(f"{bad} param(s) have non-finite grads")
    # Optimizer-routing audit (relies on the counts from [9])
    if routing.total_routed != routing.expected_trainable:
        issues.append(
            f"optimizer group routing: {routing.total_routed} routed "
            f"vs {routing.expected_trainable} trainable (orphans)"
        )
    if routing.frozen_in_groups:
        issues.append(
            f"optimizer group routing: {routing.frozen_in_groups} frozen "
            "param(s) ended up in an optimizer group"
        )
    return issues


def _verdict_warnings(sub_norms: dict[str, float]) -> list[str]:
    """Soft findings: a starved projector linear_1."""
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
    return warnings


def _print_verdict(experiment: str, issues: list[str], warnings: list[str]) -> None:
    """[11] FAIL/WARN lines, or the all-clear summary when nothing failed."""
    print("[11] Verdict:")
    for s in issues:
        print(f"    [FAIL] {s}")
    for w in warnings:
        print(f"    [WARN] {w}")
    if not issues:
        print(f"    [OK] gradient flow matches {experiment}'s freeze settings:")
        print("         - every trainable param got a non-zero grad, no frozen one did")
        print("         - all grads finite")
        print("         - optimizer groups route every trainable param exactly once")


def report(
    model: ASRModel,
    dtype: torch.dtype,
    device: str,
    experiment: str,
    knobs: dict[str, float | None],
) -> None:
    """Run one forward + backward and print every audit section, then a verdict."""
    print(f"== {experiment} gradient flow probe ({dtype}, {device}) ==\n")
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
    routing = _print_optimizer_routing(model, knobs)
    _print_effective_update(routing.groups)

    _print_verdict(
        experiment,
        _verdict_issues(enc, proj, lm, bad, routing),
        _verdict_warnings(sub_norms),
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
    experiment: Annotated[
        str,
        typer.Option(
            "--experiment",
            "-e",
            help="configs/experiments/ name: builds the fresh model and supplies "
            "the optimizer LR / WD knobs",
        ),
    ] = "stage_1",
) -> None:
    """Probe gradient flow on a checkpoint (per-component grad norms)."""
    # Dtype's values are torch attribute names by construction.
    torch_dtype: torch.dtype = getattr(torch, Dtype(dtype).value)
    # Seeds python, numpy and torch (CPU + CUDA): the fresh projector init and
    # the probe's noise audio both draw from them.
    set_seed(0)
    cfg = load_experiment_config(experiment)
    built = build_model(torch_dtype, device, cfg, model_id=model)
    report(built, torch_dtype, device, experiment, training_knobs(cfg))


if __name__ == "__main__":
    typer.run(main)
