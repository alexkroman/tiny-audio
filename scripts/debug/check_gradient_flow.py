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

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.projectors import MLPAudioProjector


def load_embedded_training_knobs() -> dict[str, float | None]:
    """Read training LR/WD knobs from configs/experiments/embedded.yaml.

    Returns a dict with keys: learning_rate, decoder_learning_rate,
    weight_decay, projector_weight_decay. Missing keys map to None;
    the caller falls back to printing actuals without the mismatch check.
    """
    repo_root = Path(__file__).resolve().parents[2]
    yaml_path = repo_root / "configs" / "experiments" / "embedded.yaml"
    if not yaml_path.exists():
        return {
            "learning_rate": None,
            "decoder_learning_rate": None,
            "weight_decay": None,
            "projector_weight_decay": None,
        }
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    training = cfg.get("training") or {}

    def _to_float(v):
        return float(v) if v is not None else None

    return {
        "learning_rate": _to_float(training.get("learning_rate")),
        "decoder_learning_rate": _to_float(training.get("decoder_learning_rate")),
        "weight_decay": _to_float(training.get("weight_decay")),
        "projector_weight_decay": _to_float(training.get("projector_weight_decay")),
    }


def build_param_groups(
    model: ASRModel,
    knobs: dict[str, float | None],
) -> list[dict]:
    """Mirror scripts/train.py ASRTrainer.create_optimizer's four-group split.

    Groups: (is_decoder, decay) for is_decoder in {False, True} and
    decay in {True, False}. is_decoder = name.startswith("language_model.").
    decay = name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) and
    "bias" not in name. Each group dict also carries `param_names` for the
    routing audit and the configured `lr` / `wd` that ASRTrainer would
    apply under embedded.yaml's knobs.
    """
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    from transformers.trainer_pt_utils import get_parameter_names

    forbidden = list(ALL_LAYERNORM_LAYERS) + [Qwen3RMSNorm, LlamaRMSNorm]
    decay_set = set(get_parameter_names(model, forbidden))
    decay_set = {n for n in decay_set if "bias" not in n}

    base_lr = knobs["learning_rate"]
    base_wd = knobs["weight_decay"]
    dec_lr = (
        knobs["decoder_learning_rate"] if knobs["decoder_learning_rate"] is not None else base_lr
    )
    dec_wd = base_wd  # ASRTrainer falls back to args.weight_decay when no decoder override
    proj_wd = (
        knobs["projector_weight_decay"] if knobs["projector_weight_decay"] is not None else base_wd
    )

    buckets: dict[tuple[bool, bool], list[tuple[str, torch.nn.Parameter]]] = {
        (False, True): [],
        (False, False): [],
        (True, True): [],
        (True, False): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = (name.startswith("language_model."), name in decay_set)
        buckets[key].append((name, param))

    labels = {
        (False, True): ("projector / decay", base_lr, proj_wd),
        (False, False): ("projector / no-decay", base_lr, 0.0),
        (True, True): ("decoder   / decay", dec_lr, dec_wd),
        (True, False): ("decoder   / no-decay", dec_lr, 0.0),
    }
    groups: list[dict] = []
    for key, items in buckets.items():
        label, lr, wd = labels[key]
        groups.append(
            {
                "label": label,
                "params": [p for _, p in items],
                "param_names": [n for n, _ in items],
                "lr": lr,
                "wd": wd,
            }
        )
    return groups


def effective_update_norms(groups: list[dict]) -> dict[str, float]:
    """SGD-style lr × ||grad|| approximation aggregated to projector/decoder.

    NOT Adam's true update: Adam normalizes per-parameter by sqrt(v) + eps,
    which we don't have at step 0. This is a back-of-envelope sanity check
    on whether the LR split is producing roughly the per-step motion ratio
    it was tuned for.
    """
    proj_total = 0.0
    dec_total = 0.0
    for g in groups:
        if g["lr"] is None or not g["params"]:
            continue
        gn = grad_norm(g["params"])
        contribution = g["lr"] * gn
        if g["label"].startswith("projector"):
            proj_total += contribution
        else:
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
    model.to(device=device, dtype=dtype)
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
    """Build a batch shaped exactly like train.DataCollator output."""
    sr = model.feature_extractor.sampling_rate
    n_samples = int(audio_seconds * sr)
    audio_arrays = [torch.randn(n_samples).numpy() for _ in range(batch_size)]
    audio_out = model.feature_extractor(
        audio_arrays,
        sampling_rate=sr,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
    )

    enc_lengths = model._compute_encoder_output_lengths(audio_out.attention_mask)
    token_counts = model.projector.get_output_length(enc_lengths).to(torch.long)

    tok = model.tokenizer
    samples_input_ids: list[list[int]] = []
    samples_labels: list[list[int]] = []
    for i in range(batch_size):
        n_audio = int(token_counts[i].item())
        user = ("<audio>" * n_audio) + " Transcribe the speech to text"
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
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
        labels = labels[: len(full_ids)]
        samples_input_ids.append(list(full_ids))
        samples_labels.append(labels)

    max_len = max(len(x) for x in samples_input_ids)
    pad_id = tok.pad_token_id
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(samples_input_ids, samples_labels)):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention_mask[i, : len(ids)] = 1
        labels[i, : len(lab)] = torch.tensor(lab)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_features": audio_out.input_features,
        "audio_attention_mask": audio_out.attention_mask,
        "audio_token_counts": token_counts,
    }


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def projector_submodule_norms(model: ASRModel) -> dict[str, float] | None:
    """Per-named-submodule grad norms for the MLP projector.

    Returns None for non-MLP projectors so the caller can print a skip note;
    the other projector types (MOSA, MoE, QFormer) have different internal
    structures and would each need their own carve-up (out of scope per the
    design spec).
    """
    if not isinstance(model.projector, MLPAudioProjector):
        return None

    norms: dict[str, float] = {}
    for name, param in model.projector.named_parameters():
        if param.grad is None:
            continue
        norms[name] = float(param.grad.detach().float().pow(2).sum().sqrt().item())
    return norms


def report(model: ASRModel, dtype: torch.dtype, device: str) -> None:
    print(f"== embedded.yaml gradient flow probe ({dtype}, {device}) ==\n")

    print("[1] Parameter summary (trainable / total):")
    for top, (tr, tot) in sorted(parameter_summary(model).items()):
        pct = 100 * tr / tot if tot else 0.0
        print(f"    {top:18s} {tr:>14,d} / {tot:>14,d}  ({pct:5.1f}%)")
    print()

    batch = synthetic_batch(model)
    batch = {k: v.to(device) for k, v in batch.items()}
    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(dtype)

    print("[2] Forward pass...")
    model.train()
    outputs = model(**batch)
    loss = outputs.loss
    print(f"    loss = {loss.item():.4f}  finite={torch.isfinite(loss).item()}")
    print(
        f"    logits: shape={tuple(outputs.logits.shape)}  "
        f"finite={torch.isfinite(outputs.logits).all().item()}"
    )
    print()

    print("[3] Backward pass...")
    loss.backward()

    encoder_params = list(model.audio_tower.parameters())
    projector_params = list(model.projector.parameters())
    lm_params = list(model.language_model.parameters())

    enc_with_grad = sum(1 for p in encoder_params if p.grad is not None)
    proj_with_grad = sum(1 for p in projector_params if p.grad is not None)
    lm_with_grad = sum(1 for p in lm_params if p.grad is not None)

    print(
        f"    encoder   params with .grad: {enc_with_grad}/{len(encoder_params)} "
        f"(expected 0 — frozen)"
    )
    print(
        f"    projector params with .grad: {proj_with_grad}/{len(projector_params)} (expected all)"
    )
    print(f"    decoder   params with .grad: {lm_with_grad}/{len(lm_params)} (expected all)")
    print()

    enc_norm = grad_norm(encoder_params)
    proj_norm = grad_norm(projector_params)
    lm_norm = grad_norm(lm_params)
    print("[4] Gradient norms:")
    print(f"    ||grad_encoder||   = {enc_norm:.6e}")
    print(f"    ||grad_projector|| = {proj_norm:.6e}")
    print(f"    ||grad_decoder||   = {lm_norm:.6e}")
    if lm_norm > 0:
        print(f"    projector / decoder norm ratio = {proj_norm / lm_norm:.3f}")
    print()

    sub_norms = projector_submodule_norms(model)
    if sub_norms is None:
        print(
            "[4b] Projector submodule gradient norms: "
            f"(skipped — projector is {type(model.projector).__name__}, not MLPAudioProjector)"
        )
    else:
        print("[4b] Projector submodule gradient norms:")
        max_norm = max(sub_norms.values())
        for name, n in sub_norms.items():
            ratio = (n / max_norm) if max_norm > 0 else 0.0
            print(f"    {name:18s} ||grad|| = {n:.6e}  ({ratio:5.2f}x of max)")
    print()

    print("[5] Per-submodule gradient norms (decoder breakdown):")
    decoder_groups: dict[str, list] = defaultdict(list)
    for name, p in model.language_model.named_parameters():
        if p.grad is None:
            continue
        if "embed_tokens" in name:
            key = "embed_tokens"
        elif name.endswith("lm_head.weight"):
            key = "lm_head"
        elif ".self_attn." in name:
            key = "attn"
        elif ".mlp." in name:
            key = "mlp"
        elif "norm" in name:
            key = "norm"
        else:
            key = "other"
        decoder_groups[key].append(p)
    for key in ("embed_tokens", "attn", "mlp", "norm", "lm_head", "other"):
        if key in decoder_groups:
            print(f"    {key:14s} ||grad|| = {grad_norm(decoder_groups[key]):.6e}")
    print()

    print("[6] <audio>-token row gradient (sanity check):")
    audio_id = model.audio_token_id
    embed = model.language_model.get_input_embeddings()
    if embed.weight.grad is not None:
        row_grad = embed.weight.grad[audio_id].detach().float()
        print(f"    embed_tokens[<audio>] ||grad|| = {row_grad.norm().item():.6e}")
        print("    (should be zero if labels mask user prompt and no assistant token is <audio>)")
    out_emb = model.language_model.get_output_embeddings()
    if (
        out_emb is not None
        and out_emb.weight.grad is not None
        and not torch.equal(
            out_emb.weight.data_ptr() == embed.weight.data_ptr() and embed.weight,
            embed.weight,  # silence linter; we just want pointer equality below
        )
    ):
        # Untied head — separate gradient meaningful.
        row_grad = out_emb.weight.grad[audio_id].detach().float()
        print(f"    lm_head[<audio>]      ||grad|| = {row_grad.norm().item():.6e}")
    else:
        tied = out_emb is not None and out_emb.weight.data_ptr() == embed.weight.data_ptr()
        print(f"    lm_head tied to embed_tokens: {tied}")
    print()

    print("[7] NaN / Inf scan over all grads:")
    bad = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad += 1
            print(f"    !! non-finite grad in {name}")
    if bad == 0:
        print("    all grads finite")
    print()

    print("[9] Optimizer param-group routing (mirrors scripts/train.py ASRTrainer):")
    knobs = load_embedded_training_knobs()
    groups = build_param_groups(model, knobs)
    print(
        f"    {'group':22s} {'params':>6s} {'numel':>14s}  {'||grad||':>12s}  {'lr':>8s}  {'wd':>6s}"
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

    encoder_in_groups = 0
    encoder_param_ptrs = {p.data_ptr() for p in model.audio_tower.parameters()}
    for g in groups:
        for p in g["params"]:
            if p.data_ptr() in encoder_param_ptrs:
                encoder_in_groups += 1
    if encoder_in_groups:
        print(f"    !! encoder params in optimizer groups: {encoder_in_groups} (should be 0)")
    print()

    # Stash for the verdict in [11] to consult without re-running build_param_groups.
    report._last_routing_audit = {
        "total_routed": total_routed,
        "expected_trainable": expected_trainable,
        "encoder_in_groups": encoder_in_groups,
        "knobs": knobs,
    }

    eff = effective_update_norms(groups)
    print("[10] Effective per-step update estimate (lr × ||grad||, SGD-style approximation):")
    print(f"    projector contribution: {eff['projector']:.2e}")
    print(f"    decoder   contribution: {eff['decoder']:.2e}")
    ratio_str = "inf (decoder=0)" if math.isinf(eff["ratio"]) else f"{eff['ratio']:.3f}"
    print(f"    ratio (projector / decoder): {ratio_str}")
    print("    note: Adam's per-parameter normalization will modify these; this is a")
    print("          sanity check on the LR split, not a literal step magnitude.")
    print()

    print("[11] Verdict:")
    issues = []
    warnings: list[str] = []

    if enc_with_grad != 0 or enc_norm != 0.0:
        issues.append("encoder is receiving gradient (should be frozen)")
    if proj_with_grad != len(projector_params) or proj_norm == 0.0:
        issues.append("projector grads incomplete or zero")
    if lm_with_grad != len(lm_params) or lm_norm == 0.0:
        issues.append("decoder grads incomplete or zero")
    if bad:
        issues.append(f"{bad} param(s) have non-finite grads")

    # New: projector linear_1 starvation check (relies on [4b])
    if sub_norms:
        max_sub = max(sub_norms.values())
        linear_1_norm = sub_norms.get("linear_1.weight", 0.0)
        if max_sub > 0 and linear_1_norm < 0.01 * max_sub:
            warnings.append(
                "projector linear_1 < 1% of max submodule grad — RMSNorm-init "
                "claim at projectors.py:30 may be wrong; verify before relying on it"
            )

    # New: optimizer-routing audit (relies on [9])
    audit = getattr(report, "_last_routing_audit", None)
    if audit is not None:
        if audit["total_routed"] != audit["expected_trainable"]:
            issues.append(
                f"optimizer group routing: {audit['total_routed']} routed "
                f"vs {audit['expected_trainable']} trainable (orphans)"
            )
        if audit["encoder_in_groups"]:
            issues.append(
                f"optimizer group routing: {audit['encoder_in_groups']} frozen "
                "encoder param(s) ended up in an optimizer group"
            )
        # LR/WD mismatch vs embedded.yaml. Only check when knobs were readable.
        knobs = audit["knobs"]
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

    if issues:
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


def main(
    model_id: str | None = None,
    dtype: str = "float32",
    device: str = "cpu",
) -> None:
    """Run gradient-flow probe.

    Args:
        model_id: Hub repo id or local path to a trained checkpoint. If
            omitted, build a fresh model from base-LM weights + randomly-
            initialized projector (verifies plumbing, not training state).
        dtype: float32 / bfloat16 / float16.
        device: cpu / cuda / mps.
    """
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    torch.manual_seed(0)
    model = build_model(torch_dtype, device, model_id=model_id)
    report(model, torch_dtype, device)


def _argparse_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hub repo id or local path; omit to build a fresh model.",
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(model_id=args.model_id, dtype=args.dtype, device=args.device)


if __name__ == "__main__":
    _argparse_main()
