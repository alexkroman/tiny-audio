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

import torch

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel


def build_model(dtype: torch.dtype, device: str) -> ASRModel:
    """Mirror configs/experiments/embedded.yaml model section."""
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

    print("[8] Verdict:")
    issues = []
    if enc_with_grad != 0 or enc_norm != 0.0:
        issues.append("encoder is receiving gradient (should be frozen)")
    if proj_with_grad != len(projector_params) or proj_norm == 0.0:
        issues.append("projector grads incomplete or zero")
    if lm_with_grad != len(lm_params) or lm_norm == 0.0:
        issues.append("decoder grads incomplete or zero")
    if bad:
        issues.append(f"{bad} param(s) have non-finite grads")
    if issues:
        for s in issues:
            print(f"    [FAIL] {s}")
    else:
        print("    [OK] gradient flow matches embedded.yaml's intent:")
        print("         - encoder frozen (no grad)")
        print("         - projector + decoder fully trainable, all params got grad")
        print("         - all grads finite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        args.dtype
    ]
    torch.manual_seed(0)
    model = build_model(dtype, args.device)
    report(model, dtype, args.device)


if __name__ == "__main__":
    main()
