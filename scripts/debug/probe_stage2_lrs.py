"""Sweep stage-2 LR candidates and report effective per-step update ratios.

Reuses scripts/debug/check_gradient_flow.py's model build + synthetic batch
plumbing, runs one forward+backward against mazesmazes/tiny-audio-next, then
evaluates `lr × ||grad||` (SGD-style sanity check, not literal Adam step) for
each candidate (projector_lr, decoder_lr) pair so we can pick LRs whose
projector/decoder update ratio actually matches the recipe's intent.
"""

from __future__ import annotations

import torch

from scripts.debug.check_gradient_flow import (
    build_model,
    build_param_groups,
    grad_norm,
    parameter_summary,
    synthetic_batch,
)


def main() -> None:
    device = "cpu"
    dtype = torch.float32
    model_id = "mazesmazes/tiny-audio-next-plus"

    print(f"Loading {model_id} (this takes ~30s)...")
    torch.manual_seed(0)
    model = build_model(dtype, device, model_id=model_id)

    print("\nParameter summary:")
    for top, (tr, tot) in parameter_summary(model).items():
        print(f"  {top:20s} {tr:>13,} / {tot:>13,} ({100 * tr / tot:5.1f}%)")

    print("\nRunning forward+backward on synthetic batch...")
    model.zero_grad(set_to_none=True)
    batch = synthetic_batch(model, batch_size=2, audio_seconds=4.0)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch)
    print(f"  loss = {out.loss.item():.4f}  finite = {torch.isfinite(out.loss).item()}")
    out.loss.backward()

    # Aggregate grad norms per logical bucket (projector decay/no-decay,
    # decoder decay/no-decay). Use build_param_groups with placeholder LRs to
    # get the exact same routing as ASRTrainer.create_optimizer.
    placeholder_knobs = {
        "learning_rate": 1.0,
        "decoder_learning_rate": 1.0,
        "weight_decay": 0.0,
        "projector_weight_decay": 0.0,
    }
    groups = build_param_groups(model, placeholder_knobs)

    print("\nPer-group ||grad|| (lr-independent; computed once):")
    bucket_grad = {}
    for g in groups:
        if not g["params"]:
            continue
        gn = grad_norm(g["params"])
        bucket_grad[g["label"]] = gn
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['label']:25s} params={len(g['params']):4d} numel={n:>12,}  ||grad||={gn:.4e}")

    # Sweep LR candidates and report effective update ratios.
    candidates = [
        ("A conservative  ", 1e-4, 1e-5),
        ("B moderate (rec)", 3e-4, 3e-5),
        ("C aggressive    ", 5e-4, 5e-5),
        ("D proj-keep     ", 1e-3, 1e-5),
        ("E proj-keep gap5", 1e-3, 5e-5),
        ("F decoder-lean  ", 3e-4, 1e-5),
        ("G symmetric step", 3e-4, 1e-5),
    ]

    print()
    print(
        f"{'recipe':17s} {'proj_lr':>9s} {'dec_lr':>9s} "
        f"{'proj_upd':>11s} {'dec_upd':>11s} {'p/d ratio':>10s} {'note':s}"
    )
    print("-" * 100)
    for label, proj_lr, dec_lr in candidates:
        proj_upd = 0.0
        dec_upd = 0.0
        for g in groups:
            gn = bucket_grad.get(g["label"], 0.0)
            if g["label"].startswith("projector"):
                proj_upd += proj_lr * gn
            else:
                dec_upd += dec_lr * gn
        ratio = proj_upd / dec_upd if dec_upd > 0 else float("inf")
        note = ""
        if 0.5 <= ratio <= 2.0:
            note = "✅ balanced"
        elif ratio > 2.0:
            note = "⚠ projector dominates"
        else:
            note = "⚠ decoder dominates"
        print(
            f"{label:17s} {proj_lr:>9.0e} {dec_lr:>9.0e} "
            f"{proj_upd:>11.4e} {dec_upd:>11.4e} {ratio:>10.3f} {note}"
        )

    print()
    print("Notes:")
    print("  - 'proj_upd' / 'dec_upd' are SGD-style lr × ||grad|| sums.")
    print("  - Adam normalization will compress per-parameter step magnitude")
    print("    toward `lr × sign(grad)` once moments warm up; the SGD")
    print("    approximation overstates the decoder's relative early dominance.")
    print("  - 'Balanced' (0.5–2.0) means projector and decoder move at")
    print("    similar magnitudes — the recipe's actual goal varies by stage.")


if __name__ == "__main__":
    main()
