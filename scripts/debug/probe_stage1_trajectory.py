"""Probe gradient norms across stage_1's training trajectory.

Loads tiny-audio-next-plus at multiple Hub revisions (= W&B run zot1zdu9
checkpoints) and reports ||grad_projector|| / ||grad_decoder|| under a
deterministic synthetic batch — the same shape probe_stage2_lrs.py
runs, just iterated across revisions.

Answers: at post-warmup (step 6000) what was the decoder grad norm?
If it was already in the 100+ range, stage_1's decoder_learning_rate=1e-4
was applying ~10× the corrective step magnitude that 1e-5 would have
given.

Usage:
    poetry run python scripts/debug/probe_stage1_trajectory.py
"""

from __future__ import annotations

import torch

from scripts.debug.check_gradient_flow import (
    build_param_groups,
    grad_norm,
    synthetic_batch,
)
from tiny_audio.asr_modeling import ASRModel

# (step, hub_commit_sha) — one entry per representative point in the
# zot1zdu9 trajectory. Pick the "Training in progress -" variant for each
# step (HF Trainer pushes two commits per save_steps; both contain the
# same weights).
CHECKPOINTS = [
    ("step  2000 (early warmup)", "f7b57a92aeb3065f04ad3927dc27f2202a13f3be"),
    ("step  6000 (post-warmup)", "63c5f7d0fee03e3a7b11b4ab4b64361b449ee6e3"),
    ("step 30000 (mid-cosine)", "b15a328929fbe959fa299bd80336f84253463412"),
    ("step 48000 (late-cosine)", "9a1dc2f7e11d8a5c78133696c670efde57f6eecc"),
    ("step 85172 (final, tip)", None),  # current main
]


def probe_one(label: str, revision: str | None) -> dict[str, float]:
    torch.manual_seed(0)
    kwargs = {"attn_implementation": "eager"}
    if revision is not None:
        kwargs["revision"] = revision
    model = ASRModel.from_pretrained("mazesmazes/tiny-audio-next-plus", **kwargs)
    for p in model.language_model.parameters():
        p.requires_grad_(True)
    model.to(device="cpu", dtype=torch.float32)

    model.zero_grad(set_to_none=True)
    batch = synthetic_batch(model, batch_size=2, audio_seconds=4.0)
    out = model(**batch)
    loss_val = float(out.loss.item())
    out.loss.backward()

    groups = build_param_groups(
        model,
        {
            "learning_rate": 1.0,
            "decoder_learning_rate": 1.0,
            "weight_decay": 0.0,
            "projector_weight_decay": 0.0,
        },
    )
    norms = {g["label"]: grad_norm(g["params"]) for g in groups if g["params"]}

    # Combined projector / decoder norms (decay + no-decay buckets).
    proj = sum(v for k, v in norms.items() if k.startswith("projector"))
    dec = sum(v for k, v in norms.items() if k.startswith("decoder"))

    # Free model before next iteration.
    del model
    import gc

    gc.collect()

    return {
        "label": label,
        "loss": loss_val,
        "proj_grad": proj,
        "dec_grad": dec,
        "ratio_d_p": dec / proj if proj > 0 else float("inf"),
    }


def main() -> None:
    rows: list[dict[str, float]] = []
    for label, rev in CHECKPOINTS:
        print(f"\nLoading {label} ...")
        rows.append(probe_one(label, rev))
        r = rows[-1]
        print(
            f"  loss={r['loss']:.4f}  ||grad_proj||={r['proj_grad']:.4e}  "
            f"||grad_dec||={r['dec_grad']:.4e}  dec/proj={r['ratio_d_p']:.1f}×"
        )

    print()
    print("=" * 95)
    print(
        f"{'checkpoint':30s} {'loss':>8s} {'||grad_proj||':>14s} "
        f"{'||grad_dec||':>14s} {'dec/proj':>10s}"
    )
    print("-" * 95)
    for r in rows:
        print(
            f"{r['label']:30s} {r['loss']:>8.4f} {r['proj_grad']:>14.4e} "
            f"{r['dec_grad']:>14.4e} {r['ratio_d_p']:>10.1f}×"
        )

    print()
    print("Interpretation cheat-sheet (SGD-style update under stage_1 LRs):")
    print("  proj_upd = 1e-3 × ||grad_proj||")
    print("  dec_upd  = 1e-4 × ||grad_dec||    (stage_1's actual decoder LR)")
    print("  dec_upd' = 1e-5 × ||grad_dec||    (counterfactual: lower decoder LR)")
    print(
        f"  {'checkpoint':30s} {'proj_upd':>12s} {'dec_upd@1e-4':>14s} "
        f"{'dec_upd@1e-5':>14s} {'p/d @1e-4':>11s} {'p/d @1e-5':>11s}"
    )
    print("-" * 100)
    for r in rows:
        proj_upd = 1e-3 * r["proj_grad"]
        dec_upd_e4 = 1e-4 * r["dec_grad"]
        dec_upd_e5 = 1e-5 * r["dec_grad"]
        ratio_e4 = proj_upd / dec_upd_e4 if dec_upd_e4 > 0 else float("inf")
        ratio_e5 = proj_upd / dec_upd_e5 if dec_upd_e5 > 0 else float("inf")
        print(
            f"  {r['label']:30s} {proj_upd:>12.4e} {dec_upd_e4:>14.4e} "
            f"{dec_upd_e5:>14.4e} {ratio_e4:>10.2f}× {ratio_e5:>10.2f}×"
        )


if __name__ == "__main__":
    main()
