---
name: project-guardrails
description: >-
  Tiny Audio's settled decisions: dependency pins, training-config choices and gate rules that
  were paid for once and must not be undone casually. Load before changing pyproject.toml,
  requirements files, configs/ or the model code.
user-invocable: false
---

# Tiny Audio guardrails

These are settled decisions. Don't reverse one silently; if a task seems to require it, stop and ask
the user first. Rules marked *(enforced)* fail `ta dev quality invariants` (`scripts/invariants.py`)
when broken; the rest are held only by review, so they matter more.

## Dependencies

- **liger-kernel stays at `>=0.8.0`.** *(enforced)* Older releases lack
  `apply_liger_kernel_to_qwen3_5`; without it there is no fused linear cross-entropy and every
  labelled forward materializes the full logits tensor (~29 GiB at batch 48), which is the OOM that
  killed several `granite_qwen_lora` launches.
- **nltk stays `<3.10`.** *(enforced)* 3.10 splits a leading apostrophe into its own token and
  truecase re-glues it onto the previous word, so labels come out as "get' em"
  (`tests/test_label_normalization.py`).
- **flash-linear-attention, tilelang and causal-conv1d stay in the optional `hybrid-kernels`
  group.** *(enforced)* fla without tilelang on Hopper with Triton 3.4–3.7 raises in backward, so a
  half-install turns a slow run into a crashed one. `scripts/deploy/runpod.py` installs and verifies
  the pair on pods.
- **transformers is pinned to one git commit**, the same in `pyproject.toml` and `requirements.txt`
  (uploaded next to the model on the Hub). *(enforced)* Move both together, and only to a commit the
  pipeline was verified on.
- **torchaudio's major.minor matches torch's.** *(enforced)* Pods never pin or replace torch: the
  deploy image provides it and the exported requirements drop `torch==`.
- **CI and the web-session hook use the same Poetry and Python.** *(enforced)*

## Training configs

- **`auto_find_batch_size` stays false.** HF halves the per-device batch on OOM without touching
  gradient accumulation, silently changing the effective batch.
- **Batch shapes are measured, not planned.** The comments in `configs/experiments/*.yaml` record
  the GPU and peak memory each shape was measured on. Don't reuse a shape on a smaller card; step
  down before a run, because the seeded data order replays an OOM on resume.
- **Numbers in config comments are evidence.** Don't change a cap, ratio or learning rate whose
  comment cites a measurement without a new measurement.

## Gates

- **Ratchets only move one way.** Baselines under `quality/` and the floors in `scripts/dev.py` and
  `pyproject.toml` may be raised, never lowered. The Claude hook blocks hand edits to
  `quality/*.json` and `poetry.lock`; regenerate them with their `--update` command or
  `poetry lock`.
- **Don't silence a checker to pass.** A new `# noqa` or `# pyright: ignore` spends a budget in
  `quality/escape_hatches.json`; fixing the finding is almost always the right move.
- **Tests stay offline.** Anything a test needs from the Hub is served from `tests/fixtures/` by
  `tests/offline_assets.py`.
