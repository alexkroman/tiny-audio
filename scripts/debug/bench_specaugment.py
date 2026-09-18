"""Benchmark the SpecAugment implementation in ASRModel._mask_input_features.

Compares three implementations at realistic training shapes:
  1. current  — the (B, N, T) broadcast-and-reduce sampler in asr_modeling.py
  2. scatter  — sample starts, scatter span markers, cumulative-sum to a mask
  3. upstream — transformers.models.whisper.modeling_whisper._compute_mask_indices
                via numpy (the previous implementation, kept for reference)

Reports median wall-clock per call and the peak intermediate tensor size for
each. Run with: poetry run python scripts/debug/bench_specaugment.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Shape:
    batch: int
    n_mels: int
    mel_len: int


# Realistic shapes from configs/:
#   encoder_train.yaml: per_device_train_batch_size=69, whisper-medium.en (80 mels)
#   production.yaml:    per_device_train_batch_size=100
# Mel length 3000 = 30s @ 100 fps, which is the Whisper feature extractor default.
SHAPES = [
    Shape(batch=8, n_mels=80, mel_len=3000),
    Shape(batch=32, n_mels=80, mel_len=3000),
    Shape(batch=69, n_mels=80, mel_len=3000),
    Shape(batch=100, n_mels=80, mel_len=3000),
    Shape(batch=100, n_mels=128, mel_len=3000),
]

MASK_TIME_PROB = 0.05
MASK_TIME_LENGTH = 10
MASK_TIME_MIN = 2
MASK_FEATURE_PROB = 0.05  # benchmark with both axes active
MASK_FEATURE_LENGTH = 10
MASK_FEATURE_MIN = 0


# --- Implementation 1: current (broadcast-and-reduce) -----------------------


def sample_mask_current(
    batch_size: int,
    axis_length: int,
    mask_prob: float,
    mask_length: int,
    min_masks: int,
    device: torch.device,
) -> torch.Tensor:
    num_masked_spans = max(int(mask_prob * axis_length / mask_length + 0.5), min_masks)
    if num_masked_spans == 0:
        return torch.zeros(batch_size, axis_length, device=device, dtype=torch.bool)
    max_start = max(axis_length - mask_length + 1, 1)
    starts = torch.randint(0, max_start, (batch_size, num_masked_spans), device=device)
    positions = torch.arange(axis_length, device=device).view(1, 1, -1)
    starts_b = starts.unsqueeze(-1)
    span_mask = (positions >= starts_b) & (positions < starts_b + mask_length)
    return span_mask.any(dim=1)


# --- Implementation 2: scatter + cumsum -------------------------------------
#
# Idea: instead of materializing (B, N, T), only mark span start / end deltas
# in a (B, T) int8 buffer and use cumsum to fill the interior. Peak memory is
# O(B * T) regardless of how many spans we sample.


def sample_mask_scatter(
    batch_size: int,
    axis_length: int,
    mask_prob: float,
    mask_length: int,
    min_masks: int,
    device: torch.device,
) -> torch.Tensor:
    num_masked_spans = max(int(mask_prob * axis_length / mask_length + 0.5), min_masks)
    if num_masked_spans == 0:
        return torch.zeros(batch_size, axis_length, device=device, dtype=torch.bool)
    max_start = max(axis_length - mask_length + 1, 1)
    starts = torch.randint(0, max_start, (batch_size, num_masked_spans), device=device)
    # Buffer with +1 slot for end markers that land at axis_length.
    deltas = torch.zeros(batch_size, axis_length + 1, device=device, dtype=torch.int16)
    # +1 at start, -1 at start+length; multiple spans accumulate via scatter_add.
    ones = torch.ones_like(starts, dtype=torch.int16)
    deltas.scatter_add_(1, starts, ones)
    ends = (starts + mask_length).clamp_max_(axis_length)
    deltas.scatter_add_(1, ends, -ones)
    coverage = deltas[:, :axis_length].cumsum(dim=1)
    return coverage > 0


# --- Implementation 3: upstream numpy (Whisper reference) -------------------


def sample_mask_upstream(
    batch_size: int,
    axis_length: int,
    mask_prob: float,
    mask_length: int,
    min_masks: int,
    device: torch.device,
) -> torch.Tensor:
    # Direct translation of transformers _compute_mask_indices, no attention_mask.
    epsilon = float(np.random.rand(1).item())
    num_masked_span = int(mask_prob * axis_length / mask_length + epsilon)
    num_masked_span = max(num_masked_span, min_masks)
    if num_masked_span * mask_length > axis_length:
        num_masked_span = axis_length // mask_length
    spec_aug_mask = np.zeros((batch_size, axis_length), dtype=bool)
    if num_masked_span == 0:
        return torch.from_numpy(spec_aug_mask).to(device)
    spec_aug_mask_idxs = []
    max_start = axis_length - mask_length
    for _ in range(batch_size):
        idxs = np.random.choice(max_start, num_masked_span, replace=False)
        idxs = np.broadcast_to(idxs[:, None], (num_masked_span, mask_length))
        offsets = np.arange(mask_length)[None, :]
        idxs = (idxs + offsets).reshape(-1)
        spec_aug_mask_idxs.append(idxs)
    spec_aug_mask_idxs = np.stack(spec_aug_mask_idxs)
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, axis=-1)
    return torch.from_numpy(spec_aug_mask).to(device)


# --- Full _mask_input_features wrappers -------------------------------------


def mask_features(sampler, x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    batch_size, hidden_size, sequence_length = x.size()
    device = x.device
    mt = sampler(
        batch_size,
        sequence_length,
        MASK_TIME_PROB,
        MASK_TIME_LENGTH,
        MASK_TIME_MIN,
        device,
    )
    x.masked_fill_(mt.unsqueeze(1), 0)
    mf = sampler(
        batch_size,
        hidden_size,
        MASK_FEATURE_PROB,
        MASK_FEATURE_LENGTH,
        MASK_FEATURE_MIN,
        device,
    )
    x.masked_fill_(mf.unsqueeze(-1), 0)
    return x


# --- Benchmark loop ---------------------------------------------------------


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench_one(sampler, x: torch.Tensor, iters: int) -> tuple[float, float]:
    device = x.device
    # Warm-up — first call triggers kernel compile / autotune on accelerators.
    for _ in range(5):
        mask_features(sampler, x)
    sync(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mask_features(sampler, x)
        sync(device)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    median = samples[len(samples) // 2]
    p90 = samples[int(len(samples) * 0.9)]
    return median * 1e3, p90 * 1e3  # ms


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    iters = 50
    for device in devices:
        print(f"\n=== device={device} ===")
        print(
            f"{'shape':<24} {'current(ms)':>14} {'scatter(ms)':>14} {'upstream(ms)':>14}"
            f" {'speedup_scatter':>16}"
        )
        for shape in SHAPES:
            torch.manual_seed(0)
            np.random.seed(0)
            x = torch.randn(shape.batch, shape.n_mels, shape.mel_len, device=device)

            cur_med, _ = bench_one(sample_mask_current, x, iters)
            sca_med, _ = bench_one(sample_mask_scatter, x, iters)
            ups_med, _ = bench_one(sample_mask_upstream, x, iters)

            speedup = cur_med / sca_med if sca_med > 0 else float("inf")
            label = f"B{shape.batch} M{shape.n_mels} T{shape.mel_len}"
            print(
                f"{label:<24} {cur_med:>14.3f} {sca_med:>14.3f} {ups_med:>14.3f} {speedup:>15.2f}x"
            )

    print("\nIntermediate sizes for largest shape (B=100, T=3000, mask_len=10):")
    print("  current  : (B, N, T) bool = 100 * ~15 * 3000 = 4.50 MB")
    print("  scatter  : (B, T+1) int16  = 100 * 3001 * 2  = 0.60 MB")
    print("  upstream : numpy (B, N*L) int64 = 100 * ~15*10 * 8 = 0.12 MB")
    print("            + H2D copy of bool mask per call (sync stall)")


if __name__ == "__main__":
    main()
