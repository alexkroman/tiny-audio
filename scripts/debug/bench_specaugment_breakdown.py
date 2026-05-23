"""Per-op breakdown of ASRModel._mask_input_features.

Splits the cost of (a) cloning the input tensor, (b) sampling time masks,
(c) sampling feature masks, (d) the two masked_fill_ ops. Helps tell whether
the sampler itself or the bulk-tensor ops dominate.
"""

# Debug benchmark — tolerate ML tensor-axis naming (B/M/T), and lambdas that
# capture loop-local vars (they're invoked inline by time_op before the loop
# moves on, so the late-binding warning is a false positive here).
# ruff: noqa: N806, B023, RET504

from __future__ import annotations

import time

import torch

MASK_TIME_PROB = 0.05
MASK_TIME_LENGTH = 10
MASK_TIME_MIN = 2
MASK_FEATURE_PROB = 0.05
MASK_FEATURE_LENGTH = 10
MASK_FEATURE_MIN = 0


def sample_mask(
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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_op(fn, device, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    shape = (100, 80, 3000)
    B, M, T = shape

    for device in devices:
        print(f"\n=== device={device}, shape=B{B} M{M} T{T} ===")
        x = torch.randn(*shape, device=device)

        t_clone = time_op(lambda: x.clone(), device)

        t_sample_time = time_op(
            lambda: sample_mask(B, T, MASK_TIME_PROB, MASK_TIME_LENGTH, MASK_TIME_MIN, device),
            device,
        )

        t_sample_feat = time_op(
            lambda: sample_mask(
                B, M, MASK_FEATURE_PROB, MASK_FEATURE_LENGTH, MASK_FEATURE_MIN, device
            ),
            device,
        )

        # Pre-compute masks so masked_fill_ is timed in isolation.
        mask_time = sample_mask(
            B, T, MASK_TIME_PROB, MASK_TIME_LENGTH, MASK_TIME_MIN, device
        ).unsqueeze(1)
        mask_feat = sample_mask(
            B, M, MASK_FEATURE_PROB, MASK_FEATURE_LENGTH, MASK_FEATURE_MIN, device
        ).unsqueeze(-1)

        def fill_time():
            y = x.clone()
            y.masked_fill_(mask_time, 0)

        def fill_both():
            y = x.clone()
            y.masked_fill_(mask_time, 0)
            y.masked_fill_(mask_feat, 0)

        def fill_combined_where():
            # Alternative: build union mask once, single pass.
            y = torch.where(mask_time | mask_feat, x.new_zeros(()), x)
            return y

        def fill_no_clone():
            x.masked_fill(mask_time, 0).masked_fill_(mask_feat, 0)

        t_fill_time_only = time_op(fill_time, device)
        t_fill_both = time_op(fill_both, device)
        t_where = time_op(fill_combined_where, device)
        t_no_clone = time_op(fill_no_clone, device)

        print(f"  clone (B*M*T fp32 = {B * M * T * 4 / 1e6:.1f} MB):  {t_clone:7.3f} ms")
        print(f"  sample_mask (time, T={T}):              {t_sample_time:7.3f} ms")
        print(f"  sample_mask (feat, M={M}):              {t_sample_feat:7.3f} ms")
        print(f"  clone + masked_fill_(time):             {t_fill_time_only:7.3f} ms")
        print(f"  clone + masked_fill_(time)+(feat):      {t_fill_both:7.3f} ms")
        print(f"  where(union_mask, 0, x) (single pass):  {t_where:7.3f} ms")
        print(f"  out-of-place + in-place (no clone):     {t_no_clone:7.3f} ms")
        total = t_clone + t_sample_time + t_sample_feat + (t_fill_both - t_clone)
        print(f"  sum-of-parts estimate:                  {total:7.3f} ms")


if __name__ == "__main__":
    main()
