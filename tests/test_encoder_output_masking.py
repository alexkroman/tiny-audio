"""Tests for _time_mask_encoder_output — SpecAugment-style time masking
on encoder output features used for frozen-encoder projector training."""

import torch

from tiny_audio.asr_modeling import _time_mask_encoder_output


class TestTimeMaskEncoderOutput:
    def test_disabled_when_num_masks_zero(self):
        x = torch.randn(2, 50, 8)
        out = _time_mask_encoder_output(x, num_masks=0, max_width_ratio=0.1)
        torch.testing.assert_close(out, x)

    def test_disabled_when_max_width_ratio_zero(self):
        x = torch.randn(2, 50, 8)
        out = _time_mask_encoder_output(x, num_masks=5, max_width_ratio=0.0)
        torch.testing.assert_close(out, x)

    def test_preserves_shape_and_dtype(self):
        x = torch.randn(3, 40, 16, dtype=torch.float32)
        out = _time_mask_encoder_output(x, num_masks=5, max_width_ratio=0.05)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_masked_positions_are_zero_unmasked_match_input(self):
        torch.manual_seed(0)
        x = torch.full((1, 100, 4), 7.0)
        out = _time_mask_encoder_output(x, num_masks=5, max_width_ratio=0.05)
        # Every position is either the original value (7.0) or zero.
        unique = out[0, :, 0].unique()
        assert set(unique.tolist()).issubset({0.0, 7.0})

    def test_at_least_one_position_masked_with_nonzero_settings(self):
        torch.manual_seed(0)
        x = torch.full((1, 100, 4), 1.0)
        # 5 masks of up to 5 positions each almost certainly mask >=1.
        # Run a few seeds to make the assertion robust.
        masked = False
        for seed in range(10):
            torch.manual_seed(seed)
            out = _time_mask_encoder_output(x, num_masks=5, max_width_ratio=0.05)
            if (out[0, :, 0] == 0.0).any().item():
                masked = True
                break
        assert masked

    def test_independent_masking_per_sample(self):
        torch.manual_seed(0)
        x = torch.full((4, 60, 8), 1.0)
        out = _time_mask_encoder_output(x, num_masks=5, max_width_ratio=0.1)
        # Each sample's mask pattern is independent — collect the masked
        # index set per sample and verify at least two patterns differ.
        masked_sets = []
        for b in range(4):
            zeros = torch.nonzero(out[b, :, 0] == 0.0).flatten().tolist()
            masked_sets.append(tuple(zeros))
        assert len(set(masked_sets)) > 1

    def test_dtype_preserved_under_bf16(self):
        x = torch.randn(2, 30, 4, dtype=torch.bfloat16)
        out = _time_mask_encoder_output(x, num_masks=3, max_width_ratio=0.1)
        assert out.dtype == torch.bfloat16

    def test_no_out_of_bounds_for_short_sequences(self):
        # If max_width_ratio rounds down to 0, the helper bumps it to 1 so
        # the mask is non-degenerate; verify the index math doesn't slice
        # out-of-bounds when time_len is very small.
        x = torch.randn(2, 4, 3)
        out = _time_mask_encoder_output(x, num_masks=5, max_width_ratio=0.05)
        assert out.shape == x.shape
