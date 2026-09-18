"""Tests for SpecAugment mask sampling (ASRModel._mask_input_features).

The regression these guard: mask spans used to be sampled uniformly over the
*padded* time axis, ignoring attention_mask. Under `padding="longest"` that
gave a short clip batched with a long one almost no real augmentation, since
most of its spans landed in padding.
"""

from types import SimpleNamespace

import pytest
import torch

from tiny_audio.asr_modeling import ASRModel


def make_masker(**config_fields):
    """Minimal stand-in exposing just what _mask_input_features touches."""
    defaults = {
        "audio_features_time_major": False,
        "mask_time_prob": 0.05,
        "mask_time_length": 10,
        "mask_time_min_masks": 2,
        "mask_feature_prob": 0.0,
        "mask_feature_length": 10,
        "mask_feature_min_masks": 0,
    }
    defaults.update(config_fields)
    stub = SimpleNamespace(config=SimpleNamespace(**defaults))
    stub._sample_mask_indices = ASRModel._sample_mask_indices
    return stub


class TestSampleMaskIndices:
    """_sample_mask_indices: shape, span count, and valid-length confinement."""

    def test_shape_and_dtype(self):
        mask = ASRModel._sample_mask_indices(
            batch_size=3,
            axis_length=200,
            mask_prob=0.05,
            mask_length=10,
            min_masks=2,
            device=torch.device("cpu"),
        )
        assert mask.shape == (3, 200)
        assert mask.dtype == torch.bool

    def test_no_valid_lengths_spans_full_axis(self):
        """Without valid_lengths the whole axis is fair game."""
        mask = ASRModel._sample_mask_indices(
            batch_size=1,
            axis_length=1000,
            mask_prob=0.5,
            mask_length=10,
            min_masks=2,
            device=torch.device("cpu"),
        )
        # 50 spans of 10 over 1000 frames: the tail cannot stay untouched
        # across the whole axis.
        assert mask[0, 500:].any()

    def test_valid_lengths_confine_spans(self):
        """Every masked position must sit inside the sample's real length."""
        valid = 100
        for _ in range(25):  # sampling is random; repeat to catch leakage
            mask = ASRModel._sample_mask_indices(
                batch_size=1,
                axis_length=3000,
                mask_prob=0.05,
                mask_length=10,
                min_masks=2,
                device=torch.device("cpu"),
                valid_lengths=torch.tensor([valid]),
            )
            assert not mask[0, valid:].any(), "span leaked into padding"
            assert mask[0, :valid].any(), "short sample got no augmentation"

    def test_span_count_scales_with_valid_length(self):
        """A long sample earns more spans than a short one in the same batch."""
        mask = ASRModel._sample_mask_indices(
            batch_size=2,
            axis_length=3000,
            mask_prob=0.05,
            mask_length=10,
            min_masks=2,
            device=torch.device("cpu"),
            valid_lengths=torch.tensor([100, 3000]),
        )
        # short: max(int(0.05*100/10 + 0.5), 2) = 2 spans -> <= 20 frames
        # long:  int(0.05*3000/10 + 0.5) = 15 spans -> ~150 frames
        assert mask[0].sum() <= 2 * 10
        assert mask[1].sum() > mask[0].sum()

    def test_min_masks_floor_applies_to_short_samples(self):
        """min_masks still holds when the proportional count rounds to zero."""
        mask = ASRModel._sample_mask_indices(
            batch_size=1,
            axis_length=3000,
            mask_prob=0.001,
            mask_length=10,
            min_masks=2,
            device=torch.device("cpu"),
            valid_lengths=torch.tensor([50]),
        )
        assert mask[0].any()

    def test_zero_spans_returns_empty_mask(self):
        mask = ASRModel._sample_mask_indices(
            batch_size=2,
            axis_length=100,
            mask_prob=0.0,
            mask_length=10,
            min_masks=0,
            device=torch.device("cpu"),
        )
        assert not mask.any()

    def test_valid_length_shorter_than_mask_length(self):
        """A clip shorter than one span still gets masked, without indexing off."""
        mask = ASRModel._sample_mask_indices(
            batch_size=1,
            axis_length=3000,
            mask_prob=0.05,
            mask_length=10,
            min_masks=2,
            device=torch.device("cpu"),
            valid_lengths=torch.tensor([4]),
        )
        assert mask[0, :10].any()


class TestMaskInputFeatures:
    """_mask_input_features: axis handling and attention_mask plumbing."""

    def test_attention_mask_confines_time_masks(self):
        """The padded-axis regression: a 1s clip batched with a 19s clip."""
        masker = make_masker()
        features = torch.ones(2, 80, 3000)
        attention_mask = torch.zeros(2, 3000)
        attention_mask[0, :100] = 1  # ~1s
        attention_mask[1, :1900] = 1  # ~19s

        out = ASRModel._mask_input_features(masker, features, attention_mask)

        zeroed = out == 0
        # Short sample: everything zeroed is inside its real 100 frames.
        assert not zeroed[0, :, 100:].any()
        # And it did get augmented rather than skipped.
        assert zeroed[0, :, :100].any()

    def test_without_attention_mask_uses_full_axis(self):
        masker = make_masker(mask_time_prob=0.5)
        features = torch.ones(1, 80, 1000)
        out = ASRModel._mask_input_features(masker, features, None)
        assert (out == 0).any()

    def test_mismatched_attention_mask_raises(self):
        """Masking the wrong axis is silent, so a shape mismatch must be loud."""
        masker = make_masker()
        features = torch.ones(2, 80, 3000)
        bad_mask = torch.ones(2, 1500)
        with pytest.raises(ValueError, match="does not match input_features"):
            ASRModel._mask_input_features(masker, features, bad_mask)

    def test_time_major_layout(self):
        """Conformer encoders hand us (batch, time, feature)."""
        masker = make_masker(audio_features_time_major=True)
        features = torch.ones(2, 400, 64)
        attention_mask = torch.zeros(2, 400)
        attention_mask[0, :50] = 1
        attention_mask[1, :400] = 1

        out = ASRModel._mask_input_features(masker, features, attention_mask)

        zeroed = out == 0
        assert not zeroed[0, 50:, :].any()
        assert zeroed[0, :50, :].any()

    def test_feature_axis_masking_is_independent_of_attention_mask(self):
        """Mel bins are never padded, so the feature axis uses the full range."""
        masker = make_masker(mask_time_prob=0.0, mask_feature_prob=0.5, mask_feature_length=4)
        features = torch.ones(1, 80, 3000)
        attention_mask = torch.zeros(1, 3000)
        attention_mask[0, :100] = 1

        out = ASRModel._mask_input_features(masker, features, attention_mask)

        zeroed = out == 0
        # A masked mel bin is zeroed across all time steps, including padding.
        assert zeroed[0, :, 2999].any()

    def test_input_is_not_mutated(self):
        masker = make_masker(mask_time_prob=0.5)
        features = torch.ones(1, 80, 1000)
        original = features.clone()
        ASRModel._mask_input_features(masker, features, None)
        assert torch.equal(features, original)
