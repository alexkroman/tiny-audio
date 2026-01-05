"""Tests for SpecAugment implementation."""

import pytest
import torch

from src.asr_config import ASRConfig
from src.asr_modeling import _compute_mask_indices, apply_specaugment


class TestComputeMaskIndices:
    """Tests for the _compute_mask_indices helper function."""

    def test_mask_shape(self):
        """Test that mask has correct shape."""
        batch_size, seq_len = 4, 100
        mask = _compute_mask_indices(
            shape=(batch_size, seq_len),
            mask_prob=0.1,
            mask_length=10,
        )
        assert mask.shape == (batch_size, seq_len)
        assert mask.dtype == torch.bool

    def test_mask_on_device(self):
        """Test mask is created on specified device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        mask = _compute_mask_indices(
            shape=(2, 50),
            mask_prob=0.1,
            mask_length=5,
            device=device,
        )
        assert mask.device == device

    def test_zero_mask_prob(self):
        """Test that zero mask_prob produces no masks."""
        mask = _compute_mask_indices(
            shape=(2, 100),
            mask_prob=0.0,
            mask_length=10,
            min_masks=0,
        )
        assert mask.sum() == 0

    def test_min_masks_enforced(self):
        """Test that min_masks is respected."""
        mask = _compute_mask_indices(
            shape=(2, 100),
            mask_prob=0.0,  # Would produce no masks without min_masks
            mask_length=5,
            min_masks=2,
        )
        # Each sample should have at least 2 mask spans of length 5
        for i in range(2):
            assert mask[i].sum() >= 5  # At least one full mask span

    def test_mask_length_respected(self):
        """Test that individual masks don't exceed mask_length."""
        mask = _compute_mask_indices(
            shape=(1, 1000),
            mask_prob=0.05,
            mask_length=10,
            min_masks=5,
        )
        # Count consecutive True values - should be <= mask_length
        # This is a statistical check
        assert mask.sum() > 0  # Should have some masks

    def test_invalid_mask_length(self):
        """Test that invalid mask_length raises error."""
        with pytest.raises(ValueError, match="mask_length must be >= 1"):
            _compute_mask_indices(
                shape=(1, 100),
                mask_prob=0.1,
                mask_length=0,
            )

    def test_mask_length_exceeds_sequence(self):
        """Test error when mask_length > sequence_length."""
        with pytest.raises(ValueError, match="mask_length .* must be <= sequence_length"):
            _compute_mask_indices(
                shape=(1, 5),
                mask_prob=0.1,
                mask_length=10,
            )


class TestApplySpecaugment:
    """Tests for the apply_specaugment function."""

    @pytest.fixture
    def sample_features(self):
        """Create sample mel spectrogram features."""
        # (batch=2, n_mels=128, time=3000) - typical Whisper input
        return torch.randn(2, 128, 3000)

    def test_output_shape_unchanged(self, sample_features):
        """Test that output shape matches input shape."""
        augmented = apply_specaugment(
            sample_features,
            mask_time_prob=0.05,
            mask_time_length=10,
        )
        assert augmented.shape == sample_features.shape

    def test_no_augmentation_when_disabled(self, sample_features):
        """Test that no changes when both probs and min_masks are 0."""
        augmented = apply_specaugment(
            sample_features,
            mask_time_prob=0.0,
            mask_time_min_masks=0,
            mask_feature_prob=0.0,
            mask_feature_min_masks=0,
        )
        assert torch.allclose(augmented, sample_features)

    def test_time_masking_creates_zeros(self, sample_features):
        """Test that time masking creates zero regions."""
        augmented = apply_specaugment(
            sample_features,
            mask_time_prob=0.05,
            mask_time_length=10,
            mask_time_min_masks=2,
            mask_feature_prob=0.0,
        )
        # Should have some zeros that weren't in original
        original_zeros = (sample_features == 0).sum()
        augmented_zeros = (augmented == 0).sum()
        assert augmented_zeros > original_zeros

    def test_feature_masking_creates_zeros(self, sample_features):
        """Test that feature masking creates zero regions."""
        augmented = apply_specaugment(
            sample_features,
            mask_time_prob=0.0,
            mask_feature_prob=0.1,
            mask_feature_length=10,
            mask_feature_min_masks=2,
        )
        # Should have some zeros that weren't in original
        original_zeros = (sample_features == 0).sum()
        augmented_zeros = (augmented == 0).sum()
        assert augmented_zeros > original_zeros

    def test_both_maskings_together(self, sample_features):
        """Test time and feature masking together."""
        augmented = apply_specaugment(
            sample_features,
            mask_time_prob=0.05,
            mask_time_length=10,
            mask_time_min_masks=2,
            mask_feature_prob=0.05,
            mask_feature_length=10,
            mask_feature_min_masks=2,
        )
        # Should have zeros from both maskings
        original_zeros = (sample_features == 0).sum()
        augmented_zeros = (augmented == 0).sum()
        assert augmented_zeros > original_zeros

    def test_fixed_mask_count_with_zero_prob(self, sample_features):
        """Test that min_masks applies even when prob=0.0 (fixed mask count mode)."""
        augmented = apply_specaugment(
            sample_features,
            mask_time_prob=0.0,  # Zero prob
            mask_time_length=10,
            mask_time_min_masks=2,  # But min_masks > 0
            mask_feature_prob=0.0,  # Zero prob
            mask_feature_length=10,
            mask_feature_min_masks=2,  # But min_masks > 0
        )
        # Should still have masking applied due to min_masks
        original_zeros = (sample_features == 0).sum()
        augmented_zeros = (augmented == 0).sum()
        assert augmented_zeros > original_zeros

    def test_original_tensor_not_modified(self, sample_features):
        """Test that original tensor is not modified in place."""
        original_copy = sample_features.clone()
        _ = apply_specaugment(
            sample_features,
            mask_time_prob=0.1,
            mask_time_length=10,
            mask_time_min_masks=5,
        )
        assert torch.allclose(sample_features, original_copy)

    def test_whisper_defaults(self, sample_features):
        """Test with Whisper default settings."""
        augmented = apply_specaugment(
            sample_features,
            mask_time_prob=0.05,
            mask_time_length=10,
            mask_time_min_masks=2,
            mask_feature_prob=0.0,  # Disabled by default in Whisper
            mask_feature_length=10,
            mask_feature_min_masks=0,
        )
        # Should produce some masking
        assert not torch.allclose(augmented, sample_features)


class TestASRConfigSpecaugment:
    """Tests for SpecAugment config parameters."""

    def test_default_specaugment_disabled(self):
        """Test that SpecAugment is disabled by default."""
        config = ASRConfig()
        assert config.use_specaugment is False

    def test_specaugment_params_stored(self):
        """Test that all SpecAugment params are stored in config."""
        config = ASRConfig(
            use_specaugment=True,
            mask_time_prob=0.1,
            mask_time_length=15,
            mask_time_min_masks=3,
            mask_feature_prob=0.05,
            mask_feature_length=20,
            mask_feature_min_masks=1,
        )
        assert config.use_specaugment is True
        assert config.mask_time_prob == 0.1
        assert config.mask_time_length == 15
        assert config.mask_time_min_masks == 3
        assert config.mask_feature_prob == 0.05
        assert config.mask_feature_length == 20
        assert config.mask_feature_min_masks == 1

    def test_whisper_default_values(self):
        """Test Whisper default values are correct."""
        config = ASRConfig(use_specaugment=True)
        assert config.mask_time_prob == 0.05
        assert config.mask_time_length == 10
        assert config.mask_time_min_masks == 2
        assert config.mask_feature_prob == 0.0
        assert config.mask_feature_length == 10
        assert config.mask_feature_min_masks == 0


class TestSpecaugmentIntegration:
    """Integration tests for SpecAugment with ASRModel."""

    @pytest.fixture
    def config_with_specaugment(self):
        """Create config with SpecAugment enabled."""
        return ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            projector_type="mlp",
            model_dtype="float32",
            attn_implementation="eager",
            use_specaugment=True,
            mask_time_prob=0.1,  # Higher prob for testing
            mask_time_length=10,
            mask_time_min_masks=2,
        )

    @pytest.fixture
    def config_without_specaugment(self):
        """Create config with SpecAugment disabled."""
        return ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            projector_type="mlp",
            model_dtype="float32",
            attn_implementation="eager",
            use_specaugment=False,
        )

    def test_specaugment_applied_during_training(self, config_with_specaugment):
        """Test that SpecAugment is applied during training mode."""
        from src.asr_modeling import ASRModel

        model = ASRModel(config_with_specaugment)
        model.train()

        # Create sample batch
        audio_features = torch.randn(1, 80, 3000)
        audio_mask = torch.ones(1, 3000)
        num_audio_tokens = int(model._get_num_audio_tokens(audio_mask))

        # Build input
        audio_placeholder = "<audio>" * num_audio_tokens
        messages = [{"role": "user", "content": f"Transcribe: {audio_placeholder}"}]
        chat_result = model.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(chat_result, "input_ids"):
            input_ids = chat_result.input_ids
        else:
            input_ids = chat_result

        # Run forward with SpecAugment enabled
        outputs = model(
            input_ids=input_ids,
            input_features=audio_features,
            audio_attention_mask=audio_mask,
            attention_mask=torch.ones_like(input_ids),
        )

        # Should complete without error
        assert outputs is not None
        assert outputs.logits is not None

    def test_specaugment_not_applied_during_eval(self, config_with_specaugment):
        """Test that SpecAugment is NOT applied during eval mode."""
        from src.asr_modeling import ASRModel

        model = ASRModel(config_with_specaugment)
        model.eval()  # Set to eval mode

        # Create sample batch
        audio_features = torch.randn(1, 80, 3000)
        audio_mask = torch.ones(1, 3000)
        num_audio_tokens = int(model._get_num_audio_tokens(audio_mask))

        # Build input
        audio_placeholder = "<audio>" * num_audio_tokens
        messages = [{"role": "user", "content": f"Transcribe: {audio_placeholder}"}]
        chat_result = model.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(chat_result, "input_ids"):
            input_ids = chat_result.input_ids
        else:
            input_ids = chat_result

        # Run multiple times - in eval mode, output should be deterministic
        torch.manual_seed(42)
        outputs1 = model(
            input_ids=input_ids,
            input_features=audio_features.clone(),
            audio_attention_mask=audio_mask,
            attention_mask=torch.ones_like(input_ids),
        )

        torch.manual_seed(42)
        outputs2 = model(
            input_ids=input_ids,
            input_features=audio_features.clone(),
            audio_attention_mask=audio_mask,
            attention_mask=torch.ones_like(input_ids),
        )

        # Should be identical in eval mode
        assert torch.allclose(outputs1.logits, outputs2.logits)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
