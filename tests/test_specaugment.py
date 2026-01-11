"""Tests for SpecAugment implementation."""

import pytest
import torch

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import apply_specaugment


class TestApplySpecaugment:
    """Tests for the apply_specaugment function."""

    @pytest.fixture
    def sample_features(self):
        """Create sample mel spectrogram features."""
        return torch.randn(2, 128, 3000)

    def test_output_shape_unchanged(self, sample_features):
        """Test that output shape matches input shape."""
        augmented = apply_specaugment(sample_features, num_time_masks=2)
        assert augmented.shape == sample_features.shape

    def test_no_augmentation_when_disabled(self, sample_features):
        """Test that no changes when masks are 0."""
        augmented = apply_specaugment(sample_features, num_time_masks=0, num_freq_masks=0)
        assert torch.allclose(augmented, sample_features)

    def test_time_masking_creates_zeros(self, sample_features):
        """Test that time masking creates zero regions."""
        augmented = apply_specaugment(sample_features, num_time_masks=2, num_freq_masks=0)
        original_zeros = (sample_features == 0).sum()
        augmented_zeros = (augmented == 0).sum()
        assert augmented_zeros > original_zeros

    def test_freq_masking_creates_zeros(self, sample_features):
        """Test that frequency masking creates zero regions."""
        augmented = apply_specaugment(sample_features, num_time_masks=0, num_freq_masks=2)
        original_zeros = (sample_features == 0).sum()
        augmented_zeros = (augmented == 0).sum()
        assert augmented_zeros > original_zeros

    def test_both_maskings_together(self, sample_features):
        """Test time and frequency masking together."""
        augmented = apply_specaugment(sample_features, num_time_masks=2, num_freq_masks=2)
        original_zeros = (sample_features == 0).sum()
        augmented_zeros = (augmented == 0).sum()
        assert augmented_zeros > original_zeros


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
            num_time_masks=3,
            time_mask_length=15,
            num_freq_masks=2,
            freq_mask_length=20,
        )
        assert config.use_specaugment is True
        assert config.num_time_masks == 3
        assert config.time_mask_length == 15
        assert config.num_freq_masks == 2
        assert config.freq_mask_length == 20

    def test_default_values(self):
        """Test default values are correct."""
        config = ASRConfig(use_specaugment=True)
        assert config.num_time_masks == 2
        assert config.time_mask_length == 10
        assert config.num_freq_masks == 0
        assert config.freq_mask_length == 10


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
            num_time_masks=2,
            time_mask_length=10,
        )

    @pytest.mark.slow
    def test_specaugment_applied_during_training(self, config_with_specaugment):
        """Test that SpecAugment is applied during training mode."""
        from tiny_audio.asr_modeling import ASRModel

        model = ASRModel(config_with_specaugment)
        model.train()

        audio_features = torch.randn(1, 80, 3000)
        audio_mask = torch.ones(1, 3000)
        num_audio_tokens = int(model._get_num_audio_tokens(audio_mask))

        audio_placeholder = "<audio>" * num_audio_tokens
        messages = [{"role": "user", "content": f"Transcribe: {audio_placeholder}"}]
        chat_result = model.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = chat_result.input_ids if hasattr(chat_result, "input_ids") else chat_result

        outputs = model(
            input_ids=input_ids,
            input_features=audio_features,
            audio_attention_mask=audio_mask,
            attention_mask=torch.ones_like(input_ids),
        )

        assert outputs is not None
        assert outputs.logits is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
