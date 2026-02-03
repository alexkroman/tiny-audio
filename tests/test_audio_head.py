"""Tests for the CosyVoice-based Audio Head module with distillation training."""

import pytest

from tiny_audio.audio_head import AudioHead


class MockAudioHeadConfig:
    """Mock config for AudioHead initialization in tests."""

    def __init__(self, **kwargs):
        self.llm_dim = kwargs.get("llm_dim", 512)
        self.audio_head_hidden_dim = kwargs.get("audio_head_hidden_dim", 256)
        self.freeze_cosy_llm = kwargs.get("freeze_cosy_llm", True)
        self.distillation_loss_weight = kwargs.get("distillation_loss_weight", 1.0)


@pytest.fixture
def audio_head_config():
    """Factory fixture for creating audio head configs."""
    return MockAudioHeadConfig


class TestAudioHeadInit:
    """Tests for AudioHead initialization."""

    def test_default_init(self, audio_head_config):
        """Test AudioHead initializes with default config."""
        config = audio_head_config()
        head = AudioHead(config)

        assert head.llm_dim == 512
        assert head.hidden_dim == 256
        assert head.distillation_loss_weight == 1.0

        # Lazy loading - model should not be loaded yet
        assert head._cosy_llm is None
        assert head._bridge is None

    def test_custom_dimensions(self, audio_head_config):
        """Test AudioHead with custom dimensions."""
        config = audio_head_config(
            llm_dim=1536,
            audio_head_hidden_dim=512,
            distillation_loss_weight=0.5,
        )
        head = AudioHead(config)

        assert head.llm_dim == 1536
        assert head.hidden_dim == 512
        assert head.distillation_loss_weight == 0.5


class TestAudioHeadOutputLength:
    """Tests for get_output_length method."""

    def test_output_length_estimate(self, audio_head_config):
        """Test output length estimation."""
        config = audio_head_config()
        head = AudioHead(config)

        # Default multiplier is 5x
        assert head.get_output_length(100) == 500
        assert head.get_output_length(10) == 50

    def test_output_length_scaling(self, audio_head_config):
        """Test output length scales linearly with input."""
        config = audio_head_config()
        head = AudioHead(config)

        len_1 = head.get_output_length(100)
        len_2 = head.get_output_length(200)

        assert len_2 == 2 * len_1


class TestAudioHeadStateDict:
    """Tests for state dict handling (saves only bridge weights)."""

    def test_state_dict_empty_before_load(self, audio_head_config):
        """Test that state_dict is empty before model is loaded."""
        config = audio_head_config()
        head = AudioHead(config)

        state = head.state_dict()
        assert state == {}

    def test_state_dict_prefix(self, audio_head_config):
        """Test that state_dict keys have 'bridge.' prefix.

        Note: Actual key verification is done in integration tests
        since it requires loading the CosyVoice model.
        """
        # This test validates that AudioHead can be instantiated
        # Key prefix verification requires integration tests with model loading
        _ = AudioHead(audio_head_config())
