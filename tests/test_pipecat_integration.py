"""Tests for Pipecat integration and streaming generation."""

import numpy as np
import pytest
import torch

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel

# Mark all tests in this module as slow (load ML models)
pytestmark = pytest.mark.slow


# Use session-scoped fixtures from conftest.py
@pytest.fixture
def config(base_asr_config):
    """Alias for session-scoped base_asr_config."""
    return base_asr_config


@pytest.fixture
def model(base_asr_model):
    """Alias for session-scoped base_asr_model."""
    return base_asr_model


@pytest.fixture
def audio_input(model):
    """Create audio input for testing."""
    # Whisper expects 3000 mel frames
    audio_features = torch.randn(1, 80, 3000)
    audio_mask = torch.ones(1, 3000)
    return audio_features, audio_mask


class TestStreamingGeneration:
    """Tests for generate_streaming method."""

    def test_generate_streaming_yields_tokens(self, model, audio_input):
        """Test that generate_streaming yields partial text."""
        audio_features, audio_mask = audio_input

        tokens = list(
            model.generate_streaming(
                input_features=audio_features,
                audio_attention_mask=audio_mask,
                max_new_tokens=10,
            )
        )

        # Should yield at least one token
        assert len(tokens) > 0
        # Each yielded item should be a string
        for token in tokens:
            assert isinstance(token, str)

    def test_generate_streaming_same_output_as_generate(self, model, audio_input):
        """Test that streaming and non-streaming produce same final text."""
        audio_features, audio_mask = audio_input

        # Non-streaming
        output_ids = model.generate(
            input_features=audio_features,
            audio_attention_mask=audio_mask,
            max_new_tokens=20,
        )
        non_streaming_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Streaming
        streaming_tokens = list(
            model.generate_streaming(
                input_features=audio_features,
                audio_attention_mask=audio_mask,
                max_new_tokens=20,
            )
        )
        streaming_text = "".join(streaming_tokens)

        # Should produce the same text
        assert streaming_text.strip() == non_streaming_text.strip()

    def test_generate_streaming_is_iterator(self, model, audio_input):
        """Test that generate_streaming returns an iterator."""
        audio_features, audio_mask = audio_input

        result = model.generate_streaming(
            input_features=audio_features,
            audio_attention_mask=audio_mask,
            max_new_tokens=5,
        )

        # Should be an iterator/generator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_generate_streaming_respects_max_tokens(self, model, audio_input):
        """Test that streaming respects max_new_tokens."""
        audio_features, audio_mask = audio_input

        # Generate with very limited tokens
        tokens = list(
            model.generate_streaming(
                input_features=audio_features,
                audio_attention_mask=audio_mask,
                max_new_tokens=3,
            )
        )

        # Join and check token count (approximate since tokenizer may chunk differently)
        full_text = "".join(tokens)
        # Re-tokenize to count
        token_ids = model.tokenizer.encode(full_text, add_special_tokens=False)
        # Should be at most max_new_tokens (might be less due to EOS)
        assert len(token_ids) <= 5  # Allow some slack for tokenizer differences


class TestAudioConversion:
    """Tests for audio byte conversion used in Pipecat adapter."""

    def test_int16_to_float32_conversion(self):
        """Test that 16-bit PCM converts correctly to float32."""
        # Create test audio as int16
        audio_int16 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        audio_bytes = audio_int16.tobytes()

        # Convert as the adapter does
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Check values
        assert audio_float[0] == 0.0  # Zero stays zero
        assert abs(audio_float[1] - 0.5) < 0.01  # 16384/32768 ≈ 0.5
        assert abs(audio_float[2] + 0.5) < 0.01  # -16384/32768 ≈ -0.5
        assert abs(audio_float[3] - 1.0) < 0.01  # 32767/32768 ≈ 1.0
        assert abs(audio_float[4] + 1.0) < 0.01  # -32768/32768 = -1.0

    def test_empty_audio_conversion(self):
        """Test that empty audio converts without error."""
        audio_bytes = b""
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(audio_float) == 0

    def test_audio_dtype_is_float32(self):
        """Test that converted audio is float32."""
        audio_int16 = np.array([100, 200, 300], dtype=np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        assert audio_float.dtype == np.float32


class TestPipecatSTTServiceImport:
    """Tests for Pipecat STT service import behavior."""

    def test_integrations_module_imports(self):
        """Test that integrations module can be imported."""
        from tiny_audio import integrations

        assert hasattr(integrations, "TinyAudioSTTService")

    def test_pipecat_stt_available_without_pipecat(self):
        """Test that TinyAudioSTTService is None when pipecat not installed."""
        # This tests the graceful degradation in __init__.py
        from tiny_audio.integrations import TinyAudioSTTService

        # If pipecat is not installed, this should be None
        # If pipecat is installed, this should be the class
        # Either way, import should not fail
        assert TinyAudioSTTService is None or callable(TinyAudioSTTService)


class TestFeatureExtractorIntegration:
    """Tests for feature extractor usage pattern in STT adapter."""

    def test_feature_extractor_accepts_numpy_array(self, model):
        """Test that feature extractor accepts numpy arrays."""
        # Create audio as numpy array (like the adapter does)
        audio_array = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz

        inputs = model.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )

        assert "input_features" in inputs
        assert "attention_mask" in inputs

    def test_feature_extractor_output_shape(self, model):
        """Test feature extractor output shape."""
        audio_array = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds

        inputs = model.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Should be (batch, n_mels, mel_frames)
        assert len(inputs.input_features.shape) == 3
        assert inputs.input_features.shape[0] == 1  # batch size
        assert inputs.input_features.shape[1] == 80  # n_mels for whisper-tiny


class TestModelLazyLoading:
    """Tests for lazy model loading pattern used in adapter."""

    def test_model_can_be_loaded_after_init(self):
        """Test that model can be loaded lazily."""
        # Simulate lazy loading pattern
        model = None

        def ensure_model():
            nonlocal model
            if model is None:
                config = ASRConfig(
                    audio_model_id="openai/whisper-tiny",
                    text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
                    projector_type="mlp",
                    model_dtype="float32",
                    attn_implementation="eager",
                )
                model = ASRModel(config)
            return model

        # First call loads model
        m1 = ensure_model()
        assert m1 is not None

        # Second call returns same instance
        m2 = ensure_model()
        assert m1 is m2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
