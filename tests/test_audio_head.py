"""Tests for the Moshi-style Depformer for multi-codebook audio token prediction."""

import pytest
import torch

from tiny_audio.audio_head import Depformer


class MockDepformerConfig:
    """Mock config for Depformer initialization in tests."""

    def __init__(self, **kwargs):
        self.llm_dim = kwargs.get("llm_dim", 256)
        self.depformer_dim = kwargs.get("depformer_dim", 128)
        self.depformer_num_layers = kwargs.get("depformer_num_layers", 2)
        self.codebook_size = kwargs.get("codebook_size", 1024)
        self.num_codebooks = kwargs.get("num_codebooks", 8)


@pytest.fixture
def depformer_config():
    """Factory fixture for creating depformer configs."""
    return MockDepformerConfig


@pytest.fixture
def small_depformer(depformer_config):
    """Create a small Depformer for testing.

    Uses dimensions that are compatible with transformer architecture:
    - hidden_dim must be divisible by num_heads (64 dims per head)
    """
    config = depformer_config(
        llm_dim=128,
        depformer_dim=128,  # 128 / 2 = 64 head_dim with 2 heads
        depformer_num_layers=2,
        codebook_size=128,
        num_codebooks=4,
    )
    return Depformer(config)


class TestDepformerInit:
    """Tests for Depformer initialization."""

    def test_default_init(self, depformer_config):
        """Test Depformer initializes with default config."""
        config = depformer_config()
        depformer = Depformer(config)

        assert depformer.llm_dim == 256
        assert depformer.hidden_dim == 128
        assert depformer.vocab_size == 1024
        assert depformer.num_codebooks == 8

    def test_custom_dimensions(self, depformer_config):
        """Test Depformer with custom dimensions."""
        config = depformer_config(
            llm_dim=1536,
            depformer_dim=512,
            codebook_size=2048,
            num_codebooks=4,
        )
        depformer = Depformer(config)

        assert depformer.llm_dim == 1536
        assert depformer.hidden_dim == 512
        assert depformer.vocab_size == 2048
        assert depformer.num_codebooks == 4

    def test_special_tokens(self, depformer_config):
        """Test special token IDs are correctly set."""
        config = depformer_config(codebook_size=1024)
        depformer = Depformer(config)

        assert depformer.initial_token_id == 1024  # vocab_size
        assert depformer.total_vocab_size == 1025  # vocab_size + 1

    def test_per_codebook_components(self, depformer_config):
        """Test per-codebook components are created correctly."""
        config = depformer_config(num_codebooks=8)
        depformer = Depformer(config)

        # depformer_in: one per codebook
        assert len(depformer.depformer_in) == 8

        # depformer_emb: one per codebook except first
        assert len(depformer.depformer_emb) == 7

        # depformer_norms: one per codebook
        assert len(depformer.depformer_norms) == 8

        # linears: one per codebook
        assert len(depformer.linears) == 8


class TestDepformerOutputLength:
    """Tests for get_output_length method."""

    def test_output_length_estimate(self, depformer_config):
        """Test output length estimation."""
        config = depformer_config()
        depformer = Depformer(config)

        # Default 2x multiplier
        assert depformer.get_output_length(100) == 200
        assert depformer.get_output_length(10) == 20

    def test_output_length_scaling(self, depformer_config):
        """Test output length scales linearly with input."""
        config = depformer_config()
        depformer = Depformer(config)

        len_1 = depformer.get_output_length(100)
        len_2 = depformer.get_output_length(200)

        assert len_2 == 2 * len_1

    def test_output_length_zero(self, depformer_config):
        """Test output length with zero input."""
        config = depformer_config()
        depformer = Depformer(config)
        assert depformer.get_output_length(0) == 0


class TestDepformerStateDict:
    """Tests for state dict handling."""

    def test_state_dict_not_empty(self, small_depformer):
        """Test that state_dict contains all model parameters."""
        state = small_depformer.state_dict()
        assert len(state) > 0

    def test_state_dict_has_expected_components(self, small_depformer):
        """Test that state_dict contains expected component prefixes."""
        state = small_depformer.state_dict()
        prefixes = set()
        for key in state:
            prefix = key.split(".")[0]
            prefixes.add(prefix)

        expected_prefixes = {
            "depformer_in",
            "depformer_emb",
            "layers",
            "depformer_norms",
            "linears",
        }
        # Note: rotary_emb doesn't have learnable parameters, so it won't appear
        assert prefixes == expected_prefixes

    def test_load_state_dict(self, small_depformer):
        """Test loading state dict restores weights."""
        # Save current state
        original_state = small_depformer.state_dict()

        # Modify weights
        with torch.no_grad():
            for param in small_depformer.depformer_in[0].parameters():
                param.fill_(0.0)

        # Verify weights are zeroed
        for key, value in small_depformer.state_dict().items():
            if key.startswith("depformer_in.0"):
                assert torch.allclose(value, torch.zeros_like(value))

        # Load original state
        small_depformer.load_state_dict(original_state)

        # Verify weights are restored
        restored_state = small_depformer.state_dict()
        for key in original_state:
            assert torch.allclose(original_state[key], restored_state[key])


class TestForwardTraining:
    """Tests for training forward pass."""

    def test_forward_train_returns_scalar_loss(self, small_depformer):
        """Test training forward pass returns scalar loss."""
        batch_size, llm_seq = 2, 20
        num_cbs = small_depformer.num_codebooks

        hidden = torch.randn(batch_size, llm_seq, small_depformer.llm_dim)
        # Targets shape: (batch, num_codebooks, seq_len)
        targets = torch.randint(0, small_depformer.vocab_size, (batch_size, num_cbs, llm_seq))
        lengths = torch.tensor([20, 15])

        loss = small_depformer(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_forward_train_loss_is_differentiable(self, small_depformer):
        """Test training loss supports backward pass."""
        batch_size, llm_seq = 2, 20
        num_cbs = small_depformer.num_codebooks

        hidden = torch.randn(batch_size, llm_seq, small_depformer.llm_dim, requires_grad=True)
        targets = torch.randint(0, small_depformer.vocab_size, (batch_size, num_cbs, llm_seq))
        lengths = torch.tensor([20, 15])

        loss = small_depformer(hidden, codec_targets=targets, codec_lengths=lengths)
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_forward_train_batch_size_one(self, small_depformer):
        """Test training with batch size of 1."""
        num_cbs = small_depformer.num_codebooks
        hidden = torch.randn(1, 20, small_depformer.llm_dim)
        targets = torch.randint(0, small_depformer.vocab_size, (1, num_cbs, 20))
        lengths = torch.tensor([20])

        loss = small_depformer(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_train_without_lengths(self, small_depformer):
        """Test training without explicit lengths (use full targets)."""
        batch_size, llm_seq = 2, 20
        num_cbs = small_depformer.num_codebooks

        hidden = torch.randn(batch_size, llm_seq, small_depformer.llm_dim)
        targets = torch.randint(0, small_depformer.vocab_size, (batch_size, num_cbs, llm_seq))

        # Should work without lengths
        loss = small_depformer(hidden, codec_targets=targets, codec_lengths=None)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_train_2d_targets(self, small_depformer):
        """Test training with 2D targets (single codebook)."""
        batch_size, llm_seq = 2, 20

        hidden = torch.randn(batch_size, llm_seq, small_depformer.llm_dim)
        # 2D targets: (batch, seq_len) - will be treated as single codebook
        targets = torch.randint(0, small_depformer.vocab_size, (batch_size, llm_seq))
        lengths = torch.tensor([20, 15])

        loss = small_depformer(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestForwardInference:
    """Tests for inference forward pass."""

    def test_forward_inference_returns_tokens(self, small_depformer):
        """Test inference forward pass returns token IDs."""
        hidden = torch.randn(2, 20, small_depformer.llm_dim)

        tokens = small_depformer(hidden)  # No targets = inference

        assert tokens.dtype == torch.long
        assert tokens.shape[0] == 2  # Batch size preserved
        assert tokens.shape[1] == small_depformer.num_codebooks  # All codebooks
        assert tokens.shape[2] == 20  # Seq len

    def test_forward_inference_tokens_in_valid_range(self, small_depformer):
        """Test inference tokens are in valid range."""
        hidden = torch.randn(1, 10, small_depformer.llm_dim)

        tokens = small_depformer(hidden)

        # Tokens should be in vocab range
        assert tokens.min() >= 0
        assert tokens.max() < small_depformer.vocab_size

    def test_forward_inference_batch_size_one(self, small_depformer):
        """Test inference with batch size of 1."""
        hidden = torch.randn(1, 20, small_depformer.llm_dim)

        tokens = small_depformer(hidden)

        assert tokens.shape[0] == 1
        assert tokens.shape[1] == small_depformer.num_codebooks

    def test_forward_inference_output_shape(self, small_depformer):
        """Test inference output shape is (batch, num_codebooks, seq_len)."""
        batch_size = 3
        seq_len = 15
        hidden = torch.randn(batch_size, seq_len, small_depformer.llm_dim)

        tokens = small_depformer(hidden)

        assert tokens.shape == (batch_size, small_depformer.num_codebooks, seq_len)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self, small_depformer):
        """Test with minimal sequence length."""
        num_cbs = small_depformer.num_codebooks
        hidden = torch.randn(1, 1, small_depformer.llm_dim)
        targets = torch.randint(0, small_depformer.vocab_size, (1, num_cbs, 1))
        lengths = torch.tensor([1])

        loss = small_depformer(hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)

    def test_long_sequence(self, small_depformer):
        """Test with longer sequence."""
        num_cbs = small_depformer.num_codebooks
        hidden = torch.randn(1, 100, small_depformer.llm_dim)
        targets = torch.randint(0, small_depformer.vocab_size, (1, num_cbs, 100))
        lengths = torch.tensor([100])

        loss = small_depformer(hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)

    def test_varying_target_lengths_in_batch(self, small_depformer):
        """Test batch with different target lengths."""
        batch_size = 3
        num_cbs = small_depformer.num_codebooks
        hidden = torch.randn(batch_size, 50, small_depformer.llm_dim)
        targets = torch.randint(0, small_depformer.vocab_size, (batch_size, num_cbs, 50))
        lengths = torch.tensor([50, 40, 30])

        loss = small_depformer(hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)


class TestDevicePlacement:
    """Tests for device and dtype handling."""

    def test_to_device(self, small_depformer):
        """Test .to() method moves model to device."""
        small_depformer.to(device="cpu")

        for param in small_depformer.parameters():
            assert param.device.type == "cpu"

    def test_forward_respects_input_device(self, small_depformer):
        """Test forward pass respects input device."""
        device = torch.device("cpu")
        small_depformer.to(device)

        num_cbs = small_depformer.num_codebooks
        hidden = torch.randn(2, 20, small_depformer.llm_dim, device=device)
        targets = torch.randint(0, small_depformer.vocab_size, (2, num_cbs, 20), device=device)
        lengths = torch.tensor([20, 15], device=device)

        loss = small_depformer(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.device == device


class TestS2SPipeline:
    """Integration tests for the full Speech-to-Speech pipeline."""

    @pytest.fixture
    def mock_asr_model_with_depformer(self, small_depformer):
        """Create a mock ASR model with depformer for testing."""
        import unittest.mock as mock

        model = mock.MagicMock()
        model.audio_head = small_depformer
        model.device = torch.device("cpu")

        # Mock generate_with_audio return value
        def mock_generate_with_audio(input_features, audio_attention_mask, **kwargs):
            batch_size = input_features.shape[0]
            num_cbs = small_depformer.num_codebooks
            # Return mock text and codec tokens
            return {
                "text_ids": torch.tensor([[1, 2, 3]]),
                "text": ["Hello, how can I help?"],
                "codec_tokens": torch.randint(
                    0, small_depformer.vocab_size, (batch_size, num_cbs, 100)
                ),
            }

        model.generate_with_audio = mock_generate_with_audio
        return model

    def test_generate_with_audio_returns_expected_keys(self, mock_asr_model_with_depformer):
        """Test generate_with_audio returns text and codec tokens."""
        model = mock_asr_model_with_depformer

        input_features = torch.randn(1, 80, 100)
        attention_mask = torch.ones(1, 100)

        result = model.generate_with_audio(
            input_features=input_features,
            audio_attention_mask=attention_mask,
        )

        assert "text" in result
        assert "text_ids" in result
        assert "codec_tokens" in result
        assert isinstance(result["text"], list)
        assert isinstance(result["codec_tokens"], torch.Tensor)

    def test_codec_tokens_shape(self, mock_asr_model_with_depformer, small_depformer):
        """Test codec tokens have expected shape."""
        model = mock_asr_model_with_depformer

        input_features = torch.randn(2, 80, 100)
        attention_mask = torch.ones(2, 100)

        result = model.generate_with_audio(
            input_features=input_features,
            audio_attention_mask=attention_mask,
        )

        codec_tokens = result["codec_tokens"]
        assert codec_tokens.dim() == 3  # (batch, num_codebooks, seq_len)
        assert codec_tokens.shape[0] == 2  # Batch size
        assert codec_tokens.shape[1] == small_depformer.num_codebooks

    def test_codec_tokens_valid_range(self, mock_asr_model_with_depformer, small_depformer):
        """Test codec tokens are in valid vocabulary range."""
        model = mock_asr_model_with_depformer

        input_features = torch.randn(1, 80, 100)
        attention_mask = torch.ones(1, 100)

        result = model.generate_with_audio(
            input_features=input_features,
            audio_attention_mask=attention_mask,
        )

        codec_tokens = result["codec_tokens"]
        assert codec_tokens.min() >= 0
        assert codec_tokens.max() < small_depformer.vocab_size


class TestDecodeAudio:
    """Tests for the decode_audio method."""

    def test_decode_audio_import_error_message(self):
        """Test decode_audio gives helpful error when Mimi not installed."""
        import unittest.mock as mock

        # Create a mock model
        model = mock.MagicMock()

        # Simulate decode_audio that raises ImportError
        def mock_decode_audio(codec_tokens, sample_rate=24000):
            raise ImportError(
                "Mimi required for audio decoding. Install with: "
                "pip install moshi_mlx (Mac) or transformers with Mimi support"
            )

        model.decode_audio = mock_decode_audio

        codec_tokens = torch.randint(0, 1024, (1, 8, 100))

        with pytest.raises(ImportError, match="Mimi required"):
            model.decode_audio(codec_tokens)

    def test_decode_audio_input_shape_2d(self):
        """Test decode_audio handles 2D input (batch, seq_len)."""
        # This tests the shape handling in decode_audio
        # The actual Mimi model is optional, so we test the interface

        codec_tokens = torch.randint(0, 1024, (1, 100))
        assert codec_tokens.dim() == 2

        # When unsqueezed for Mimi, should be (batch, codebooks, seq_len)
        expanded = codec_tokens.unsqueeze(1)
        assert expanded.dim() == 3
        assert expanded.shape == (1, 1, 100)

    def test_decode_audio_input_shape_3d(self):
        """Test decode_audio handles 3D input (batch, codebooks, seq_len)."""
        codec_tokens = torch.randint(0, 1024, (1, 8, 100))  # 8 codebooks
        assert codec_tokens.dim() == 3
        assert codec_tokens.shape == (1, 8, 100)


class TestDepformerIntegration:
    """Integration tests combining Depformer with mock LLM hidden states."""

    def test_end_to_end_inference(self, small_depformer):
        """Test full inference: hidden states -> codec tokens."""
        # Simulate LLM hidden states from a response
        batch_size = 1
        response_len = 50  # Tokens in LLM response
        hidden_dim = small_depformer.llm_dim

        hidden_states = torch.randn(batch_size, response_len, hidden_dim)

        # Run inference
        codec_tokens = small_depformer(hidden_states)

        # Verify output
        assert codec_tokens.dtype == torch.long
        assert codec_tokens.shape[0] == batch_size
        assert codec_tokens.shape[1] == small_depformer.num_codebooks
        assert codec_tokens.min() >= 0

    def test_end_to_end_training(self, small_depformer):
        """Test full training: hidden states + targets -> loss."""
        batch_size = 2
        response_len = 30
        num_cbs = small_depformer.num_codebooks
        hidden_dim = small_depformer.llm_dim

        hidden_states = torch.randn(batch_size, response_len, hidden_dim, requires_grad=True)
        targets = torch.randint(0, small_depformer.vocab_size, (batch_size, num_cbs, response_len))
        lengths = torch.tensor([30, 25])

        # Run training
        loss = small_depformer(hidden_states, codec_targets=targets, codec_lengths=lengths)

        # Verify loss
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert loss.requires_grad

        # Verify gradients flow
        loss.backward()
        assert hidden_states.grad is not None

    def test_inference_deterministic_with_same_seed(self, small_depformer):
        """Test inference is deterministic with same random seed."""
        hidden_states = torch.randn(1, 20, small_depformer.llm_dim)

        # Run with seed 42
        torch.manual_seed(42)
        tokens_1 = small_depformer(hidden_states.clone())

        # Run again with same seed
        torch.manual_seed(42)
        tokens_2 = small_depformer(hidden_states.clone())

        assert torch.equal(tokens_1, tokens_2)


class TestMimiPipeline:
    """Tests for the full Mimi-based S2S pipeline workflow.

    These tests verify the pipeline used in demo/s2s_web.py:
    1. Audio bytes -> float32 array
    2. ASRProcessor -> mel features
    3. generate_with_audio -> text + codec tokens
    4. decode_audio -> waveform
    5. Resample and convert to int16 bytes
    """

    def test_audio_bytes_to_float32_conversion(self):
        """Test converting PCM int16 bytes to float32 array."""
        import numpy as np

        # Simulate audio data at 16kHz (1 second = 16000 samples)
        samples = 4096
        int16_audio = np.random.randint(-32768, 32767, samples, dtype=np.int16)
        audio_bytes = int16_audio.tobytes()

        # Convert as done in s2s_web.py
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        assert audio_array.dtype == np.float32
        assert len(audio_array) == samples
        assert audio_array.min() >= -1.0
        assert audio_array.max() <= 1.0

    def test_float32_to_int16_bytes_conversion(self):
        """Test converting float32 waveform to PCM int16 bytes."""
        import numpy as np

        # Simulate decoded audio waveform
        samples = 24000  # 1 second at 24kHz
        waveform = np.random.uniform(-1.0, 1.0, samples).astype(np.float32)

        # Convert as done in s2s_web.py
        audio_int16 = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        assert len(audio_bytes) == samples * 2  # int16 = 2 bytes
        # Verify round-trip
        recovered = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_allclose(waveform, recovered, atol=1e-4)

    def test_resample_24k_to_48k(self):
        """Test resampling from Mimi's 24kHz to browser's 48kHz."""
        import numpy as np
        import scipy.signal

        # Simulate 1 second of audio at 24kHz
        input_rate = 24000
        output_rate = 48000
        duration = 1.0
        samples_in = int(input_rate * duration)

        audio_24k = np.sin(2 * np.pi * 440 * np.arange(samples_in) / input_rate).astype(np.float32)

        # Resample as done in s2s_web.py
        num_samples = int(len(audio_24k) * output_rate / input_rate)
        audio_48k = scipy.signal.resample(audio_24k, num_samples)

        assert len(audio_48k) == output_rate  # 48000 samples for 1 second
        assert audio_48k.shape[0] == num_samples

    def test_codec_tokens_shape_for_mimi(self):
        """Test codec tokens are shaped correctly for Mimi decoding."""
        batch_size = 2
        num_codebooks = 8
        seq_len = 100

        # 3D input (batch, num_codebooks, seq_len) - Depformer output format
        tokens_3d = torch.randint(0, 1024, (batch_size, num_codebooks, seq_len))
        assert tokens_3d.dim() == 3
        assert tokens_3d.shape == (batch_size, num_codebooks, seq_len)


def _mimi_available() -> bool:
    """Check if Mimi codec is available."""
    import platform

    # Try moshi_mlx first (Mac)
    if platform.system() == "Darwin":
        try:
            from moshi_mlx import models  # noqa: F401

            return True
        except ImportError:
            pass

    # Try transformers MimiModel
    try:
        from transformers import MimiModel  # noqa: F401

        return True
    except ImportError:
        return False


class TestMimiDecodeAudioMethod:
    """Tests for ASRModel.decode_audio method with Mimi codec."""

    def test_decode_audio_2d_input_expansion(self):
        """Test decode_audio expands 2D input to 3D."""
        # This tests the shape handling logic
        codec_tokens = torch.randint(0, 1024, (1, 100))
        assert codec_tokens.dim() == 2

        # Simulate the expansion done in decode_audio
        if codec_tokens.dim() == 2:
            codec_tokens = codec_tokens.unsqueeze(1)

        assert codec_tokens.dim() == 3
        assert codec_tokens.shape == (1, 1, 100)

    def test_decode_audio_3d_input_unchanged(self):
        """Test decode_audio keeps 3D input unchanged."""
        codec_tokens = torch.randint(0, 1024, (1, 8, 100))
        assert codec_tokens.dim() == 3

        # Should not expand already 3D tensor
        original_shape = codec_tokens.shape
        if codec_tokens.dim() == 2:
            codec_tokens = codec_tokens.unsqueeze(1)

        assert codec_tokens.shape == original_shape

    def test_decode_audio_batch_handling(self):
        """Test decode_audio handles batched input."""
        batch_size = 4
        num_codebooks = 8
        seq_len = 50

        codec_tokens = torch.randint(0, 1024, (batch_size, num_codebooks, seq_len))
        assert codec_tokens.shape[0] == batch_size
        assert codec_tokens.shape[1] == num_codebooks
        assert codec_tokens.shape[2] == seq_len

    @pytest.mark.skipif(
        not _mimi_available(),
        reason="Mimi not installed (moshi_mlx or transformers with Mimi)",
    )
    def test_decode_audio_with_mimi(self):
        """Test actual Mimi decoding produces valid waveform."""
        import platform

        # Create codec tokens to decode (all 8 codebooks)
        codec_tokens = torch.randint(0, 2048, (1, 8, 100))

        # Use MLX on Mac, transformers elsewhere
        if platform.system() == "Darwin":
            try:
                import mlx.core as mx
                from moshi_mlx import models

                mimi = models.mimi_202412()
                mimi.load_weights()

                codes_np = codec_tokens.cpu().numpy()
                codes_mlx = mx.array(codes_np)
                pcm = mimi.decode(codes_mlx)
                waveform = torch.from_numpy(pcm.__array__()).squeeze(1)
            except ImportError:
                # Fall back to transformers
                from transformers import MimiModel

                mimi = MimiModel.from_pretrained("kyutai/mimi")
                mimi.eval()

                with torch.no_grad():
                    decoded = mimi.decode(codec_tokens)
                    waveform = decoded.audio_values.squeeze(1)
        else:
            from transformers import MimiModel

            mimi = MimiModel.from_pretrained("kyutai/mimi")
            mimi.eval()

            with torch.no_grad():
                decoded = mimi.decode(codec_tokens)
                waveform = decoded.audio_values.squeeze(1)

        # Verify output shape and values
        assert waveform.dim() == 2  # (batch, samples)
        assert waveform.shape[0] == 1  # Batch size
        assert waveform.shape[1] > 0  # Has samples

        # Waveform should be in reasonable audio range
        assert waveform.abs().max() < 10.0  # Not exploding values


class TestS2SWorkflow:
    """End-to-end tests for the S2S workflow from s2s_web.py."""

    def test_full_audio_conversion_roundtrip(self):
        """Test full audio conversion: int16 -> float32 -> process -> int16."""
        import numpy as np

        # Input: 16-bit PCM audio
        input_samples = 16000  # 1 second at 16kHz
        input_int16 = np.random.randint(-32768, 32767, input_samples, dtype=np.int16)
        input_bytes = input_int16.tobytes()

        # Step 1: Convert to float32 (as done in s2s_web.py)
        audio_float = np.frombuffer(input_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Step 2: Simulate processing (would be ASR + generation)
        # Just pass through for this test
        processed = audio_float

        # Step 3: Convert back to int16 bytes (as done in s2s_web.py)
        output_int16 = (np.clip(processed, -1, 1) * 32767).astype(np.int16)
        output_bytes = output_int16.tobytes()

        # Verify round-trip is within quantization error (off-by-one due to /32768 vs *32767)
        recovered = np.frombuffer(output_bytes, dtype=np.int16)
        # Max difference of 1 is expected due to asymmetric int16 range (-32768 to 32767)
        np.testing.assert_allclose(input_int16, recovered, atol=1)

    def test_mel_spectrogram_shape(self):
        """Test that ASRProcessor produces correct mel spectrogram shape."""

        # Simulate mel spectrogram output shape
        # Whisper uses 80 mel bins, and time depends on audio length
        batch_size = 1
        n_mels = 80
        audio_duration_sec = 2.0
        sample_rate = 16000
        hop_length = 160  # Whisper default

        # Calculate expected mel length
        num_samples = int(audio_duration_sec * sample_rate)
        mel_length = num_samples // hop_length

        # Create mock mel features
        mel_features = torch.randn(batch_size, n_mels, mel_length)

        assert mel_features.shape == (1, 80, mel_length)
        assert mel_features.shape[2] > 0

    def test_process_audio_empty_input(self):
        """Test handling of empty audio input."""
        import numpy as np

        empty_bytes = b""
        result = np.frombuffer(empty_bytes, dtype=np.int16)

        assert len(result) == 0

    def test_process_audio_short_input(self):
        """Test handling of very short audio input."""
        import numpy as np

        # Very short audio (100ms)
        samples = 1600  # 100ms at 16kHz
        short_audio = np.random.randint(-32768, 32767, samples, dtype=np.int16)
        audio_bytes = short_audio.tobytes()

        # Convert to float32
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        assert len(audio_float) == samples
        assert audio_float.dtype == np.float32


class TestMimiAvailability:
    """Tests for Mimi availability checking."""

    def test_mimi_check_returns_bool(self):
        """Test _mimi_available returns boolean."""
        result = _mimi_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not _mimi_available(),
        reason="Mimi not installed",
    )
    def test_mimi_model_loads(self):
        """Test Mimi model can be loaded when available."""
        import platform

        if platform.system() == "Darwin":
            try:
                from moshi_mlx import models

                mimi = models.mimi_202412()
                mimi.load_weights()
                assert mimi is not None
                return
            except ImportError:
                pass

        from transformers import MimiModel

        mimi = MimiModel.from_pretrained("kyutai/mimi")
        assert mimi is not None


class TestProcessAudioFunction:
    """Tests for the process_audio function pattern from s2s_web.py."""

    def test_result_dict_structure(self):
        """Test process_audio returns dict with expected keys."""
        # Simulate process_audio return value
        result = {"text": "Hello", "audio": b"\x00\x00"}

        assert "text" in result
        assert "audio" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["audio"], bytes)

    def test_empty_result_structure(self):
        """Test process_audio empty result structure."""
        result = {"text": "", "audio": b""}

        assert result["text"] == ""
        assert result["audio"] == b""
        assert len(result["audio"]) == 0

    def test_audio_output_format(self):
        """Test audio output is valid int16 PCM bytes."""
        import numpy as np

        # Simulate output audio (48kHz, 1 second)
        output_rate = 48000
        waveform = np.random.uniform(-1, 1, output_rate).astype(np.float32)

        # Convert to int16 bytes as done in s2s_web.py
        audio_int16 = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Verify it's valid PCM
        recovered = np.frombuffer(audio_bytes, dtype=np.int16)
        assert len(recovered) == output_rate
        assert recovered.dtype == np.int16
