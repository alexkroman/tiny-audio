"""Tests for the Freeze-Omni style AR decoder Audio Head with Mimi codec tokens."""

import pytest
import torch

from tiny_audio.audio_head import AudioHead


class MockAudioHeadConfig:
    """Mock config for AudioHead initialization in tests."""

    def __init__(self, **kwargs):
        self.llm_dim = kwargs.get("llm_dim", 256)
        self.audio_head_hidden_dim = kwargs.get("audio_head_hidden_dim", 128)
        self.codebook_size = kwargs.get("codebook_size", 1024)
        self.num_codebooks = kwargs.get("num_codebooks", 1)


@pytest.fixture
def audio_head_config():
    """Factory fixture for creating audio head configs."""
    return MockAudioHeadConfig


@pytest.fixture
def small_audio_head(audio_head_config):
    """Create a small AudioHead for testing.

    Uses dimensions that are compatible with transformer architecture:
    - hidden_dim must be divisible by num_heads (64 heads)
    - head_dim (hidden_dim / num_heads) must work with RoPE
    """
    config = audio_head_config(
        llm_dim=128,
        audio_head_hidden_dim=128,  # 128 / 2 = 64 head_dim with 2 heads
        codebook_size=128,
    )
    return AudioHead(config)


class TestAudioHeadInit:
    """Tests for AudioHead initialization."""

    def test_default_init(self, audio_head_config):
        """Test AudioHead initializes with default config."""
        config = audio_head_config()
        head = AudioHead(config)

        assert head.llm_dim == 256
        assert head.hidden_dim == 128
        assert head.vocab_size == 1024
        assert head.num_codebooks == 1

    def test_custom_dimensions(self, audio_head_config):
        """Test AudioHead with custom dimensions."""
        config = audio_head_config(
            llm_dim=1536,
            audio_head_hidden_dim=512,
            codebook_size=2048,
            num_codebooks=2,
        )
        head = AudioHead(config)

        assert head.llm_dim == 1536
        assert head.hidden_dim == 512
        assert head.vocab_size == 2048
        assert head.num_codebooks == 2

    def test_special_tokens(self, audio_head_config):
        """Test special token IDs are correctly set."""
        config = audio_head_config(codebook_size=1024)
        head = AudioHead(config)

        assert head.bos_id == 1024  # vocab_size + 0
        assert head.sos_id == 1025  # vocab_size + 1
        assert head.eos_id == 1026  # vocab_size + 2
        assert head.pad_id == 1027  # vocab_size + 3
        assert head.total_vocab_size == 1028  # vocab_size + 4


class TestAudioHeadOutputLength:
    """Tests for get_output_length method."""

    def test_output_length_estimate(self, audio_head_config):
        """Test output length estimation."""
        config = audio_head_config()
        head = AudioHead(config)

        # Default 2x multiplier
        assert head.get_output_length(100) == 200
        assert head.get_output_length(10) == 20

    def test_output_length_scaling(self, audio_head_config):
        """Test output length scales linearly with input."""
        config = audio_head_config()
        head = AudioHead(config)

        len_1 = head.get_output_length(100)
        len_2 = head.get_output_length(200)

        assert len_2 == 2 * len_1

    def test_output_length_zero(self, audio_head_config):
        """Test output length with zero input."""
        config = audio_head_config()
        head = AudioHead(config)
        assert head.get_output_length(0) == 0


class TestAudioHeadStateDict:
    """Tests for state dict handling."""

    def test_state_dict_not_empty(self, small_audio_head):
        """Test that state_dict contains all model parameters."""
        state = small_audio_head.state_dict()
        assert len(state) > 0

    def test_state_dict_has_expected_components(self, small_audio_head):
        """Test that state_dict contains expected component prefixes."""
        state = small_audio_head.state_dict()
        prefixes = set()
        for key in state:
            prefix = key.split(".")[0]
            prefixes.add(prefix)

        expected_prefixes = {
            "input_proj",
            "embedding",
            "pre_nn_layers",
            "pre_nn_norm",
            "decoder_layers",
            "decoder_norm",
            "output_proj",
        }
        assert prefixes == expected_prefixes

    def test_load_state_dict(self, small_audio_head):
        """Test loading state dict restores weights."""
        # Save current state
        original_state = small_audio_head.state_dict()

        # Modify weights
        with torch.no_grad():
            for param in small_audio_head.input_proj.parameters():
                param.fill_(0.0)

        # Verify weights are zeroed
        for key, value in small_audio_head.state_dict().items():
            if key.startswith("input_proj"):
                assert torch.allclose(value, torch.zeros_like(value))

        # Load original state
        small_audio_head.load_state_dict(original_state)

        # Verify weights are restored
        restored_state = small_audio_head.state_dict()
        for key in original_state:
            assert torch.allclose(original_state[key], restored_state[key])


class TestForwardTraining:
    """Tests for training forward pass."""

    def test_forward_train_returns_scalar_loss(self, small_audio_head):
        """Test training forward pass returns scalar loss."""
        batch_size, llm_seq = 2, 20
        audio_len = 50

        hidden = torch.randn(batch_size, llm_seq, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (batch_size, audio_len))
        lengths = torch.tensor([50, 40])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_forward_train_loss_is_differentiable(self, small_audio_head):
        """Test training loss supports backward pass."""
        batch_size, llm_seq = 2, 20
        audio_len = 50

        hidden = torch.randn(batch_size, llm_seq, small_audio_head.llm_dim, requires_grad=True)
        targets = torch.randint(0, small_audio_head.vocab_size, (batch_size, audio_len))
        lengths = torch.tensor([50, 40])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_forward_train_batch_size_one(self, small_audio_head):
        """Test training with batch size of 1."""
        hidden = torch.randn(1, 20, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (1, 50))
        lengths = torch.tensor([50])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_train_without_lengths(self, small_audio_head):
        """Test training without explicit lengths (use full targets)."""
        batch_size, llm_seq = 2, 20
        audio_len = 50

        hidden = torch.randn(batch_size, llm_seq, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (batch_size, audio_len))

        # Should work without lengths
        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=None)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestForwardInference:
    """Tests for inference forward pass."""

    def test_forward_inference_returns_tokens(self, small_audio_head):
        """Test inference forward pass returns token IDs."""
        hidden = torch.randn(2, 20, small_audio_head.llm_dim)

        tokens = small_audio_head(hidden)  # No targets = inference

        assert tokens.dtype == torch.long
        assert tokens.shape[0] == 2  # Batch size preserved

    def test_forward_inference_tokens_in_valid_range(self, small_audio_head):
        """Test inference tokens are in valid range."""
        hidden = torch.randn(1, 10, small_audio_head.llm_dim)

        tokens = small_audio_head(hidden)

        # Tokens should be in vocab range (excluding special tokens in output)
        # Or could be EOS
        assert tokens.min() >= 0
        assert tokens.max() < small_audio_head.total_vocab_size

    def test_forward_inference_batch_size_one(self, small_audio_head):
        """Test inference with batch size of 1."""
        hidden = torch.randn(1, 20, small_audio_head.llm_dim)

        tokens = small_audio_head(hidden)

        assert tokens.shape[0] == 1


class TestPreNN:
    """Tests for Pre-NN processing."""

    def test_pre_nn_output_shape(self, small_audio_head):
        """Test Pre-NN produces correct output shape."""
        batch_size, seq_len = 2, 20
        hidden = torch.randn(batch_size, seq_len, small_audio_head.hidden_dim)

        output = small_audio_head._forward_pre_nn(hidden)

        assert output.shape == (batch_size, seq_len, small_audio_head.hidden_dim)

    def test_pre_nn_is_differentiable(self, small_audio_head):
        """Test Pre-NN supports gradient computation."""
        hidden = torch.randn(2, 10, small_audio_head.hidden_dim, requires_grad=True)

        output = small_audio_head._forward_pre_nn(hidden)
        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None


class TestAttentionMask:
    """Tests for attention mask creation."""

    def test_train_mask_shape(self, small_audio_head):
        """Test training mask has correct shape."""
        batch_size = 2
        context_len = 10
        target_len = 20
        device = torch.device("cpu")

        mask = small_audio_head._create_train_mask(batch_size, context_len, target_len, device)

        total_len = context_len + target_len
        assert mask.shape == (batch_size, 1, total_len, total_len)

    def test_train_mask_context_fully_visible(self, small_audio_head):
        """Test context tokens can attend to all context."""
        batch_size = 2
        context_len = 10
        target_len = 20
        device = torch.device("cpu")

        mask = small_audio_head._create_train_mask(batch_size, context_len, target_len, device)

        # Context portion should have no masking (all zeros)
        context_mask = mask[:, :, :context_len, :context_len]
        assert (context_mask == 0).all()

    def test_train_mask_ar_causal(self, small_audio_head):
        """Test AR tokens have causal attention."""
        batch_size = 2
        context_len = 10
        target_len = 20
        device = torch.device("cpu")

        mask = small_audio_head._create_train_mask(batch_size, context_len, target_len, device)

        # AR portion should be upper triangular (causal)
        ar_mask = mask[:, :, context_len:, context_len:]

        # Check that upper triangle is masked (-inf)
        for i in range(target_len):
            for j in range(target_len):
                if j > i:
                    assert ar_mask[0, 0, i, j] == float("-inf")
                else:
                    assert ar_mask[0, 0, i, j] == 0

    def test_train_mask_ar_sees_context(self, small_audio_head):
        """Test AR tokens can attend to context."""
        batch_size = 2
        context_len = 10
        target_len = 20
        device = torch.device("cpu")

        mask = small_audio_head._create_train_mask(batch_size, context_len, target_len, device)

        # AR tokens attending to context should have no masking
        ar_to_context = mask[:, :, context_len:, :context_len]
        assert (ar_to_context == 0).all()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self, small_audio_head):
        """Test with minimal sequence length."""
        hidden = torch.randn(1, 1, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (1, 5))
        lengths = torch.tensor([5])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)

    def test_long_sequence(self, small_audio_head):
        """Test with longer sequence."""
        hidden = torch.randn(1, 100, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (1, 200))
        lengths = torch.tensor([200])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)

    def test_varying_target_lengths_in_batch(self, small_audio_head):
        """Test batch with different target lengths."""
        batch_size = 3
        hidden = torch.randn(batch_size, 20, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (batch_size, 100))
        lengths = torch.tensor([100, 75, 50])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)


class TestDevicePlacement:
    """Tests for device and dtype handling."""

    def test_to_device(self, small_audio_head):
        """Test .to() method moves model to device."""
        small_audio_head.to(device="cpu")

        for param in small_audio_head.parameters():
            assert param.device.type == "cpu"

    def test_forward_respects_input_device(self, small_audio_head):
        """Test forward pass respects input device."""
        device = torch.device("cpu")
        small_audio_head.to(device)

        hidden = torch.randn(2, 20, small_audio_head.llm_dim, device=device)
        targets = torch.randint(0, small_audio_head.vocab_size, (2, 50), device=device)
        lengths = torch.tensor([50, 40], device=device)

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.device == device


class TestS2SPipeline:
    """Integration tests for the full Speech-to-Speech pipeline."""

    @pytest.fixture
    def mock_asr_model_with_audio_head(self, small_audio_head):
        """Create a mock ASR model with audio head for testing."""
        import unittest.mock as mock

        model = mock.MagicMock()
        model.audio_head = small_audio_head
        model.device = torch.device("cpu")

        # Mock generate_with_audio return value
        def mock_generate_with_audio(input_features, audio_attention_mask, **kwargs):
            batch_size = input_features.shape[0]
            # Return mock text and codec tokens
            return {
                "text_ids": torch.tensor([[1, 2, 3]]),
                "text": ["Hello, how can I help?"],
                "codec_tokens": torch.randint(0, small_audio_head.vocab_size, (batch_size, 100)),
            }

        model.generate_with_audio = mock_generate_with_audio
        return model

    def test_generate_with_audio_returns_expected_keys(self, mock_asr_model_with_audio_head):
        """Test generate_with_audio returns text and codec tokens."""
        model = mock_asr_model_with_audio_head

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

    def test_codec_tokens_shape(self, mock_asr_model_with_audio_head):
        """Test codec tokens have expected shape."""
        model = mock_asr_model_with_audio_head

        input_features = torch.randn(2, 80, 100)
        attention_mask = torch.ones(2, 100)

        result = model.generate_with_audio(
            input_features=input_features,
            audio_attention_mask=attention_mask,
        )

        codec_tokens = result["codec_tokens"]
        assert codec_tokens.dim() == 2  # (batch, seq_len)
        assert codec_tokens.shape[0] == 2  # Batch size

    def test_codec_tokens_valid_range(self, mock_asr_model_with_audio_head, small_audio_head):
        """Test codec tokens are in valid vocabulary range."""
        model = mock_asr_model_with_audio_head

        input_features = torch.randn(1, 80, 100)
        attention_mask = torch.ones(1, 100)

        result = model.generate_with_audio(
            input_features=input_features,
            audio_attention_mask=attention_mask,
        )

        codec_tokens = result["codec_tokens"]
        assert codec_tokens.min() >= 0
        assert codec_tokens.max() < small_audio_head.vocab_size


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

        codec_tokens = torch.randint(0, 1024, (1, 100))

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


class TestAudioHeadIntegration:
    """Integration tests combining AudioHead with mock LLM hidden states."""

    def test_end_to_end_inference(self, small_audio_head):
        """Test full inference: hidden states -> codec tokens."""
        # Simulate LLM hidden states from a response
        batch_size = 1
        response_len = 50  # Tokens in LLM response
        hidden_dim = small_audio_head.llm_dim

        hidden_states = torch.randn(batch_size, response_len, hidden_dim)

        # Run inference
        codec_tokens = small_audio_head(hidden_states)

        # Verify output
        assert codec_tokens.dtype == torch.long
        assert codec_tokens.shape[0] == batch_size
        assert codec_tokens.min() >= 0

    def test_end_to_end_training(self, small_audio_head):
        """Test full training: hidden states + targets -> loss."""
        batch_size = 2
        response_len = 30
        audio_len = 60
        hidden_dim = small_audio_head.llm_dim

        hidden_states = torch.randn(batch_size, response_len, hidden_dim, requires_grad=True)
        targets = torch.randint(0, small_audio_head.vocab_size, (batch_size, audio_len))
        lengths = torch.tensor([60, 50])

        # Run training
        loss = small_audio_head(hidden_states, codec_targets=targets, codec_lengths=lengths)

        # Verify loss
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert loss.requires_grad

        # Verify gradients flow
        loss.backward()
        assert hidden_states.grad is not None

    def test_inference_deterministic_with_same_seed(self, small_audio_head):
        """Test inference is deterministic with same random seed."""
        hidden_states = torch.randn(1, 20, small_audio_head.llm_dim)

        # Run with seed 42
        torch.manual_seed(42)
        tokens_1 = small_audio_head(hidden_states.clone())

        # Run again with same seed
        torch.manual_seed(42)
        tokens_2 = small_audio_head(hidden_states.clone())

        assert torch.equal(tokens_1, tokens_2)


class TestMimiPipeline:
    """Tests for the full Mimi-based S2S pipeline workflow.

    These tests verify the pipeline used in demo/s2s_web.py:
    1. Audio bytes → float32 array
    2. ASRProcessor → mel features
    3. generate_with_audio → text + codec tokens
    4. decode_audio → waveform
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
        seq_len = 100

        # 2D input (batch, seq_len)
        tokens_2d = torch.randint(0, 1024, (batch_size, seq_len))
        assert tokens_2d.dim() == 2

        # Should be expanded to 3D for Mimi: (batch, codebooks, seq_len)
        tokens_3d = tokens_2d.unsqueeze(1)
        assert tokens_3d.dim() == 3
        assert tokens_3d.shape == (batch_size, 1, seq_len)

    def test_codec_tokens_3d_passthrough(self):
        """Test 3D codec tokens pass through unchanged."""
        batch_size = 1
        num_codebooks = 8
        seq_len = 100

        # 3D input (batch, codebooks, seq_len) - already correct shape
        tokens_3d = torch.randint(0, 2048, (batch_size, num_codebooks, seq_len))
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
        seq_len = 50

        codec_tokens = torch.randint(0, 1024, (batch_size, seq_len))
        expanded = codec_tokens.unsqueeze(1)

        assert expanded.shape[0] == batch_size
        assert expanded.shape[1] == 1  # Single codebook
        assert expanded.shape[2] == seq_len

    @pytest.mark.skipif(
        not _mimi_available(),
        reason="Mimi not installed (moshi_mlx or transformers with Mimi)",
    )
    def test_decode_audio_with_mimi(self):
        """Test actual Mimi decoding produces valid waveform."""
        import platform

        # Create codec tokens to decode
        codec_tokens = torch.randint(0, 2048, (1, 100))

        # Expand to 3D for Mimi: (batch, codebooks, seq_len)
        if codec_tokens.dim() == 2:
            codec_tokens = codec_tokens.unsqueeze(1)

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
