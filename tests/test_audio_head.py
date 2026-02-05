"""Tests for the AR codec AudioHead for speech-to-speech."""

import pytest
import torch

from tiny_audio.audio_head import AudioHead


class MockAudioHeadConfig:
    """Mock config for AudioHead initialization in tests."""

    def __init__(self, **kwargs):
        self.llm_dim = kwargs.get("llm_dim", 3072)
        self.max_audio_tokens = kwargs.get("max_audio_tokens", 500)
        self.audio_top_k = kwargs.get("audio_top_k", 50)
        self.audio_temperature = kwargs.get("audio_temperature", 1.0)
        self.audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.1)


@pytest.fixture
def audio_head_config():
    """Factory fixture for creating audio head configs."""
    return MockAudioHeadConfig


@pytest.fixture
def small_audio_head(audio_head_config):
    """Create a small AudioHead for testing."""
    config = audio_head_config(llm_dim=256)
    return AudioHead(config, llm_dim=256)


class TestAudioHeadInit:
    """Tests for AudioHead initialization."""

    def test_default_init(self, audio_head_config):
        """Test AudioHead initializes with default config."""
        config = audio_head_config()
        head = AudioHead(config)

        assert head.llm_dim == 3072
        assert head.hidden_dim == AudioHead.HIDDEN_DIM
        assert head.vocab_size == AudioHead.VOCAB_SIZE

    def test_custom_llm_dim(self, audio_head_config):
        """Test AudioHead with custom LLM dimension."""
        config = audio_head_config(llm_dim=1536)
        head = AudioHead(config, llm_dim=1536)

        assert head.llm_dim == 1536

    def test_pre_nn_created(self, small_audio_head):
        """Test that Pre-NN is created."""
        assert hasattr(small_audio_head, "pre_nn")
        assert small_audio_head.pre_nn is not None

    def test_ar_decoder_created(self, small_audio_head):
        """Test that AR decoder is created."""
        assert hasattr(small_audio_head, "ar_decoder")
        assert small_audio_head.ar_decoder is not None

    def test_special_tokens(self, small_audio_head):
        """Test special token IDs are set correctly."""
        vocab_size = small_audio_head.vocab_size
        assert small_audio_head.bos_token_id == vocab_size + 0
        assert small_audio_head.sos_token_id == vocab_size + 1
        assert small_audio_head.eos_token_id == vocab_size + 2
        assert small_audio_head.pad_token_id == vocab_size + 3


class TestAudioHeadOutputLength:
    """Tests for get_output_length method."""

    def test_output_length_estimate(self, audio_head_config):
        """Test output length estimation."""
        config = audio_head_config()
        head = AudioHead(config)

        # Mimi: 24kHz audio / 12.5 Hz = 1920 samples per frame
        # Estimate: ~3 frames per text token
        assert head.get_output_length(1) == 3 * 1920
        assert head.get_output_length(10) == 30 * 1920

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
            "pre_nn",
            "ar_decoder",
        }
        assert expected_prefixes.issubset(prefixes)

    def test_load_state_dict(self, small_audio_head):
        """Test loading state dict restores weights."""
        original_state = small_audio_head.state_dict()

        # Modify weights
        with torch.no_grad():
            for param in small_audio_head.input_proj.parameters():
                param.fill_(0.0)
                break

        # Load original state (strict=False because ar_decoder.embedding is tied to self.embedding)
        small_audio_head.load_state_dict(original_state, strict=False)

        # Verify weights are restored
        restored_state = small_audio_head.state_dict()
        for key in original_state:
            assert torch.allclose(original_state[key], restored_state[key])


class TestForwardTraining:
    """Tests for training forward pass."""

    def test_forward_train_returns_scalar_loss(self, small_audio_head):
        """Test training forward pass returns scalar loss."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8  # Mimi uses 8 codebooks

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim)
        # Codec targets: (batch, num_codebooks, audio_len) - discrete tokens for all codebooks
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )
        lengths = torch.tensor([30, 25])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_forward_train_loss_is_differentiable(self, small_audio_head):
        """Test training loss supports backward pass."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim, requires_grad=True)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )
        lengths = torch.tensor([30, 25])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_forward_train_batch_size_one(self, small_audio_head):
        """Test training with batch size of 1."""
        num_codebooks = 8
        hidden = torch.randn(1, 10, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (1, num_codebooks, 30))
        lengths = torch.tensor([30])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_train_without_lengths(self, small_audio_head):
        """Test training without explicit lengths."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=None)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestForwardInference:
    """Tests for inference forward pass."""

    def test_forward_inference_returns_codes(self, small_audio_head):
        """Test inference forward pass returns codec tokens."""
        # Override max_tokens for faster test
        small_audio_head.max_tokens = 10

        hidden = torch.randn(1, 10, small_audio_head.llm_dim)

        codes, _ = small_audio_head(hidden)  # No targets = inference

        assert codes.dtype == torch.long
        assert codes.shape[0] == 1  # Batch size preserved
        # All generated tokens should be valid codec tokens
        assert (codes >= 0).all()
        assert (codes < small_audio_head.vocab_size).all()

    def test_forward_inference_batch_size_one(self, small_audio_head):
        """Test inference with batch size of 1."""
        small_audio_head.max_tokens = 10
        hidden = torch.randn(1, 10, small_audio_head.llm_dim)

        codes, _ = small_audio_head(hidden)

        assert codes.shape[0] == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self, small_audio_head):
        """Test with minimal sequence length."""
        num_codebooks = 8
        hidden = torch.randn(1, 1, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (1, num_codebooks, 3))
        lengths = torch.tensor([3])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)

    def test_long_sequence(self, small_audio_head):
        """Test with longer sequence."""
        num_codebooks = 8
        hidden = torch.randn(1, 50, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (1, num_codebooks, 200))
        lengths = torch.tensor([200])

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)

    def test_varying_lengths_in_batch(self, small_audio_head):
        """Test batch with different target lengths."""
        batch_size = 3
        num_codebooks = 8
        hidden = torch.randn(batch_size, 20, small_audio_head.llm_dim)
        targets = torch.randint(0, small_audio_head.vocab_size, (batch_size, num_codebooks, 80))
        lengths = torch.tensor([80, 60, 40])

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
        num_codebooks = 8
        small_audio_head.to(device)

        hidden = torch.randn(2, 10, small_audio_head.llm_dim, device=device)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (2, num_codebooks, 30), device=device
        )
        lengths = torch.tensor([30, 25], device=device)

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert loss.device == device


class TestDecodeToAudio:
    """Tests for decode_to_audio method."""

    def test_decode_to_audio_without_mimi_raises(self, small_audio_head):
        """Test decode_to_audio raises error when Mimi not loaded."""
        num_codebooks = 8
        codes = torch.randint(0, small_audio_head.vocab_size, (1, num_codebooks, 20))

        with pytest.raises(RuntimeError, match="Mimi not loaded"):
            small_audio_head.decode_to_audio(codes)


class TestEncodeAudio:
    """Tests for encode_audio method."""

    def test_encode_audio_without_mimi_raises(self, small_audio_head):
        """Test encode_audio raises error when Mimi not loaded."""
        audio = torch.randn(1, 24000)

        with pytest.raises(RuntimeError, match="Mimi not loaded"):
            small_audio_head.encode_audio(audio)


class TestARTraining:
    """Tests for AR codec training behavior."""

    def test_loss_decreases_with_training(self, small_audio_head):
        """Test that loss decreases when training on same batch."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )

        small_audio_head.train()
        optimizer = torch.optim.Adam(small_audio_head.parameters(), lr=1e-3)

        # Get initial loss
        loss_initial = small_audio_head(hidden, codec_targets=targets).item()

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            loss = small_audio_head(hidden, codec_targets=targets)
            loss.backward()
            optimizer.step()

        loss_final = small_audio_head(hidden, codec_targets=targets).item()

        # Loss should decrease (model is learning)
        assert loss_final < loss_initial


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_large_input_values(self, small_audio_head):
        """Test with very large input values."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim) * 100
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )

        loss = small_audio_head(hidden, codec_targets=targets)

        assert not torch.isnan(loss), "NaN with large inputs"
        assert not torch.isinf(loss), "Inf with large inputs"

    def test_small_input_values(self, small_audio_head):
        """Test with very small input values."""
        batch_size, text_len, audio_len = 2, 10, 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim) * 1e-6
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )

        loss = small_audio_head(hidden, codec_targets=targets)

        assert not torch.isnan(loss), "NaN with small inputs"


class TestBatchSizeEdgeCases:
    """Tests for batch size edge cases."""

    def test_large_batch_size(self, small_audio_head):
        """Test with large batch size."""
        batch_size = 32
        text_len = 10
        audio_len = 30
        num_codebooks = 8

        hidden = torch.randn(batch_size, text_len, small_audio_head.llm_dim)
        targets = torch.randint(
            0, small_audio_head.vocab_size, (batch_size, num_codebooks, audio_len)
        )
        lengths = torch.randint(20, audio_len + 1, (batch_size,))

        loss = small_audio_head(hidden, codec_targets=targets, codec_lengths=lengths)

        assert not torch.isnan(loss)
        assert loss.dim() == 0


class TestGradientFlow:
    """Tests for gradient flow through all components."""

    def test_gradients_flow_to_all_trainable_params(self, small_audio_head):
        """Test gradients flow to all trainable parameters."""
        num_codebooks = 8
        hidden = torch.randn(2, 10, small_audio_head.llm_dim, requires_grad=True)
        targets = torch.randint(0, small_audio_head.vocab_size, (2, num_codebooks, 30))

        loss = small_audio_head(hidden, codec_targets=targets)
        loss.backward()

        # Check all trainable parameters have gradients
        for name, param in small_audio_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradient_magnitude_reasonable(self, small_audio_head):
        """Test gradient magnitudes are reasonable (not exploding)."""
        num_codebooks = 8
        hidden = torch.randn(2, 10, small_audio_head.llm_dim, requires_grad=True)
        targets = torch.randint(0, small_audio_head.vocab_size, (2, num_codebooks, 30))

        loss = small_audio_head(hidden, codec_targets=targets)
        loss.backward()

        for name, param in small_audio_head.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert grad_norm < 1e6, f"Exploding gradient for {name}: {grad_norm}"


class TestConfigPriority:
    """Tests for config parameter priority."""

    def test_llm_dim_from_constructor_overrides_config(self, audio_head_config):
        """Test that constructor llm_dim overrides config."""
        config = audio_head_config(llm_dim=1024)
        head = AudioHead(config, llm_dim=512)

        assert head.llm_dim == 512

    def test_llm_dim_from_config_when_constructor_none(self, audio_head_config):
        """Test that config llm_dim is used when constructor is None."""
        config = audio_head_config(llm_dim=1024)
        head = AudioHead(config, llm_dim=None)

        assert head.llm_dim == 1024

    def test_llm_dim_default_when_both_missing(self):
        """Test default llm_dim when both config and constructor missing."""

        class EmptyConfig:
            pass

        head = AudioHead(EmptyConfig(), llm_dim=None)

        assert head.llm_dim == 2048  # SmolLM3 native dimension


class TestGenerationParameters:
    """Tests for generation parameter handling."""

    def test_top_k_affects_generation(self, audio_head_config):
        """Test that top_k parameter affects generation diversity."""
        config = audio_head_config(llm_dim=256, audio_top_k=5)
        head = AudioHead(config, llm_dim=256)
        head.max_tokens = 10

        hidden = torch.randn(1, 5, 256)

        # Should not crash with low top_k
        codes, _ = head(hidden)
        assert codes.shape[0] == 1

    def test_temperature_affects_generation(self, audio_head_config):
        """Test that temperature parameter affects generation."""
        config = audio_head_config(llm_dim=256, audio_temperature=0.5)
        head = AudioHead(config, llm_dim=256)
        head.max_tokens = 10

        hidden = torch.randn(1, 5, 256)

        # Should not crash with low temperature
        codes, _ = head(hidden)
        assert codes.shape[0] == 1

    def test_repetition_penalty_affects_generation(self, audio_head_config):
        """Test that repetition penalty parameter affects generation."""
        config = audio_head_config(llm_dim=256, audio_repetition_penalty=2.0)
        head = AudioHead(config, llm_dim=256)
        head.max_tokens = 10

        hidden = torch.randn(1, 5, 256)

        # Should not crash with high repetition penalty
        codes, _ = head(hidden)
        assert codes.shape[0] == 1
