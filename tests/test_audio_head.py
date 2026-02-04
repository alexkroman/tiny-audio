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
