"""Tests for the flow matching AudioHead for speech-to-speech."""

import pytest
import torch

from tiny_audio.audio_head import AudioHead, lsd_decode


class MockAudioHeadConfig:
    """Mock config for AudioHead initialization in tests."""

    def __init__(self, **kwargs):
        self.llm_dim = kwargs.get("llm_dim", 2048)
        self.lsd_decode_steps = kwargs.get("lsd_decode_steps", 1)
        self.flow_temperature = kwargs.get("flow_temperature", 1.0)


@pytest.fixture
def audio_head_config():
    """Factory fixture for creating audio head configs."""
    return MockAudioHeadConfig


@pytest.fixture
def small_audio_head(audio_head_config):
    """Create a small AudioHead for testing."""
    config = audio_head_config(
        llm_dim=256,
        lsd_decode_steps=1,
        flow_temperature=1.0,
    )
    return AudioHead(config, llm_dim=256)


class TestLSDDecode:
    """Tests for the LSD decoding function."""

    def test_lsd_decode_single_step(self):
        """Test LSD decode with single step."""
        batch_size = 2
        latent_dim = 32

        # Create noise
        x_0 = torch.randn(batch_size, latent_dim)

        # Simple velocity function that returns zeros
        def v_t(s, t, x):
            return torch.zeros_like(x)

        result = lsd_decode(v_t, x_0, num_steps=1)

        assert result.shape == x_0.shape
        # With zero velocity, result should be x_0
        assert torch.allclose(result, x_0)

    def test_lsd_decode_multi_step(self):
        """Test LSD decode with multiple steps."""
        batch_size = 2
        latent_dim = 32
        x_0 = torch.randn(batch_size, latent_dim)

        # Velocity that moves towards target
        target = torch.ones(batch_size, latent_dim)

        def v_t(s, t, x):
            return target - x

        result = lsd_decode(v_t, x_0, num_steps=10)

        assert result.shape == x_0.shape
        # Should have moved towards target
        assert not torch.allclose(result, x_0)


class TestAudioHeadInit:
    """Tests for AudioHead initialization."""

    def test_default_init(self, audio_head_config):
        """Test AudioHead initializes with default config."""
        config = audio_head_config()
        head = AudioHead(config)

        assert head.llm_dim == 2048
        assert head.cond_dim == AudioHead.COND_DIM
        assert head.latent_dim == AudioHead.LATENT_DIM
        assert head.mimi_dim == AudioHead.MIMI_DIM

    def test_custom_llm_dim(self, audio_head_config):
        """Test AudioHead with custom LLM dimension."""
        config = audio_head_config(llm_dim=1536)
        head = AudioHead(config, llm_dim=1536)

        assert head.llm_dim == 1536

    def test_projections_created(self, small_audio_head):
        """Test that projections are created correctly."""
        # llm_proj: llm_dim -> cond_dim
        assert small_audio_head.llm_proj.in_features == 256
        assert small_audio_head.llm_proj.out_features == small_audio_head.cond_dim

        # latent_proj_in: mimi_dim -> latent_dim
        assert small_audio_head.latent_proj_in.in_features == small_audio_head.mimi_dim
        assert small_audio_head.latent_proj_in.out_features == small_audio_head.latent_dim

        # latent_proj_out: latent_dim -> mimi_dim
        assert small_audio_head.latent_proj_out.in_features == small_audio_head.latent_dim
        assert small_audio_head.latent_proj_out.out_features == small_audio_head.mimi_dim

    def test_flow_net_created(self, small_audio_head):
        """Test that flow network is created."""
        assert small_audio_head.flow_net is not None


class TestAudioHeadOutputLength:
    """Tests for get_output_length method."""

    def test_output_length_estimate(self, audio_head_config):
        """Test output length estimation."""
        config = audio_head_config()
        head = AudioHead(config)

        # Mimi: 24kHz audio / 12.5 Hz = 1920 samples per frame
        assert head.get_output_length(1) == 1920
        assert head.get_output_length(10) == 19200

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
            "llm_proj",
            "latent_proj_in",
            "latent_proj_out",
            "flow_net",
        }
        assert expected_prefixes.issubset(prefixes)

    def test_load_state_dict(self, small_audio_head):
        """Test loading state dict restores weights."""
        original_state = small_audio_head.state_dict()

        # Modify weights
        with torch.no_grad():
            small_audio_head.llm_proj.weight.fill_(0.0)

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
        batch_size, seq_len = 2, 20

        hidden = torch.randn(batch_size, seq_len, small_audio_head.llm_dim)
        # Mimi latent targets: (batch, seq_len, 512)
        targets = torch.randn(batch_size, seq_len, small_audio_head.mimi_dim)
        lengths = torch.tensor([20, 15])

        loss = small_audio_head(hidden, latent_targets=targets, latent_lengths=lengths)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_forward_train_loss_is_differentiable(self, small_audio_head):
        """Test training loss supports backward pass."""
        batch_size, seq_len = 2, 20

        hidden = torch.randn(batch_size, seq_len, small_audio_head.llm_dim, requires_grad=True)
        targets = torch.randn(batch_size, seq_len, small_audio_head.mimi_dim)
        lengths = torch.tensor([20, 15])

        loss = small_audio_head(hidden, latent_targets=targets, latent_lengths=lengths)
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_forward_train_batch_size_one(self, small_audio_head):
        """Test training with batch size of 1."""
        hidden = torch.randn(1, 20, small_audio_head.llm_dim)
        targets = torch.randn(1, 20, small_audio_head.mimi_dim)
        lengths = torch.tensor([20])

        loss = small_audio_head(hidden, latent_targets=targets, latent_lengths=lengths)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_train_without_lengths(self, small_audio_head):
        """Test training without explicit lengths."""
        batch_size, seq_len = 2, 20

        hidden = torch.randn(batch_size, seq_len, small_audio_head.llm_dim)
        targets = torch.randn(batch_size, seq_len, small_audio_head.mimi_dim)

        loss = small_audio_head(hidden, latent_targets=targets, latent_lengths=None)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_train_interpolation(self, small_audio_head):
        """Test training with different hidden and target lengths."""
        batch_size = 2
        hidden_len = 30
        target_len = 20

        hidden = torch.randn(batch_size, hidden_len, small_audio_head.llm_dim)
        targets = torch.randn(batch_size, target_len, small_audio_head.mimi_dim)
        lengths = torch.tensor([target_len, target_len - 5])

        loss = small_audio_head(hidden, latent_targets=targets, latent_lengths=lengths)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestForwardInference:
    """Tests for inference forward pass."""

    def test_forward_inference_returns_latents(self, small_audio_head):
        """Test inference forward pass returns latent embeddings."""
        hidden = torch.randn(2, 20, small_audio_head.llm_dim)

        latents = small_audio_head(hidden)  # No targets = inference

        assert latents.dtype == hidden.dtype
        assert latents.shape[0] == 2  # Batch size preserved
        assert latents.shape[1] == 20  # Seq len preserved
        assert latents.shape[2] == small_audio_head.mimi_dim  # 512-dim Mimi embeddings

    def test_forward_inference_batch_size_one(self, small_audio_head):
        """Test inference with batch size of 1."""
        hidden = torch.randn(1, 20, small_audio_head.llm_dim)

        latents = small_audio_head(hidden)

        assert latents.shape[0] == 1
        assert latents.shape[2] == small_audio_head.mimi_dim

    def test_forward_inference_output_shape(self, small_audio_head):
        """Test inference output shape is (batch, seq_len, mimi_dim)."""
        batch_size = 3
        seq_len = 15
        hidden = torch.randn(batch_size, seq_len, small_audio_head.llm_dim)

        latents = small_audio_head(hidden)

        assert latents.shape == (batch_size, seq_len, small_audio_head.mimi_dim)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self, small_audio_head):
        """Test with minimal sequence length."""
        hidden = torch.randn(1, 1, small_audio_head.llm_dim)
        targets = torch.randn(1, 1, small_audio_head.mimi_dim)
        lengths = torch.tensor([1])

        loss = small_audio_head(hidden, latent_targets=targets, latent_lengths=lengths)

        assert not torch.isnan(loss)

    def test_long_sequence(self, small_audio_head):
        """Test with longer sequence."""
        hidden = torch.randn(1, 100, small_audio_head.llm_dim)
        targets = torch.randn(1, 100, small_audio_head.mimi_dim)
        lengths = torch.tensor([100])

        loss = small_audio_head(hidden, latent_targets=targets, latent_lengths=lengths)

        assert not torch.isnan(loss)

    def test_varying_lengths_in_batch(self, small_audio_head):
        """Test batch with different target lengths."""
        batch_size = 3
        hidden = torch.randn(batch_size, 50, small_audio_head.llm_dim)
        targets = torch.randn(batch_size, 50, small_audio_head.mimi_dim)
        lengths = torch.tensor([50, 40, 30])

        loss = small_audio_head(hidden, latent_targets=targets, latent_lengths=lengths)

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
        targets = torch.randn(2, 20, small_audio_head.mimi_dim, device=device)
        lengths = torch.tensor([20, 15], device=device)

        loss = small_audio_head(hidden, latent_targets=targets, latent_lengths=lengths)

        assert loss.device == device


class TestDecodeToAudio:
    """Tests for decode_to_audio method."""

    def test_decode_to_audio_without_mimi_raises(self, small_audio_head):
        """Test decode_to_audio raises error when Mimi not loaded."""
        latents = torch.randn(1, 20, small_audio_head.mimi_dim)

        with pytest.raises(RuntimeError, match="Mimi decoder not loaded"):
            small_audio_head.decode_to_audio(latents)


class TestFlowMatching:
    """Tests for flow matching specific behavior."""

    def test_flow_loss_decreases_with_training(self, small_audio_head):
        """Test that loss decreases when training on same batch."""
        batch_size, seq_len = 2, 20

        hidden = torch.randn(batch_size, seq_len, small_audio_head.llm_dim)
        targets = torch.randn(batch_size, seq_len, small_audio_head.mimi_dim)

        small_audio_head.train()
        optimizer = torch.optim.Adam(small_audio_head.parameters(), lr=1e-3)

        # Get initial loss
        loss_initial = small_audio_head(hidden, latent_targets=targets).item()

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            loss = small_audio_head(hidden, latent_targets=targets)
            loss.backward()
            optimizer.step()

        loss_final = small_audio_head(hidden, latent_targets=targets).item()

        # Loss should decrease (model is learning)
        assert loss_final < loss_initial

    def test_inference_deterministic_with_same_seed(self, small_audio_head):
        """Test inference is deterministic with same random seed."""
        hidden = torch.randn(1, 10, small_audio_head.llm_dim)

        small_audio_head.eval()

        # Run with seed 42
        torch.manual_seed(42)
        latents_1 = small_audio_head(hidden.clone())

        # Run again with same seed
        torch.manual_seed(42)
        latents_2 = small_audio_head(hidden.clone())

        assert torch.equal(latents_1, latents_2)
