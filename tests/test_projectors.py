"""Tests for audio projector modules."""

import pytest
import torch

from tiny_audio.projectors import (
    PROJECTOR_CLASSES,
    MLPAudioProjector,
    MoEAudioProjector,
    MOSAProjector,
    QFormerAudioProjector,
    SimpleAdapter,
    SwiGLUExpert,
    load_balancing_loss,
    z_loss,
)


class MockConfig:
    """Mock config for projector initialization."""

    def __init__(self, **kwargs):
        self.encoder_dim = kwargs.get("encoder_dim", 256)
        self.llm_dim = kwargs.get("llm_dim", 512)
        self.projector_hidden_dim = kwargs.get("projector_hidden_dim", 1024)
        self.projector_pool_stride = kwargs.get("projector_pool_stride", 4)
        self.projector_dropout = kwargs.get("projector_dropout", 0.0)
        self.projector_num_layers = kwargs.get("projector_num_layers", 2)
        self.projector_init_std = kwargs.get("projector_init_std", 0.02)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 2)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 0.02)
        self.router_z_loss_coef = kwargs.get("router_z_loss_coef", 0.001)
        self.adapter_hidden_dim = kwargs.get("adapter_hidden_dim", 1024)
        # QFormer settings
        self.qformer_window_size = kwargs.get("qformer_window_size", 15)
        self.downsample_rate = kwargs.get("downsample_rate", 5)
        self.qformer_hidden_size = kwargs.get("qformer_hidden_size")
        self.qformer_num_layers = kwargs.get("qformer_num_layers", 2)
        self.qformer_num_heads = kwargs.get("qformer_num_heads", 8)
        self.qformer_intermediate_size = kwargs.get("qformer_intermediate_size")


# =============================================================================
# Helper Module Tests
# =============================================================================


class TestSimpleAdapter:
    """Tests for SimpleAdapter module."""

    def test_forward_shape(self):
        """Test that SimpleAdapter produces correct output shape."""
        adapter = SimpleAdapter(input_dim=256, hidden_dim=512, output_dim=128)
        x = torch.randn(2, 10, 256)
        out = adapter(x)
        assert out.shape == (2, 10, 128)

    def test_forward_dtype(self):
        """Test that SimpleAdapter preserves dtype."""
        adapter = SimpleAdapter(input_dim=256, hidden_dim=512, output_dim=128)
        x = torch.randn(2, 10, 256, dtype=torch.float32)
        out = adapter(x)
        assert out.dtype == torch.float32


class TestSwiGLUExpert:
    """Tests for SwiGLUExpert module."""

    def test_forward_shape(self):
        """Test that SwiGLUExpert produces correct output shape."""
        expert = SwiGLUExpert(input_dim=256, hidden_dim=512, output_dim=128)
        x = torch.randn(2, 10, 256)
        out = expert(x)
        assert out.shape == (2, 10, 128)

    def test_no_bias(self):
        """Test that SwiGLUExpert has no biases."""
        expert = SwiGLUExpert(input_dim=256, hidden_dim=512, output_dim=128)
        assert expert.gate_proj.bias is None
        assert expert.up_proj.bias is None
        assert expert.down_proj.bias is None


# =============================================================================
# MLP Projector Tests
# =============================================================================


class TestMLPAudioProjector:
    """Tests for MLPAudioProjector."""

    @pytest.fixture
    def config(self):
        return MockConfig(encoder_dim=256, llm_dim=512, projector_pool_stride=4)

    @pytest.fixture
    def projector(self, config):
        return MLPAudioProjector(config)

    def test_forward_shape(self, projector):
        """Test that MLP projector produces correct output shape."""
        x = torch.randn(2, 100, 256)
        out = projector(x)
        # Stride-4 frame stacking quarters sequence length
        assert out.shape == (2, 25, 512)

    def test_get_output_length(self, projector):
        """Test output length calculation (floor division)."""
        assert projector.get_output_length(100) == 25
        assert projector.get_output_length(104) == 26
        assert projector.get_output_length(4) == 1

    def test_downsampling(self, projector):
        """Test that downsampling reduces sequence length by k (must be divisible)."""
        for seq_len in [8, 48, 100, 200]:  # All divisible by k=4
            x = torch.randn(1, seq_len, 256)
            out = projector(x)
            expected_len = projector.get_output_length(seq_len)
            assert out.shape[1] == expected_len


# =============================================================================
# MOSA Projector Tests
# =============================================================================


class TestMOSAProjector:
    """Tests for MOSAProjector (dense MoE)."""

    @pytest.fixture
    def config(self):
        return MockConfig(
            encoder_dim=256,
            llm_dim=512,
            num_experts=4,
            adapter_hidden_dim=512,
            router_aux_loss_coef=0.02,
            router_z_loss_coef=0.001,
        )

    @pytest.fixture
    def projector(self, config):
        return MOSAProjector(config)

    def test_forward_shape(self, projector):
        """Test that MOSA projector produces correct output shape."""
        x = torch.randn(2, 100, 256)
        out = projector(x)
        # Stride-4 total (2 conv layers with stride-2 each)
        assert out.shape == (2, 25, 512)

    def test_get_output_length(self, projector):
        """Test output length calculation for stride-4 (floor division)."""
        assert projector.get_output_length(100) == 25
        assert projector.get_output_length(101) == 25  # Floor division
        assert projector.get_output_length(4) == 1
        assert projector.get_output_length(5) == 1  # Floor division


# =============================================================================
# MoE Projector Tests
# =============================================================================


class TestMoEAudioProjector:
    """Tests for MoEAudioProjector."""

    @pytest.fixture
    def config(self):
        return MockConfig(
            encoder_dim=256,
            llm_dim=512,
            projector_hidden_dim=512,
            projector_pool_stride=4,
            num_experts=4,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.02,
            router_z_loss_coef=0.001,
        )

    @pytest.fixture
    def projector(self, config):
        return MoEAudioProjector(config)

    def test_forward_shape(self, projector):
        """Test that SharedMoE projector produces correct output shape."""
        x = torch.randn(2, 100, 256)
        out = projector(x)
        assert out.shape == (2, 25, 512)

    def test_get_output_length(self, projector):
        """Test output length calculation."""
        assert projector.get_output_length(100) == 25
        assert projector.get_output_length(101) == 26

    def test_has_shared_expert(self, projector):
        """Test that projector has a shared expert."""
        assert hasattr(projector.moe, "shared_expert")
        assert projector.moe.shared_expert is not None

    def test_aux_loss(self, projector):
        """Test auxiliary loss computation."""
        x = torch.randn(2, 100, 256)
        _ = projector(x)
        aux_loss = projector.get_aux_loss()
        assert aux_loss.numel() == 1
        assert aux_loss >= 0


# =============================================================================
# QFormer Projector Tests
# =============================================================================


class TestQFormerAudioProjector:
    """Tests for QFormerAudioProjector."""

    @pytest.fixture
    def config(self):
        return MockConfig(
            encoder_dim=256,
            llm_dim=512,
            qformer_window_size=15,
            downsample_rate=5,
            qformer_num_layers=2,
            qformer_num_heads=8,
        )

    @pytest.fixture
    def projector(self, config):
        return QFormerAudioProjector(config)

    def test_forward_shape(self, projector):
        """Test that QFormer projector produces correct output shape."""
        x = torch.randn(2, 100, 256)
        out = projector(x)
        # 100 frames / 15 window = 7 windows (rounded up), 3 queries each = 21 tokens
        # Actually: ceil(100/15) = 7, 7 * 3 = 21
        expected_len = projector.get_output_length(100)
        assert out.shape == (2, expected_len, 512)

    def test_get_output_length(self, projector):
        """Test output length calculation."""
        # window_size=15, downsample_rate=5 -> num_queries=3
        # nblocks = ceil(input/15), output = nblocks * 3
        assert projector.get_output_length(15) == 3  # 1 window
        assert projector.get_output_length(16) == 6  # 2 windows
        assert projector.get_output_length(30) == 6  # 2 windows
        assert projector.get_output_length(100) == 21  # 7 windows

    def test_learnable_queries(self, projector):
        """Test that queries are learnable parameters."""
        assert projector.query.requires_grad
        assert projector.query.shape == (1, 3, 256)  # num_queries=15//5=3


# =============================================================================
# Loss Function Tests
# =============================================================================


class TestLossFunctions:
    """Tests for auxiliary loss functions."""

    def test_load_balancing_loss_uniform(self):
        """Test that uniform distribution gives minimal load balancing loss."""
        num_experts = 4
        # Uniform probabilities
        probs = torch.ones(100, num_experts) / num_experts
        loss = load_balancing_loss(probs, num_experts, top_k=num_experts)
        assert loss < 1e-6

    def test_load_balancing_loss_imbalanced(self):
        """Test that imbalanced distribution gives higher loss."""
        num_experts = 4
        # All probability on one expert
        probs = torch.zeros(100, num_experts)
        probs[:, 0] = 1.0
        loss = load_balancing_loss(probs, num_experts, top_k=num_experts)
        assert loss > 0

    def test_z_loss_small_logits(self):
        """Test that small logits give small z-loss."""
        logits = torch.randn(100, 4) * 0.1  # Small logits
        loss = z_loss(logits)
        assert loss < 5.0

    def test_z_loss_large_logits(self):
        """Test that large logits give larger z-loss."""
        logits = torch.randn(100, 4) * 100  # Large logits
        loss = z_loss(logits)
        assert loss > 100


# =============================================================================
# Registry Tests
# =============================================================================


class TestProjectorRegistry:
    """Tests for projector registry."""

    def test_all_projectors_registered(self):
        """Test that all projector types are in the registry."""
        expected = {"mlp", "mosa", "moe", "qformer"}
        assert set(PROJECTOR_CLASSES.keys()) == expected

    def test_registry_instantiation(self):
        """Test that all registered projectors can be instantiated."""
        config = MockConfig()
        for _name, cls in PROJECTOR_CLASSES.items():
            projector = cls(config)
            assert hasattr(projector, "forward")
            assert hasattr(projector, "get_output_length")


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests for gradient flow through projectors."""

    @pytest.mark.parametrize("projector_type", ["mlp", "mosa", "moe"])
    def test_gradients_flow(self, projector_type):
        """Test that gradients flow through projector."""
        config = MockConfig()
        projector = PROJECTOR_CLASSES[projector_type](config)
        projector.train()

        x = torch.randn(2, 100, 256, requires_grad=True)
        out = projector(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
