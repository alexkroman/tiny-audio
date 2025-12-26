"""Tests for audio projector modules."""

import pytest
import torch

from src.projectors import (
    PROJECTOR_CLASSES,
    MLPAudioProjector,
    MOSAProjector,
    QFormerAudioProjector,
    ResidualAudioProjector,
    SharedMoEAudioProjector,
    SimpleAdapter,
    SwiGLU,
    SwiGLUAudioProjector,
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
        self.qformer_hidden_size = kwargs.get("qformer_hidden_size", None)
        self.qformer_num_layers = kwargs.get("qformer_num_layers", 2)
        self.qformer_num_heads = kwargs.get("qformer_num_heads", 8)
        self.qformer_intermediate_size = kwargs.get("qformer_intermediate_size", None)


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


class TestSwiGLU:
    """Tests for SwiGLU module."""

    def test_forward_shape(self):
        """Test that SwiGLU produces correct output shape."""
        swiglu = SwiGLU(in_features=256, hidden_features=512, out_features=128)
        x = torch.randn(2, 10, 256)
        out = swiglu(x)
        assert out.shape == (2, 10, 128)


# =============================================================================
# MLP Projector Tests
# =============================================================================


class TestMLPAudioProjector:
    """Tests for MLPAudioProjector."""

    @pytest.fixture
    def config(self):
        return MockConfig(encoder_dim=256, llm_dim=512)

    @pytest.fixture
    def projector(self, config):
        return MLPAudioProjector(config)

    def test_forward_shape(self, projector):
        """Test that MLP projector produces correct output shape."""
        x = torch.randn(2, 100, 256)
        out = projector(x)
        # Stride-2 conv halves sequence length
        assert out.shape == (2, 50, 512)

    def test_get_output_length(self, projector):
        """Test output length calculation."""
        # Stride-2 conv: (n + 1) // 2
        assert projector.get_output_length(100) == 50
        assert projector.get_output_length(101) == 51
        assert projector.get_output_length(1) == 1

    def test_downsampling(self, projector):
        """Test that downsampling reduces sequence length by 2."""
        for seq_len in [10, 50, 100, 101]:
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
        """Test output length calculation for stride-4."""
        assert projector.get_output_length(100) == 25
        assert projector.get_output_length(101) == 26  # Padded to 104
        assert projector.get_output_length(4) == 1
        assert projector.get_output_length(5) == 2  # Padded to 8

    def test_routing_weights_stored(self, projector):
        """Test that router logits and weights are stored after forward."""
        x = torch.randn(2, 100, 256)
        _ = projector(x)
        assert projector.last_router_logits is not None
        assert projector.last_routing_weights is not None
        # Shape should be (batch, seq//4, num_experts)
        assert projector.last_routing_weights.shape == (2, 25, 4)

    def test_routing_weights_sum_to_one(self, projector):
        """Test that routing weights sum to 1 (softmax)."""
        x = torch.randn(2, 100, 256)
        _ = projector(x)
        sums = projector.last_routing_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_aux_loss(self, projector):
        """Test auxiliary loss computation."""
        x = torch.randn(2, 100, 256)
        _ = projector(x)
        aux_loss = projector.get_aux_loss()
        assert aux_loss.numel() == 1
        assert aux_loss >= 0

    def test_aux_loss_zero_when_disabled(self, config):
        """Test aux loss is zero when coefficients are zero."""
        config.router_aux_loss_coef = 0.0
        config.router_z_loss_coef = 0.0
        projector = MOSAProjector(config)
        x = torch.randn(2, 100, 256)
        _ = projector(x)
        aux_loss = projector.get_aux_loss()
        assert aux_loss == 0.0


# =============================================================================
# SwiGLU Projector Tests
# =============================================================================


class TestSwiGLUAudioProjector:
    """Tests for SwiGLUAudioProjector."""

    @pytest.fixture
    def config(self):
        return MockConfig(
            encoder_dim=256, llm_dim=512, projector_hidden_dim=512, projector_pool_stride=4
        )

    @pytest.fixture
    def projector(self, config):
        return SwiGLUAudioProjector(config)

    def test_forward_shape(self, projector):
        """Test that SwiGLU projector produces correct output shape."""
        x = torch.randn(2, 100, 256)
        out = projector(x)
        # Stride-4 pooling
        assert out.shape == (2, 25, 512)

    def test_get_output_length(self, projector):
        """Test output length calculation."""
        assert projector.get_output_length(100) == 25
        assert projector.get_output_length(101) == 26  # Padded to 104
        assert projector.get_output_length(4) == 1

    def test_handles_padding(self, projector):
        """Test that projector handles sequences not divisible by stride."""
        for seq_len in [99, 100, 101, 102, 103]:
            x = torch.randn(1, seq_len, 256)
            out = projector(x)
            expected_len = projector.get_output_length(seq_len)
            assert out.shape[1] == expected_len


# =============================================================================
# Residual Projector Tests
# =============================================================================


class TestResidualAudioProjector:
    """Tests for ResidualAudioProjector."""

    @pytest.fixture
    def config(self):
        return MockConfig(
            encoder_dim=256,
            llm_dim=512,
            projector_hidden_dim=1024,
            projector_pool_stride=4,
            projector_num_layers=2,
        )

    @pytest.fixture
    def projector(self, config):
        return ResidualAudioProjector(config)

    def test_forward_shape(self, projector):
        """Test that Residual projector produces correct output shape."""
        x = torch.randn(2, 100, 256)
        out = projector(x)
        assert out.shape == (2, 25, 512)

    def test_get_output_length(self, projector):
        """Test output length calculation."""
        assert projector.get_output_length(100) == 25
        assert projector.get_output_length(101) == 26

    def test_num_layers(self, config):
        """Test that projector respects num_layers config."""
        config.projector_num_layers = 3
        projector = ResidualAudioProjector(config)
        assert len(projector.layers) == 3
        assert len(projector.layer_norms) == 3


# =============================================================================
# Shared MoE Projector Tests
# =============================================================================


class TestSharedMoEAudioProjector:
    """Tests for SharedMoEAudioProjector."""

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
        return SharedMoEAudioProjector(config)

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
        expected = {"mlp", "mosa", "swiglu", "residual", "shared_moe", "qformer"}
        assert set(PROJECTOR_CLASSES.keys()) == expected

    def test_registry_instantiation(self):
        """Test that all registered projectors can be instantiated."""
        config = MockConfig()
        for name, cls in PROJECTOR_CLASSES.items():
            projector = cls(config)
            assert hasattr(projector, "forward")
            assert hasattr(projector, "get_output_length")


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests for gradient flow through projectors."""

    @pytest.mark.parametrize("projector_type", ["mlp", "mosa", "swiglu", "residual", "shared_moe"])
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
