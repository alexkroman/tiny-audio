"""Tests for audio projector modules.

Uses MockProjectorConfig from conftest.py for projector initialization.
"""

import pytest
import torch
from conftest import MockProjectorConfig

from tiny_audio.projectors import (
    PROJECTOR_CLASSES,
    MLPAudioProjector,
    MoEAudioProjector,
    MOSAProjector,
    QFormerAudioProjector,
    SimpleAdapter,
)

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


# =============================================================================
# MLP Projector Tests
# =============================================================================


class TestMLPAudioProjector:
    """Tests for MLPAudioProjector."""

    @pytest.fixture
    def config(self):
        return MockProjectorConfig(encoder_dim=256, llm_dim=512, projector_pool_stride=4)

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
        for seq_len in [8, 48, 100, 200]:
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
        return MockProjectorConfig(
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
        """Test output length calculation for Conv1d downsampling (4x reduction)."""
        # Conv1d formula: out = (in + 2*pad - kernel) // stride + 1
        # With kernel=3, stride=2, pad=1: out = (in - 1) // 2 + 1
        # Applied twice for 4x total reduction
        assert projector.get_output_length(100) == 25
        assert projector.get_output_length(101) == 26
        assert projector.get_output_length(4) == 1
        assert projector.get_output_length(5) == 2


# =============================================================================
# MoE Projector Tests
# =============================================================================


class TestMoEAudioProjector:
    """Tests for MoEAudioProjector."""

    @pytest.fixture
    def config(self):
        return MockProjectorConfig(
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
        """Test output length calculation (matches MLP frame stacking)."""
        # Formula: (L - k) // k + 1 where k=4
        assert projector.get_output_length(100) == 25  # (100-4)//4+1 = 25
        assert projector.get_output_length(101) == 25  # (101-4)//4+1 = 25

    def test_has_shared_expert(self, projector):
        """Test that projector has a shared expert."""
        assert hasattr(projector, "shared_expert")
        assert projector.shared_expert is not None

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
        return MockProjectorConfig(
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
        expected_len = projector.get_output_length(100)
        assert out.shape == (2, expected_len, 512)

    def test_get_output_length(self, projector):
        """Test output length calculation."""
        # window_size=15, downsample_rate=5 -> num_queries=3
        # nblocks = ceil(input/15), output = nblocks * 3
        assert projector.get_output_length(15) == 3
        assert projector.get_output_length(16) == 6
        assert projector.get_output_length(30) == 6
        assert projector.get_output_length(100) == 21

    def test_learnable_queries(self, projector):
        """Test that queries are learnable parameters."""
        assert projector.query.requires_grad
        assert projector.query.shape == (1, 3, 256)


# =============================================================================
# Registry Tests
# =============================================================================


class TestProjectorRegistry:
    """Tests for projector registry."""

    def test_core_projectors_registered(self):
        """Test that core projector types are in the registry."""
        assert "mlp" in PROJECTOR_CLASSES
        assert "mosa" in PROJECTOR_CLASSES
        assert "moe" in PROJECTOR_CLASSES
        assert "qformer" in PROJECTOR_CLASSES

    def test_registry_instantiation(self):
        """Test that all registered projectors can be instantiated."""
        config = MockProjectorConfig()
        for _name, cls in PROJECTOR_CLASSES.items():
            projector = cls(config)
            assert hasattr(projector, "forward")
            assert hasattr(projector, "get_output_length")


# =============================================================================
# Gradient Flow Tests
# =============================================================================


# Projector types to test
GRADIENT_TEST_PROJECTORS = ["mlp", "mosa", "moe"]


class TestGradientFlow:
    """Tests for gradient flow through projectors."""

    @pytest.mark.parametrize("projector_type", GRADIENT_TEST_PROJECTORS)
    def test_gradients_flow(self, projector_type):
        """Test that gradients flow through projector."""
        config = MockProjectorConfig()
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
