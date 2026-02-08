"""Tests for audio projector modules.

Uses MockProjectorConfig from conftest.py for projector initialization.
"""

import pytest
import torch
from conftest import MockProjectorConfig

from tiny_audio.projectors import (
    PROJECTOR_CLASSES,
    MLPAudioProjector,
)


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


class TestProjectorRegistry:
    """Tests for projector registry."""

    def test_mlp_registered(self):
        """Test that MLP projector is in the registry."""
        assert "mlp" in PROJECTOR_CLASSES

    def test_registry_instantiation(self):
        """Test that all registered projectors can be instantiated."""
        config = MockProjectorConfig()
        for cls in PROJECTOR_CLASSES.values():
            projector = cls(config)
            assert hasattr(projector, "forward")
            assert hasattr(projector, "get_output_length")


class TestGradientFlow:
    """Tests for gradient flow through projectors."""

    def test_gradients_flow(self):
        """Test that gradients flow through MLP projector."""
        config = MockProjectorConfig()
        projector = MLPAudioProjector(config)
        projector.train()

        x = torch.randn(2, 100, 256, requires_grad=True)
        out = projector(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
