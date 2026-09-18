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

    def test_parameter_layout(self, projector):
        """Norm on the stacked input, biased linears, nothing after linear_2."""
        assert set(dict(projector.named_parameters())) == {
            "input_norm.weight",
            "linear_1.weight",
            "linear_1.bias",
            "linear_2.weight",
            "linear_2.bias",
        }

    def test_input_norm_sized_for_stacked_frames(self, projector):
        """input_norm runs after frame stacking, so it spans encoder_dim * k."""
        assert projector.input_norm.weight.shape == (256 * 4,)

    def test_output_scale_is_observable(self, projector):
        """Scaling linear_2 must scale the output.

        This is the property a trailing RMSNorm destroyed: it renormalized
        linear_2's output, so the loss could not see that weight's magnitude
        and the magnitude inflated freely. Biases are zeroed here so the
        relationship is exactly proportional.
        """
        with torch.no_grad():
            projector.linear_1.bias.zero_()
            projector.linear_2.bias.zero_()
        projector.eval()
        x = torch.randn(2, 100, 256)
        before = projector(x)
        with torch.no_grad():
            projector.linear_2.weight.mul_(3.0)
        assert torch.allclose(projector(x), before * 3.0, atol=1e-5)

    def test_input_scale_is_normalized_away(self, projector):
        """input_norm makes the projection invariant to encoder output scale."""
        projector.eval()
        x = torch.randn(2, 100, 256)
        assert torch.allclose(projector(x * 7.0), projector(x), atol=1e-4)


# =============================================================================
# Registry Tests
# =============================================================================


class TestProjectorRegistry:
    """Tests for projector registry."""

    def test_core_projectors_registered(self):
        """Test that core projector types are in the registry."""
        assert "mlp" in PROJECTOR_CLASSES

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
GRADIENT_TEST_PROJECTORS = ["mlp"]


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
