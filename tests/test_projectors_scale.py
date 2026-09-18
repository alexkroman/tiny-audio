"""Tests for MLPAudioProjector's output-scale calibration and gradient probe."""

from types import SimpleNamespace

import pytest
import torch

from tiny_audio.projectors import MLPAudioProjector, _frame_stack, _frame_stack_length


def scaled_config(target) -> SimpleNamespace:
    """Projector config carrying an explicit `projector_output_rms`."""
    return SimpleNamespace(
        encoder_dim=256,
        llm_dim=512,
        projector_hidden_dim=1024,
        projector_pool_stride=4,
        projector_output_rms=target,
    )


class TestFrameStacking:
    """Frame stacking drops the trailing partial window."""

    @pytest.mark.parametrize(
        ("length", "k", "expected"), [(8, 4, 2), (7, 4, 1), (3, 4, 0), (9, 2, 4)]
    )
    def test_length_rule(self, length, k, expected):
        assert _frame_stack_length(length, k) == expected

    def test_length_rule_on_tensor(self):
        out = _frame_stack_length(torch.tensor([8, 9, 3]), 4)
        assert out.tolist() == [2, 2, 0]

    def test_stack_concatenates_adjacent_frames(self):
        x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        out = _frame_stack(x, 2)
        assert out.shape == (2, 2, 6)
        # Frame 0 and 1 of sample 0, concatenated along the feature dim.
        assert torch.equal(out[0, 0], torch.cat([x[0, 0], x[0, 1]]))
        # The fifth frame does not fill a window and is dropped.
        assert torch.equal(out[0, 1], torch.cat([x[0, 2], x[0, 3]]))


class TestOutputScaleCalibration:
    """`output_scale` is set only for a concrete positive RMS target."""

    @pytest.mark.parametrize("target", ["auto", None, True, 0.0, -1.0, "0.5"])
    def test_non_numeric_or_non_positive_target_keeps_unit_scale(self, target):
        projector = MLPAudioProjector(scaled_config(target))
        assert projector.output_scale.item() == pytest.approx(1.0)

    def test_float_target_sets_output_rms(self):
        torch.manual_seed(0)
        projector = MLPAudioProjector(scaled_config(0.015))
        assert projector.output_scale.item() != pytest.approx(1.0)
        # The probe is a fresh random draw, so allow a few percent of noise.
        assert projector.measure_output_rms() == pytest.approx(0.015, rel=0.1)

    def test_scale_is_a_persistent_buffer(self):
        projector = MLPAudioProjector(scaled_config(0.5))
        state = projector.state_dict()
        assert "output_scale" in state
        assert not any(p is projector.output_scale for p in projector.parameters())
        # The gradient probe's accumulators are neither parameters nor buffers.
        assert "scale_grad" not in state
        assert "scale_grad_count" not in state

    def test_scale_round_trips_through_state_dict(self):
        src = MLPAudioProjector(scaled_config(0.02))
        dst = MLPAudioProjector(scaled_config("auto"))
        dst.load_state_dict(src.state_dict())
        assert dst.output_scale.item() == pytest.approx(src.output_scale.item())

    def test_output_is_linear_in_scale(self, projector_config):
        projector = MLPAudioProjector(projector_config()).eval()
        x = torch.randn(1, 16, 256)
        with torch.no_grad():
            base = projector(x)
            projector.output_scale.fill_(3.0)
            scaled = projector(x)
        assert torch.allclose(scaled, 3.0 * base, atol=1e-5)


class TestScaleGradientProbe:
    """The backward hook accumulates dL/d(log c) only while training with grad."""

    def test_accumulates_in_training_mode(self, projector_config):
        projector = MLPAudioProjector(projector_config()).train()
        x = torch.randn(2, 8, 256)
        weights = torch.randn(2, 2, 512)

        out = projector(x)
        # L = sum(W * out) => dL/dout = W, so the probe equals L itself.
        loss = (out * weights).sum()
        loss.backward()

        assert projector.scale_grad_count == 1
        assert projector.scale_grad is not None
        assert projector.scale_grad.item() == pytest.approx(loss.item(), rel=1e-4)

    def test_accumulates_across_forwards(self, projector_config):
        projector = MLPAudioProjector(projector_config()).train()
        total = 0.0
        for _ in range(3):
            out = projector(torch.randn(1, 8, 256))
            loss = out.sum()
            loss.backward()
            total += loss.item()
        assert projector.scale_grad_count == 3
        assert projector.scale_grad.item() == pytest.approx(total, rel=1e-4)

    def test_silent_in_eval_mode(self, projector_config):
        projector = MLPAudioProjector(projector_config()).eval()
        out = projector(torch.randn(1, 8, 256))
        out.sum().backward()
        assert projector.scale_grad is None
        assert projector.scale_grad_count == 0

    def test_silent_without_grad(self, projector_config):
        projector = MLPAudioProjector(projector_config()).train()
        with torch.no_grad():
            projector(torch.randn(1, 8, 256))
        assert projector.scale_grad is None
        assert projector.scale_grad_count == 0
