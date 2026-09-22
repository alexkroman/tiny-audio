"""Tests for the projector's frame-stacking helpers."""

import pytest
import torch

from tiny_audio.projectors import _frame_stack, _frame_stack_length


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
