"""Tests for tiny_audio.turns: marker margins, thresholds, streaming helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_audio.turns import (
    SAMPLE_RATE,
    first_fire_times,
    marker_margins,
    set_end_of_turn_threshold,
    trailing_is_silent,
)

EOT, IM_END = 9, 7


def test_margin_read_at_first_eot_or_im_end():
    # row 0 fires at step 2; row 1 ends at step 1; row 2 never ends
    new = torch.tensor([[3, 4, EOT, IM_END], [3, IM_END, 7, 7], [3, 4, 5, 6]])
    pair = torch.zeros(3, 4, 2)
    pair[0, 2] = torch.tensor([5.0, 1.5])
    pair[1, 1] = torch.tensor([-1.0, 2.0])
    margins = marker_margins(new, pair, EOT, IM_END)
    assert margins[0].item() == pytest.approx(3.5)
    assert margins[1].item() == pytest.approx(-3.0)
    assert torch.isnan(margins[2])


def test_set_end_of_turn_threshold_round_trips(tmp_path):
    from transformers import GenerationConfig

    config = GenerationConfig(sequence_bias=[[[42], 1.5]])
    set_end_of_turn_threshold(config, 151705, 2.5)
    config.save_pretrained(tmp_path)
    back = GenerationConfig.from_pretrained(tmp_path)
    assert back.sequence_bias == [[[42], 1.5], [[151705], -2.5]]  # other biases kept
    set_end_of_turn_threshold(back, 151705, 2.0)
    assert back.sequence_bias == [[[42], 1.5], [[151705], -2.0]]  # replaced, not stacked
    set_end_of_turn_threshold(back, 151705, 0.0)
    assert back.sequence_bias == [[[42], 1.5]]


def test_trailing_is_silent():
    rng = np.random.default_rng(0)
    speech = (0.1 * rng.standard_normal(SAMPLE_RATE)).astype(np.float32)
    audio = np.concatenate([speech, np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)])
    assert trailing_is_silent(audio, 0.3)
    assert not trailing_is_silent(audio, 0.6)


def test_first_fire_times_per_threshold():
    times = [0.16, 0.32, 0.48, 0.64]
    margins = [-5.0, 1.0, float("nan"), 3.5]
    fires = first_fire_times(times, margins, [-6.0, 0.0, 2.0, 5.0])
    assert fires == {-6.0: 0.16, 0.0: 0.32, 2.0: 0.64, 5.0: None}


def test_turns_package_never_imports_the_rest_of_tiny_audio():
    """Keeps tiny_audio/turns/ liftable into its own package unchanged."""
    root = Path(__file__).resolve().parents[1] / "tiny_audio" / "turns"
    allowed = {"__future__", "collections", "typing", "numpy", "torch", "transformers"}
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                continue  # relative imports stay inside turns/
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module]
            else:
                continue
            for name in names:
                assert name.split(".")[0] in allowed, f"{path.name} imports {name}"
