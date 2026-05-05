"""Tests for ForcedAligner — Viterbi trellis, backtrack, and align()."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


class TestGetTrellis:
    """_get_trellis builds a forward-DP trellis."""

    def test_trellis_shape(self):
        from tiny_audio.alignment import ForcedAligner

        # 5 frames, 3-class emission (blank=0, char_a=1, char_b=2)
        emission = torch.tensor(
            [
                [0.0, -1.0, -1.0],  # mostly blank
                [-1.0, 0.0, -1.0],  # mostly 'a'
                [-1.0, 0.0, -1.0],  # mostly 'a'
                [-1.0, -1.0, 0.0],  # mostly 'b'
                [0.0, -1.0, -1.0],  # mostly blank
            ]
        )
        tokens = [1, 2]  # Target: emit token 1 (a), then token 2 (b)
        trellis = ForcedAligner._get_trellis(emission, tokens, blank_id=0)
        # shape: (num_frames + 1, num_tokens + 1)
        assert trellis.shape == (6, 3)

    def test_trellis_starts_at_zero(self):
        from tiny_audio.alignment import ForcedAligner

        emission = torch.zeros(3, 3)
        tokens = [1, 2]
        trellis = ForcedAligner._get_trellis(emission, tokens, blank_id=0)
        assert trellis[0, 0].item() == 0.0
        # All other initial cells are -inf (only [0,0] is reachable at frame 0)
        assert trellis[0, 1].item() == float("-inf")


class TestBacktrack:
    """_backtrack returns one (token_id, start_frame, end_frame) per token."""

    def test_backtrack_returns_one_span_per_token(self):
        from tiny_audio.alignment import ForcedAligner

        # Same emissions as the trellis test: token 'a' (1) at frames 1-2, 'b' (2) at frame 3
        emission = torch.tensor(
            [
                [0.0, -1.0, -1.0],
                [-1.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0],
                [-1.0, -1.0, 0.0],
                [0.0, -1.0, -1.0],
            ]
        )
        tokens = [1, 2]
        trellis = ForcedAligner._get_trellis(emission, tokens, blank_id=0)
        spans = ForcedAligner._backtrack(trellis, emission, tokens, blank_id=0)
        assert len(spans) == 2
        # First span = token 1, second span = token 2
        assert spans[0][0] == 1
        assert spans[1][0] == 2
        # Token 1 should come before token 2 (monotonic)
        assert spans[0][1] <= spans[1][1]

    def test_backtrack_empty_tokens(self):
        from tiny_audio.alignment import ForcedAligner

        emission = torch.zeros(5, 3)
        trellis = ForcedAligner._get_trellis(emission, [], blank_id=0)
        spans = ForcedAligner._backtrack(trellis, emission, [], blank_id=0)
        assert spans == []

    def test_backtrack_falls_back_when_alignment_fails(self):
        """When trellis is all -inf at the end, falls back to uniform distribution."""
        from tiny_audio.alignment import ForcedAligner

        # All -inf emission means no path can reach the end of token sequence
        emission = torch.full((4, 3), float("-inf"))
        tokens = [1, 2]
        trellis = ForcedAligner._get_trellis(emission, tokens, blank_id=0)
        spans = ForcedAligner._backtrack(trellis, emission, tokens, blank_id=0)
        # Should fall back to uniform: 4 frames / 2 tokens = 2 frames each
        assert len(spans) == 2
        assert spans[0] == (1, 0.0, 2.0)
        assert spans[1] == (2, 2.0, 4.0)


@pytest.fixture
def reset_aligner_singleton():
    """Reset ForcedAligner class-level singleton state before/after each test."""
    from tiny_audio.alignment import ForcedAligner

    original = (
        ForcedAligner._bundle,
        ForcedAligner._model,
        ForcedAligner._labels,
        ForcedAligner._dictionary,
    )
    yield
    (
        ForcedAligner._bundle,
        ForcedAligner._model,
        ForcedAligner._labels,
        ForcedAligner._dictionary,
    ) = original


class TestAlign:
    """align() — full path with mocked torchaudio bundle."""

    def test_align_returns_word_list(self, reset_aligner_singleton):
        """align() with mocked emissions produces dicts with word/start/end keys."""
        from tiny_audio.alignment import ForcedAligner

        # Build a fake bundle/model that returns synthetic log-probs
        fake_labels = ("-", "|", "H", "E", "L", "O")
        # Vocab indices: blank=0, |=1, H=2, E=3, L=4, O=5
        # For text "HE", we want emission to favor H then E
        fake_emission = torch.tensor(
            [
                [
                    [-3.0, -3.0, 0.0, -3.0, -3.0, -3.0],  # H
                    [-3.0, -3.0, 0.0, -3.0, -3.0, -3.0],  # H
                    [-3.0, -3.0, -3.0, 0.0, -3.0, -3.0],  # E
                    [-3.0, -3.0, -3.0, 0.0, -3.0, -3.0],  # E
                    [0.0, -3.0, -3.0, -3.0, -3.0, -3.0],  # blank
                ]
            ]
        )

        fake_model = MagicMock()
        fake_model.return_value = (fake_emission, None)

        fake_bundle = MagicMock()
        fake_bundle.sample_rate = 16000

        ForcedAligner._bundle = fake_bundle
        ForcedAligner._model = fake_model
        ForcedAligner._labels = fake_labels
        ForcedAligner._dictionary = {c: i for i, c in enumerate(fake_labels)}

        audio = np.zeros(16000, dtype=np.float32)
        words = ForcedAligner.align(audio, "HE", sample_rate=16000)
        assert isinstance(words, list)
        for w in words:
            assert "word" in w
            assert "start" in w
            assert "end" in w
            assert w["start"] >= 0.0
            assert w["end"] >= w["start"]

    def test_align_empty_text_returns_empty(self, reset_aligner_singleton):
        """Empty token list short-circuits to []."""
        from tiny_audio.alignment import ForcedAligner

        fake_labels = ("-", "|")
        fake_model = MagicMock()
        fake_model.return_value = (torch.zeros(1, 3, 2), None)
        fake_bundle = MagicMock()
        fake_bundle.sample_rate = 16000

        ForcedAligner._bundle = fake_bundle
        ForcedAligner._model = fake_model
        ForcedAligner._labels = fake_labels
        ForcedAligner._dictionary = {c: i for i, c in enumerate(fake_labels)}

        audio = np.zeros(16000, dtype=np.float32)
        # Text with no chars present in dictionary → empty tokens
        words = ForcedAligner.align(audio, "@@@", sample_rate=16000)
        assert words == []


class TestGetInstance:
    """get_instance is a singleton — second call returns the same model."""

    def test_singleton_returns_cached(self):
        from tiny_audio.alignment import ForcedAligner

        # Pre-populate
        sentinel_model = MagicMock()
        sentinel_labels = ("-", "|")
        sentinel_dict = {"-": 0, "|": 1}

        original = (
            ForcedAligner._bundle,
            ForcedAligner._model,
            ForcedAligner._labels,
            ForcedAligner._dictionary,
        )
        try:
            ForcedAligner._bundle = MagicMock()
            ForcedAligner._model = sentinel_model
            ForcedAligner._labels = sentinel_labels
            ForcedAligner._dictionary = sentinel_dict

            model, labels, dictionary = ForcedAligner.get_instance(device="cpu")
            assert model is sentinel_model
            assert labels is sentinel_labels
            assert dictionary is sentinel_dict
        finally:
            (
                ForcedAligner._bundle,
                ForcedAligner._model,
                ForcedAligner._labels,
                ForcedAligner._dictionary,
            ) = original
