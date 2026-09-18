"""Tests for ForcedAligner — Viterbi trellis, backtrack, and align()."""

from typing import ClassVar
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


@pytest.mark.usefixtures("reset_aligner_singleton")
class TestAlign:
    """align() — full path with mocked torchaudio bundle."""

    def test_align_returns_word_list(self):
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

    def test_align_empty_text_returns_empty(self):
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


class TestTokenizeWords:
    """_tokenize_words keeps words and CTC token groups in lockstep."""

    # blank=0, separator=1, then the letters.
    LABELS = ("-", "|", "H", "E", "L", "O", "W", "R", "D")
    DICT: ClassVar[dict[str, int]] = {c: i for i, c in enumerate(LABELS)}

    def test_plain_words_are_separator_joined(self):
        from tiny_audio.alignment import ForcedAligner

        words, tokens = ForcedAligner._tokenize_words("hello world", self.DICT)

        assert words == ["hello", "world"]
        assert tokens.count(self.DICT["|"]) == 1

    def test_unrepresentable_word_is_dropped_not_shifted(self):
        """ "--" has no representable characters, so it cannot be aligned.

        Leaving it in `words` while it contributes no token group shifted every
        later word onto the previous word's timing and dropped the last one.
        """
        from tiny_audio.alignment import ForcedAligner

        words, tokens = ForcedAligner._tokenize_words("hello -- world", self.DICT)

        assert words == ["hello", "world"]
        # Exactly one separator: no empty group between the two real words.
        assert tokens.count(self.DICT["|"]) == 1

    def test_one_group_per_word(self):
        """Token groups split on the separator must match `words` 1:1."""
        from tiny_audio.alignment import ForcedAligner

        words, tokens = ForcedAligner._tokenize_words("hello 123 world ... do", self.DICT)

        groups = []
        current: list[int] = []
        for t in tokens:
            if t == self.DICT["|"]:
                groups.append(current)
                current = []
            else:
                current.append(t)
        groups.append(current)

        assert words == ["hello", "world", "do"]
        assert len(groups) == len(words)
        assert all(groups)

    def test_no_representable_characters_at_all(self):
        from tiny_audio.alignment import ForcedAligner

        assert ForcedAligner._tokenize_words("123 ---", self.DICT) == ([], [])


class TestAlignWordPairing:
    """align() must label each timing span with the word it came from."""

    LABELS = ("-", "|", "H", "E", "L", "O", "W", "R", "D")

    def _emission(self, token_ids: list[int]) -> torch.Tensor:
        """One high-probability frame per token, blank-separated."""
        num_labels = len(self.LABELS)
        frames = []
        for tid in token_ids:
            row = [-8.0] * num_labels
            row[tid] = 0.0
            frames.append(row)
            blank = [-8.0] * num_labels
            blank[0] = 0.0
            frames.append(blank)
        return torch.tensor([frames])

    def test_dashes_do_not_shift_word_labels(self):
        """Replays the reported failure: "hello -- world" mislabeled the spans."""
        from tiny_audio.alignment import ForcedAligner

        dictionary = {c: i for i, c in enumerate(self.LABELS)}
        # HELLO | WORLD
        target = [dictionary[c] for c in ["H", "E", "L", "L", "O", "|", "W", "O", "R", "L", "D"]]

        fake_model = MagicMock()
        fake_model.return_value = (self._emission(target), None)
        fake_bundle = MagicMock()
        fake_bundle.sample_rate = 16000

        ForcedAligner._bundle = fake_bundle
        ForcedAligner._model = fake_model
        ForcedAligner._labels = self.LABELS
        ForcedAligner._dictionary = dictionary

        words = ForcedAligner.align(np.zeros(16000, dtype=np.float32), "hello -- world")

        assert [w["word"] for w in words] == ["hello", "world"]
        assert words[0]["end"] <= words[1]["end"]
        for w in words:
            assert w["end"] >= w["start"] >= 0.0


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
