"""Tests for ForcedAligner — token alignment, word pairing, and align()."""

from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


class TestAlignTokens:
    """_align_tokens wraps torchaudio's CTC forced_align and returns one span per token."""

    def test_returns_one_span_per_token_in_order(self):
        from tiny_audio.alignment import ForcedAligner

        # 5 frames, 3-class emission (blank=0, char_a=1, char_b=2):
        # 'a' at frames 1-2, 'b' at frame 3, blank elsewhere.
        emission = torch.tensor(
            [
                [0.0, -1.0, -1.0],
                [-1.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0],
                [-1.0, -1.0, 0.0],
                [0.0, -1.0, -1.0],
            ]
        )
        spans = ForcedAligner._align_tokens(emission, [1, 2], blank_id=0)
        assert [s[0] for s in spans] == [1, 2]
        # 'a' spans both of its frames; 'b' its single frame; ends are exclusive.
        assert spans[0][1:] == (1.0, 3.0)
        assert spans[1][1:] == (3.0, 4.0)

    def test_repeated_tokens_are_separated_by_blank(self):
        """CTC needs a blank between identical consecutive targets ("LL")."""
        from tiny_audio.alignment import ForcedAligner

        emission = torch.full((5, 2), -5.0)
        emission[0, 1] = 0.0  # L
        emission[1, 0] = 0.0  # blank
        emission[2, 1] = 0.0  # L
        emission[3:, 0] = 0.0
        spans = ForcedAligner._align_tokens(emission, [1, 1], blank_id=0)
        assert [(s[0], s[1], s[2]) for s in spans] == [(1, 0.0, 1.0), (1, 2.0, 3.0)]

    def test_empty_tokens(self):
        from tiny_audio.alignment import ForcedAligner

        assert ForcedAligner._align_tokens(torch.zeros(5, 3), [], blank_id=0) == []

    def test_falls_back_to_uniform_when_no_path_has_probability(self):
        """An all -inf emission leaves no valid path; spread tokens uniformly."""
        from tiny_audio.alignment import ForcedAligner

        emission = torch.full((4, 3), float("-inf"))
        spans = ForcedAligner._align_tokens(emission, [1, 2], blank_id=0)
        assert spans == [(1, 0.0, 2.0), (2, 2.0, 4.0)]

    def test_falls_back_to_uniform_when_more_tokens_than_frames(self):
        """torchaudio raises when the targets cannot fit; align must not."""
        from tiny_audio.alignment import ForcedAligner

        emission = torch.log_softmax(torch.zeros(2, 4), dim=-1)
        spans = ForcedAligner._align_tokens(emission, [1, 2, 3], blank_id=0)
        assert [s[0] for s in spans] == [1, 2, 3]
        assert spans[0][1] == 0.0
        assert spans[-1][2] == 2.0


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
