"""Tests for the Nemotron + stock Qwen3-ASR pipeline's word-to-speaker assignment."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.speaker_asr.longform import LongFormResult
from scripts.speaker_asr.nemotron import (
    assign_speakers,
    fill_gaps,
    group_turns,
    time_all_words,
    transcribe_diarized,
)


def activity(spans, seconds=4.0, speakers=8):
    """(frames, speakers) at 10 ms with 1.0 over each (speaker, start_s, end_s)."""
    act = np.zeros((int(seconds * 100), speakers), dtype=np.float32)
    for speaker, start, end in spans:
        act[int(start * 100) : int(end * 100), speaker] = 1.0
    return act


class TestAssignSpeakers:
    def test_picks_most_active_speaker_over_word_span(self):
        act = activity([(0, 0.0, 1.0), (1, 1.0, 2.0)])
        assert assign_speakers([(0.2, 0.5), (1.2, 1.6)], act) == [0, 1]

    def test_word_straddling_turn_goes_to_majority(self):
        act = activity([(0, 0.0, 1.0), (1, 1.0, 2.0)])
        assert assign_speakers([(0.9, 1.5)], act) == [1]

    def test_silence_and_untimed_words_are_unassigned(self):
        act = activity([(0, 0.0, 1.0)])
        assert assign_speakers([(2.0, 2.5), (None, None)], act) == [None, None]

    def test_word_past_end_of_activity_uses_last_frame(self):
        act = activity([(3, 3.5, 4.0)])
        assert assign_speakers([(4.2, 4.4)], act) == [3]


class TestFillGaps:
    @pytest.mark.parametrize(
        ("speakers", "expected"),
        [
            ([0, None, 1, None], [0, 0, 1, 1]),
            ([None, None, 2], [2, 2, 2]),  # leading gap takes the first known speaker
            ([None, None], [0, 0]),  # nobody heard at all: one speaker
        ],
    )
    def test_fill(self, speakers, expected):
        assert fill_gaps(speakers) == expected


class TestTimeAllWords:
    def test_unaligned_words_are_kept_untimed_and_offset_applied(self):
        aligned = [
            {"word": "hello,", "start": 0.1, "end": 0.4},
            {"word": "world.", "start": 0.6, "end": 0.9},
        ]
        words = time_all_words("hello, -- world.", aligned, offset=10.0)
        assert [w for w, _, _ in words] == ["hello,", "--", "world."]
        assert words[0][1:] == pytest.approx((10.1, 10.4))
        assert words[1][1:] == (None, None)
        assert words[2][1:] == pytest.approx((10.6, 10.9))


class TestGroupTurns:
    def test_consecutive_words_merge_into_turns(self):
        words = [("a", 0.0, 0.1), ("b", 0.2, 0.3), ("c", 0.4, 0.5), ("d", None, None)]
        turns = group_turns(words, [5, 5, 2, 2])
        assert [(t.speaker, t.text, t.start, t.end) for t in turns] == [
            (5, "a b", 0.0, 0.3),
            (2, "c d", 0.4, 0.5),  # untimed word keeps the turn's last known end
        ]


class TestTranscribeDiarized:
    def test_end_to_end_with_stubbed_models(self, monkeypatch):
        """Words from two chunks land on their Nemotron speakers, numbered by first appearance."""

        class StubDiarizer:
            def activity(self, audio):
                return activity([(4, 0.0, 15.0), (7, 15.0, 30.0)], seconds=30.0)

        texts = iter(["hi there", "bye now"])

        def align(audio, text):
            return [
                {"word": w, "start": 1.0 + i, "end": 1.5 + i} for i, w in enumerate(text.split())
            ]

        audio = np.zeros(30 * 16000, dtype=np.float32)
        audio[15 * 16000] = 1.0  # quiet everywhere: force the cut via chunk_s
        result = transcribe_diarized(
            lambda chunk: next(texts),
            audio,
            diarizer=StubDiarizer(),
            chunk_s=20.0,
            min_chunk_s=15.0,
            align=align,
        )
        assert isinstance(result, LongFormResult)
        assert result.text == "<SPK_1>hi there<SPK_2>bye now"


class TestLocalEvaluatorSpeakers:
    """On a speaker dataset, a tiny-audio checkpoint is scored through its own pipeline."""

    @staticmethod
    def evaluator(result):
        from scripts.eval.evaluators.asr import LocalEvaluator

        calls = []

        def pipe(audio, **kwargs):
            calls.append(kwargs)
            return result

        ev = LocalEvaluator.__new__(LocalEvaluator)
        ev.pipe, ev.user_prompt, ev.speakers = pipe, None, True
        return ev, calls

    def test_words_become_speaker_turns(self):
        words = [
            {"word": "hi", "start": 0.0, "end": 0.2, "speaker": "SPEAKER_1"},
            {"word": "there", "start": 0.3, "end": 0.5, "speaker": "SPEAKER_1"},
            {"word": "yes", "start": 1.0, "end": 1.2, "speaker": "SPEAKER_0"},
        ]
        ev, calls = self.evaluator({"text": "hi there yes", "words": words})
        text, _, _ = ev.transcribe(np.zeros(16000, dtype=np.float32))
        assert calls[0]["return_speakers"] is True
        assert text == "<SPK_1>hi there<SPK_2>yes"  # numbered by first appearance

    def test_diarization_error_is_raised_not_scored(self):
        ev, _ = self.evaluator({"text": "hi", "words": [], "diarization_error": "no nemotron"})
        with pytest.raises(RuntimeError, match="no nemotron"):
            ev.transcribe(np.zeros(16000, dtype=np.float32))
