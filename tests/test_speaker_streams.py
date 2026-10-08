"""Speaker streams (`return_speakers=True`): per-speaker masked ASR.

The diarizer, the ASR call and the aligner are all stubbed, so these run
offline: Nemotron activity is a hand-built array, each ASR call returns the
next scripted transcript, and the aligner times words from a lookup by text.
"""

from collections.abc import Sequence
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt
import pytest
import transformers

from tiny_audio.alignment import AlignedWord
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.asr_pipeline import (
    ASRPipeline,
    NemotronDiarizer,
    QwenForcedAligner,
    stream_chunks,
)
from tiny_audio.diarization import masked_audio

SR = 16000


def activity(
    spans: list[tuple[int, float, float, float]], seconds: float = 5.0
) -> npt.NDArray[np.float32]:
    """(frames, 8) activity: each `(col, start_s, end_s, p)` sets that column to `p`."""
    act = np.zeros((int(seconds * 100), 8), dtype=np.float32)
    for col, start, end, p in spans:
        act[int(start * 100) : int(end * 100), col] = p
    return act


def frames(*on: tuple[int, int], n: int = 20) -> npt.NDArray[np.bool_]:
    out = np.zeros(n, dtype=bool)
    for s, e in on:
        out[s:e] = True
    return out


class TestSampleSpans:
    def test_frames_to_samples(self) -> None:
        spans = NemotronDiarizer.sample_spans(frames((3, 5), (10, 12)), SR, 20 * 160)
        assert spans == [(3 * 160, 5 * 160), (10 * 160, 12 * 160)]

    def test_other_sample_rates(self) -> None:
        assert NemotronDiarizer.sample_spans(frames((3, 5)), 8000, 20 * 80) == [(240, 400)]

    def test_runs_at_both_edges(self) -> None:
        """A run from frame 0 starts at sample 0; one to the last frame runs to the end."""
        n = 20 * 160 + 37  # the model's frames fall a few samples short of the audio
        assert NemotronDiarizer.sample_spans(frames((0, 2), (17, 20)), SR, n) == [
            (0, 320),
            (17 * 160, n),
        ]

    def test_no_activity_no_spans(self) -> None:
        assert NemotronDiarizer.sample_spans(frames(), SR, 20 * 160) == []


class TestMaskedAudio:
    def test_zeros_outside_spans_in_the_slice(self) -> None:
        audio = np.arange(1, 11, dtype=np.float32)
        out = masked_audio(audio, [(1, 3), (6, 8)], 2, 9)
        assert out.tolist() == [3, 0, 0, 0, 7, 8, 0]

    def test_never_writes_to_the_input(self) -> None:
        audio = np.ones(10, dtype=np.float32)
        masked_audio(audio, [(0, 2)], 0, 10)
        assert audio.tolist() == [1.0] * 10


class TestSpanIsActive:
    def test_word_inside_and_outside_the_mask(self) -> None:
        mask = frames((10, 20), n=50)
        assert NemotronDiarizer.span_is_active(mask, 0.12, 0.15)
        assert NemotronDiarizer.span_is_active(mask, 0.05, 0.11)  # partly inside
        assert not NemotronDiarizer.span_is_active(mask, 0.25, 0.4)

    def test_zero_length_word_and_past_the_end(self) -> None:
        mask = frames((10, 20), n=50)
        assert NemotronDiarizer.span_is_active(mask, 0.15, 0.15)
        assert not NemotronDiarizer.span_is_active(mask, 9.0, 9.5)


class TestStreamChunks:
    def test_packs_turns_up_to_the_limit(self) -> None:
        audio = np.ones(60 * SR, dtype=np.float32)
        spans = [(0, 2 * SR), (5 * SR, 7 * SR), (17 * SR, 19 * SR), (30 * SR, 31 * SR)]
        assert stream_chunks(audio, spans, SR) == [(0, 7 * SR), (17 * SR, 31 * SR)]

    def test_long_turn_is_cut_at_its_quietest_point(self) -> None:
        audio = np.random.default_rng(0).normal(0, 0.1, 40 * SR).astype(np.float32)
        audio[14 * SR : int(14.2 * SR)] = 0.0  # 12 s into the turn
        ranges = stream_chunks(audio, [(2 * SR, 32 * SR)], SR)
        assert len(ranges) == 2
        assert ranges[0][0] == 2 * SR
        assert ranges[-1][1] == 32 * SR
        assert 14 * SR <= ranges[0][1] <= int(14.2 * SR)
        assert all((e - s) / SR <= 18.0 for s, e in ranges)


class TestDedupeWords:
    def test_keeps_the_more_active_speaker(self) -> None:
        act = activity([(0, 0, 2, 0.9), (1, 1, 3, 0.6)])
        words = [
            {"word": "Yes.", "start": 1.2, "end": 1.5, "speaker": "SPEAKER_0"},
            {"word": "yes", "start": 1.3, "end": 1.5, "speaker": "SPEAKER_1"},
            {"word": "no", "start": 1.3, "end": 1.6, "speaker": "SPEAKER_1"},
        ]
        out = NemotronDiarizer.dedupe_words(words, act, [0, 1])
        assert [(w["word"], w["speaker"]) for w in out] == [
            ("Yes.", "SPEAKER_0"),
            ("no", "SPEAKER_1"),
        ]

    def test_same_speaker_and_non_overlapping_repeats_stay(self) -> None:
        act = activity([(0, 0, 2, 0.9), (1, 1, 3, 0.6)])
        words = [
            {"word": "so", "start": 0.1, "end": 0.3, "speaker": "SPEAKER_0"},
            {"word": "so", "start": 0.2, "end": 0.4, "speaker": "SPEAKER_0"},
            {"word": "so", "start": 1.5, "end": 1.7, "speaker": "SPEAKER_1"},
        ]
        assert NemotronDiarizer.dedupe_words(words, act, [0, 1]) == words


class TestStreamsPipeline:
    """`ASRPipeline(..., return_speakers=True)` end to end, with every model stubbed."""

    @pytest.fixture
    def pipeline(self, base_asr_model: ASRModel) -> ASRPipeline:
        # device="cpu": see TestPipelineCall.pipeline in test_asr_pipeline.py.
        return ASRPipeline(
            model=base_asr_model,
            feature_extractor=base_asr_model.feature_extractor,
            tokenizer=base_asr_model.tokenizer,
            device="cpu",
        )

    @pytest.fixture
    def asr_inputs(self, monkeypatch: pytest.MonkeyPatch) -> list[npt.NDArray[np.float32]]:
        """Record every ASR input; call k returns `self.script[k]`."""
        calls: list[npt.NDArray[np.float32]] = []

        def fake_call(_self: object, inputs: dict[str, Any], **kwargs: Any) -> dict[str, str]:
            calls.append(np.asarray(inputs["raw"]))
            return {"text": self.script[len(calls) - 1]}

        monkeypatch.setattr(transformers.AutomaticSpeechRecognitionPipeline, "__call__", fake_call)
        return calls

    script: list[str]

    @staticmethod
    def stub_aligner(
        monkeypatch: pytest.MonkeyPatch, times: dict[str, tuple[float, float]]
    ) -> list[int]:
        """Align each word at `times[word]` (chunk-relative); returns chunk lengths seen."""
        seen: list[int] = []

        def fake_align(
            chunks: Sequence[tuple[npt.NDArray[np.float32], str]], sample_rate: int = SR
        ) -> list[list[AlignedWord]]:
            seen.extend(len(a) for a, _ in chunks)
            return [
                [{"word": w, "start": times[w][0], "end": times[w][1]} for w in text.split()]
                for _, text in chunks
            ]

        monkeypatch.setattr(QwenForcedAligner, "align_chunks", fake_align)
        return seen

    @staticmethod
    def stub_activity(monkeypatch: pytest.MonkeyPatch, act: npt.NDArray[np.float32]) -> None:
        def fake_activity(audio: npt.ArrayLike, sample_rate: int = SR) -> npt.NDArray[np.float32]:
            return act

        monkeypatch.setattr(NemotronDiarizer, "activity", fake_activity)

    @staticmethod
    def speech(seconds: float = 5.0) -> dict[str, Any]:
        audio = np.random.default_rng(0).normal(0, 0.1, int(seconds * SR)).astype(np.float32)
        return {"array": audio, "sampling_rate": SR}

    # Columns are in arrival order: column 1 is SPEAKER_0 (0-1 s and 3-4 s), column 4 is
    # SPEAKER_1 (1.5-2.5 s). Names follow position in `keep`, not the column index.
    TWO_SPEAKERS = ((1, 0.0, 1.0, 0.9), (1, 3.0, 4.0, 0.9), (4, 1.5, 2.5, 0.9))

    def test_single_speaker_is_not_masked(
        self,
        pipeline: ASRPipeline,
        asr_inputs: list[npt.NDArray[np.float32]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.script = ["hello there", "hello there"]
        self.stub_aligner(monkeypatch, {"hello": (0.5, 0.9), "there": (1.0, 1.4)})
        self.stub_activity(monkeypatch, activity([(3, 0.2, 4.0, 0.9), (6, 1.0, 1.2, 0.3)]))

        timed = pipeline(self.speech(), return_timestamps=True)
        streamed = pipeline(self.speech(), return_speakers=True)

        assert streamed["text"] == timed["text"] == "hello there"
        assert streamed["words"] == [{**w, "speaker": "SPEAKER_0"} for w in timed["words"]]
        assert all(len(a) == 5 * SR for a in asr_inputs)  # unmasked, whole clip

    def test_two_speakers_each_word_from_its_stream(
        self,
        pipeline: ASRPipeline,
        asr_inputs: list[npt.NDArray[np.float32]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.script = ["a1 a2", "b1"]  # SPEAKER_0's stream is transcribed first
        aligned = self.stub_aligner(
            monkeypatch, {"a1": (0.2, 0.5), "a2": (3.2, 3.6), "b1": (0.1, 0.6)}
        )
        self.stub_activity(monkeypatch, activity(list(self.TWO_SPEAKERS)))

        result = pipeline(self.speech(), return_speakers=True)

        # Only talk time reaches the model: 0-4 s for SPEAKER_0, 1.5-2.5 s for SPEAKER_1.
        assert [len(a) for a in asr_inputs] == [4 * SR, 1 * SR]
        assert aligned == [4 * SR, 1 * SR]  # one batched call over both streams
        assert not asr_inputs[0][1 * SR : 3 * SR].any()  # SPEAKER_1's turn is silenced
        assert asr_inputs[0][: 1 * SR].any()
        assert [(w["word"], w["speaker"]) for w in result["words"]] == [
            ("a1", "SPEAKER_0"),
            ("b1", "SPEAKER_1"),
            ("a2", "SPEAKER_0"),
        ]
        assert [w["start"] for w in result["words"]] == pytest.approx([0.2, 1.6, 3.2])
        assert result["text"] == "a1 b1 a2"
        assert result["speaker_segments"] == NemotronDiarizer.segments(
            activity(list(self.TWO_SPEAKERS)), [1, 4]
        )

    def test_fully_masked_audio_never_reaches_the_model(
        self,
        pipeline: ASRPipeline,
        asr_inputs: list[npt.NDArray[np.float32]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On a long recording only chunks holding the speaker's own turns are transcribed."""
        self.script = ["a", "b"]
        self.stub_aligner(monkeypatch, {"a": (0.1, 0.3), "b": (0.1, 0.3)})
        self.stub_activity(
            monkeypatch, activity([(0, 1.0, 2.0, 0.9), (1, 50.0, 51.0, 0.9)], seconds=60)
        )
        pipeline(self.speech(60.0), return_speakers=True)
        assert [len(a) for a in asr_inputs] == [SR, SR]

    def test_words_inside_masked_audio_are_dropped(
        self,
        pipeline: ASRPipeline,
        asr_inputs: list[npt.NDArray[np.float32]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.script = ["a1 ghost a2", "b1"]
        # "ghost" lands at 2.0 s, in SPEAKER_0's silenced 1-3 s gap.
        self.stub_aligner(
            monkeypatch,
            {"a1": (0.2, 0.5), "ghost": (2.0, 2.3), "a2": (3.2, 3.6), "b1": (0.1, 0.6)},
        )
        self.stub_activity(monkeypatch, activity(list(self.TWO_SPEAKERS)))

        result = pipeline(self.speech(), return_speakers=True)
        assert result["text"] == "a1 b1 a2"
        assert "ghost" not in {w["word"] for w in result["words"]}

    def test_word_heard_by_two_streams_is_kept_once(
        self,
        pipeline: ASRPipeline,
        asr_inputs: list[npt.NDArray[np.float32]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Overlapping speakers both hear "yes"; dedupe keeps the louder stream's copy."""
        self.script = ["well Yes.", "yes"]
        self.stub_aligner(
            monkeypatch,
            {"well": (0.1, 0.4), "Yes.": (1.2, 1.5), "yes": (0.25, 0.5)},
        )
        # SPEAKER_0 0-2 s at 0.9; SPEAKER_1 overlaps 1-3 s at 0.6. B's chunk starts at 1.0.
        self.stub_activity(monkeypatch, activity([(0, 0.0, 2.0, 0.9), (1, 1.0, 3.0, 0.6)]))

        result = pipeline(self.speech(), return_speakers=True)
        assert len(asr_inputs) == 2  # both streams were transcribed
        assert result["text"] == "well Yes."
        assert [w["speaker"] for w in result["words"]] == ["SPEAKER_0", "SPEAKER_0"]

    def test_diarization_error_is_reported_not_raised(
        self,
        pipeline: ASRPipeline,
        asr_inputs: list[npt.NDArray[np.float32]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.script = ["a1 a2"]
        self.stub_aligner(monkeypatch, {"a1": (0.2, 0.5), "a2": (3.2, 3.6)})

        def boom(audio: npt.ArrayLike, sample_rate: int = SR) -> NoReturn:
            msg = "no nemotron"
            raise RuntimeError(msg)

        monkeypatch.setattr(NemotronDiarizer, "activity", boom)
        result = pipeline(self.speech(), return_speakers=True)
        assert result["diarization_error"] == "no nemotron"
        assert result["speaker_segments"] == []
        assert result["text"] == "a1 a2"
        assert [w["word"] for w in result["words"]] == ["a1", "a2"]

    def test_alignment_error_is_reported_not_raised(
        self,
        pipeline: ASRPipeline,
        asr_inputs: list[npt.NDArray[np.float32]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.script = ["a1 a2", "b1"]
        self.stub_activity(monkeypatch, activity(list(self.TWO_SPEAKERS)))

        def boom(*args: object, **kwargs: object) -> NoReturn:
            msg = "no aligner"
            raise RuntimeError(msg)

        monkeypatch.setattr(QwenForcedAligner, "align_chunks", boom)
        result = pipeline(self.speech(), return_speakers=True)
        assert result["timestamp_error"] == "no aligner"
        assert result["words"] == []
        assert result["text"] == "a1 a2 b1"  # chunk transcripts by start time
