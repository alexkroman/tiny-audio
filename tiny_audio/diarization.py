"""Speaker diarization with NVIDIA Nemotron-3-Diarization."""

import importlib.util
import re
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

if TYPE_CHECKING:
    # In transformers main after 5.17; loaded through the Auto classes at runtime
    # so an older build reaches get_instance's install hint instead of failing here.
    from transformers import (
        Nemotron3DiarizationForAudioFrameClassification,
        Nemotron3DiarizationProcessor,
    )

    from .alignment import AlignedWord, get_device, to_16k
else:
    try:
        from .alignment import get_device, to_16k
    except ImportError:  # flat layout on the Hub: sibling modules, no package
        from alignment import get_device, to_16k


class SpeakerSegment(TypedDict):
    """One speaker turn, in seconds."""

    speaker: str
    start: float
    end: float


def _label_runs(labels: npt.NDArray[np.generic]) -> list[tuple[int, int, int]]:
    """Split a 1-D label array into maximal constant runs.

    Returns `(value, start, end)` triples with `end` exclusive, in order. This
    is the frames-to-segments step for speaker activity: a boundary is wherever
    the label changes, and the last run always closes at `len(labels)`, so a
    trailing run needs no special case.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [labels.size]))
    return [(labels[s].item(), int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def masked_audio(
    audio: npt.NDArray[np.float32], spans: list[tuple[int, int]], start: int, end: int
) -> npt.NDArray[np.float32]:
    """`audio[start:end]` with every sample outside `spans` set to 0.0.

    `spans` are absolute `(start, end)` sample ranges, end exclusive. Only the
    slice is copied, so a speaker stream of a long recording never holds a
    full-length masked copy. Zeros are digital silence: the encoder sees its
    silence floor there, and `is_silent` skips a chunk that is nothing else.
    """
    out = np.zeros(end - start, dtype=np.float32)
    for s, e in spans:
        lo, hi = max(s, start), min(e, end)
        if lo < hi:
            out[lo - start : hi - start] = audio[lo:hi]
    return out


def pack_spans(spans: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    """Merge consecutive sorted `(start, end)` spans into ranges at most `limit` long.

    Each range starts and ends on a span, so a speaker's chunk never begins or
    ends in audio where they are silent. A span already over `limit` stays alone.
    """
    ranges: list[tuple[int, int]] = []
    for s, e in spans:
        if ranges and e - ranges[-1][0] <= limit:
            ranges[-1] = (ranges[-1][0], e)
        else:
            ranges.append((s, e))
    return ranges


class StreamChunk(TypedDict):
    """One stretch of one speaker's masked audio: kept column, absolute start, samples."""

    col: int
    start: int
    audio: npt.NDArray[np.float32]


_NON_WORD_RE = re.compile(r"[^\w']+")


def _norm_word(word: str) -> str:
    """Lowercase `word` without punctuation, for matching one word heard in two streams."""
    return _NON_WORD_RE.sub("", word.lower())


class NemotronDiarizer:
    """Speaker activity from NVIDIA Nemotron-3-Diarization, per 10 ms frame.

    One offline pass over the whole recording: the model chunks internally
    (27.2 s + 3.2 s look-ahead) and carries a speaker cache across chunks, so
    speaker ids are recording-level for audio of any length, with no
    embedding or clustering step. Overlap is native (several speakers can be
    active in one frame). Tracks at most 8 speakers, numbered by arrival.

    Needs a transformers build with `nemotron3_diarization` (main as of
    2026-10, after 5.17).

    Example:
        >>> activity = NemotronDiarizer.activity(audio)  # (frames, 8) probabilities
        >>> keep = NemotronDiarizer.top_speakers(activity, max_speakers=2)
        >>> segments = NemotronDiarizer.segments(activity, keep)

    `ASRPipeline(..., return_speakers=True)` turns this into speaker streams:
    `sample_spans` gives each kept speaker a mask, the ASR runs once per
    speaker on audio with everyone else silenced, and `span_is_active` /
    `dedupe_words` clean up the merge.
    """

    MODEL_ID = "nvidia/Nemotron-3-Diarization"
    FRAME_S = 0.01
    SEGMENT_THRESHOLD = 0.5  # a speaker's segment = frames above this
    _model: "Nemotron3DiarizationForAudioFrameClassification | None" = None
    _processor: "Nemotron3DiarizationProcessor | None" = None

    @classmethod
    def get_instance(
        cls,
    ) -> "tuple[Nemotron3DiarizationForAudioFrameClassification, Nemotron3DiarizationProcessor]":
        """Load the diarization model and processor once, then return the cached pair."""
        if cls._model is None or cls._processor is None:
            if importlib.util.find_spec("transformers.models.nemotron3_diarization") is None:
                msg = (
                    "Speaker diarization uses Nemotron-3-Diarization, which needs a transformers "
                    "build with `nemotron3_diarization`: "
                    "pip install git+https://github.com/huggingface/transformers"
                )
                raise ImportError(msg)
            model: Nemotron3DiarizationForAudioFrameClassification = (
                AutoModelForAudioFrameClassification.from_pretrained(cls.MODEL_ID)
            )
            # PreTrainedModel.to is functools.wraps'd, which pyright cannot bind as a method.
            model.to(get_device())  # pyright: ignore[reportArgumentType]
            model.eval()
            cls._model = model
            processor: Nemotron3DiarizationProcessor = AutoProcessor.from_pretrained(cls.MODEL_ID)
            cls._processor = processor
        return cls._model, cls._processor

    @classmethod
    @torch.inference_mode()
    def activity(cls, audio: npt.ArrayLike, sample_rate: int = 16000) -> npt.NDArray[np.float32]:
        """(frames, 8) speech probabilities at 10 ms; columns in arrival order."""
        model, processor = cls.get_instance()
        inputs = processor(to_16k(audio, sample_rate), sampling_rate=16000)
        inputs = inputs.to(model.device, dtype=model.dtype)
        probs: npt.NDArray[np.float32] = model(**inputs).logits[0].float().sigmoid().cpu().numpy()
        return probs

    @classmethod
    def top_speakers(
        cls,
        activity: npt.NDArray[np.float32],
        num_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[int]:
        """Columns that count as speakers, in arrival order.

        Every column that is ever above SEGMENT_THRESHOLD, capped by
        `num_speakers` (exact, when known) or `max_speakers` to the columns
        with the most speech. A dropped column gets no stream of its own.
        """
        active = [
            c for c in range(activity.shape[1]) if (activity[:, c] > cls.SEGMENT_THRESHOLD).any()
        ]
        cap = num_speakers or max_speakers
        if cap is not None and len(active) > cap:
            mass = activity.sum(axis=0)
            active = sorted(sorted(active, key=lambda c: -mass[c])[:cap])
        return active or [0]

    @staticmethod
    def speaker_names(keep: list[int]) -> dict[int, str]:
        """Column -> label: `SPEAKER_i` for the column's position in `keep`."""
        return {c: f"SPEAKER_{i}" for i, c in enumerate(keep)}

    @classmethod
    def segments(cls, activity: npt.NDArray[np.float32], keep: list[int]) -> list[SpeakerSegment]:
        """Speaker turns `{"speaker", "start", "end"}` by start time; overlaps allowed."""
        names = cls.speaker_names(keep)
        out: list[SpeakerSegment] = []
        for c in keep:
            for on, s, e in _label_runs(activity[:, c] > cls.SEGMENT_THRESHOLD):
                if on:
                    out.append(
                        {"speaker": names[c], "start": s * cls.FRAME_S, "end": e * cls.FRAME_S}
                    )
        return sorted(out, key=lambda seg: (seg["start"], seg["end"]))

    @classmethod
    def _frame_range(cls, start: float, end: float, n_frames: int) -> tuple[int, int]:
        """Frames `[lo, hi)` under the span `start..end` s: at least one, clamped to the array."""
        lo = min(int(start / cls.FRAME_S), n_frames - 1)
        hi = max(lo + 1, min(int(np.ceil(end / cls.FRAME_S)), n_frames))
        return lo, hi

    @classmethod
    def sample_spans(
        cls, frames: npt.NDArray[np.bool_], sample_rate: int, n_samples: int
    ) -> list[tuple[int, int]]:
        """The active runs of a frame mask as `(start, end)` sample ranges, end exclusive.

        One frame covers `FRAME_S * sample_rate` samples. A run that reaches the
        last frame runs to `n_samples`: the model's frame count can fall a few
        samples short of the audio, and that tail belongs to the run.
        """
        hop = cls.FRAME_S * sample_rate
        spans: list[tuple[int, int]] = []
        for on, s, e in _label_runs(frames):
            start = min(round(s * hop), n_samples)
            end = n_samples if e == len(frames) else min(round(e * hop), n_samples)
            if on and start < end:
                spans.append((start, end))
        return spans

    @classmethod
    def span_is_active(cls, frames: npt.NDArray[np.bool_], start: float, end: float) -> bool:
        """True if any frame of the span `start..end` s is in the mask `frames`."""
        if len(frames) == 0:
            return False
        lo, hi = cls._frame_range(start, end, len(frames))
        return bool(frames[lo:hi].any())

    @classmethod
    def stream_words(
        cls,
        chunks: list[StreamChunk],
        aligned: "list[list[AlignedWord]]",
        sample_rate: int,
        masks: dict[int, npt.NDArray[np.bool_]],
        keep: list[int],
    ) -> list[dict[str, Any]]:
        """Every stream's aligned words on the recording timeline, labelled, by start.

        A word placed wholly in its own speaker's silenced audio (`masks`) is a
        hallucination on silence and is dropped.
        """
        names = cls.speaker_names(keep)
        words: list[dict[str, Any]] = []
        for ch, chunk_words in zip(chunks, aligned, strict=True):
            offset = ch["start"] / sample_rate
            for w in chunk_words:
                start, end = w["start"] + offset, w["end"] + offset
                if cls.span_is_active(masks[ch["col"]], start, end):
                    words.append({**w, "start": start, "end": end, "speaker": names[ch["col"]]})
        return sorted(words, key=lambda w: (w["start"], w["end"]))

    @classmethod
    def _span_activity(
        cls, activity: npt.NDArray[np.float32], col: int, start: float, end: float
    ) -> float:
        """Mean activity of column `col` over the span `start..end` s."""
        lo, hi = cls._frame_range(start, end, len(activity))
        return float(activity[lo:hi, col].mean())

    @classmethod
    def dedupe_words(
        cls, words: list[dict[str, Any]], activity: npt.NDArray[np.float32], keep: list[int]
    ) -> list[dict[str, Any]]:
        """Drop one copy of a word two speaker streams both heard.

        `words` are sorted by start and labelled `SPEAKER_i` (position in
        `keep`). Two words from different speakers with the same text (case
        and punctuation ignored) and overlapping spans are one word that
        leaked into both masks: the copy whose speaker is more active over its
        span stays, the earlier word on a tie.
        """
        cols = {name: c for c, name in cls.speaker_names(keep).items()}
        norms = [_norm_word(w["word"]) for w in words]
        dropped: set[int] = set()
        for i, a in enumerate(words):
            for j in range(i + 1, len(words)):
                b = words[j]
                if b["start"] >= a["end"]:
                    break
                if i in dropped or j in dropped or a["speaker"] == b["speaker"]:
                    continue
                if not norms[i] or norms[i] != norms[j]:
                    continue
                score_a = cls._span_activity(activity, cols[a["speaker"]], a["start"], a["end"])
                score_b = cls._span_activity(activity, cols[b["speaker"]], b["start"], b["end"])
                dropped.add(j if score_a >= score_b else i)
        return [w for k, w in enumerate(words) if k not in dropped]
