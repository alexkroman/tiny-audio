"""Speaker diarization with NVIDIA Nemotron-3-Diarization."""

import importlib.util
from typing import Any, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
import torch
import torchaudio
from transformers import (
    AutoModelForAudioFrameClassification,
    AutoProcessor,
    BatchFeature,
    PreTrainedModel,
)


class SpeakerSegment(TypedDict):
    """One speaker turn, in seconds."""

    speaker: str
    start: float
    end: float


class _AudioProcessor(Protocol):
    """What `activity` uses of the Nemotron processor (its class ships after 5.17)."""

    def __call__(self, audio: npt.NDArray[np.float32], *, sampling_rate: int) -> BatchFeature:
        """Turn 16 kHz mono audio into model inputs."""
        ...


def get_device() -> torch.device:
    """Get best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        >>> words = NemotronDiarizer.assign_speakers_to_words(words, activity, keep)
    """

    MODEL_ID = "nvidia/Nemotron-3-Diarization"
    FRAME_S = 0.01
    SEGMENT_THRESHOLD = 0.5  # a speaker's segment = frames above this
    MIN_WORD_ACTIVITY = 0.1  # below this nobody is heard: the word inherits its neighbour
    _model: PreTrainedModel | None = None
    _processor: _AudioProcessor | None = None

    @classmethod
    def get_instance(cls) -> tuple[PreTrainedModel, _AudioProcessor]:
        """Load the diarization model and processor once, then return the cached pair."""
        if cls._model is None or cls._processor is None:
            if importlib.util.find_spec("transformers.models.nemotron3_diarization") is None:
                msg = (
                    "Speaker diarization uses Nemotron-3-Diarization, which needs a transformers "
                    "build with `nemotron3_diarization`: "
                    "pip install git+https://github.com/huggingface/transformers"
                )
                raise ImportError(msg)
            model: PreTrainedModel = AutoModelForAudioFrameClassification.from_pretrained(
                cls.MODEL_ID
            )
            model.to(
                get_device()
            )  # pyright: ignore[reportArgumentType]  # `to` is functools.wraps'd
            model.eval()
            cls._model = model
            processor: _AudioProcessor = AutoProcessor.from_pretrained(cls.MODEL_ID)
            cls._processor = processor
        return cls._model, cls._processor

    @classmethod
    @torch.inference_mode()
    def activity(cls, audio: npt.ArrayLike, sample_rate: int = 16000) -> npt.NDArray[np.float32]:
        """(frames, 8) speech probabilities at 10 ms; columns in arrival order."""
        if sample_rate != 16000:
            audio = np.asarray(
                torchaudio.functional.resample(
                    torch.as_tensor(audio, dtype=torch.float32), sample_rate, 16000
                ),
                dtype=np.float32,
            )
        model, processor = cls.get_instance()
        inputs = processor(np.asarray(audio, dtype=np.float32), sampling_rate=16000)
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
        with the most speech. Capping is how a known count is applied:
        a dropped column's words go to the most active remaining speaker.
        """
        active = [
            c for c in range(activity.shape[1]) if (activity[:, c] > cls.SEGMENT_THRESHOLD).any()
        ]
        cap = num_speakers or max_speakers
        if cap is not None and len(active) > cap:
            mass = activity.sum(axis=0)
            active = sorted(sorted(active, key=lambda c: -mass[c])[:cap])
        return active or [0]

    @classmethod
    def segments(cls, activity: npt.NDArray[np.float32], keep: list[int]) -> list[SpeakerSegment]:
        """Speaker turns `{"speaker", "start", "end"}` by start time; overlaps allowed."""
        names = {c: f"SPEAKER_{i}" for i, c in enumerate(keep)}
        out: list[SpeakerSegment] = []
        for c in keep:
            for on, s, e in _label_runs(activity[:, c] > cls.SEGMENT_THRESHOLD):
                if on:
                    out.append(
                        {"speaker": names[c], "start": s * cls.FRAME_S, "end": e * cls.FRAME_S}
                    )
        return sorted(out, key=lambda seg: (seg["start"], seg["end"]))

    @classmethod
    def speaker_columns(
        cls,
        spans: list[tuple[float | None, float | None]],
        activity: npt.NDArray[np.float32],
        keep: list[int],
    ) -> list[int]:
        """The kept activity column that said each `(start_s, end_s)` word.

        The kept speaker most active over the word's span. A word with no
        time, or where no kept speaker reaches MIN_WORD_ACTIVITY (it sits in
        a pause), takes the previous word's speaker -- the next word's at the
        start of the recording.
        """
        cols = np.asarray(keep)
        picked: list[int | None] = []
        for start, end in spans:
            if start is None or end is None:
                picked.append(None)
                continue
            lo = min(int(start / cls.FRAME_S), len(activity) - 1)
            hi = max(lo + 1, min(int(np.ceil(end / cls.FRAME_S)), len(activity)))
            mean = activity[lo:hi, cols].mean(axis=0)
            best = int(mean.argmax())
            picked.append(int(cols[best]) if mean[best] >= cls.MIN_WORD_ACTIVITY else None)
        known = [p for p in picked if p is not None]
        last = known[0] if known else keep[0]
        out: list[int] = []
        for p in picked:
            last = p if p is not None else last
            out.append(last)
        return out

    @classmethod
    def assign_speakers_to_words(
        cls, words: list[dict[str, Any]], activity: npt.NDArray[np.float32], keep: list[int]
    ) -> list[dict[str, Any]]:
        """Label each `{"start", "end"}` word `SPEAKER_i` by `speaker_columns`."""
        names = {c: f"SPEAKER_{i}" for i, c in enumerate(keep)}
        spans = [(w["start"], w["end"]) for w in words]
        for word, col in zip(words, cls.speaker_columns(spans, activity, keep), strict=True):
            word["speaker"] = names[col]
        return words
