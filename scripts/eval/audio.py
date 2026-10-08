"""Audio utilities and text normalization for ASR evaluation."""

import functools
import io
import json
import re
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeGuard, runtime_checkable

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

AudioArray = npt.NDArray[np.floating[Any] | np.signedinteger[Any]]


@runtime_checkable
class DecodedAudioSamples(Protocol):
    """torchcodec.AudioSamples: an already-decoded waveform."""

    data: torch.Tensor
    sample_rate: float


@runtime_checkable
class LazyAudioDecoder(Protocol):
    """torchcodec.AudioDecoder: decodes on `get_all_samples()`."""

    def get_all_samples(self) -> DecodedAudioSamples: ...


@runtime_checkable
class _ArrayAudio(Protocol):
    array: AudioArray
    sampling_rate: int


@runtime_checkable
class _PathAudio(Protocol):
    path: str | None


def is_str_dict(value: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict)


def to_mono(audio_array: AudioArray | torch.Tensor, channel_axis: int = 0) -> AudioArray:
    """Collapse a waveform to 1-D, averaging channels rather than keeping one.

    `channel_axis` is 0 for torchcodec and `datasets` (channels-first) and 1
    for `soundfile.read` (frames-first). Size-1 axes are squeezed first so a
    `(1, n)` or `(n, 1)` mono clip passes through untouched whichever layout
    it arrived in. Only genuine multichannel audio is averaged: squeezing alone
    leaves `(2, n)` 2-D, which `sf.write` then reads as 2 frames of n channels.
    """
    array: AudioArray = (
        audio_array.detach().cpu().numpy() if isinstance(audio_array, torch.Tensor) else audio_array
    )
    if array.ndim > 1:
        array = array.squeeze()
    if array.ndim > 1:
        # Keep the dtype: averaging int16 PCM yields float64 in int16 range,
        # which sf.write would then clip as out-of-range float samples.
        array = array.mean(axis=channel_axis).astype(array.dtype)
    return array


def audio_to_wav_bytes(
    audio_array: AudioArray | torch.Tensor, sample_rate: int, channel_axis: int = 0
) -> bytes:
    """Encode a waveform as mono 16-bit WAV bytes (see `to_mono` for `channel_axis`)."""
    buffer = io.BytesIO()
    sf.write(
        buffer, to_mono(audio_array, channel_axis), sample_rate, format="WAV", subtype="PCM_16"
    )
    buffer.seek(0)
    return buffer.getvalue()


def _read_path_to_wav(path: str) -> bytes:
    audio_array, sample_rate = sf.read(path)
    return audio_to_wav_bytes(audio_array, sample_rate, channel_axis=1)


def _torchcodec_samples(wav_data: object) -> DecodedAudioSamples | None:
    """The decoded samples of a torchcodec decoder or samples object, else None."""
    # torchcodec.AudioDecoder (lazy decoder emitted by recent `datasets`):
    # call get_all_samples() to materialize.
    if isinstance(wav_data, LazyAudioDecoder):
        return wav_data.get_all_samples()
    # torchcodec.AudioSamples (already-decoded form).
    if isinstance(wav_data, DecodedAudioSamples):
        return wav_data
    return None


def prepare_wav_bytes(wav_data: object) -> bytes:
    """Convert various audio formats to mono WAV bytes."""
    samples = _torchcodec_samples(wav_data)
    if samples is not None:
        return audio_to_wav_bytes(samples.data, int(samples.sample_rate))

    if is_str_dict(wav_data):
        if "array" in wav_data and "sampling_rate" in wav_data:
            return audio_to_wav_bytes(wav_data["array"], wav_data["sampling_rate"])
        if "bytes" in wav_data:
            raw: bytes = wav_data["bytes"]
            return raw
        if wav_data.get("path"):
            return _read_path_to_wav(wav_data["path"])
    else:
        if isinstance(wav_data, _ArrayAudio):
            return audio_to_wav_bytes(wav_data.array, wav_data.sampling_rate)

        if isinstance(wav_data, _PathAudio) and wav_data.path:
            return _read_path_to_wav(wav_data.path)

    msg = f"Unsupported audio format: {type(wav_data)}"
    raise ValueError(msg)


def as_16k_array(audio: object) -> AudioArray:
    """Decode any container `prepare_wav_bytes` accepts to a 16 kHz array."""
    array: AudioArray
    sample_rate: float
    samples = _torchcodec_samples(audio)
    if samples is not None:
        # Already a float waveform: use it directly instead of quantizing to
        # PCM_16 WAV and decoding it straight back, which cost a full encode +
        # decode per clip and threw away precision below 16 bits.
        array = to_mono(samples.data)
        sample_rate = samples.sample_rate
    elif is_str_dict(audio) and ("array" in audio or "raw" in audio):
        array = audio["array"] if "array" in audio else audio["raw"]
        sample_rate = audio.get("sampling_rate", 16000)
    else:
        array, sample_rate = sf.read(io.BytesIO(prepare_wav_bytes(audio)))

    if sample_rate != 16000:
        array = librosa.resample(array, orig_sr=sample_rate, target_sr=16000)
    return array


class TextNormalizer:
    """Whisper-based text normalizer for ASR evaluation.

    Uses EnglishTextNormalizer which handles:
    - Lowercase and punctuation removal
    - Number normalization ("three" <-> "3")
    - British to American spelling ("colour" -> "color")
    - Disfluency removal ("uh", "um", "hmm")
    - Tag removal ("<inaudible>", "<COMMA>", etc.)
    - Contraction expansion: "don't" -> "do not", "we'd" -> "we would",
      and ALL "'s" -> " is" (which incorrectly mangles possessives:
      "john's car" -> "john is car"). This is a known Whisper limitation;
      since both reference and prediction get the same treatment, WER stays
      symmetric. Don't try to "fix" by re-expanding "'s" — that's a no-op
      because Whisper has already done it.

    Additional project-level fixes (Whisper leaves these alone):
    - "okay" -> "ok"
    - "all right" -> "alright"
    - "kinda" -> "kind of"
    """

    def __init__(self) -> None:
        self._normalizer = _english_normalizer()

    _SPELLING_FIXES: ClassVar[dict[str, str]] = {
        "okay": "ok",
        "all right": "alright",
        "kinda": "kind of",
    }
    # Word-bounded, like `_FILLER_RE` below: raw substring replacement turned
    # "the overall rights" into "the overalrights" and "okayed" into "oked".
    # `\s+` inside a key tolerates any run of whitespace; the match is
    # whitespace-collapsed before the dict lookup so it still finds its entry.
    _SPELLING_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:"
        + "|".join(r"\s+".join(map(re.escape, k.split())) for k in _SPELLING_FIXES)
        + r")\b"
    )

    # Whisper's normalizer drops `uh`, `um` and `hmm` but NOT `ah`, which is an
    # inconsistency rather than a decision: GigaSpeech references carry 125 `ah`
    # tokens per 500 rows and both our model and AssemblyAI's delete ~82% of
    # them, costing 0.86 and 0.91 WER points respectively -- ~9% of GigaSpeech
    # WER, scored as recognition error on both systems when it is a filler-token
    # convention mismatch.
    #
    # Word-bounded for the same reason as `_SPELLING_RE`: raw substring removal
    # would turn `ahead` into `ed` and `ahmed` into `med`. It is a separate
    # pattern because it deletes rather than maps.
    _FILLER_RE: ClassVar[re.Pattern[str]] = re.compile(r"\bah\b")

    def normalize(self, text: str) -> str:
        """Normalize text for WER calculation."""
        text = self._normalizer(text)
        text = self._SPELLING_RE.sub(
            lambda m: self._SPELLING_FIXES[" ".join(m.group().split())], text
        )
        # Applied after Whisper's pass so it sees the already-lowercased,
        # punctuation-stripped surface. Collapse whitespace so a removed token
        # does not leave a double space that shifts tokenisation.
        text = self._FILLER_RE.sub(" ", text)
        return " ".join(text.split())


@functools.cache
def _english_normalizer() -> EnglishTextNormalizer:
    """Build Whisper's English normalizer once per process.

    A `ta eval -d all` run builds one TextNormalizer per evaluator per
    dataset. Only the spelling map is needed, so fetch whisper-tiny's
    `normalizer.json` rather than its whole tokenizer -- that is all
    `WhisperTokenizer.english_spelling_normalizer` reads.
    """
    spelling = Path(hf_hub_download("openai/whisper-tiny", "normalizer.json"))
    return EnglishTextNormalizer(json.loads(spelling.read_text(encoding="utf-8")))
