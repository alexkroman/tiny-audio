"""Audio utilities and text normalization for ASR evaluation."""

import io

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperTokenizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


def audio_to_wav_bytes(audio_array: np.ndarray | torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes using soundfile."""
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.numpy()
    if audio_array.ndim > 1:
        audio_array = audio_array.squeeze()

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.getvalue()


def _read_path_to_wav(path: str) -> bytes:
    audio_array, sample_rate = sf.read(path)
    return audio_to_wav_bytes(audio_array, sample_rate)


def prepare_wav_bytes(wav_data) -> bytes:
    """Convert various audio formats to WAV bytes."""
    # torchcodec.AudioDecoder (lazy decoder emitted by recent `datasets`):
    # call get_all_samples() to materialize, then encode.
    if hasattr(wav_data, "get_all_samples"):
        samples = wav_data.get_all_samples()
        return audio_to_wav_bytes(samples.data, int(samples.sample_rate))

    # torchcodec.AudioSamples (already-decoded form).
    if hasattr(wav_data, "data") and hasattr(wav_data, "sample_rate"):
        return audio_to_wav_bytes(wav_data.data, int(wav_data.sample_rate))

    if isinstance(wav_data, dict):
        if "array" in wav_data and "sampling_rate" in wav_data:
            return audio_to_wav_bytes(wav_data["array"], wav_data["sampling_rate"])
        if "bytes" in wav_data:
            return wav_data["bytes"]
        if "path" in wav_data and wav_data["path"]:
            return _read_path_to_wav(wav_data["path"])

    if hasattr(wav_data, "array") and hasattr(wav_data, "sampling_rate"):
        return audio_to_wav_bytes(wav_data.array, wav_data.sampling_rate)

    if hasattr(wav_data, "path") and wav_data.path:
        return _read_path_to_wav(wav_data.path)

    raise ValueError(f"Unsupported audio format: {type(wav_data)}")


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

    def __init__(self):
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        self._normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)

    _SPELLING_FIXES = {
        "okay": "ok",
        "all right": "alright",
        "kinda": "kind of",
    }

    def normalize(self, text: str) -> str:
        """Normalize text for WER calculation."""
        text = self._normalizer(text)
        for src, dst in self._SPELLING_FIXES.items():
            text = text.replace(src, dst)
        return text
