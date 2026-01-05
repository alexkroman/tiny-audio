"""Audio utilities and text normalization for ASR evaluation."""

import io
import re

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


def prepare_wav_bytes(wav_data) -> bytes:
    """Convert various audio formats to WAV bytes."""
    # Dict with array (most common HF datasets format)
    if isinstance(wav_data, dict):
        if "array" in wav_data and "sampling_rate" in wav_data:
            return audio_to_wav_bytes(wav_data["array"], wav_data["sampling_rate"])
        if "bytes" in wav_data:
            return wav_data["bytes"]
        if "path" in wav_data and wav_data["path"]:
            audio_array, sample_rate = sf.read(wav_data["path"])
            return audio_to_wav_bytes(audio_array, sample_rate)

    # Audio object with array/sampling_rate attributes
    if hasattr(wav_data, "array") and hasattr(wav_data, "sampling_rate"):
        return audio_to_wav_bytes(wav_data.array, wav_data.sampling_rate)

    # AudioDecoder - try to get path and load with soundfile
    if hasattr(wav_data, "path") and wav_data.path:
        audio_array, sample_rate = sf.read(wav_data.path)
        return audio_to_wav_bytes(audio_array, sample_rate)

    raise ValueError(f"Unsupported audio format: {type(wav_data)}")


class TextNormalizer:
    """Whisper-based text normalizer for ASR evaluation.

    Uses EnglishTextNormalizer which handles:
    - Lowercase and punctuation removal
    - Number normalization ("three" <-> "3")
    - British to American spelling ("colour" -> "color")
    - Disfluency removal ("uh", "um", "hmm")
    - Tag removal ("<inaudible>", etc.)

    Additional normalizations:
    - "okay" -> "ok"
    - "all right" -> "alright"
    """

    def __init__(self):
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        self._normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)

    def normalize(self, text: str) -> str:
        """Normalize text for WER calculation."""
        text = self._normalizer(text)
        # Normalize common spelling variants
        text = text.replace("okay", "ok")
        text = text.replace("all right", "alright")
        text = text.replace("kinda", "kind of")
        return re.sub(r"'s\b", " is", text)
