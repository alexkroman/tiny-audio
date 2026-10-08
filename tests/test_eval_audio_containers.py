"""Tests for the audio container adapters in scripts.eval.audio."""

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf
import torch
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from scripts.eval.audio import TextNormalizer, as_16k_array, audio_to_wav_bytes, prepare_wav_bytes


def decode(wav: bytes) -> tuple[sf.AudioData, int]:
    return sf.read(io.BytesIO(wav))


@pytest.fixture
def tone() -> npt.NDArray[np.float32]:
    t = np.arange(16000, dtype=np.float32) / 16000
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestAudioToWavBytes:
    """Tensor inputs and channel squeezing."""

    def test_torch_tensor_is_accepted(self, tone: npt.NDArray[np.float32]) -> None:
        wav = audio_to_wav_bytes(torch.as_tensor(tone), 16000)
        array, sr = decode(wav)
        assert sr == 16000
        assert array.shape == (16000,)
        assert np.allclose(array, tone, atol=1e-3)

    def test_leading_channel_dim_is_squeezed(self, tone: npt.NDArray[np.float32]) -> None:
        array, _ = decode(audio_to_wav_bytes(tone[None, :], 16000))
        assert array.ndim == 1

    def test_channels_first_stereo_is_averaged(self, tone: npt.NDArray[np.float32]) -> None:
        """Squeezing alone left (2, n), which sf.write reads as 2 frames of n channels."""
        array, _ = decode(audio_to_wav_bytes(np.stack([tone, np.zeros_like(tone)]), 16000))
        assert array.shape == (16000,)
        assert np.allclose(array, tone / 2, atol=1e-3)

    def test_stereo_file_is_averaged(self, tone: npt.NDArray[np.float32], tmp_path: Path) -> None:
        """soundfile is frames-first, so the file path averages over axis 1."""
        path = tmp_path / "stereo.wav"
        sf.write(path, np.stack([tone, np.zeros_like(tone)], axis=1), 16000)
        array, _ = decode(prepare_wav_bytes({"path": str(path)}))
        assert array.shape == (16000,)
        assert np.allclose(array, tone / 2, atol=1e-3)

    def test_multichannel_int16_keeps_its_dtype(self) -> None:
        pcm = np.full((2, 100), 16384, dtype=np.int16)
        array, _ = decode(audio_to_wav_bytes(pcm, 16000))
        assert np.allclose(array, 0.5, atol=1e-3)


class TestPrepareWavBytes:
    """Each container shape `datasets` / torchcodec can hand us."""

    def test_torchcodec_decoder_like_object(self, tone: npt.NDArray[np.float32]) -> None:
        samples = SimpleNamespace(data=torch.as_tensor(tone), sample_rate=16000)
        decoder = SimpleNamespace(get_all_samples=lambda: samples)
        array, sr = decode(prepare_wav_bytes(decoder))
        assert sr == 16000
        assert array.shape == (16000,)

    def test_torchcodec_samples_like_object(self, tone: npt.NDArray[np.float32]) -> None:
        samples = SimpleNamespace(data=torch.as_tensor(tone), sample_rate=8000)
        _, sr = decode(prepare_wav_bytes(samples))
        assert sr == 8000

    def test_dict_with_path(self, tone: npt.NDArray[np.float32], tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"
        sf.write(path, tone, 16000)
        array, sr = decode(prepare_wav_bytes({"path": str(path)}))
        assert sr == 16000
        assert array.shape == (16000,)

    def test_dict_with_empty_path_is_unsupported(self) -> None:
        with pytest.raises(ValueError, match="Unsupported audio format"):
            prepare_wav_bytes({"path": ""})

    def test_object_with_path_attribute(
        self, tone: npt.NDArray[np.float32], tmp_path: Path
    ) -> None:
        path = tmp_path / "clip.wav"
        sf.write(path, tone, 16000)
        array, _ = decode(prepare_wav_bytes(SimpleNamespace(path=str(path))))
        assert array.shape == (16000,)

    def test_bytes_dict_passes_through_untouched(self) -> None:
        assert prepare_wav_bytes({"bytes": b"RIFF"}) == b"RIFF"


class TestAs16kArray:
    """Everything ends up as a 16 kHz numpy array."""

    def test_array_dict_at_16k_is_returned_as_is(self, tone: npt.NDArray[np.float32]) -> None:
        out = as_16k_array({"array": tone, "sampling_rate": 16000})
        assert out is tone

    def test_raw_key_is_accepted(self, tone: npt.NDArray[np.float32]) -> None:
        out = as_16k_array({"raw": tone})
        assert out is tone

    def test_lower_rate_is_resampled(self, tone: npt.NDArray[np.float32]) -> None:
        out = as_16k_array({"array": tone[:8000], "sampling_rate": 8000})
        assert out.shape == (16000,)

    def test_wav_bytes_are_decoded(self, tone: npt.NDArray[np.float32]) -> None:
        out = as_16k_array({"bytes": audio_to_wav_bytes(tone, 16000)})
        assert out.shape == (16000,)
        assert np.allclose(out, tone, atol=1e-3)

    def test_torchcodec_decoder_skips_the_pcm16_round_trip(
        self, tone: npt.NDArray[np.float32]
    ) -> None:
        """Exact float samples back, not a 16-bit-quantized copy."""
        samples = SimpleNamespace(data=torch.as_tensor(tone)[None, :], sample_rate=16000.0)
        out = as_16k_array(SimpleNamespace(get_all_samples=lambda: samples))
        assert out.shape == (16000,)
        assert np.array_equal(out, tone)

    def test_torchcodec_stereo_is_averaged_then_resampled(self) -> None:
        stereo = torch.stack([torch.full((8000,), 0.5), torch.zeros(8000)])
        out = as_16k_array(SimpleNamespace(data=stereo, sample_rate=8000.0))
        assert out.shape == (16000,)
        assert np.allclose(out[1000:-1000], 0.25, atol=1e-3)


class _LowercaseNormalizer(EnglishTextNormalizer):
    """Stand-in for Whisper's normalizer: lowercases, loads no spelling table."""

    def __init__(self) -> None:
        pass

    def __call__(self, s: str) -> str:
        return s.lower()


class TestTextNormalizerFixes:
    """Project-level spelling fixes run after Whisper's normalizer."""

    def test_spelling_fixes_apply_after_base_normalizer(self) -> None:
        normalizer = object.__new__(TextNormalizer)
        normalizer._normalizer = _LowercaseNormalizer()
        assert normalizer.normalize("Okay, ALL RIGHT kinda") == "ok, alright kind of"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Regression: raw substring replacement corrupted these.
            ("the overall rights", "the overall rights"),
            ("okayed", "okayed"),
            ("kindaish", "kindaish"),
            ("ball rightly", "ball rightly"),
            # Whole words still map, including at string edges.
            ("okay okay", "ok ok"),
            ("it's all right now", "it's alright now"),
            ("all   right", "alright"),
        ],
    )
    def test_spelling_fixes_are_word_bounded(self, text: str, expected: str) -> None:
        normalizer = object.__new__(TextNormalizer)
        normalizer._normalizer = _LowercaseNormalizer()
        assert normalizer.normalize(text) == expected
