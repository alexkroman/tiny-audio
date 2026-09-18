"""Tests for the audio container adapters in scripts.eval.audio."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.eval.audio import TextNormalizer, as_16k_array, audio_to_wav_bytes, prepare_wav_bytes


def decode(wav: bytes) -> tuple[np.ndarray, int]:
    return sf.read(io.BytesIO(wav))


@pytest.fixture
def tone() -> np.ndarray:
    t = np.arange(16000, dtype=np.float32) / 16000
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestAudioToWavBytes:
    """Tensor inputs and channel squeezing."""

    def test_torch_tensor_is_accepted(self, tone):
        wav = audio_to_wav_bytes(torch.from_numpy(tone), 16000)
        array, sr = decode(wav)
        assert sr == 16000
        assert array.shape == (16000,)
        assert np.allclose(array, tone, atol=1e-3)

    def test_leading_channel_dim_is_squeezed(self, tone):
        array, _ = decode(audio_to_wav_bytes(tone[None, :], 16000))
        assert array.ndim == 1


class TestPrepareWavBytes:
    """Each container shape `datasets` / torchcodec can hand us."""

    def test_torchcodec_decoder_like_object(self, tone):
        samples = SimpleNamespace(data=torch.from_numpy(tone), sample_rate=16000)
        decoder = SimpleNamespace(get_all_samples=lambda: samples)
        array, sr = decode(prepare_wav_bytes(decoder))
        assert sr == 16000
        assert array.shape == (16000,)

    def test_torchcodec_samples_like_object(self, tone):
        samples = SimpleNamespace(data=torch.from_numpy(tone), sample_rate=8000)
        _, sr = decode(prepare_wav_bytes(samples))
        assert sr == 8000

    def test_dict_with_path(self, tone, tmp_path):
        path = tmp_path / "clip.wav"
        sf.write(path, tone, 16000)
        array, sr = decode(prepare_wav_bytes({"path": str(path)}))
        assert sr == 16000
        assert array.shape == (16000,)

    def test_dict_with_empty_path_is_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported audio format"):
            prepare_wav_bytes({"path": ""})

    def test_object_with_path_attribute(self, tone, tmp_path):
        path = tmp_path / "clip.wav"
        sf.write(path, tone, 16000)
        array, _ = decode(prepare_wav_bytes(SimpleNamespace(path=str(path))))
        assert array.shape == (16000,)

    def test_bytes_dict_passes_through_untouched(self):
        assert prepare_wav_bytes({"bytes": b"RIFF"}) == b"RIFF"


class TestAs16kArray:
    """Everything ends up as a 16 kHz numpy array."""

    def test_array_dict_at_16k_is_returned_as_is(self, tone):
        out = as_16k_array({"array": tone, "sampling_rate": 16000})
        assert out is tone

    def test_raw_key_is_accepted(self, tone):
        out = as_16k_array({"raw": tone})
        assert out is tone

    def test_lower_rate_is_resampled(self, tone):
        out = as_16k_array({"array": tone[:8000], "sampling_rate": 8000})
        assert out.shape == (16000,)

    def test_wav_bytes_are_decoded(self, tone):
        out = as_16k_array({"bytes": audio_to_wav_bytes(tone, 16000)})
        assert out.shape == (16000,)
        assert np.allclose(out, tone, atol=1e-3)


class TestTextNormalizerFixes:
    """Project-level spelling fixes run after Whisper's normalizer."""

    def test_spelling_fixes_apply_after_base_normalizer(self):
        normalizer = object.__new__(TextNormalizer)
        normalizer._normalizer = str.lower  # stand-in for Whisper's normalizer
        assert normalizer.normalize("Okay, ALL RIGHT kinda") == "ok, alright kind of"
