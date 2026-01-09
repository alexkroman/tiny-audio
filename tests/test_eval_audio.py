"""Tests for scripts/eval/audio.py - audio utilities and text normalization."""

import io

import numpy as np
import pytest
import soundfile as sf

from scripts.eval.audio import TextNormalizer, audio_to_wav_bytes, prepare_wav_bytes


class TestAudioToWavBytes:
    """Tests for audio_to_wav_bytes function."""

    def test_numpy_array_mono(self):
        """Test conversion of mono numpy array."""
        audio = np.random.randn(16000).astype(np.float32)
        wav_bytes = audio_to_wav_bytes(audio, 16000)

        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0

        # Verify it's valid WAV
        audio_back, sr = sf.read(io.BytesIO(wav_bytes))
        assert sr == 16000
        assert len(audio_back) == 16000

    def test_numpy_array_stereo_squeezed(self):
        """Test that stereo arrays are squeezed to mono."""
        audio = np.random.randn(1, 16000).astype(np.float32)
        wav_bytes = audio_to_wav_bytes(audio, 16000)

        audio_back, sr = sf.read(io.BytesIO(wav_bytes))
        assert audio_back.ndim == 1

    def test_different_sample_rates(self):
        """Test various sample rates."""
        for sr in [8000, 16000, 22050, 44100]:
            audio = np.random.randn(sr).astype(np.float32)
            wav_bytes = audio_to_wav_bytes(audio, sr)
            audio_back, sr_back = sf.read(io.BytesIO(wav_bytes))
            assert sr_back == sr


class TestPrepareWavBytes:
    """Tests for prepare_wav_bytes function."""

    def test_dict_with_array_and_sampling_rate(self):
        """Test dict format with array and sampling_rate keys."""
        audio_dict = {
            "array": np.random.randn(16000).astype(np.float32),
            "sampling_rate": 16000,
        }
        wav_bytes = prepare_wav_bytes(audio_dict)
        assert isinstance(wav_bytes, bytes)

    def test_dict_with_bytes(self):
        """Test dict format with bytes key."""
        # Create valid WAV bytes first
        audio = np.random.randn(16000).astype(np.float32)
        original_bytes = audio_to_wav_bytes(audio, 16000)

        audio_dict = {"bytes": original_bytes}
        wav_bytes = prepare_wav_bytes(audio_dict)
        assert wav_bytes == original_bytes

    def test_object_with_array_attribute(self):
        """Test object with array and sampling_rate attributes."""

        class AudioObject:
            def __init__(self):
                self.array = np.random.randn(16000).astype(np.float32)
                self.sampling_rate = 16000

        wav_bytes = prepare_wav_bytes(AudioObject())
        assert isinstance(wav_bytes, bytes)

    def test_unsupported_format_raises_error(self):
        """Test that unsupported formats raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported audio format"):
            prepare_wav_bytes("not_a_valid_audio")

        with pytest.raises(ValueError, match="Unsupported audio format"):
            prepare_wav_bytes(12345)


class TestTextNormalizer:
    """Tests for TextNormalizer class."""

    @pytest.fixture
    def normalizer(self):
        """Create a TextNormalizer instance."""
        return TextNormalizer()

    def test_lowercase(self, normalizer):
        """Test that text is lowercased."""
        result = normalizer.normalize("HELLO WORLD")
        assert result == result.lower()

    def test_okay_normalization(self, normalizer):
        """Test 'okay' -> 'ok' normalization."""
        result = normalizer.normalize("okay")
        assert "ok" in result
        assert "okay" not in result

    def test_alright_normalization(self, normalizer):
        """Test 'all right' -> 'alright' normalization."""
        result = normalizer.normalize("all right")
        assert "alright" in result

    def test_kinda_normalization(self, normalizer):
        """Test 'kinda' -> 'kind of' normalization."""
        result = normalizer.normalize("kinda")
        assert "kind of" in result

    def test_contractions_expansion(self, normalizer):
        """Test that 's contractions are expanded."""
        result = normalizer.normalize("it's")
        assert "it is" in result

    def test_empty_string(self, normalizer):
        """Test empty string handling."""
        result = normalizer.normalize("")
        assert result == ""

    def test_whitespace_only(self, normalizer):
        """Test whitespace-only strings."""
        result = normalizer.normalize("   ")
        assert result.strip() == ""

    def test_numbers_preserved(self, normalizer):
        """Test that numbers are handled consistently."""
        # The Whisper normalizer may convert numbers to words or vice versa
        result = normalizer.normalize("I have 3 apples")
        assert len(result) > 0

    def test_punctuation_removal(self, normalizer):
        """Test that punctuation is removed."""
        result = normalizer.normalize("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
