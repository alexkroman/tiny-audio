"""Tests for ASR pipeline word-level timestamp functionality."""

import numpy as np
import pytest


def create_audio_sample(duration_sec: float = 2.0, sample_rate: int = 16000) -> dict:
    """Create a sample audio dict with a simple tone."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples)
    # Create a simple 440Hz sine wave
    audio_array = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    return {"array": audio_array, "sampling_rate": sample_rate}


@pytest.fixture
def pipeline():
    """Load the ASR pipeline with a small model for testing."""
    from transformers import pipeline as hf_pipeline

    # Use a small model for faster tests
    return hf_pipeline(
        "automatic-speech-recognition",
        model="mazesmazes/tiny-audio",
        trust_remote_code=True,
    )


class TestReturnTimestamps:
    """Test word-level timestamp functionality."""

    def test_basic_transcription_without_timestamps(self, pipeline):
        """Verify basic transcription still works."""
        audio = create_audio_sample(duration_sec=1.0)
        result = pipeline(audio)

        assert "text" in result
        assert isinstance(result["text"], str)
        # Should not have words key when timestamps not requested
        assert "words" not in result

    def test_return_timestamps_adds_words_key(self, pipeline):
        """Verify return_timestamps=True adds words to output."""
        pytest.importorskip("ctc_forced_aligner", reason="ctc-forced-aligner not installed")

        audio = create_audio_sample(duration_sec=2.0)
        result = pipeline(audio, return_timestamps=True)

        assert "text" in result
        assert "words" in result
        assert isinstance(result["words"], list)

    def test_word_timestamps_format(self, pipeline):
        """Verify word timestamps have correct format."""
        pytest.importorskip("ctc_forced_aligner", reason="ctc-forced-aligner not installed")

        audio = create_audio_sample(duration_sec=2.0)
        result = pipeline(audio, return_timestamps=True)

        # If there's text, there should be words
        if result["text"].strip():
            assert len(result["words"]) > 0

            # Check format of each word entry
            for word_entry in result["words"]:
                assert "word" in word_entry
                assert "start" in word_entry
                assert "end" in word_entry
                assert isinstance(word_entry["word"], str)
                assert isinstance(word_entry["start"], (int, float))
                assert isinstance(word_entry["end"], (int, float))
                # End should be >= start
                assert word_entry["end"] >= word_entry["start"]

    def test_timestamps_are_sequential(self, pipeline):
        """Verify word timestamps are in sequential order."""
        pytest.importorskip("ctc_forced_aligner", reason="ctc-forced-aligner not installed")

        audio = create_audio_sample(duration_sec=3.0)
        result = pipeline(audio, return_timestamps=True)

        words = result.get("words", [])
        if len(words) > 1:
            for i in range(1, len(words)):
                # Each word should start at or after the previous word started
                assert words[i]["start"] >= words[i - 1]["start"], (
                    f"Words not sequential: {words[i-1]} -> {words[i]}"
                )

    def test_empty_transcription_returns_empty_words(self, pipeline):
        """Verify empty transcription returns empty words list."""
        pytest.importorskip("ctc_forced_aligner", reason="ctc-forced-aligner not installed")

        # Very short audio that might produce empty transcription
        audio = create_audio_sample(duration_sec=0.1)
        result = pipeline(audio, return_timestamps=True)

        assert "words" in result
        # If text is empty, words should be empty
        if not result["text"].strip():
            assert result["words"] == []


class TestForcedAligner:
    """Test the ForcedAligner class directly."""

    def test_aligner_lazy_loading(self):
        """Verify aligner model is lazy-loaded."""
        pytest.importorskip("ctc_forced_aligner", reason="ctc-forced-aligner not installed")

        from src.asr_pipeline import ForcedAligner

        # Reset class state
        ForcedAligner._model = None
        ForcedAligner._tokenizer = None

        # Model should not be loaded yet
        assert ForcedAligner._model is None

        # Call get_instance to load
        model, tokenizer = ForcedAligner.get_instance("cpu")

        # Now model should be loaded
        assert ForcedAligner._model is not None
        assert model is not None
        assert tokenizer is not None

    def test_aligner_align_method(self):
        """Test the align method directly."""
        pytest.importorskip("ctc_forced_aligner", reason="ctc-forced-aligner not installed")

        from src.asr_pipeline import ForcedAligner

        # Create a simple audio array
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        # Align with some text
        text = "hello world"
        words = ForcedAligner.align(audio, text, sample_rate=sample_rate)

        assert isinstance(words, list)
        # Should have entries for the words
        for word_entry in words:
            assert "word" in word_entry
            assert "start" in word_entry
            assert "end" in word_entry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
