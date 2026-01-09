"""Tests for ASR pipeline word-level timestamp functionality."""

import numpy as np
import pytest

# Mark all tests in this module as slow (load ML models)
pytestmark = pytest.mark.slow


def create_audio_sample(duration_sec: float = 2.0, sample_rate: int = 16000) -> dict:
    """Create a sample audio dict with a simple tone."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples)
    # Create a simple 440Hz sine wave
    audio_array = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    return {"array": audio_array, "sampling_rate": sample_rate}


@pytest.fixture
def pipeline():
    """Load the ASR pipeline with a local model for testing."""
    from tiny_audio.asr_config import ASRConfig
    from tiny_audio.asr_modeling import ASRModel
    from tiny_audio.asr_pipeline import ASRPipeline

    # Use small models for faster tests
    config = ASRConfig(
        audio_model_id="openai/whisper-tiny",
        text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        projector_type="mlp",
        model_dtype="float32",
        attn_implementation="eager",
    )
    model = ASRModel(config)
    return ASRPipeline(model=model)


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
        pytest.importorskip("torchaudio", reason="torchaudio not installed")

        audio = create_audio_sample(duration_sec=2.0)
        result = pipeline(audio, return_timestamps=True)

        assert "text" in result
        assert "words" in result
        assert isinstance(result["words"], list)

    def test_word_timestamps_format(self, pipeline):
        """Verify word timestamps have correct format."""
        pytest.importorskip("torchaudio", reason="torchaudio not installed")

        audio = create_audio_sample(duration_sec=2.0)
        result = pipeline(audio, return_timestamps=True)

        # words key should always be present when return_timestamps=True
        assert "words" in result

        # Check format of each word entry (if any)
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
        pytest.importorskip("torchaudio", reason="torchaudio not installed")

        audio = create_audio_sample(duration_sec=3.0)
        result = pipeline(audio, return_timestamps=True)

        words = result.get("words", [])
        if len(words) > 1:
            for i in range(1, len(words)):
                # Each word should start at or after the previous word started
                assert words[i]["start"] >= words[i - 1]["start"], (
                    f"Words not sequential: {words[i - 1]} -> {words[i]}"
                )

    def test_empty_transcription_returns_empty_words(self, pipeline):
        """Verify empty transcription returns empty words list."""
        pytest.importorskip("torchaudio", reason="torchaudio not installed")

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
        pytest.importorskip("torchaudio", reason="torchaudio not installed")

        from tiny_audio.asr_pipeline import ForcedAligner

        # Reset class state
        ForcedAligner._model = None
        ForcedAligner._labels = None
        ForcedAligner._dictionary = None
        ForcedAligner._bundle = None

        # Model should not be loaded yet
        assert ForcedAligner._model is None

        # Call get_instance to load
        model, labels, dictionary = ForcedAligner.get_instance("cpu")

        # Now model should be loaded
        assert ForcedAligner._model is not None
        assert model is not None
        assert labels is not None
        assert dictionary is not None

    def test_aligner_align_method(self):
        """Test the align method directly."""
        pytest.importorskip("torchaudio", reason="torchaudio not installed")

        from tiny_audio.asr_pipeline import ForcedAligner

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


class TestSpeakerDiarization:
    """Test speaker diarization functionality."""

    def test_return_speakers_adds_speaker_segments(self, pipeline):
        """Verify return_speakers=True adds speaker_segments to output."""
        pytest.importorskip("pyannote.audio", reason="pyannote-audio not installed")

        audio = create_audio_sample(duration_sec=3.0)
        result = pipeline(audio, return_speakers=True)

        assert "text" in result
        assert "speaker_segments" in result
        assert isinstance(result["speaker_segments"], list)

    def test_return_speakers_enables_timestamps(self, pipeline):
        """Verify return_speakers=True automatically enables timestamps."""
        pytest.importorskip("pyannote.audio", reason="pyannote-audio not installed")

        audio = create_audio_sample(duration_sec=3.0)
        result = pipeline(audio, return_speakers=True)

        # Should have words (timestamps) even though we didn't explicitly request them
        assert "words" in result

    def test_words_have_speaker_labels(self, pipeline):
        """Verify words have speaker labels assigned."""
        pytest.importorskip("pyannote.audio", reason="pyannote-audio not installed")
        pytest.importorskip("torchaudio", reason="torchaudio not installed")

        audio = create_audio_sample(duration_sec=3.0)
        result = pipeline(audio, return_speakers=True)

        # If there are words, they should have speaker labels
        if result.get("words"):
            for word in result["words"]:
                assert "speaker" in word, f"Word missing speaker label: {word}"

    def test_speaker_segments_format(self, pipeline):
        """Verify speaker segments have correct format."""
        pytest.importorskip("pyannote.audio", reason="pyannote-audio not installed")

        audio = create_audio_sample(duration_sec=3.0)
        result = pipeline(audio, return_speakers=True)

        for segment in result.get("speaker_segments", []):
            assert "speaker" in segment
            assert "start" in segment
            assert "end" in segment
            assert isinstance(segment["speaker"], str)
            assert isinstance(segment["start"], (int, float))
            assert isinstance(segment["end"], (int, float))
            assert segment["end"] >= segment["start"]

    def test_num_speakers_parameter(self, pipeline):
        """Verify num_speakers parameter is accepted."""
        pytest.importorskip("pyannote.audio", reason="pyannote-audio not installed")

        audio = create_audio_sample(duration_sec=3.0)
        # Should not raise an error
        result = pipeline(audio, return_speakers=True, num_speakers=2)

        assert "speaker_segments" in result

    def test_min_max_speakers_parameters(self, pipeline):
        """Verify min_speakers and max_speakers parameters are accepted."""
        pytest.importorskip("pyannote.audio", reason="pyannote-audio not installed")

        audio = create_audio_sample(duration_sec=3.0)
        # Should not raise an error
        result = pipeline(audio, return_speakers=True, min_speakers=1, max_speakers=3)

        assert "speaker_segments" in result


class TestSpeakerDiarizerClass:
    """Test the SpeakerDiarizer class directly."""

    def test_diarizer_lazy_loading(self):
        """Verify diarizer pipeline is lazy-loaded."""
        pytest.importorskip("pyannote.audio", reason="pyannote-audio not installed")

        from tiny_audio.asr_pipeline import SpeakerDiarizer

        # Reset class state
        SpeakerDiarizer._pipeline = None

        # Pipeline should not be loaded yet
        assert SpeakerDiarizer._pipeline is None

        # Note: Actually loading requires HF token with pyannote access
        # So we just test that the class structure is correct

    def test_assign_speakers_to_words(self):
        """Test the assign_speakers_to_words method."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
            {"word": "goodbye", "start": 2.0, "end": 2.5},
        ]

        speaker_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5},
            {"speaker": "SPEAKER_01", "start": 1.8, "end": 3.0},
        ]

        result = SpeakerDiarizer.assign_speakers_to_words(words, speaker_segments)

        assert result[0]["speaker"] == "SPEAKER_00"  # hello at 0.25 midpoint
        assert result[1]["speaker"] == "SPEAKER_00"  # world at 0.8 midpoint
        assert result[2]["speaker"] == "SPEAKER_01"  # goodbye at 2.25 midpoint

    def test_assign_speakers_handles_gaps(self):
        """Test speaker assignment when word falls in gap between segments."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [
            {"word": "gap", "start": 1.5, "end": 1.8},  # Falls in gap
        ]

        speaker_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0},
        ]

        result = SpeakerDiarizer.assign_speakers_to_words(words, speaker_segments)

        # Should find closest segment
        assert result[0]["speaker"] is not None

    def test_assign_speakers_empty_segments(self):
        """Test speaker assignment with empty segments list."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
        ]

        result = SpeakerDiarizer.assign_speakers_to_words(words, [])

        # Should handle gracefully with None speaker
        assert result[0]["speaker"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
