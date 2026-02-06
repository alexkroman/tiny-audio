"""Tests for ASRPipeline and helper classes."""

import numpy as np
import pytest


class TestExtractAudio:
    """Tests for ASRPipeline._extract_audio method."""

    @pytest.fixture
    def extract_audio(self):
        """Get the extract_audio function without loading a model."""
        from tiny_audio.asr_pipeline import ASRPipeline

        class MockPipeline:
            _extract_audio = ASRPipeline._extract_audio

        return MockPipeline()

    def test_dict_with_array(self, extract_audio):
        """Dict with 'array' key should extract audio."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_audio._extract_audio({"array": audio, "sampling_rate": 16000})
        assert result is not None
        assert "array" in result
        assert result["sampling_rate"] == 16000

    def test_dict_with_raw(self, extract_audio):
        """Dict with 'raw' key should extract audio."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_audio._extract_audio({"raw": audio, "sampling_rate": 16000})
        assert result is not None
        assert "array" in result

    def test_numpy_array(self, extract_audio):
        """Numpy array should be extracted with default sample rate."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_audio._extract_audio(audio)
        assert result is not None
        assert result["sampling_rate"] == 16000

    def test_unsupported_returns_none(self, extract_audio):
        """Unsupported input type should return None."""
        result = extract_audio._extract_audio(12345)
        assert result is None


class TestSpeakerAssignment:
    """Tests for LocalSpeakerDiarizer.assign_speakers_to_words."""

    def test_exact_overlap(self):
        """Words within speaker segments get assigned correctly."""
        from tiny_audio.diarization import LocalSpeakerDiarizer

        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]

        result = LocalSpeakerDiarizer.assign_speakers_to_words(words, segments)

        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_00"

    def test_multiple_speakers(self):
        """Words should be assigned to correct speakers."""
        from tiny_audio.diarization import LocalSpeakerDiarizer

        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "hi", "start": 2.0, "end": 2.5},
        ]
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 1.5, "end": 3.0},
        ]

        result = LocalSpeakerDiarizer.assign_speakers_to_words(words, segments)

        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_closest_segment_fallback(self):
        """Words outside segments should be assigned to closest speaker."""
        from tiny_audio.diarization import LocalSpeakerDiarizer

        words = [{"word": "hello", "start": 1.0, "end": 1.5}]  # Between segments
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0},
        ]

        result = LocalSpeakerDiarizer.assign_speakers_to_words(words, segments)
        # Midpoint is 1.25, closer to SPEAKER_01 (midpoint 2.5) than SPEAKER_00 (midpoint 0.25)
        # Actually: |1.25 - 0.25| = 1.0, |1.25 - 2.5| = 1.25, so SPEAKER_00 is closer
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_empty_segments(self):
        """Empty segments should result in None speaker."""
        from tiny_audio.diarization import LocalSpeakerDiarizer

        words = [{"word": "hello", "start": 0.0, "end": 0.5}]
        segments = []

        result = LocalSpeakerDiarizer.assign_speakers_to_words(words, segments)

        assert result[0]["speaker"] is None


class TestSanitizeParameters:
    """Tests for ASRPipeline._sanitize_parameters."""

    def test_removes_custom_params(self):
        """Custom params should be removed before parent validation."""
        from unittest.mock import patch

        from tiny_audio.asr_pipeline import ASRPipeline

        # Create a mock pipeline that won't call parent __init__
        with patch.object(ASRPipeline, "__init__", return_value=None):
            pipeline = ASRPipeline.__new__(ASRPipeline)

        # Mock the parent's _sanitize_parameters
        with patch(
            "transformers.AutomaticSpeechRecognitionPipeline._sanitize_parameters"
        ) as mock_parent:
            mock_parent.return_value = ({}, {}, {})

            # Call with custom params
            pipeline._sanitize_parameters(
                return_timestamps=True,
                return_speakers=True,
                num_speakers=2,
                min_speakers=1,
                max_speakers=3,
                hf_token="test",
                other_param="value",
            )

            # Parent should be called without our custom params
            call_kwargs = mock_parent.call_args[1]
            assert "return_timestamps" not in call_kwargs
            assert "return_speakers" not in call_kwargs
            assert "num_speakers" not in call_kwargs
            assert "hf_token" not in call_kwargs
            # But other params should remain
            assert call_kwargs.get("other_param") == "value"


class TestPostprocess:
    """Tests for ASRPipeline.postprocess method."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with postprocess method."""
        from unittest.mock import MagicMock

        from tiny_audio.asr_pipeline import ASRPipeline

        # Create instance without calling __init__
        pipeline = object.__new__(ASRPipeline)

        # Create mock tokenizer
        pipeline.tokenizer = MagicMock()
        pipeline.tokenizer.decode.return_value = "hello world"

        return pipeline

    def test_handles_list_outputs(self, mock_pipeline):
        """Should handle list of outputs from chunking."""
        import torch

        result = mock_pipeline.postprocess([{"tokens": torch.tensor([1, 2, 3])}])
        assert "text" in result

    def test_handles_tensor_tokens(self, mock_pipeline):
        """Should handle tensor tokens."""
        import torch

        result = mock_pipeline.postprocess({"tokens": torch.tensor([[1, 2, 3]])})
        assert "text" in result
        mock_pipeline.tokenizer.decode.assert_called()

    def test_strips_think_tags(self, mock_pipeline):
        """Should strip <think>...</think> tags from output."""
        mock_pipeline.tokenizer.decode.return_value = "<think>reasoning</think> hello world"
        import torch

        result = mock_pipeline.postprocess({"tokens": torch.tensor([1, 2, 3])})
        assert "<think>" not in result["text"]
        assert "reasoning" not in result["text"]
        assert "hello world" in result["text"]
