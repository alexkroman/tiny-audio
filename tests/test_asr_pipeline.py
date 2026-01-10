"""Tests for ASRPipeline and helper classes."""

import numpy as np
import pytest


class TestPostProcessPrediction:
    """Tests for ASRPipeline._post_process_prediction method."""

    @pytest.fixture
    def post_process(self):
        """Get the post-processing function without loading a model."""
        # Import the class to access the method
        from tiny_audio.asr_pipeline import ASRPipeline

        # Create a minimal mock to call the method
        class MockPipeline:
            _truncate_trailing_repeats = ASRPipeline._truncate_trailing_repeats
            _post_process_prediction = ASRPipeline._post_process_prediction

        return MockPipeline()

    def test_lowercase(self, post_process):
        """Text should be lowercased."""
        result = post_process._post_process_prediction("HELLO WORLD")
        assert result == "hello world"

    def test_acronym_combining(self, post_process):
        """Single letters should be combined into acronyms."""
        result = post_process._post_process_prediction("the U S A is great")
        assert result == "the usa is great"

    def test_currency_normalization(self, post_process):
        """EUR X should become X euros."""
        result = post_process._post_process_prediction("it costs EUR 100")
        assert result == "it costs 100 euros"

    def test_trailing_repeat_truncation(self, post_process):
        """Trailing repeats should be removed."""
        result = post_process._post_process_prediction("hello world world world")
        assert result == "hello world"

    def test_whitespace_normalization(self, post_process):
        """Multiple spaces should be collapsed."""
        result = post_process._post_process_prediction("hello   world")
        assert result == "hello world"

    def test_empty_string(self, post_process):
        """Empty string should return empty."""
        result = post_process._post_process_prediction("")
        assert result == ""

    def test_combined_processing(self, post_process):
        """All processing steps should work together."""
        result = post_process._post_process_prediction("The U S A costs EUR 50 fifty fifty")
        assert result == "the usa costs 50 euros fifty"


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
    """Tests for SpeakerDiarizer.assign_speakers_to_words."""

    def test_exact_overlap(self):
        """Words within speaker segments get assigned correctly."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]

        result = SpeakerDiarizer.assign_speakers_to_words(words, segments)

        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_00"

    def test_multiple_speakers(self):
        """Words should be assigned to correct speakers."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "hi", "start": 2.0, "end": 2.5},
        ]
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 1.5, "end": 3.0},
        ]

        result = SpeakerDiarizer.assign_speakers_to_words(words, segments)

        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_closest_segment_fallback(self):
        """Words outside segments should be assigned to closest speaker."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [{"word": "hello", "start": 1.0, "end": 1.5}]  # Between segments
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0},
        ]

        result = SpeakerDiarizer.assign_speakers_to_words(words, segments)
        # Midpoint is 1.25, closer to SPEAKER_01 (midpoint 2.5) than SPEAKER_00 (midpoint 0.25)
        # Actually: |1.25 - 0.25| = 1.0, |1.25 - 2.5| = 1.25, so SPEAKER_00 is closer
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_empty_segments(self):
        """Empty segments should result in None speaker."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [{"word": "hello", "start": 0.0, "end": 0.5}]
        segments = []

        result = SpeakerDiarizer.assign_speakers_to_words(words, segments)

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
