"""Tests for ASR pipeline post-processing functionality."""

import pytest


class TestTruncateTrailingRepeats:
    """Test the _truncate_trailing_repeats method."""

    @pytest.fixture
    def pipeline(self):
        """Create a minimal pipeline instance for testing post-processing."""
        from tiny_audio.asr_pipeline import ASRPipeline

        # Create pipeline instance without full initialization
        return object.__new__(ASRPipeline)

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            # Single word repeats
            ("hello world world world", "hello world"),
            ("the the the", "the"),
            ("a a a a a", "a"),
            # Two word repeats
            ("I think I think I think", "I think"),
            ("go home go home go home", "go home"),
            # Three word repeats
            ("I think that I think that I think that", "I think that"),
            ("one two three one two three one two three", "one two three"),
            # Four word repeats
            ("a b c d a b c d a b c d", "a b c d"),
            # Mixed - should handle fox fox fox fox as 1-gram repeats
            ("the quick brown fox fox fox fox", "the quick brown fox"),
            # No repeats - unchanged
            ("no repeats here", "no repeats here"),
            ("hello world", "hello world"),
            # Edge cases
            ("test", "test"),
            ("", ""),
            ("a", "a"),
            ("a b", "a b"),
            # Repeats not at end - unchanged
            ("hello hello world", "hello hello world"),
        ],
    )
    def test_truncate_trailing_repeats(self, pipeline, input_text, expected):
        """Test that trailing repeats are correctly truncated."""
        result = pipeline._truncate_trailing_repeats(input_text)
        assert result == expected


class TestPostProcessPrediction:
    """Test the full _post_process_prediction method."""

    @pytest.fixture
    def pipeline(self):
        """Create a minimal pipeline instance for testing post-processing."""
        from tiny_audio.asr_pipeline import ASRPipeline

        return object.__new__(ASRPipeline)

    def test_lowercase(self, pipeline):
        """Test that output is lowercased."""
        result = pipeline._post_process_prediction("Hello World")
        assert result == "hello world"

    def test_acronym_combining(self, pipeline):
        """Test that spaced letters are combined."""
        result = pipeline._post_process_prediction("u s a")
        assert result == "usa"

    def test_currency_normalization(self, pipeline):
        """Test EUR currency conversion."""
        result = pipeline._post_process_prediction("eur 100")
        assert result == "100 euros"

    def test_trailing_repeat_truncation(self, pipeline):
        """Test that trailing repeats are truncated in post-processing."""
        result = pipeline._post_process_prediction("Hello World World World")
        assert result == "hello world"

    def test_whitespace_normalization(self, pipeline):
        """Test that extra whitespace is normalized."""
        result = pipeline._post_process_prediction("hello   world")
        assert result == "hello world"

    def test_empty_string(self, pipeline):
        """Test handling of empty string."""
        result = pipeline._post_process_prediction("")
        assert result == ""

    def test_combined_processing(self, pipeline):
        """Test multiple post-processing steps together."""
        # Uppercase + repeats + extra whitespace
        result = pipeline._post_process_prediction("THE  QUICK  FOX FOX FOX")
        assert result == "the quick fox"
