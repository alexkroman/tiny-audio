"""Tests for _truncate_repetitions post-processing function."""

from tiny_audio.asr_pipeline import _truncate_repetitions


class TestTruncateRepetitions:
    """Test repetition truncation post-processing."""

    def test_repeated_characters(self) -> None:
        """Should truncate repeated characters at end."""
        assert _truncate_repetitions("444444") == "4"
        assert _truncate_repetitions("hello worldddd") == "hello world"
        assert _truncate_repetitions("testttt") == "test"

    def test_repeated_words(self) -> None:
        """Should truncate repeated words at end."""
        assert _truncate_repetitions("the the the the") == "the"
        assert _truncate_repetitions("hello world world world world") == "hello world"

    def test_repeated_phrases(self) -> None:
        """Should truncate repeated phrases at end."""
        assert _truncate_repetitions("i am sorry i am sorry i am sorry") == "i am sorry"
        assert (
            _truncate_repetitions("hello there i am sorry i am sorry i am sorry")
            == "hello there i am sorry"
        )

    def test_long_repeated_phrases(self) -> None:
        """Should handle long repeated phrases (like hallucinations)."""
        phrase = "i am sorry but i cannot speak for the other members of the council"
        repeated = " ".join([phrase] * 5)
        assert _truncate_repetitions(repeated) == phrase

    def test_punctuated_and_cased_loops(self) -> None:
        """Loops the model punctuates or capitalizes still collapse (served eval rows)."""
        assert (
            _truncate_repetitions("will start at 4:20. The The The The The The The The")
            == "will start at 4:20. The"
        )
        assert _truncate_repetitions("Yeah Yeah Yeah Yeah. Yeah. Yeah. Yeah,") == "Yeah"
        assert (
            _truncate_repetitions("to be quick, quick, quick, quick, quick, quick, quick.")
            == "to be quick,"
        )

    def test_short_punctuated_repetition_is_speech(self) -> None:
        """'No, no, no,' is what was said, not a loop: punctuated runs under six stay."""
        assert _truncate_repetitions("No, no, no,") == "No, no, no,"
        assert _truncate_repetitions("Practice, practice, practice.") == (
            "Practice, practice, practice."
        )

    def test_no_repetition(self) -> None:
        """Should leave text unchanged when no repetition."""
        assert _truncate_repetitions("this is fine") == "this is fine"
        assert _truncate_repetitions("hello world") == "hello world"

    def test_edge_cases(self) -> None:
        """Should handle edge cases."""
        assert _truncate_repetitions("") == ""
        assert _truncate_repetitions("single") == "single"
        assert _truncate_repetitions("two words") == "two words"

    def test_min_repeats_threshold(self) -> None:
        """Should respect min_repeats threshold."""
        # Default is 3 repeats
        assert _truncate_repetitions("word word") == "word word"  # Only 2, not truncated
        assert _truncate_repetitions("word word word") == "word"  # 3, truncated
