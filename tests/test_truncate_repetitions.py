"""Tests for truncate_repetitions post-processing function."""

from tiny_audio.asr_text import truncate_repetitions, truncate_runaway_loop


class TestTruncateRepetitions:
    """Test repetition truncation post-processing."""

    def test_repeated_characters(self) -> None:
        """Should truncate repeated characters at end."""
        assert truncate_repetitions("444444") == "4"
        assert truncate_repetitions("hello worldddd") == "hello world"
        assert truncate_repetitions("testttt") == "test"

    def test_repeated_words(self) -> None:
        """Should truncate repeated words at end."""
        assert truncate_repetitions("the the the the") == "the"
        assert truncate_repetitions("hello world world world world") == "hello world"

    def test_repeated_phrases(self) -> None:
        """Should truncate repeated phrases at end."""
        assert truncate_repetitions("i am sorry i am sorry i am sorry") == "i am sorry"
        assert (
            truncate_repetitions("hello there i am sorry i am sorry i am sorry")
            == "hello there i am sorry"
        )

    def test_long_repeated_phrases(self) -> None:
        """Should handle long repeated phrases (like hallucinations)."""
        phrase = "i am sorry but i cannot speak for the other members of the council"
        repeated = " ".join([phrase] * 5)
        assert truncate_repetitions(repeated) == phrase

    def test_punctuated_and_cased_loops(self) -> None:
        """Loops the model punctuates or capitalizes still collapse (served eval rows)."""
        assert (
            truncate_repetitions("will start at 4:20. The The The The The The The The")
            == "will start at 4:20. The"
        )
        assert truncate_repetitions("Yeah Yeah Yeah Yeah. Yeah. Yeah. Yeah,") == "Yeah"

    def test_long_punctuated_phrase_loop(self) -> None:
        """A 5+ word phrase looped with drifting punctuation: longer than a runaway unit."""
        copies = [
            "Thank you very much, Chair.",
            "thank you, very much chair",
            "Thank you very much Chair,",
        ]
        loop = " ".join(copies * 2)
        assert (
            truncate_repetitions("Good morning. " + loop)
            == "Good morning. Thank you very much, Chair."
        )
        assert (
            truncate_repetitions("to be quick, quick, quick, quick, quick, quick, quick.")
            == "to be quick,"
        )

    def test_short_punctuated_repetition_is_speech(self) -> None:
        """'No, no, no,' is what was said, not a loop: punctuated runs under six stay."""
        assert truncate_repetitions("No, no, no,") == "No, no, no,"
        assert truncate_repetitions("Practice, practice, practice.") == (
            "Practice, practice, practice."
        )

    def test_no_repetition(self) -> None:
        """Should leave text unchanged when no repetition."""
        assert truncate_repetitions("this is fine") == "this is fine"
        assert truncate_repetitions("hello world") == "hello world"

    def test_edge_cases(self) -> None:
        """Should handle edge cases."""
        assert truncate_repetitions("") == ""
        assert truncate_repetitions("single") == "single"
        assert truncate_repetitions("two words") == "two words"

    def test_min_repeats_threshold(self) -> None:
        """Should respect min_repeats threshold."""
        # Default is 3 repeats
        assert truncate_repetitions("word word") == "word word"  # Only 2, not truncated
        assert truncate_repetitions("word word word") == "word"  # 3, truncated


class TestTruncateRunawayLoop:
    """Mid-transcript loops that drift never form an exact trailing run (served eval rows)."""

    def test_drifting_loop_is_cut_at_its_first_copy(self) -> None:
        text = (
            "for the atmosphere. That's where you live. " + "It's It's It's It's It's It's It " * 4
        )
        assert truncate_runaway_loop(text) == "for the atmosphere. That's where you live. It's"

    def test_phrase_loop_with_varied_case_is_cut(self) -> None:
        loop = "a little bit of a Little bit of a little bit of a little bit of " * 3
        assert (
            truncate_runaway_loop("I think we should have " + loop)
            == "I think we should have a little bit of"
        )

    def test_speech_that_recovers_is_kept(self) -> None:
        text = (
            "How does your wellness? No, no, no, no, no, no, no, sir, which traditional "
            "remedy or practice do you still swear by and why?"
        )
        assert truncate_runaway_loop(text) == text

    def test_short_runs_are_kept(self) -> None:
        text = "No, no, no, no, no. ow, ow, you bit me hard."
        assert truncate_runaway_loop(text) == text

    def test_applied_by_truncate_repetitions(self) -> None:
        assert truncate_repetitions("Yes " + "it is it is it is " * 5 + "it it") == "Yes it is"

    def test_repeated_words_without_a_loop_are_kept(self) -> None:
        text = "the cat and the dog and the bird and the fish"
        assert truncate_repetitions(text) == text
