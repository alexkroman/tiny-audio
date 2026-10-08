"""Tests for scripts.labels.normalize_label and _needs_truecase.

Covers the Ultravox-style training-label normalizer:
- Gigaspeech punct-tag restoration (<COMMA> -> ',', etc.)
- Gigaspeech garbage-tag sample drops (<MUSIC>/<NOISE>/<SIL>/<OTHER>)
- Conditional truecasing (mono-case sources lifted, already-cased preserved)
- Residual marker stripping (<unk>, <laugh>, TEDLIUM brackets)
- TEDLIUM edge audience-event words (laughter / applause)
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from scripts.labels import (
    TEXT_CASE_CASED,
    TEXT_CASE_MONO,
    _needs_truecase,
    normalize_label,
)


class TestGigaspeechPunctTags:
    def test_apostrophe_preserved_with_punct_tags(self) -> None:
        # Gigaspeech preserves contractions; tag restoration shouldn't disturb them.
        result = normalize_label("THEY'RE LEAVING <COMMA> AREN'T THEY <QUESTIONMARK>")
        assert "they're" in result.lower()
        assert "aren't" in result.lower()


class TestNonSpeechTags:
    """Gigaspeech ships <SIL>/<NOISE>/<MUSIC>/<OTHER> for non-speech moments.
    We strip them (not whole-sample drop) so partial transcripts of mixed
    speech+non-speech segments survive. The collator's empty-label filter
    catches the entire-label-was-a-tag edge case. Whole-sample drop would
    nuke eval batches that happen to draw a small set of tagged samples.
    """

    @pytest.mark.parametrize("tag", ["<SIL>", "<NOISE>", "<MUSIC>", "<OTHER>"])
    def test_tag_stripped_keeps_surrounding_speech(self, tag: str) -> None:
        result = normalize_label(f"SOME SPEECH {tag} MORE SPEECH")
        assert tag not in result
        assert "speech" in result.lower()
        assert result, "Sample with speech around the tag should NOT be dropped"

    def test_lowercase_tag_stripped(self) -> None:
        result = normalize_label("some speech <music> more")
        assert "<music>" not in result
        assert "speech" in result

    def test_tag_alone_becomes_empty(self) -> None:
        # The entire label was just the tag — strip leaves nothing, returns "",
        # collator's empty-label filter then drops the sample.
        assert normalize_label("<MUSIC>") == ""
        assert normalize_label("<NOISE>") == ""


class TestConditionalTruecase:
    def test_all_uppercase_gets_recased(self) -> None:
        # Gigaspeech / AMI style — should be lifted to proper case.
        result = normalize_label("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG")
        assert result[0].isupper()
        # At least some downstream tokens should be lowercase now.
        assert any(c.islower() for c in result)

    def test_all_lowercase_gets_recased(self) -> None:
        # TEDLIUM / Peoples / Switchboard style.
        result = normalize_label("the quick brown fox jumps over the lazy dog")
        # Truecase should at least capitalize the sentence-initial word.
        assert result[0].isupper()

    def test_already_cased_text_preserved(self) -> None:
        # LibriHeavy / CV / VoxPopuli / SPGI style — already proper-cased.
        # Critical: truecase should NOT damage "McClarnon" -> "Mcclarnon".
        text = "It was also confirmed that McClarnon is still a member of the band."
        result = normalize_label(text)
        assert "McClarnon" in result, f"Truecase damaged proper noun: {result}"

    def test_short_text_skipped(self) -> None:
        # "yeah" / "OH" / "MM" — too short to recase usefully.
        assert normalize_label("yeah") == "yeah"
        assert normalize_label("OH").lower() == "oh"

    def test_needs_truecase_heuristic(self) -> None:
        assert _needs_truecase("HELLO WORLD HOW ARE YOU TODAY") is True  # all caps
        assert _needs_truecase("hello world how are you today") is True  # zero caps
        assert _needs_truecase("Hello world, how are you today.") is False  # cased
        assert _needs_truecase("yeah") is False  # too short


class TestResidualMarkers:
    def test_switchboard_laugh_stripped(self) -> None:
        assert "<LAUGH>" not in normalize_label("yeah <LAUGH> you know to death")
        assert "<laugh>" not in normalize_label("yeah <laugh> you know to death")

    def test_inaudible_stripped(self) -> None:
        result = normalize_label("we walked <inaudible> down the street")
        assert "<inaudible>" not in result


class TestUnicodeCleanup:
    """ftfy + NFKC pass — defensive layer for mojibake, smart quotes,
    HTML entities, composed/decomposed forms, and width variants."""

    def test_mojibake_fixed(self) -> None:
        # Common UTF-8-double-encoded apostrophe corruption.
        result = normalize_label("Itâ€™s a good idea")
        assert "'" in result
        assert "â€™" not in result

    def test_smart_quotes_normalized(self) -> None:
        # Curly quotes folded to straight (NFKC + ftfy).
        result = normalize_label("She said “yes” and walked away.")
        assert "“" not in result
        assert "”" not in result
        assert '"' in result

    def test_curly_apostrophe_normalized(self) -> None:
        result = normalize_label("I don\u2019t think so.")
        assert "\u2019" not in result
        assert "'" in result

    def test_full_width_latin_normalized(self) -> None:
        # NFKC folds full-width to half-width.
        result = normalize_label("\uff28\uff25\uff2c\uff2c\uff2f \uff37\uff2f\uff32\uff2c\uff24")
        assert "HELLO" in result.upper() or "Hello" in result

    def test_html_entity_decoded(self) -> None:
        result = normalize_label("Tom &amp; Jerry")
        assert "&amp;" not in result
        assert "&" in result

    def test_clean_ascii_passthrough(self) -> None:
        # Already clean — should be a no-op aside from truecase / regex.
        text = "It's already clean."
        assert normalize_label(text) == text


class TestEdgeCases:
    def test_none_input(self) -> None:
        assert normalize_label(None) == ""

    def test_only_markers_becomes_empty(self) -> None:
        # If all that remains after stripping is whitespace, return empty
        # so the collator's empty-label filter discards the sample.
        assert normalize_label("<unk> <unk> <unk>") == ""

    def test_only_brackets_becomes_empty(self) -> None:
        assert normalize_label("[ stage direction ]") == ""


class TestDeclaredTextCase:
    """Per-source casing policy (`text_case`) overrides the per-row heuristic.

    The heuristic cannot classify a single row correctly, because a lowercase
    FRAGMENT of an already-cased source is character-identical to a row from a
    genuinely uncased source. Measured on 300 real rows per source: it
    truecased 13% of SPGISpeech (injecting proper nouns) and left 21% of AMI
    as ALL-CAPS.
    """

    # Real SPGISpeech rows that the heuristic misclassifies as mono-case:
    # sliding-window fragments that happen to contain no capital letter.
    SPGI_FRAGMENTS: ClassVar[list[str]] = [
        (
            "with which we will work and be able to clear out all the various "
            "permissions as we move to finalize the bankable feasibility study"
        ),
        (
            "and we stay committed to maintaining sustainable profitability and "
            "building value for all stakeholders."
        ),
        "and not just click on a website because clearly, what's good for everyone is",
    ]

    @pytest.mark.parametrize("text", SPGI_FRAGMENTS)
    def test_cased_source_fragments_are_left_alone(self, text: str) -> None:
        assert normalize_label(text, TEXT_CASE_CASED) == text

    @pytest.mark.parametrize("text", SPGI_FRAGMENTS)
    def test_heuristic_would_have_corrupted_these(self, text: str) -> None:
        """Guard: pins the bug the declaration exists to prevent.

        If the heuristic ever stops misfiring here, the parametrized cases
        above are no longer exercising anything and should be revisited.
        """
        assert normalize_label(text, None) != text

    def test_truecase_invents_proper_nouns_on_fragments(self) -> None:
        """The concrete damage: ordinary common nouns promoted to proper."""
        text = (
            "with which we will work and be able to clear out all the various "
            "permissions as we move to finalize the bankable feasibility study"
        )
        heuristic = normalize_label(text, None)
        assert "Permissions" in heuristic
        assert "Bankable" in heuristic
        assert normalize_label(text, TEXT_CASE_CASED) == text

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("YEAH", "Yeah"), ("OKAY", "Okay"), ("HMM", "Hmm"), ("NO", "No"), ("IT'S", "It's")],
    )
    def test_short_monocase_rows_are_recased(self, raw: str, expected: str) -> None:
        """Real AMI rows: 21% fall under the truecase floor and used to pass
        through verbatim, shipping ALL-CAPS training labels."""
        assert normalize_label(raw, TEXT_CASE_MONO) == expected

    def test_short_rows_bypass_the_statistical_truecaser(self) -> None:
        """Deterministic recase, so a backchannel cannot become a proper noun."""
        assert normalize_label("SO UH", TEXT_CASE_MONO) == "So uh"

    def test_long_monocase_rows_still_truecase(self) -> None:
        result = normalize_label("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", TEXT_CASE_MONO)
        assert result[0].isupper()
        assert any(c.islower() for c in result)

    def test_unset_policy_preserves_legacy_behavior(self) -> None:
        for text in ["YEAH", "the quick brown fox jumps over the lazy dog", "Already Cased Text."]:
            assert normalize_label(text, None) == normalize_label(text)


class TestSpelledLetterRuns:
    """A period closing a run of spelled-out letters is not a sentence start.

    AMI writes acronyms inline ("S. S. H.") and carries no real sentence
    punctuation, so every period in it is part of an acronym.
    """

    def test_acronym_run_does_not_capitalize_next_word(self) -> None:
        result = normalize_label(
            "IF YOU IF YOU S. S. H. AND THEY HAVE THIS BIG WARNING ABOUT DOING NOTHING",
            TEXT_CASE_MONO,
        )
        assert "S. S. H. and" in result, result
        assert "S. S. H. And" not in result

    def test_real_sentence_boundary_still_capitalizes(self) -> None:
        """Gigaspeech's tag-derived boundaries must keep working."""
        result = normalize_label("six tomatoes. the next thing we tried", TEXT_CASE_MONO)
        assert "tomatoes. The" in result, result

    def test_single_letter_without_a_run_still_capitalizes(self) -> None:
        """'e t. the video game' — the letter before the period carries no
        period of its own, so this is a boundary, not an acronym run.

        Truecase also lifts the bare letters, so the assertion is on the
        boundary itself rather than on the preceding token's case.
        """
        result = normalize_label("e t. the video game", TEXT_CASE_MONO)
        assert ". The" in result, result


class TestTedliumEdgeEvents:
    """TEDLIUM writes audience events as bare lowercase words at segment edges."""

    def test_leading_laughter_stripped(self) -> None:
        out = normalize_label("laughter so i had to add handcuffs", TEXT_CASE_MONO)
        assert "laughter" not in out.lower()
        assert out.lower().startswith("so i had")

    def test_trailing_chain_stripped(self) -> None:
        out = normalize_label("walking around like crazy laughter applause", TEXT_CASE_MONO)
        assert out.lower().endswith("like crazy")

    def test_event_only_label_becomes_empty(self) -> None:
        assert normalize_label("applause", TEXT_CASE_MONO) == ""

    def test_mid_segment_real_speech_kept(self) -> None:
        out = normalize_label("the first time i felt applause on the vest", TEXT_CASE_MONO)
        assert "applause" in out.lower()

    def test_word_prefix_not_matched(self) -> None:
        out = normalize_label("laughteresque moments", TEXT_CASE_MONO)
        assert out.lower().startswith("laughteresque")

    def test_cased_source_edge_word_kept(self) -> None:
        text = "The hall erupted in laughter"
        assert normalize_label(text, TEXT_CASE_CASED) == text

    def test_all_caps_mono_source_kept(self) -> None:
        out = normalize_label("THEY ROARED WITH LAUGHTER", TEXT_CASE_MONO)
        assert out.lower().endswith("laughter")
