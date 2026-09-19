"""Tests for scripts.train._normalize_label and _needs_truecase.

Covers the Ultravox-style training-label normalizer:
- Gigaspeech punct-tag restoration (<COMMA> -> ',', etc.)
- Gigaspeech garbage-tag sample drops (<MUSIC>/<NOISE>/<SIL>/<OTHER>)
- Conditional truecasing (mono-case sources lifted, already-cased preserved)
- Residual marker stripping (<unk>, <laugh>, TEDLIUM brackets)
- Percent canonicalization carryover from the prior normalizer
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from scripts.train import (
    TEXT_CASE_CASED,
    TEXT_CASE_MONO,
    _needs_truecase,
    _normalize_label,
)


class TestGigaspeechPunctTags:
    def test_comma_tag_becomes_comma(self):
        result = _normalize_label("HELLO <COMMA> WORLD <PERIOD>")
        assert "," in result
        assert "." in result
        assert "<COMMA>" not in result
        assert "<PERIOD>" not in result

    def test_question_and_exclamation(self):
        result = _normalize_label("WHAT IS THIS <QUESTIONMARK> AMAZING <EXCLAMATIONPOINT>")
        assert "?" in result
        assert "!" in result

    def test_punct_tags_lowercase(self):
        # IGNORECASE: tags may appear in lowercase too.
        result = _normalize_label("hello <comma> world <period>")
        assert "," in result
        assert "." in result

    def test_apostrophe_preserved_with_punct_tags(self):
        # Gigaspeech preserves contractions; tag restoration shouldn't disturb them.
        result = _normalize_label("THEY'RE LEAVING <COMMA> AREN'T THEY <QUESTIONMARK>")
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
    def test_tag_stripped_keeps_surrounding_speech(self, tag):
        result = _normalize_label(f"SOME SPEECH {tag} MORE SPEECH")
        assert tag not in result
        assert "speech" in result.lower()
        assert result, "Sample with speech around the tag should NOT be dropped"

    def test_lowercase_tag_stripped(self):
        result = _normalize_label("some speech <music> more")
        assert "<music>" not in result
        assert "speech" in result

    def test_tag_alone_becomes_empty(self):
        # The entire label was just the tag — strip leaves nothing, returns "",
        # collator's empty-label filter then drops the sample.
        assert _normalize_label("<MUSIC>") == ""
        assert _normalize_label("<NOISE>") == ""


class TestConditionalTruecase:
    def test_all_uppercase_gets_recased(self):
        # Gigaspeech / AMI style — should be lifted to proper case.
        result = _normalize_label("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG")
        assert result[0].isupper()
        # At least some downstream tokens should be lowercase now.
        assert any(c.islower() for c in result)

    def test_all_lowercase_gets_recased(self):
        # TEDLIUM / Peoples / Switchboard style.
        result = _normalize_label("the quick brown fox jumps over the lazy dog")
        # Truecase should at least capitalize the sentence-initial word.
        assert result[0].isupper()

    def test_already_cased_text_preserved(self):
        # LibriHeavy / CV / VoxPopuli / SPGI style — already proper-cased.
        # Critical: truecase should NOT damage "McClarnon" -> "Mcclarnon".
        text = "It was also confirmed that McClarnon is still a member of the band."
        result = _normalize_label(text)
        assert "McClarnon" in result, f"Truecase damaged proper noun: {result}"

    def test_libriheavy_passthrough(self):
        text = "My, what imaginations these children have developed!"
        assert _normalize_label(text) == text

    def test_short_text_skipped(self):
        # "yeah" / "OH" / "MM" — too short to recase usefully.
        assert _normalize_label("yeah") == "yeah"
        assert _normalize_label("OH").lower() == "oh" or _normalize_label("OH") == "OH"

    def test_needs_truecase_heuristic(self):
        assert _needs_truecase("HELLO WORLD HOW ARE YOU TODAY") is True  # all caps
        assert _needs_truecase("hello world how are you today") is True  # zero caps
        assert _needs_truecase("Hello world, how are you today.") is False  # cased
        assert _needs_truecase("yeah") is False  # too short


class TestResidualMarkers:
    def test_tedlium_unk_stripped(self):
        # _normalize_label sees `<unk> i thought ...` -> strip <unk>, truecase
        result = _normalize_label("<unk> i thought i would read poems today")
        assert "<unk>" not in result
        # Should still have the rest of the sentence.
        assert "thought" in result.lower()

    def test_switchboard_laugh_stripped(self):
        assert "<LAUGH>" not in _normalize_label("yeah <LAUGH> you know to death")
        assert "<laugh>" not in _normalize_label("yeah <laugh> you know to death")

    def test_tedlium_bracket_stripped(self):
        result = _normalize_label("the topic of [ medicine ] is important today")
        assert "[" not in result
        assert "medicine" not in result

    def test_inaudible_stripped(self):
        result = _normalize_label("we walked <inaudible> down the street")
        assert "<inaudible>" not in result


class TestPercentCanonicalization:
    """Only the `per cent` -> `percent` spelling collapse survives. The `%`
    character is preserved: rewriting it to " percent" destroyed `%` in 100%
    of training targets and cost ~93% of a measured 66%-vs-94% raw-text ITN
    gap. The removal is WER-neutral — Whisper's EnglishTextNormalizer maps
    "five percent" and "5%" to the same string on both sides of the score.
    """

    def test_percent_sign_is_preserved(self):
        result = _normalize_label("inflation rose to five % this year")
        assert "%" in result

    def test_per_cent_becomes_single_word(self):
        result = _normalize_label("inflation rose to five per cent this year")
        assert "percent" in result.lower()
        assert "per cent" not in result.lower()


class TestUnicodeCleanup:
    """ftfy + NFKC pass — defensive layer for mojibake, smart quotes,
    HTML entities, composed/decomposed forms, and width variants."""

    def test_mojibake_fixed(self):
        # Common UTF-8-double-encoded apostrophe corruption.
        result = _normalize_label("Itâ€™s a good idea")
        assert "'" in result
        assert "â€™" not in result

    def test_smart_quotes_normalized(self):
        # Curly quotes folded to straight (NFKC + ftfy).
        result = _normalize_label("She said “yes” and walked away.")
        assert "“" not in result
        assert "”" not in result
        assert '"' in result

    def test_curly_apostrophe_normalized(self):
        result = _normalize_label("I don’t think so.")
        assert "’" not in result
        assert "'" in result

    def test_full_width_latin_normalized(self):
        # NFKC folds full-width to half-width.
        result = _normalize_label("ＨＥＬＬＯ ＷＯＲＬＤ")
        assert "HELLO" in result.upper() or "Hello" in result

    def test_html_entity_decoded(self):
        result = _normalize_label("Tom &amp; Jerry")
        assert "&amp;" not in result
        assert "&" in result

    def test_clean_ascii_passthrough(self):
        # Already clean — should be a no-op aside from truecase / regex.
        text = "It's already clean."
        assert _normalize_label(text) == text


class TestEdgeCases:
    def test_empty_input(self):
        assert _normalize_label("") == ""
        assert _normalize_label(None) == ""  # type: ignore[arg-type]
        assert _normalize_label("   ") == ""

    def test_only_markers_becomes_empty(self):
        # If all that remains after stripping is whitespace, return empty
        # so the collator's empty-label filter discards the sample.
        assert _normalize_label("<unk> <unk> <unk>") == ""

    def test_only_garbage_tag(self):
        assert _normalize_label("<MUSIC>") == ""

    def test_only_brackets_becomes_empty(self):
        assert _normalize_label("[ stage direction ]") == ""


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
    def test_cased_source_fragments_are_left_alone(self, text):
        assert _normalize_label(text, TEXT_CASE_CASED) == text

    @pytest.mark.parametrize("text", SPGI_FRAGMENTS)
    def test_heuristic_would_have_corrupted_these(self, text):
        """Guard: pins the bug the declaration exists to prevent.

        If the heuristic ever stops misfiring here, the parametrized cases
        above are no longer exercising anything and should be revisited.
        """
        assert _normalize_label(text, None) != text

    def test_truecase_invents_proper_nouns_on_fragments(self):
        """The concrete damage: ordinary common nouns promoted to proper."""
        text = (
            "with which we will work and be able to clear out all the various "
            "permissions as we move to finalize the bankable feasibility study"
        )
        heuristic = _normalize_label(text, None)
        assert "Permissions" in heuristic
        assert "Bankable" in heuristic
        assert _normalize_label(text, TEXT_CASE_CASED) == text

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("YEAH", "Yeah"), ("OKAY", "Okay"), ("HMM", "Hmm"), ("NO", "No"), ("IT'S", "It's")],
    )
    def test_short_monocase_rows_are_recased(self, raw, expected):
        """Real AMI rows: 21% fall under the truecase floor and used to pass
        through verbatim, shipping ALL-CAPS training labels."""
        assert _normalize_label(raw, TEXT_CASE_MONO) == expected

    def test_short_rows_bypass_the_statistical_truecaser(self):
        """Deterministic recase, so a backchannel cannot become a proper noun."""
        assert _normalize_label("SO UH", TEXT_CASE_MONO) == "So uh"

    def test_long_monocase_rows_still_truecase(self):
        result = _normalize_label("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", TEXT_CASE_MONO)
        assert result[0].isupper()
        assert any(c.islower() for c in result)

    def test_unset_policy_preserves_legacy_behavior(self):
        for text in ["YEAH", "the quick brown fox jumps over the lazy dog", "Already Cased Text."]:
            assert _normalize_label(text, None) == _normalize_label(text)


class TestSpelledLetterRuns:
    """A period closing a run of spelled-out letters is not a sentence start.

    AMI writes acronyms inline ("S. S. H.") and carries no real sentence
    punctuation, so every period in it is part of an acronym.
    """

    def test_acronym_run_does_not_capitalize_next_word(self):
        result = _normalize_label(
            "IF YOU IF YOU S. S. H. AND THEY HAVE THIS BIG WARNING ABOUT DOING NOTHING",
            TEXT_CASE_MONO,
        )
        assert "S. S. H. and" in result, result
        assert "S. S. H. And" not in result

    def test_real_sentence_boundary_still_capitalizes(self):
        """Gigaspeech's tag-derived boundaries must keep working."""
        result = _normalize_label("six tomatoes. the next thing we tried", TEXT_CASE_MONO)
        assert "tomatoes. The" in result, result

    def test_single_letter_without_a_run_still_capitalizes(self):
        """'e t. the video game' — the letter before the period carries no
        period of its own, so this is a boundary, not an acronym run.

        Truecase also lifts the bare letters, so the assertion is on the
        boundary itself rather than on the preceding token's case.
        """
        result = _normalize_label("e t. the video game", TEXT_CASE_MONO)
        assert ". The" in result, result
