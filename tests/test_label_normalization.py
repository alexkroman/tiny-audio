"""Tests for _normalize_label - the per-sample label-normalization helper
applied at training time in DataCollator._build_sample.

After the Ultravox-style normalizer rewrite the pipeline is:
  ftfy + NFKC → Gigaspeech punct-tag → marker strip → TEDLIUM bracket strip →
  percent canon → whitespace collapse → conditional truecase

Tests below capture both the marker-stripping correctness (deterministic)
and the truecase output (which depends on the truecase library's NLTK-backed
vocab). The truecase-dependent expected values were captured from the live
normalizer on 2026-05-13; if the truecase library updates and these expected
values shift, update with `python -c "from scripts.train import _normalize_label; print(_normalize_label(...))"`.
"""

import pytest

from scripts.train import _normalize_label


class TestPercentCanonicalization:
    def test_percent_symbol_becomes_word(self):
        # Lowercase input → truecase fires → sentence-initial cap.
        assert _normalize_label("we grew 25%") == "We grew 25 percent"

    def test_percent_symbol_mid_sentence(self):
        # Truecase capitalizes acronym-shaped tokens like "q2" → "Q2".
        assert _normalize_label("a 25% margin in q2") == "A 25 percent margin in Q2"

    def test_decimal_percent(self):
        assert _normalize_label("decreasing 0.4% quarter") == "Decreasing 0.4 percent quarter"

    def test_per_cent_two_word_form(self):
        assert _normalize_label("we grew 25 per cent") == "We grew 25 percent"

    def test_already_spoken_form_unchanged_except_truecase(self):
        # Same percent canon path; lowercase input gets sentence-initial cap.
        assert _normalize_label("we grew 25 percent") == "We grew 25 percent"


class TestGigaspeechPunctRestoration:
    """Gigaspeech ships punct as <PERIOD>/<COMMA>/<QUESTIONMARK>/<EXCLAMATIONPOINT>.
    The normalizer maps each to real punctuation BEFORE the truecase pass.
    """

    def test_period_at_end(self):
        assert _normalize_label("BOOST PIPES <PERIOD>") == "Boost pipes."

    def test_comma_mid_sentence(self):
        assert _normalize_label("PIPES <COMMA> AND THE FLOWERS") == "Pipes, and the flowers"

    def test_multiple_punct_tags(self):
        assert _normalize_label("HELLO <COMMA> WORLD <PERIOD>") == "Hello, world."

    def test_question_tag(self):
        assert _normalize_label("REALLY <QUESTIONMARK>") == "Really?"

    def test_exclamation_tag(self):
        # Short input (<5 letters total after punct) skips truecase.
        assert _normalize_label("WOW <EXCLAMATIONPOINT>") == "WOW!"


class TestNonSpeechTagStripping:
    """Gigaspeech non-speech segment tags (<SIL>/<MUSIC>/<NOISE>/<OTHER>) are
    stripped — not whole-sample dropped. The empty-label filter at the
    collator catches the entire-label-was-just-a-tag edge case.
    """

    def test_music_stripped(self):
        assert _normalize_label("hello <music> world") == "Hello world"

    def test_noise_at_start(self):
        assert _normalize_label("<noise> hello") == "Hello"

    def test_other_at_end(self):
        assert _normalize_label("hello <other>") == "Hello"

    def test_sil_stripped(self):
        assert _normalize_label("hello <sil> world") == "Hello world"

    def test_tag_alone_becomes_empty(self):
        # Tag-only label → strip leaves nothing → empty string returned.
        # Collator's empty-label filter then drops the row.
        assert _normalize_label("<music>") == ""
        assert _normalize_label("<NOISE>") == ""


class TestGigaspeechEdgeCases:
    def test_punct_tag_at_start_no_leading_space(self):
        # `\s*` in the regex matches zero-width at string start.
        # Resulting `.Boost` is unusual but a literal consequence of the
        # mapping rule; downstream WER scoring normalizes regardless.
        assert _normalize_label("<period>boost") == ".Boost"

    def test_unknown_angle_bracket_token_preserved(self):
        # We only strip the documented marker set. <foo> is unrecognized
        # so it stays in the label — truecase capitalizes the F since
        # it's at sentence start.
        assert _normalize_label("<foo> hello") == "<Foo> hello"


class TestCombinedNormalization:
    def test_percent_and_gigaspeech_marker(self):
        # Mixed case after percent insertion (lowercase 'percent' appended to
        # UPPER text) → upper_frac in (0.05, 0.9) → truecase SKIPPED →
        # output preserves mixed case as-is.
        assert _normalize_label("WE GREW 25% <PERIOD>") == "WE GREW 25 percent."

    def test_marker_then_percent(self):
        assert _normalize_label("HELLO <COMMA> WE GREW 25%") == "HELLO, WE GREW 25 percent"


class TestHygiene:
    def test_strips_leading_trailing_whitespace(self):
        assert _normalize_label("   hello world   ") == "Hello world"

    def test_collapses_internal_whitespace(self):
        assert _normalize_label("hello    world") == "Hello world"

    def test_empty_input(self):
        assert _normalize_label("") == ""

    def test_only_whitespace(self):
        assert _normalize_label("   ") == ""

    def test_only_marker_remains_after_strip(self):
        # <period> → '.' via Gigaspeech-tag map; whole label becomes just '.'.
        assert _normalize_label("<period>") == "."

    def test_already_cased_passes_through(self):
        # Mixed case (sentence-initial caps + proper nouns) → upper_count > 0
        # → truecase SKIPPED → output preserves existing casing.
        assert (
            _normalize_label("My, what imaginations these children have developed!")
            == "My, what imaginations these children have developed!"
        )


class TestTedliumNormalization:
    """TEDLIUM ships <unk> in ~92% of train rows and [...] editorial brackets
    in ~0.25%. Both stripped; surrounding lowercase prose gets truecased.
    """

    def test_unk_at_start(self):
        assert (
            _normalize_label("<unk> i thought i would read poems") == "I thought I would read poems"
        )

    def test_unk_mid_sentence(self):
        assert _normalize_label("hello <unk> world") == "Hello world"

    def test_unk_at_end(self):
        assert (
            _normalize_label("washing my mouth out with soap <unk>")
            == "Washing my mouth out with soap"
        )

    def test_bracket_block_stripped(self):
        assert _normalize_label("she said [ medicine ] and laughed") == "She said and laughed"

    def test_long_bracket_block_stripped(self):
        assert (
            _normalize_label("then [ her face and hands stood out ] she paused")
            == "Then she paused"
        )

    def test_unk_and_bracket_combined(self):
        assert _normalize_label("<unk> hello [ aside ] world") == "Hello world"


class TestEdAccNormalization:
    """EdAcc ships <overlap>/<laugh>/<dtmf>/<foreign>/<no-speech>/<lipsmack>
    in ~20% of rows. All stripped; surrounding text gets recased per truecase.
    """

    def test_overlap_marker_stripped(self):
        # Truecase capitalizes "Promo" (recognized as a proper-noun-ish token).
        assert (
            _normalize_label("YOU'RE A BIG PROMO <OVERLAP> YOU'RE THE BIG PROMOTER")
            == "You're a big Promo you're the big promoter"
        )

    def test_laugh_marker_stripped(self):
        # Mid-sentence "Yeah" cap is a known truecase library quirk
        # (sentence-boundary heuristic over-capitalizes interjections).
        assert (
            _normalize_label("ANALYZING THIS CONVERSATION BUT ANYWAY <LAUGH> YEAH")
            == "Analyzing this conversation but anyway Yeah"
        )

    def test_dtmf_marker_stripped(self):
        assert (
            _normalize_label("EVERYBODY IS GOING THERE AND <DTMF> A LITTLE BIT GRIM")
            == "Everybody is going there and a little bit grim"
        )

    def test_foreign_marker_stripped(self):
        assert _normalize_label("HE SAID <FOREIGN> AND LAUGHED") == "He said and laughed"

    def test_no_speech_marker_stripped(self):
        # Hyphenated tag form must be caught literally.
        assert _normalize_label("OKAY <NO-SPEECH> RIGHT") == "Okay right"

    def test_lipsmack_marker_stripped(self):
        # Short total letter count (< 5 after strip) → truecase SKIPPED.
        assert _normalize_label("UM <LIPSMACK> SO") == "UM SO"

    @pytest.mark.parametrize(
        "marker",
        ["overlap", "laugh", "dtmf", "foreign", "no-speech", "lipsmack"],
    )
    def test_lowercase_form_also_stripped(self, marker):
        # _CORPUS_MARKER_RE has re.IGNORECASE; verify lowercase variant.
        assert _normalize_label(f"hello <{marker}> world") == "Hello world"


class TestEarnings22Normalization:
    """Earnings22 ships <clear_throat>/<inaudible>/<crosstalk> in ~3% of rows."""

    def test_clear_throat_marker_stripped(self):
        assert _normalize_label("um <clear_throat> as i was saying") == "Um as I was saying"

    def test_inaudible_marker_stripped(self):
        # "q4" → "Q4" by truecase (recognized acronym shape).
        assert _normalize_label("the revenue <inaudible> in q4") == "The revenue in Q4"

    def test_crosstalk_marker_stripped(self):
        assert _normalize_label("yeah <crosstalk> i agree") == "Yeah I agree"
