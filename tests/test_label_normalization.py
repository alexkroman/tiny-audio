"""Tests for _normalize_label - the per-sample label-normalization helper
applied at training time in DataCollator._build_sample.

Mirrors (a subset of) scripts/analysis.py:normalize_text on the percent rule
so train and eval agree on canonical surface form for percent values.
"""

import pytest

from scripts.train import _normalize_label


class TestPercentCanonicalization:
    def test_percent_symbol_becomes_word(self):
        assert _normalize_label("we grew 25%") == "we grew 25 percent"

    def test_percent_symbol_mid_sentence(self):
        assert _normalize_label("a 25% margin in q2") == "a 25 percent margin in q2"

    def test_decimal_percent(self):
        assert _normalize_label("decreasing 0.4% quarter") == "decreasing 0.4 percent quarter"

    def test_per_cent_two_word_form(self):
        assert _normalize_label("we grew 25 per cent") == "we grew 25 percent"

    def test_already_spoken_form_unchanged(self):
        assert _normalize_label("we grew 25 percent") == "we grew 25 percent"


class TestGigaspeechMarkerStripping:
    def test_period_marker_at_end(self):
        assert _normalize_label("BOOST PIPES <PERIOD>") == "boost pipes"

    def test_comma_marker_mid_sentence(self):
        # Original: "PIPES <COMMA> AND THE FLOWERS" — no double space after strip.
        assert _normalize_label("PIPES <COMMA> AND THE FLOWERS") == "pipes and the flowers"

    def test_multiple_markers(self):
        assert _normalize_label("HELLO <COMMA> WORLD <PERIOD>") == "hello world"

    def test_question_and_exclamation_markers(self):
        assert _normalize_label("REALLY <QUESTIONMARK>") == "really"
        assert _normalize_label("WOW <EXCLAMATIONPOINT>") == "wow"

    def test_audio_event_markers_stripped(self):
        # <SIL>/<MUSIC>/<NOISE>/<OTHER> aren't real words; strip them too.
        assert _normalize_label("hello <music> world") == "hello world"
        assert _normalize_label("<noise> hello") == "hello"
        assert _normalize_label("hello <other>") == "hello"
        assert _normalize_label("hello <sil> world") == "hello world"

    def test_marker_at_start_no_leading_space(self):
        # The \s* in the regex must allow zero-width match at string start.
        assert _normalize_label("<period>boost") == "boost"

    def test_unknown_angle_bracket_token_preserved(self):
        # We only strip the documented Gigaspeech marker set. An unrelated
        # <foo> stays in the label; eval will normalize it later if needed.
        assert _normalize_label("<foo> hello") == "<foo> hello"


class TestCombinedNormalization:
    def test_percent_and_marker_together(self):
        assert _normalize_label("WE GREW 25% <PERIOD>") == "we grew 25 percent"

    def test_marker_then_percent(self):
        assert _normalize_label("HELLO <COMMA> WE GREW 25%") == "hello we grew 25 percent"


class TestHygiene:
    def test_lowercases(self):
        assert _normalize_label("HELLO World") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert _normalize_label("   hello world   ") == "hello world"

    def test_collapses_internal_whitespace(self):
        assert _normalize_label("hello    world") == "hello world"

    def test_empty_input(self):
        assert _normalize_label("") == ""

    def test_only_whitespace(self):
        assert _normalize_label("   ") == ""

    def test_only_marker(self):
        assert _normalize_label("<period>") == ""

    @pytest.mark.parametrize(
        "raw",
        [
            "i am convinced that it will make the rest of the world more united",
            "in the us it was a decision taken only by one person",
            "and that is why scarlett says",
        ],
    )
    def test_clean_inputs_unchanged(self, raw):
        # Realistic clean labels (already lowercased, no markers, no %) should
        # round-trip identically. Guards against accidental over-normalization.
        assert _normalize_label(raw) == raw


class TestTedliumNormalization:
    def test_unk_marker_stripped_at_start(self):
        assert (
            _normalize_label("<unk> i thought i would read poems") == "i thought i would read poems"
        )

    def test_unk_marker_stripped_mid_sentence(self):
        assert _normalize_label("hello <unk> world") == "hello world"

    def test_unk_marker_stripped_at_end(self):
        assert (
            _normalize_label("washing my mouth out with soap <unk>")
            == "washing my mouth out with soap"
        )

    def test_bracket_block_stripped(self):
        assert _normalize_label("she said [ medicine ] and laughed") == "she said and laughed"

    def test_long_bracket_block_stripped(self):
        assert (
            _normalize_label("then [ her face and hands stood out ] she paused")
            == "then she paused"
        )

    def test_unk_and_bracket_combined(self):
        assert _normalize_label("<unk> hello [ aside ] world") == "hello world"


class TestEdAccNormalization:
    def test_overlap_marker_stripped(self):
        # EdAcc convention; absent from the test split it scores against.
        assert (
            _normalize_label("YOU'RE A BIG PROMO <OVERLAP> YOU'RE THE BIG PROMOTER")
            == "you're a big promo you're the big promoter"
        )

    def test_laugh_marker_stripped(self):
        assert (
            _normalize_label("ANALYZING THIS CONVERSATION BUT ANYWAY <LAUGH> YEAH")
            == "analyzing this conversation but anyway yeah"
        )

    def test_dtmf_marker_stripped(self):
        assert (
            _normalize_label("EVERYBODY IS GOING THERE AND <DTMF> A LITTLE BIT GRIM")
            == "everybody is going there and a little bit grim"
        )

    def test_foreign_marker_stripped(self):
        # 4% of EdAcc validation rows ship <foreign> for code-switched segments.
        assert _normalize_label("HE SAID <FOREIGN> AND LAUGHED") == "he said and laughed"

    def test_no_speech_marker_stripped(self):
        # EdAcc placeholder for non-speech regions. Hyphenated form must be
        # caught by the regex literally.
        assert _normalize_label("OKAY <NO-SPEECH> RIGHT") == "okay right"

    def test_lipsmack_marker_stripped(self):
        assert _normalize_label("UM <LIPSMACK> SO") == "um so"

    def test_lowercase_form_also_stripped(self):
        # _CORPUS_MARKER_RE is IGNORECASE; verify both surface forms.
        assert _normalize_label("hello <overlap> world") == "hello world"
        assert _normalize_label("hello <laugh> world") == "hello world"
        assert _normalize_label("hello <dtmf> world") == "hello world"
        assert _normalize_label("hello <foreign> world") == "hello world"
        assert _normalize_label("hello <no-speech> world") == "hello world"
        assert _normalize_label("hello <lipsmack> world") == "hello world"


class TestEarnings22Normalization:
    def test_clear_throat_marker_stripped(self):
        assert _normalize_label("um <clear_throat> as i was saying") == "um as i was saying"

    def test_inaudible_marker_stripped(self):
        assert _normalize_label("the revenue <inaudible> in q4") == "the revenue in q4"

    def test_crosstalk_marker_stripped(self):
        assert _normalize_label("yeah <crosstalk> i agree") == "yeah i agree"
