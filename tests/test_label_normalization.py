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
