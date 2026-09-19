"""Tests for scripts.eval.formatting — casing/punctuation scoring.

These metrics exist because every WER number in the repo is computed after
Whisper's normalizer lowercases and strips punctuation from both sides, making
WER structurally blind to the formatting the LLM decoder exists to produce.
"""

import pytest

from scripts.eval.formatting import (
    MIN_SCORABLE_SAMPLES,
    compute_formatting_metrics,
    is_case_scorable,
    is_orthographic_reference,
    is_punct_scorable,
    light_normalize,
    score_case,
    score_punct,
)


class TestScorability:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Hello world", True),
            ("HELLO WORLD", False),  # LibriSpeech / AMI convention
            ("hello world", False),  # Peoples / TEDLIUM convention
            ("", False),
            ("123 456", False),
        ],
    )
    def test_case_scorable(self, text, expected):
        assert is_case_scorable(text) is expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("Hello, world.", True), ("hello world", False), ("Really?", True), ("", False)],
    )
    def test_punct_scorable(self, text, expected):
        assert is_punct_scorable(text) is expected

    def test_orthographic_requires_both(self):
        assert is_orthographic_reference("Hello, world.") is True
        assert is_orthographic_reference("HELLO WORLD.") is False  # cased? no
        assert is_orthographic_reference("Hello world") is False  # punctuated? no


class TestLightNormalize:
    def test_preserves_case_and_punctuation(self):
        assert light_normalize("  Hello,   World!  ") == "Hello, World!"

    def test_reattaches_detached_punctuation(self):
        """GigaSpeech dev ships `"yeah ."`. Without reattachment every mark
        reads as a missing trailing punct plus a spurious token, which drove
        GigaSpeech punct F1 to 4.6 as a pure tokenization artifact.
        """
        assert light_normalize("yeah .") == "yeah."
        assert light_normalize("hello , world .") == "hello, world."


class TestCaseScoring:
    def test_perfect_casing(self):
        assert score_case("Hello World", "Hello World") == (0, 2)

    def test_counts_only_case_differences(self):
        errors, comparable = score_case("Hello World", "hello World")
        assert (errors, comparable) == (1, 2)

    def test_recognition_errors_are_excluded(self):
        """A word we got wrong cannot also be a casing error — otherwise the
        metric would double-count substitutions."""
        errors, comparable = score_case("Hello World", "Hello Planet")
        assert comparable == 1
        assert errors == 0

    def test_punctuation_does_not_count_as_a_case_error(self):
        assert score_case("Hello World.", "Hello World") == (0, 2)

    def test_truecase_style_spurious_capital_is_caught(self):
        """The exact defect this metric was built to expose."""
        errors, comparable = score_case("and subscriptions revenues", "and Subscriptions revenues")
        assert errors == 1
        assert comparable == 3


class TestPunctScoring:
    def test_exact_match(self):
        tp, pred, act = score_punct("Hello, world.", "Hello, world.")
        assert (tp, pred, act) == (2, 2, 2)

    def test_missing_punctuation_is_a_recall_miss(self):
        tp, pred, act = score_punct("Hello, world.", "Hello world")
        assert (tp, pred, act) == (0, 0, 2)

    def test_spurious_punctuation_is_a_precision_miss(self):
        tp, pred, act = score_punct("Hello world", "Hello, world.")
        assert (tp, pred, act) == (0, 2, 0)

    def test_wrong_mark_is_not_a_true_positive(self):
        tp, _, act = score_punct("Really?", "Really.")
        assert tp == 0
        assert act == 1


class TestAggregate:
    @staticmethod
    def _pairs(n, ref, hyp):
        return [(ref, hyp)] * n

    def test_monocase_corpus_reports_no_casing_number(self):
        """An ALL-CAPS corpus must omit the metric, not report a fake 0.0."""
        m = compute_formatting_metrics(self._pairs(50, "HELLO WORLD", "Hello world"))
        assert "case_error_rate" not in m
        assert "orthographic_wer" not in m

    def test_below_floor_is_omitted(self):
        m = compute_formatting_metrics(
            self._pairs(MIN_SCORABLE_SAMPLES - 1, "Hello, world.", "Hello, world.")
        )
        assert "orthographic_wer" not in m
        assert "punct_f1" not in m

    def test_at_floor_is_reported(self):
        m = compute_formatting_metrics(
            self._pairs(MIN_SCORABLE_SAMPLES, "Hello, world.", "Hello, world.")
        )
        assert m["orthographic_wer"] == pytest.approx(0.0)
        assert m["punct_f1"] == pytest.approx(100.0)
        assert m["case_error_rate"] == pytest.approx(0.0)
        assert m["orthographic_scored_samples"] == MIN_SCORABLE_SAMPLES

    def test_allcaps_reference_does_not_inflate_orthographic_wer(self):
        """Ungated, an ALL-CAPS reference scores ~100% orthographic WER against
        correctly-cased output and poisons the pooled number (observed: 48.19
        pooled for a system whose scorable orthographic WER is 10.14)."""
        good = self._pairs(20, "Hello, world.", "Hello, world.")
        allcaps = self._pairs(20, "HELLO WORLD", "Hello world")
        assert compute_formatting_metrics(good + allcaps)["orthographic_wer"] == pytest.approx(0.0)

    def test_empty_input_is_safe(self):
        assert compute_formatting_metrics([]) == {}
