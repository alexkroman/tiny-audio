"""Tests for the paired bootstrap CI on corpus-WER deltas.

Nothing in this repo computed a confidence interval before this, so every
margin quoted in the experiment configs was a bare point estimate. That is how
one granite_qwen_top4 checkpoint came to be cited at 11.95 / 11.76 / 10.94 /
10.50 across sweeps -- a ~1.5pt spread -- while recipes were being accepted or
rejected on differences smaller than that.
"""

import math

from scripts.analysis import _paired_bootstrap_delta


class TestPointEstimate:
    """The point estimate must be corpus WER, not a mean of per-utterance WERs."""

    def test_delta_is_a_ratio_of_totals(self):
        # 3/10 = 30% against 1/10 = 10%.
        point, _, _ = _paired_bootstrap_delta([1, 2], [5, 5], [1, 0], [5, 5], resamples=200)
        assert point == 20.0

    def test_long_references_are_not_downweighted(self):
        """Averaging per-utterance rates would give 25%; the corpus rate is 10%."""
        # One long clean utterance (0/90) and one short bad one (1/2).
        point, _, _ = _paired_bootstrap_delta([0, 1], [90, 2], [0, 0], [90, 2], resamples=200)
        assert math.isclose(point, 1 / 92 * 100, rel_tol=1e-9)

    def test_sign_convention_is_a_minus_b(self):
        worse, _, _ = _paired_bootstrap_delta([5], [10], [1], [10], resamples=200)
        assert worse > 0


class TestPairing:
    """Both models resample on the same indices, so shared difficulty cancels."""

    def test_identical_models_have_a_zero_width_interval(self):
        errs, refs = [1, 4, 0, 2], [10, 10, 10, 10]
        point, lo, hi = _paired_bootstrap_delta(errs, refs, errs, refs, resamples=500)
        assert (point, lo, hi) == (0.0, 0.0, 0.0)

    def test_a_one_word_difference_does_not_separate(self):
        refs = [10] * 50
        a = [2] * 50
        b = [2] * 49 + [1]
        _, lo, hi = _paired_bootstrap_delta(a, refs, b, refs, resamples=2000)
        assert lo <= 0 <= hi

    def test_a_large_consistent_gap_does_separate(self):
        # Varied per utterance: with identical rows every resample returns the
        # same WER and the interval collapses to a point (correctly, but it
        # would not exercise the percentile path).
        refs = [6 + (i % 7) for i in range(200)]
        a = [3 + (i % 2) for i in range(200)]
        b = [i % 2 for i in range(200)]
        point, lo, hi = _paired_bootstrap_delta(a, refs, b, refs, resamples=2000)
        assert lo > 0
        assert lo < point < hi

    def test_uniform_rows_collapse_the_interval_to_a_point(self):
        """Every resample is the same corpus, so there is nothing to vary."""
        refs = [10] * 200
        point, lo, hi = _paired_bootstrap_delta([4] * 200, refs, [1] * 200, refs, resamples=500)
        assert point == lo == hi
        assert math.isclose(point, 30.0, rel_tol=1e-9)


class TestIntervalProperties:
    def test_interval_brackets_the_point_estimate(self):
        refs = [8] * 100
        a = [3 if i % 3 else 1 for i in range(100)]
        b = [2 if i % 4 else 0 for i in range(100)]
        point, lo, hi = _paired_bootstrap_delta(a, refs, b, refs, resamples=3000)
        assert lo <= point <= hi

    def test_more_samples_give_a_tighter_interval(self):
        def width(n):
            refs = [8] * n
            a = [3 if i % 3 else 1 for i in range(n)]
            b = [2 if i % 4 else 0 for i in range(n)]
            _, lo, hi = _paired_bootstrap_delta(a, refs, b, refs, resamples=3000, seed=1)
            return hi - lo

        assert width(400) < width(50)

    def test_is_deterministic_for_a_fixed_seed(self):
        refs, a, b = [7] * 40, [2] * 40, [1] * 40
        first = _paired_bootstrap_delta(a, refs, b, refs, resamples=800, seed=7)
        second = _paired_bootstrap_delta(a, refs, b, refs, resamples=800, seed=7)
        assert first == second

    def test_empty_corpus_returns_nan_rather_than_dividing_by_zero(self):
        point, lo, hi = _paired_bootstrap_delta([], [], [], [], resamples=100)
        assert all(math.isnan(v) for v in (point, lo, hi))


class TestChunking:
    """The index matrix is chunked; results must not depend on the chunk size."""

    def test_large_resample_count_still_brackets_the_point(self):
        refs = [12] * 300
        a = [4] * 300
        b = [3] * 300
        point, lo, hi = _paired_bootstrap_delta(a, refs, b, refs, resamples=12_000)
        assert lo <= point <= hi
        assert math.isclose(point, (4 / 12 - 3 / 12) * 100, rel_tol=1e-9)
