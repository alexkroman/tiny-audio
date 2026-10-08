"""Tests for pattern-based ITN scoring (scripts/itn.py)."""

from scripts.itn import ITN_CLASSES, find_itn_spans, merge_scores, score_sample


class TestFindItnSpans:
    """Tests for ITN class detection over raw reference text."""

    def test_detects_each_class(self):
        """Every class in the registry fires on a representative string."""
        cases = {
            "email": "write to info@example.com today",
            "url": "see www.example.com for details",
            "phone": "call 555-123-4567 now",
            "version": "upgrade to v2.1.3 please",
            "clock_time": "the 3:00 p.m. meeting",
            "currency": "a $1,250.00 invoice",
            "percent": "up 9% this year",
            "temperature": "it hit 72°F outside",
            "unit": "a 64 inches display",
            "thousands_sep": "about 1,250 units",
            "fraction": "roughly 3/4 done",
            "range_score": "a 5-3 vote",
            "decimal": "exactly 22.5 points",
            "acronym_dotted": "back in the U.S.A.",
            "title_abbrev": "Dr. Smith arrived",
            "alphanum_id": "the B737 fleet",
            "year": "back in 2019",
            "integer": "all 42 of them",
        }
        assert set(cases) == {name for name, _ in ITN_CLASSES}

        for expected_class, text in cases.items():
            found = {name for _, _, name, _ in find_itn_spans(text)}
            assert expected_class in found, f"{expected_class!r} missed in {text!r} (got {found})"

    def test_ordinals_are_deliberately_unclaimed(self):
        """`ordinal` was removed: word<->digit is unscoreable by `_loose_tokens`.

        Guard that nothing else silently picks the span up -- `integer`
        (`\\b\\d+\\b`) does not match "1st" because a word character follows the
        digit, and `alphanum_id` requires a leading letter. If a future class
        claims it, `loose - exact` starts reporting recognition errors for
        what is only a formatting choice.
        """
        for text in ("january 1st through december 31st", "the 21st of June"):
            claimed = {name for _, _, name, _ in find_itn_spans(text)}
            assert "ordinal" not in claimed
            reclaimed = {"integer", "alphanum_id"} & claimed
            assert not reclaimed, f"ordinal span re-claimed in {text!r} by {reclaimed}"

    def test_spans_do_not_overlap(self):
        """A specific class claims the span; later classes cannot re-count it."""
        spans = find_itn_spans("a $1,250.00 invoice")
        assert [(name, text) for _, _, name, text in spans] == [("currency", "$1,250.00")]

    def test_no_spans_in_plain_text(self):
        assert find_itn_spans("the quick brown fox jumped over it") == []


class TestScoreSample:
    """Tests for the exact / loose / miss three-way split."""

    def test_exact_match_counts_both(self):
        stats = score_sample("it cost $1,250", "it cost $1,250")
        assert stats["currency"] == {"total": 1, "exact": 1, "loose": 1}

    def test_formatting_difference_is_loose_only(self):
        """Right value, wrong formatting -- a formatting error, not a miss."""
        stats = score_sample("it cost $1,250", "it cost 1250")
        assert stats["currency"] == {"total": 1, "exact": 0, "loose": 1}

    def test_spelled_out_number_is_a_miss(self):
        """The whole point: spelled-out output must not be credited."""
        stats = score_sample("it cost $25", "it cost twenty five dollars")
        assert stats["currency"] == {"total": 1, "exact": 0, "loose": 0}

    def test_case_insensitive_exact(self):
        stats = score_sample("the B737 fleet", "the b737 fleet")
        assert stats["alphanum_id"]["exact"] == 1

    def test_counts_repeated_occurrences(self):
        stats = score_sample("9% then 15%", "9% then 15%")
        assert stats["percent"] == {"total": 2, "exact": 2, "loose": 2}

    def test_empty_prediction_scores_zero(self):
        stats = score_sample("it cost $25", "")
        assert stats["currency"] == {"total": 1, "exact": 0, "loose": 0}

    def test_reference_without_itn_yields_nothing(self):
        assert score_sample("hello there", "hello there") == {}


class TestMergeScores:
    """Tests for corpus-level accumulation."""

    def test_merge_accumulates_per_class(self):
        target = {}
        merge_scores(target, score_sample("it cost $25", "it cost $25"))
        merge_scores(target, score_sample("it cost $30", "it cost thirty dollars"))
        assert target["currency"] == {"total": 2, "exact": 1, "loose": 1}

    def test_merge_returns_target(self):
        target = {}
        assert merge_scores(target, {"percent": {"total": 1, "exact": 1, "loose": 1}}) is target
