"""Tests for the table sort keys in scripts/analysis.py."""

from scripts.analysis import _sort_key, _sort_key_desc


class TestSortKey:
    """Ascending key: lower is better (WER, latency)."""

    def test_parses_percentages(self):
        assert _sort_key("12.50%") == 12.5

    def test_parses_plain_numbers(self):
        assert _sort_key("3.25") == 3.25

    def test_missing_sorts_last(self):
        assert _sort_key("-") == float("inf")

    def test_unparseable_sorts_last(self):
        assert _sort_key("n/a") == float("inf")


class TestSortKeyDesc:
    """Descending key: higher is better (confidence margin)."""

    def test_orders_high_to_low(self):
        rows = ["1.50", "3.25", "2.00"]
        assert sorted(rows, key=_sort_key_desc) == ["3.25", "2.00", "1.50"]

    def test_missing_values_sort_last(self):
        """The regression: `-_sort_key("-")` is -inf, which sorted to the top."""
        rows = ["1.50", "-", "3.25"]
        assert sorted(rows, key=_sort_key_desc) == ["3.25", "1.50", "-"]

    def test_all_missing_is_stable(self):
        assert sorted(["-", "-"], key=_sort_key_desc) == ["-", "-"]

    def test_unparseable_sorts_last(self):
        assert sorted(["2.00", "n/a"], key=_sort_key_desc) == ["2.00", "n/a"]

    def test_handles_percentages(self):
        assert sorted(["10.00%", "20.00%"], key=_sort_key_desc) == ["20.00%", "10.00%"]
