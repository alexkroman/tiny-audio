"""Tests for scripts/analysis/tables.py: the column order every compare table shares."""

from typing import Any, cast

from scripts.analysis.metrics import ModelMetrics
from scripts.analysis.tables import EXCLUDED_DATASETS, ordered_datasets


def _models(*dataset_sets: set[str]) -> dict[str, ModelMetrics]:
    """Stand-in metrics: only `datasets` is read by ordered_datasets."""
    return {
        f"m{i}": cast(ModelMetrics, {"datasets": dict.fromkeys(ds, cast(Any, {}))})
        for i, ds in enumerate(dataset_sets)
    }


def test_canonical_datasets_lead_then_the_rest_alphabetically() -> None:
    models = _models({"zeta", "ami", "earnings22"}, {"alpha", "ami"})
    assert ordered_datasets(models) == ["earnings22", "ami", "alpha", "zeta"]


def test_excluded_datasets_never_get_a_column() -> None:
    excluded = next(iter(EXCLUDED_DATASETS))
    assert ordered_datasets(_models({excluded, "ami"})) == ["ami"]
