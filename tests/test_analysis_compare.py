"""`ta analysis compare`: one sweep per model, paired, rendered as comparison tables."""

from pathlib import Path

import pytest
from eval_outputs import write_run
from typer.testing import CliRunner

from scripts.analysis import app, common
from scripts.analysis.compare import _cpwer_series, _verdict, print_corpus_cis
from scripts.analysis.metrics import ModelMetrics, collect_model_metrics

runner = CliRunner()

GOOD = [("alpha beta gamma delta", "alpha beta gamma delta")] * 20
BAD = [("alpha beta gamma delta", "alpha beta gamma epsilon")] * 20


@pytest.fixture
def outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an outputs dir where modelA is perfect and modelB misses 1 word in 4.

    Each model has one sweep over ami and tedlium. The test also runs from tmp_path
    and widens the console so table titles stay on one line.
    """
    monkeypatch.chdir(tmp_path)  # no outputs/keywords.json: entity columns stay empty
    monkeypatch.setattr(common.console, "width", 400)  # keep table titles on one line
    root = tmp_path / "outputs"
    for ds in ("ami", "tedlium"):
        write_run(root, "20260101_000000", "modelA", ds, "ra", GOOD)
        write_run(root, "20260101_010000", "modelB", ds, "rb", BAD)
    return root


class TestCompare:
    """The compare command prints its tables, needs a sweep per model and honours excludes."""

    def test_prints_every_table_and_separates_a_clear_gap(self, outputs: Path) -> None:
        result = runner.invoke(app, ["compare", "modelA", "modelB", "-o", str(outputs)])
        assert result.exit_code == 0, result.output
        out = result.output
        assert "Accuracy by WER (corpus = 2/2 datasets" in out
        # Insertions, deletions, substitutions in that order; the ITN table is gone.
        assert out.index("Insertion Rate") < out.index("Deletion Rate")
        assert out.index("Deletion Rate") < out.index("Substitution Rate")
        assert "ITN Formatting" not in out
        assert "Entity" not in out  # the entity table is gone
        # The WER bootstrap table is gone; the per-dataset WER table carries the gap.
        assert "Corpus WER delta" not in out
        assert any("modelB" in line and "25.00%" in line for line in out.splitlines())

    def test_a_model_without_a_sweep_exits(self, outputs: Path) -> None:
        result = runner.invoke(app, ["compare", "modelA", "nobody", "-o", str(outputs)])
        assert result.exit_code == 1
        assert "No Run ID found for: nobody" in result.output

    def test_excluded_dataset_leaves_the_corpus(self, outputs: Path) -> None:
        args = ["compare", "modelA", "modelB", "-o", str(outputs), "--exclude-dataset", "tedlium"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "corpus = 1/1 datasets" in result.output


@pytest.mark.parametrize(
    ("lo", "hi", "verdict"),
    [(0.5, 1.0, "separated"), (-1.0, 1.0, "not separated"), (-1.0, -0.5, "inconsistent")],
)
def test_verdict(lo: float, hi: float, verdict: str) -> None:
    assert verdict in _verdict(lo, hi)


def test_series_need_their_corpus_metric(tmp_path: Path) -> None:
    empty = collect_model_metrics("m", tmp_path)  # no runs: no corpus keys
    assert _cpwer_series(empty) is None
    full: ModelMetrics = {
        **empty,
        "corpus_cpwer": 12.5,
        "corpus_cp_utt_errors": [1],
        "corpus_cp_utt_ref_words": [8],
    }
    assert _cpwer_series(full) == (12.5, [1], [8])


def test_corpus_cis_need_two_models(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    only: ModelMetrics = {
        **collect_model_metrics("m", tmp_path),
        "corpus_cpwer": 1.0,
        "corpus_cp_utt_errors": [1],
        "corpus_cp_utt_ref_words": [8],
    }
    print_corpus_cis({"m": only}, "cpWER", _cpwer_series)
    assert capsys.readouterr().out == ""


def test_corpus_cis_skip_a_model_scored_on_other_rows(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A model whose per-utterance series has another length is left out, not mis-paired."""
    base = collect_model_metrics("m", tmp_path)
    best: ModelMetrics = {
        **base,
        "display_name": "best",
        "corpus_cpwer": 1.0,
        "corpus_cp_utt_errors": [0, 1],
        "corpus_cp_utt_ref_words": [8, 8],
    }
    other: ModelMetrics = {
        **base,
        "display_name": "other",
        "corpus_cpwer": 9.0,
        "corpus_cp_utt_errors": [3],
        "corpus_cp_utt_ref_words": [8],
    }
    print_corpus_cis({"best": best, "other": other}, "cpWER", _cpwer_series)
    out = capsys.readouterr().out
    assert "Corpus cpWER delta vs best" in out
    assert "other" not in out.split("Corpus cpWER delta vs best")[1]
