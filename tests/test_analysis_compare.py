"""`ta analysis compare`: one sweep per model, paired, rendered as comparison tables."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.analysis import app, common
from scripts.analysis.compare import _cpwer_series, _verdict, _wer_series, print_corpus_cis
from scripts.analysis.metrics import ModelMetrics

runner = CliRunner()

GOOD = [("alpha beta gamma delta", "alpha beta gamma delta")] * 20
BAD = [("alpha beta gamma delta", "alpha beta gamma epsilon")] * 20


def _write(
    root: Path, ts: str, model: str, ds: str, run_id: str, pairs: list[tuple[str, str]]
) -> None:
    """One `ta eval` run directory: metrics.txt carrying a Run ID, and results.txt."""
    d = root / f"{ts}_{model}_{ds}"
    d.mkdir(parents=True)
    header = f"Model: {model}\nDataset: {ds}\nTimestamp: {ts}\nRun ID: {run_id}\n"
    (d / "metrics.txt").write_text(header + "-" * 40 + f"\nwer: 0.0\nnum_samples: {len(pairs)}\n")
    (d / "results.txt").write_text(
        "".join(
            f"Sample {i} - WER: 0.00%\nGround Truth: {r}\nPrediction: {p}\n"
            f"Ground Truth Raw: {r}\nPrediction Raw: {p}\n" + "-" * 80 + "\n"
            for i, (r, p) in enumerate(pairs, 1)
        )
    )


@pytest.fixture
def outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # no outputs/keywords.json: entity columns stay empty
    monkeypatch.setattr(common.console, "width", 400)  # keep table titles on one line
    root = tmp_path / "outputs"
    for ds in ("ami", "tedlium"):
        _write(root, "20260101_000000", "modelA", ds, "ra", GOOD)
        _write(root, "20260101_010000", "modelB", ds, "rb", BAD)
    return root


class TestCompare:
    def test_prints_every_table_and_separates_a_clear_gap(self, outputs: Path) -> None:
        result = runner.invoke(app, ["compare", "modelA", "modelB", "-o", str(outputs)])
        assert result.exit_code == 0, result.output
        out = result.output
        assert "Accuracy by WER (corpus = 2/2 datasets" in out
        assert "Corpus WER delta vs modelA" in out
        # modelB misses one word in four on every row: a 25-point gap, every resample.
        assert any(
            "modelB" in line and "+25.00" in line and line.rstrip().endswith("│ separated │")
            for line in out.splitlines()
        )

    def test_a_model_without_a_sweep_exits(self, outputs: Path) -> None:
        result = runner.invoke(app, ["compare", "modelA", "nobody", "-o", str(outputs)])
        assert result.exit_code == 1
        assert "No Run ID found for: nobody" in result.output

    def test_excluded_dataset_leaves_the_corpus(self, outputs: Path) -> None:
        args = ["compare", "modelA", "modelB", "-o", str(outputs), "--exclude-dataset", "tedlium"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "corpus = 1/1 datasets" in result.output


def _metrics() -> ModelMetrics:
    return {
        "display_name": "m",
        "sweep": "s",
        "datasets": {},
        "by_length": {},
        "entity_errors": {},
        "itn": {},
        "itn_raw_samples": 0,
    }


@pytest.mark.parametrize(
    ("lo", "hi", "verdict"),
    [(0.5, 1.0, "separated"), (-1.0, 1.0, "not separated"), (-1.0, -0.5, "inconsistent")],
)
def test_verdict(lo: float, hi: float, verdict: str) -> None:
    assert verdict in _verdict(lo, hi)


def test_series_need_their_corpus_metric() -> None:
    empty = _metrics()
    assert _wer_series(empty) is None
    assert _cpwer_series(empty) is None
    full: ModelMetrics = {
        **empty,
        "corpus_cpwer": 12.5,
        "corpus_cp_utt_errors": [1],
        "corpus_cp_utt_ref_words": [8],
    }
    assert _cpwer_series(full) == (12.5, [1], [8])


def test_corpus_cis_need_two_models(capsys: pytest.CaptureFixture[str]) -> None:
    only: ModelMetrics = {**_metrics(), "corpus_wer": 1.0, "corpus_utt_errors": [1]}
    print_corpus_cis({"m": only}, "WER", _wer_series)
    assert capsys.readouterr().out == ""
