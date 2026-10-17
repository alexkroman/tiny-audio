"""`ta analysis high-wer` and `ta analysis entity-errors`: per-sample reports from results.txt."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.analysis import app, common, samples

runner = CliRunner()


def _write_run(root: Path, name: str, rows: list[tuple[str, str, float]]) -> None:
    """One run directory whose results.txt holds (reference, prediction, wer) rows."""
    run = root / name
    run.mkdir(parents=True)
    (run / "results.txt").write_text(
        "".join(
            f"Sample {i} - WER: {wer:.2f}%\nGround Truth: {ref}\nPrediction: {hyp}\n"
            + "-" * 80
            + "\n"
            for i, (ref, hyp, wer) in enumerate(rows, 1)
        )
    )


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    _write_run(
        tmp_path,
        "20260101_000000_modelA_ami",
        [
            ("alice met bob in paris", "alice met bob in paris", 0.0),
            ("the meeting is on monday", "a meeting on sunday", 60.0),
        ],
    )
    _write_run(
        tmp_path,
        "20260101_000100_modelA_tedlium",
        [("visit london next week", "visit lumber next week", 25.0)],
    )
    _write_run(tmp_path, "20260101_000000_modelB_ami", [("x", "y", 100.0)])
    return tmp_path


class TestHighWer:
    def test_reports_rows_at_or_above_the_threshold_worst_first(
        self, outputs: Path, tmp_path: Path
    ) -> None:
        report = tmp_path / "report.txt"
        args = ["high-wer", "modelA", "-o", str(outputs), "--threshold", "25"]
        result = runner.invoke(app, [*args, "--output-file", str(report)])
        assert result.exit_code == 0, result.output
        text = report.read_text()
        assert "# Total: 2 samples" in text
        assert text.index("WER: 60.0%") < text.index("WER: 25.0%")
        assert "Dataset: ami | Sample: 2" in text
        assert "alice met bob" not in text  # 0% WER is below the threshold

    def test_unknown_model_exits_with_an_error(self, outputs: Path) -> None:
        result = runner.invoke(app, ["high-wer", "nobody", "-o", str(outputs)])
        assert result.exit_code == 1
        assert "No results files found" in result.output


class TestEntityErrors:
    @pytest.fixture
    def keywords(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        path = tmp_path / "keywords.json"
        refs = {
            "alice met bob in paris": [
                {"text": "alice", "label": "PERSON"},
                {"text": "paris", "label": "GPE"},
            ],
            "the meeting is on monday": [{"text": "monday", "label": "DATE"}],
            "visit london next week": [{"text": "london", "label": "GPE"}],
        }
        path.write_text(
            json.dumps({"references": [{"text": t, "entities": e} for t, e in refs.items()]})
        )
        monkeypatch.setattr(common, "KEYWORDS_FILE", path)

    @pytest.mark.usefixtures("keywords")
    def test_lists_samples_that_missed_an_entity(self, outputs: Path, tmp_path: Path) -> None:
        report = tmp_path / "report.txt"
        args = ["entity-errors", "modelA", "-o", str(outputs), "--output-file", str(report)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        text = report.read_text()
        assert "# Total: 2 samples" in text
        assert "Missing: monday (DATE)" in text
        assert "Missing: london (GPE)" in text
        assert "alice" not in text  # every entity of that row was transcribed

    @pytest.mark.usefixtures("keywords")
    def test_entity_type_filters_case_insensitively(self, outputs: Path, tmp_path: Path) -> None:
        report = tmp_path / "report.txt"
        args = ["entity-errors", "modelA", "-o", str(outputs), "--entity-type", "gpe"]
        result = runner.invoke(app, [*args, "--output-file", str(report)])
        assert result.exit_code == 0, result.output
        text = report.read_text()
        assert "# Entity Type: gpe" in text
        assert "# Total: 1 samples" in text
        assert "monday" not in text

    def test_no_keywords_file_exits_with_an_error(
        self, outputs: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(common, "KEYWORDS_FILE", tmp_path / "missing.json")
        result = runner.invoke(app, ["entity-errors", "modelA", "-o", str(outputs)])
        assert result.exit_code == 1
        assert "No reference entities" in result.output


def test_missing_entities_matches_type_and_text() -> None:
    entities: list[common.Entity] = [
        {"text": "Paris", "label": "GPE"},
        {"text": "Bob", "label": "PERSON"},
    ]
    assert samples._missing_entities(entities, "bob went home", "") == [entities[0]]
    assert samples._missing_entities(entities, "bob went home", "person") == []


def test_report_prints_without_an_output_file(capsys: pytest.CaptureFixture[str]) -> None:
    samples._emit(["line one"], None)
    assert "line one" in capsys.readouterr().out
