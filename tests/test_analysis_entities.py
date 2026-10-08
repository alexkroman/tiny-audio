"""Tests for the per-sample reports and the missed-entity table (scripts/analysis/)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.analysis import app
from scripts.analysis.common import ITN_COVERED_ENTITY_TYPES

runner = CliRunner()

Sample = dict[str, str | float]


def _write_results(path: Path, samples: list[Sample], run_id: str = "testrun") -> None:
    """Write a results.txt in the format `parse_results_file` expects.

    Also writes the minimal metrics.txt carrying a `Run ID`, because
    `latest_sweep` reads runs by sweep and drops directories without one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.parent.name
    stamp = name[:15]
    dataset = name.rsplit("_", 1)[-1]
    (path.parent / "metrics.txt").write_text(
        f"Model: {name}\nDataset: {dataset}\nTimestamp: {stamp}\nRun ID: {run_id}\n"
        + "-" * 40
        + f"\nwer: 0.0\nnum_samples: {len(samples)}\n"
    )
    blocks: list[str] = []
    for i, s in enumerate(samples, start=1):
        lines = [
            f"Sample {i} - WER: {s.get('wer', 0.0)}%",
            f"Ground Truth: {s['gt']}",
            f"Prediction: {s['pred']}",
        ]
        if "gt_raw" in s:
            lines.append(f"Ground Truth Raw: {s['gt_raw']}")
            lines.append(f"Prediction Raw: {s['pred_raw']}")
        blocks.append("\n".join(lines))
    path.write_text(("\n" + "-" * 80 + "\n").join(blocks))


@pytest.fixture
def outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An outputs dir with one run whose samples carry raw transcripts."""
    monkeypatch.setattr("scripts.analysis.common.KEYWORDS_FILE", tmp_path / "keywords.json")
    _write_results(
        tmp_path / "20250101_120000_modela_librispeech" / "results.txt",
        [
            {
                "gt": "barack obama visited paris",
                "pred": "barack obama visited paris",
                "gt_raw": "Barack Obama visited Paris.",
                "pred_raw": "Barack Obama visited Paris.",
            },
            {
                "gt": "angela merkel spoke in berlin",
                "pred": "angela mercle spoke in berlin",
                "gt_raw": "Angela Merkel spoke in Berlin.",
                "pred_raw": "Angela Mercle spoke in Berlin.",
                "wer": 20.0,
            },
        ],
    )
    (tmp_path / "keywords.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "text": "barack obama visited paris",
                        "entities": [
                            {"text": "Barack Obama", "label": "PERSON"},
                            {"text": "Paris", "label": "GPE"},
                            {"text": "1961", "label": "DATE"},
                            {"text": "three", "label": "CARDINAL"},
                        ],
                    },
                    {
                        "text": "angela merkel spoke in berlin",
                        "entities": [
                            {"text": "Angela Merkel", "label": "PERSON"},
                            {"text": "Berlin", "label": "GPE"},
                        ],
                    },
                ]
            }
        )
    )
    return tmp_path


class TestEntityErrors:
    def _run(self, outputs: Path, *extra: str) -> str:
        result = runner.invoke(app, ["entity-errors", "modela", "-o", str(outputs), *extra])
        assert result.exit_code == 0, result.output
        return result.output

    def test_reports_only_missed_entities(self, outputs: Path) -> None:
        """Every type counts here, numeric ones included -- unlike the compare table."""
        output = self._run(outputs)
        assert "Missing: Angela Merkel (PERSON)" in output
        assert "Missing: 1961 (DATE), three (CARDINAL)" in output
        assert "Berlin (GPE)" not in output
        assert "Paris (GPE)" not in output

    def test_entity_type_filter(self, outputs: Path) -> None:
        output = self._run(outputs, "--entity-type", "person")
        assert "Total: 1 samples" in output
        assert "Missing: Angela Merkel (PERSON)" in output
        assert "Total: 0 samples" in self._run(outputs, "--entity-type", "gpe")

    def test_missing_keywords_file_fails(self, outputs: Path) -> None:
        (outputs / "keywords.json").unlink()
        result = runner.invoke(app, ["entity-errors", "modela", "-o", str(outputs)])
        assert result.exit_code == 1


class TestHighWer:
    def test_filters_and_writes_report(self, outputs: Path) -> None:
        report = outputs / "report.txt"
        args = ["high-wer", "modela", "-o", str(outputs), "--threshold", "10"]
        result = runner.invoke(app, [*args, "--output-file", str(report)])
        assert result.exit_code == 0, result.output
        text = report.read_text()
        assert "Total: 1 samples" in text
        assert "WER: 20.0%" in text
        assert "angela mercle" in text

    def test_unknown_model_fails(self, outputs: Path) -> None:
        result = runner.invoke(app, ["high-wer", "nomodel", "-o", str(outputs)])
        assert result.exit_code == 1


class TestEntityTableColumns:
    """The table reports semantic recall; numeric types belong to the ITN table."""

    @pytest.fixture
    def entity_section(self, outputs: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        """Just the missed-entity table, so ITN's own columns cannot satisfy an assert.

        `MIN_CLASS_SUPPORT` is lowered because these assert on which COLUMNS
        appear, not on whether a handful of entities is enough to report.
        """
        monkeypatch.setattr("scripts.analysis.tables.MIN_CLASS_SUPPORT", 1)
        result = runner.invoke(app, ["compare", "modela", "--output-dir", str(outputs)])
        assert result.exit_code == 0, result.output
        assert "Missed Entity Errors" in result.output
        return result.output.split("Missed Entity Errors")[1].split("ITN Formatting")[0]

    def test_semantic_types_are_shown(self, entity_section: str) -> None:
        assert "PERSON" in entity_section
        assert "GPE" in entity_section

    def test_itn_covered_types_are_dropped(self, entity_section: str) -> None:
        for etype in ITN_COVERED_ENTITY_TYPES:
            assert etype not in entity_section, f"{etype} is scored in the ITN table"

    def test_average_excludes_dropped_types(self, entity_section: str) -> None:
        """DATE and CARDINAL were missed; counting them would move the average.

        One of four semantic entities (Angela Merkel) is missed: 25% on
        average, 50% of PERSON, 0% of GPE. Counting the two numeric entities
        would read 50% on average.
        """
        row = next(line for line in entity_section.splitlines() if "modela" in line)
        # Cell-wise: "50.00%" contains "0.00%", so a substring check proves nothing.
        cells = [c.strip() for c in row.strip("│").split("│")]
        assert cells == ["modela", "25.00%", "0.00%", "50.00%"]  # Average, GPE, PERSON

    def test_thin_support_skips_the_table(
        self, outputs: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("scripts.analysis.tables.MIN_CLASS_SUPPORT", 99)
        result = runner.invoke(app, ["compare", "modela", "--output-dir", str(outputs)])
        assert result.exit_code == 0, result.output
        assert "Entity table skipped" in result.output


def test_compare_without_a_sweep_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compare", "nomodel", "--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No Run ID found" in result.output


def test_speaker_tables_for_tagged_runs(tmp_path: Path) -> None:
    rows: list[Sample] = [
        {
            "gt": "alpha beta gamma delta",
            "pred": "alpha beta gamma delta",
            "gt_raw": "<SPK_1>alpha beta<SPK_2>gamma delta",
            "pred_raw": raw,
        }
        for raw in ("<SPK_1>alpha<SPK_2>beta gamma delta", "<SPK_1>alpha beta<SPK_2>gamma delta")
    ]
    _write_results(tmp_path / "20250101_120000_modela_ami-speakers-long" / "results.txt", rows)
    _write_results(
        tmp_path / "20250101_130000_modelb_ami-speakers-long" / "results.txt", rows[::-1]
    )
    result = runner.invoke(app, ["compare", "modela", "modelb", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Speaker Attribution" in result.output
    assert "Corpus cpWER delta" in result.output
