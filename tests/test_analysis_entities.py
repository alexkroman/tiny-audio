"""Tests for entity extraction and the missed-entity table (scripts/analysis.py).

Entity extraction runs spaCy over the *raw* reference, not the
`EnglishTextNormalizer` output the rest of `results.txt` is scored against, so
these tests stub spaCy with a recognizer that only fires on capitalized text --
a fake that cannot produce a single entity if the normalized (lowercased) text
is passed by mistake.
"""

import json
import re
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.analysis import ITN_COVERED_ENTITY_TYPES, app

runner = CliRunner()

# Capitalized runs of words, so lowercase (normalized) input yields nothing.
_CAPITALIZED = re.compile(r"\b[A-Z][\w.]*(?:\s+[A-Z][\w.]*)*")


class _FakeEnt:
    def __init__(self, text, label, start, end):
        self.text = text
        self.label_ = label
        self.start_char = start
        self.end_char = end


class _FakeDoc:
    def __init__(self, text):
        self.ents = [
            _FakeEnt(m.group(0), "PERSON", m.start(), m.end()) for m in _CAPITALIZED.finditer(text)
        ]


@pytest.fixture
def fake_spacy(monkeypatch):
    """Install a stub `spacy` module; `extract-entities` imports it lazily."""
    module = types.ModuleType("spacy")
    module.load = lambda name: _FakeDoc
    monkeypatch.setitem(sys.modules, "spacy", module)
    return module


def _write_results(path: Path, samples: list[dict]) -> None:
    """Write a results.txt in the format `parse_results_file` expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
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
def outputs(tmp_path, monkeypatch):
    """An outputs dir with one run whose samples carry raw transcripts."""
    monkeypatch.setattr("scripts.analysis.KEYWORDS_FILE", str(tmp_path / "keywords.json"))
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
            },
        ],
    )
    return tmp_path


class TestExtractEntities:
    """`extract-entities` reads the raw reference, keyed by the normalized one."""

    def _run(self, outputs, extra=None):
        result = runner.invoke(
            app,
            ["extract-entities", "--output-dir", str(outputs), "--min-count", "1", *(extra or [])],
        )
        assert result.exit_code == 0, result.output
        return json.loads((outputs / "keywords.json").read_text())

    def test_entities_come_from_raw_text(self, fake_spacy, outputs):
        """The regression: spaCy was fed lowercased, unpunctuated text."""
        keywords = self._run(outputs)
        found = {e["text"] for ref in keywords["references"] for e in ref["entities"]}
        assert "Barack Obama" in found
        assert "Angela Merkel" in found

    def test_references_keyed_by_normalized_text(self, fake_spacy, outputs):
        """Consumers look references up by the normalized ground truth."""
        keywords = self._run(outputs)
        assert {ref["text"] for ref in keywords["references"]} == {
            "barack obama visited paris",
            "angela merkel spoke in berlin",
        }

    def test_records_raw_as_the_source(self, fake_spacy, outputs):
        assert self._run(outputs)["source_text"] == "raw"

    def test_skips_references_without_raw_text(self, fake_spacy, outputs):
        """Old runs predate raw transcripts; they are skipped, not normalized-scored."""
        _write_results(
            outputs / "20240101_120000_modelb_tedlium" / "results.txt",
            [{"gt": "an older run with no raw line", "pred": "an older run with no raw line"}],
        )
        keywords = self._run(outputs)
        assert keywords["references_missing_raw"] == 1
        assert "an older run with no raw line" not in {r["text"] for r in keywords["references"]}

    def test_raw_in_any_run_covers_the_reference(self, fake_spacy, outputs):
        """A reference skipped in one file is still extracted from another that has raw."""
        _write_results(
            outputs / "20240101_120000_modelb_librispeech" / "results.txt",
            [{"gt": "barack obama visited paris", "pred": "barack obama visited paris"}],
        )
        keywords = self._run(outputs)
        assert keywords["references_missing_raw"] == 0
        assert "barack obama visited paris" in {r["text"] for r in keywords["references"]}

    def test_rare_types_reported_below_threshold(self, fake_spacy, outputs):
        keywords = self._run(outputs, extra=["--min-count", "99"])
        assert keywords["excluded_types"] == {"PERSON": 4}
        assert keywords["entity_counts_by_type"] == {}


class TestEntityTableColumns:
    """The table reports semantic recall; numeric types belong to the ITN table."""

    @pytest.fixture
    def compared(self, outputs):
        """Run `compare` against a keywords file holding both type families."""
        (outputs / "keywords.json").write_text(
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
                        }
                    ]
                }
            )
        )
        result = runner.invoke(app, ["compare", "modela", "--output-dir", str(outputs)])
        assert result.exit_code == 0, result.output
        return result.output

    @pytest.fixture
    def entity_section(self, compared):
        """Just the missed-entity table, so ITN's own columns cannot satisfy an assert."""
        assert "Missed Entity Errors" in compared
        return compared.split("Missed Entity Errors")[1].split("ITN Formatting")[0]

    def test_semantic_types_are_shown(self, entity_section):
        assert "PERSON" in entity_section
        assert "GPE" in entity_section

    def test_itn_covered_types_are_dropped(self, entity_section):
        for etype in ITN_COVERED_ENTITY_TYPES:
            assert etype not in entity_section, f"{etype} is scored in the ITN table"

    def test_average_excludes_dropped_types(self, entity_section):
        """DATE and CARDINAL were missed; counting them would move the average.

        Only `Paris` and `Barack Obama` are in the prediction, so semantic recall
        is 100% -- an average over all four reference entities would read 50%.
        """
        row = next(line for line in entity_section.splitlines() if "modela" in line)
        # Cell-wise: "50.00%" contains "0.00%", so a substring check proves nothing.
        cells = [c.strip() for c in row.strip("\u2502").split("\u2502")]
        assert cells == ["modela", "0.00%", "0.00%", "0.00%"]  # Average, GPE, PERSON
