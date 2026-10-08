"""The quality ratchets in scripts/quality.py."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts import quality

runner = CliRunner()


class TestCodeLines:
    def test_blank_and_comment_only_lines_do_not_count(self) -> None:
        text = "import os\n\n# comment\n    # indented comment\nx = 1  # trailing\n"
        assert quality.code_lines(text) == 2


class TestFileLengthViolations:
    CAP = quality.MAX_CODE_LINES

    def test_new_file_over_the_cap_fails(self) -> None:
        (problem,) = quality.file_length_violations({"a.py": self.CAP + 1}, {})
        assert "over the" in problem

    def test_file_at_the_cap_passes(self) -> None:
        assert quality.file_length_violations({"a.py": self.CAP}, {}) == []

    def test_grandfathered_file_may_stay_or_shrink(self) -> None:
        baseline = {"a.py": self.CAP + 50}
        assert quality.file_length_violations({"a.py": self.CAP + 50}, baseline) == []
        assert quality.file_length_violations({"a.py": self.CAP + 10}, baseline) == []

    def test_grandfathered_file_may_not_grow(self) -> None:
        (problem,) = quality.file_length_violations(
            {"a.py": self.CAP + 51}, {"a.py": self.CAP + 50}
        )
        assert "grew past" in problem


class TestCoverage:
    @staticmethod
    def report(**files: tuple[int, int]) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
        return {
            "files": {
                path: {"summary": {"num_statements": total, "covered_lines": covered}}
                for path, (covered, total) in files.items()
            }
        }

    def test_percent_per_file_and_empty_files_count_as_covered(self) -> None:
        cov = quality.file_coverage(self.report(**{"a.py": (3, 4), "empty.py": (0, 0)}))
        assert cov == {"a.py": 75.0, "empty.py": 100.0}

    def test_recorded_floor_is_enforced(self) -> None:
        (problem,) = quality.coverage_violations({"a.py": 70.0}, {"a.py": 75.0})
        assert "recorded floor" in problem
        assert quality.coverage_violations({"a.py": 75.0}, {"a.py": 75.0}) == []

    def test_new_file_needs_the_new_file_floor(self) -> None:
        floor = quality.NEW_FILE_COVERAGE_FLOOR
        (problem,) = quality.coverage_violations({"new.py": floor - 1}, {})
        assert "new-file floor" in problem
        assert quality.coverage_violations({"new.py": floor}, {}) == []

    def test_update_records_floors_the_same_run_passes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        report, baseline = tmp_path / "coverage.json", tmp_path / "floors.json"
        report.write_text(json.dumps(self.report(**{"a.py": (2, 3)})))  # 66.666...%
        monkeypatch.setattr(quality, "COVERAGE_REPORT", report)
        monkeypatch.setattr(quality, "COVERAGE_BASELINE", baseline)
        monkeypatch.setattr(quality, "ROOT", tmp_path)

        assert runner.invoke(quality.app, ["coverage-floors", "--update"]).exit_code == 0
        assert json.loads(baseline.read_text()) == {"a.py": 66.6}
        assert runner.invoke(quality.app, ["coverage-floors"]).exit_code == 0

    def test_missing_report_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(quality, "COVERAGE_REPORT", tmp_path / "missing.json")
        result = runner.invoke(quality.app, ["coverage-floors"])
        assert result.exit_code == 1
        assert "ta dev test" in result.output


class TestTestsWithoutAssertions:
    @pytest.mark.parametrize(
        "body",
        [
            "assert x",
            "with pytest.raises(ValueError):\n        f()",
            "mock.assert_called_once()",
            "_assert_counts(1)",
            "_require_fused_ce(cfg)",
            "check_disk(conn)",
        ],
    )
    def test_assertion_shapes_are_recognized(self, body: str) -> None:
        assert quality.tests_without_assertions(f"def test_x():\n    {body}\n") == []

    def test_a_test_that_only_runs_code_is_reported(self) -> None:
        source = "def test_runs():\n    f()\n\ndef test_checks():\n    assert f()\n"
        assert quality.tests_without_assertions(source) == [(1, "test_runs")]

    def test_helpers_that_assert_count_transitively(self) -> None:
        source = (
            "def _inner(x):\n    assert x\n\n"
            "def _outer(x):\n    _inner(x)\n\n"
            "class TestX:\n    def test_uses_helper(self):\n        _outer(1)\n"
        )
        assert quality.tests_without_assertions(source) == []

    def test_the_repository_suite_passes(self) -> None:
        result = runner.invoke(quality.app, ["test-assertions"])
        assert result.exit_code == 0, result.output
