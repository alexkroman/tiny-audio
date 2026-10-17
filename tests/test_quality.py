"""The quality ratchets in scripts/quality.py."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from scripts import quality

runner = CliRunner()


class TestCodeLines:
    """`code_lines` counts only lines that hold code."""

    def test_blank_and_comment_only_lines_do_not_count(self) -> None:
        text = "import os\n\n# comment\n    # indented comment\nx = 1  # trailing\n"
        assert quality.code_lines(text) == 2


class TestFileLengthViolations:
    """New files are capped; grandfathered files may shrink but never grow."""

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
    """Per-file coverage floors from coverage.py's JSON report."""

    @staticmethod
    def report(**files: tuple[int, int]) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
        """A coverage.py JSON report with `(covered, total)` lines per file."""
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

    def test_update_records_floors_the_same_run_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
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
    """Tests that run code but check nothing are reported."""

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


class TestEscapeHatches:
    """Per-file budgets for checker-silencing comments, which only go down."""

    def test_only_real_comments_count(self) -> None:
        source = (
            "import os  # noqa: F401\n"
            "x = f()  # pyright: ignore[reportUnknownVariableType]\n"
            "y = 1  # type: ignore\n"
            's = "# noqa inside a string"\n'
            "def g():  # pragma: no cover\n    run()  # nosec B603\n"
        )
        assert quality.count_escape_hatches(source) == {
            "noqa": 1,
            "pyright: ignore": 1,
            "type: ignore": 1,
            "pragma: no cover": 1,
            "nosec": 1,
        }

    def test_new_file_has_no_budget(self) -> None:
        (problem,) = quality.escape_hatch_violations({"noqa": {"a.py": 1}}, {})
        assert "budget is 0" in problem

    def test_budget_may_be_met_but_not_exceeded(self) -> None:
        budget = {"noqa": {"a.py": 2}}
        assert quality.escape_hatch_violations({"noqa": {"a.py": 2}}, budget) == []
        assert quality.escape_hatch_violations({"noqa": {"a.py": 1}}, budget) == []
        assert quality.escape_hatch_violations({"noqa": {"a.py": 3}}, budget)

    def test_update_lowers_budgets_but_refuses_to_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        baseline = tmp_path / "escape_hatches.json"
        monkeypatch.setattr(quality, "ESCAPE_HATCH_BASELINE", baseline)
        monkeypatch.setattr(quality, "ROOT", tmp_path)
        counts = {"noqa": {"a.py": 1}}
        monkeypatch.setattr(quality, "escape_hatch_counts", lambda: counts)

        baseline.write_text(json.dumps({"noqa": {"a.py": 3, "gone.py": 1}}))
        assert runner.invoke(quality.app, ["escape-hatches", "--update"]).exit_code == 0
        assert json.loads(baseline.read_text()) == counts

        counts = {"noqa": {"a.py": 2}}
        result = runner.invoke(quality.app, ["escape-hatches", "--update"])
        assert result.exit_code == 1
        assert json.loads(baseline.read_text()) == {"noqa": {"a.py": 1}}

    def test_the_repository_is_within_budget(self) -> None:
        assert runner.invoke(quality.app, ["escape-hatches"]).exit_code == 0


class TestModuleTests:
    """Every module must be named by some test; the grandfathered list only shrinks."""

    def test_imports_and_dotted_strings_are_references(self) -> None:
        source = (
            "import scripts.a\n"
            "from scripts import b\n"
            "from tiny_audio.c import thing\n"
            'monkeypatch.setattr("scripts.d.subprocess.call", f)\n'
        )
        refs = quality.referenced_names(source)
        modules = ["scripts.a", "scripts.b", "tiny_audio.c", "scripts.d", "scripts.e"]
        assert quality.untested_modules(modules, refs) == ["scripts.e"]

    def test_a_prefix_is_not_a_reference(self) -> None:
        assert quality.untested_modules(["scripts.ab"], {"scripts.a"}) == ["scripts.ab"]

    def _run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, allowed: list[str]
    ) -> tuple[int, Path]:
        """Run `module-tests` over a two-module tree where only `scripts.a` is tested."""
        baseline = tmp_path / "untested.json"
        baseline.write_text(json.dumps(allowed))
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_a.py").write_text("from scripts import a\n")
        monkeypatch.setattr(quality, "UNTESTED_MODULES_BASELINE", baseline)
        monkeypatch.setattr(quality, "ROOT", tmp_path)
        monkeypatch.setattr(quality, "_modules_in_scope", lambda: ["scripts.a", "scripts.b"])
        return runner.invoke(quality.app, ["module-tests"]).exit_code, baseline

    def test_new_untested_module_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert self._run(monkeypatch, tmp_path, [])[0] == 1

    def test_grandfathered_module_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert self._run(monkeypatch, tmp_path, ["scripts.b"])[0] == 0

    def test_a_newly_tested_module_must_leave_the_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        code, baseline = self._run(monkeypatch, tmp_path, ["scripts.a", "scripts.b"])
        assert code == 1
        assert runner.invoke(quality.app, ["module-tests", "--update"]).exit_code == 0
        assert json.loads(baseline.read_text()) == ["scripts.b"]

    def test_the_repository_passes(self) -> None:
        assert runner.invoke(quality.app, ["module-tests"]).exit_code == 0


class TestTypeCompleteness:
    """`pyright --verifytypes` scores per package, held to a floor that only rises."""

    @staticmethod
    def fake_pyright(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
        """Make every subprocess call return `stdout` as pyright's output."""

        def run(*_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(stdout=stdout, stderr="")

        monkeypatch.setattr(quality.subprocess, "run", run)

    @staticmethod
    def report(score: float) -> str:
        """A `--verifytypes --outputjson` report with the given completeness score."""
        return json.dumps({"typeCompleteness": {"completenessScore": score}})

    def test_floor_is_enforced(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        baseline = tmp_path / "type_completeness.json"
        baseline.write_text(json.dumps(dict.fromkeys(quality.TYPED_PACKAGES, 80.0)))
        monkeypatch.setattr(quality, "TYPE_COMPLETENESS_BASELINE", baseline)
        self.fake_pyright(monkeypatch, self.report(0.81))
        assert runner.invoke(quality.app, ["type-completeness"]).exit_code == 0
        self.fake_pyright(monkeypatch, self.report(0.79))
        assert runner.invoke(quality.app, ["type-completeness"]).exit_code == 1

    def test_update_records_a_floor_the_same_run_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        baseline = tmp_path / "type_completeness.json"
        monkeypatch.setattr(quality, "TYPE_COMPLETENESS_BASELINE", baseline)
        monkeypatch.setattr(quality, "ROOT", tmp_path)
        self.fake_pyright(monkeypatch, self.report(0.83279))
        assert runner.invoke(quality.app, ["type-completeness", "--update"]).exit_code == 0
        assert json.loads(baseline.read_text()) == dict.fromkeys(quality.TYPED_PACKAGES, 83.2)
        assert runner.invoke(quality.app, ["type-completeness"]).exit_code == 0

    def test_no_score_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fake_pyright(monkeypatch, "not json")
        result = runner.invoke(quality.app, ["type-completeness"])
        assert result.exit_code == 1
        assert "no score" in result.output


class TestTrackedIgnored:
    """A tracked file matched by a .gitignore rule fails the gate."""

    def test_a_tracked_ignored_file_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def run(*_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(stdout="results.json\n")

        monkeypatch.setattr(quality.subprocess, "run", run)
        result = runner.invoke(quality.app, ["tracked-ignored"])
        assert result.exit_code == 1
        assert "results.json" in result.output

    def test_the_repository_passes(self) -> None:
        assert runner.invoke(quality.app, ["tracked-ignored"]).exit_code == 0
