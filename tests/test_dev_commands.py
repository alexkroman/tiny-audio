"""Tests for `ta dev`: command wiring plus guards on the quality ratchets.

The ratchet tests pin the *minimum* each threshold may take. Raising a floor
is a one-line edit here too; lowering one fails CI, which is the point.
"""

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from scripts import dev
from scripts.utils import get_project_root

runner = CliRunner()


@pytest.fixture
def recorded_runs(monkeypatch):
    """Replace `dev.run` with a recorder that returns exit code 0."""
    calls: list[tuple[str, ...]] = []

    def fake_run(*args: str) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(dev, "run", fake_run)
    return calls


@pytest.fixture
def pyproject() -> dict:
    return tomllib.loads((get_project_root() / "pyproject.toml").read_text())


class TestRunHelpers:
    """`run` and `run_all` semantics."""

    def test_run_returns_subprocess_exit_code(self, monkeypatch):
        seen = {}

        def fake_call(args):
            seen["args"] = args
            return 7

        monkeypatch.setattr(dev.subprocess, "call", fake_call)
        assert dev.run("echo", "hi") == 7
        assert seen["args"] == ("echo", "hi")

    def test_run_all_stops_at_first_failure(self, monkeypatch):
        codes = iter([0, 3, 0])
        executed = []

        def fake_run(*args):
            executed.append(args)
            return next(codes)

        monkeypatch.setattr(dev, "run", fake_run)
        assert dev.run_all(["a"], ["b"], ["c"]) == 3
        assert executed == [("a",), ("b",)]

    def test_run_all_succeeds_when_everything_passes(self, recorded_runs):
        assert dev.run_all(["a"], ["b"]) == 0
        assert recorded_runs == [("a",), ("b",)]


class TestCommandWiring:
    """Each subcommand runs exactly the command list it advertises."""

    def test_check_runs_every_check_in_order(self, recorded_runs):
        result = runner.invoke(dev.app, ["check"])
        assert result.exit_code == 0
        assert recorded_runs == [tuple(cmd) for cmd in dev.CHECK_COMMANDS]

    def test_lint_runs_lint_commands(self, recorded_runs):
        assert runner.invoke(dev.app, ["lint"]).exit_code == 0
        assert recorded_runs == [tuple(cmd) for cmd in dev.LINT_COMMANDS]

    def test_type_check_runs_both_checkers(self, recorded_runs):
        assert runner.invoke(dev.app, ["type-check"]).exit_code == 0
        assert [c[0] for c in recorded_runs] == ["mypy", "pyright"]

    def test_test_enforces_coverage(self, recorded_runs):
        assert runner.invoke(dev.app, ["test"]).exit_code == 0
        (cmd,) = recorded_runs
        assert cmd[0] == "pytest"
        assert "--cov=tiny_audio" in cmd
        assert "--cov=scripts" in cmd
        assert "--cov-report=xml" in cmd

    def test_coverage_adds_html_report(self, recorded_runs):
        assert runner.invoke(dev.app, ["coverage"]).exit_code == 0
        (cmd,) = recorded_runs
        assert cmd[: len(dev.TEST_COMMAND)] == tuple(dev.TEST_COMMAND)
        assert cmd[-1] == "--cov-report=html"

    def test_precommit_formats_then_checks_tests_and_builds(self, recorded_runs, monkeypatch):
        monkeypatch.setattr(dev, "format_code", lambda: recorded_runs.append(("<format>",)))
        assert runner.invoke(dev.app, ["precommit"]).exit_code == 0
        assert recorded_runs[0] == ("<format>",)
        assert recorded_runs[1:] == [
            *(tuple(cmd) for cmd in dev.CHECK_COMMANDS),
            tuple(dev.TEST_COMMAND),
            ("poetry", "build"),
        ]

    def test_docstrings_is_verbose(self, recorded_runs):
        assert runner.invoke(dev.app, ["docstrings"]).exit_code == 0
        assert all(cmd[0] == "interrogate" and cmd[-1] == "-v" for cmd in recorded_runs)
        assert len(recorded_runs) == len(dev.DOCSTRINGS_COMMANDS)

    def test_dead_code_uses_the_shared_command(self, recorded_runs):
        assert runner.invoke(dev.app, ["dead-code"]).exit_code == 0
        assert recorded_runs == [tuple(dev.DEAD_CODE_COMMAND)]

    def test_failure_exit_code_propagates(self, monkeypatch):
        monkeypatch.setattr(dev, "run", lambda *args: 5)
        assert runner.invoke(dev.app, ["lint"]).exit_code == 5


class TestFormatCode:
    """Markdown formatting only touches tracked files outside the excluded set."""

    def test_only_tracked_unexcluded_markdown_is_formatted(self, recorded_runs, monkeypatch):
        listing = (
            "README.md\ndocs/course/01.md\nMODEL_CARD.md\ndemo/README.md\ndocs/QUICKSTART.md\n"
        )
        monkeypatch.setattr(dev.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=listing))
        dev.format_code()
        md_calls = [c for c in recorded_runs if c[0] == "mdformat"]
        assert md_calls == [("mdformat", "README.md", "docs/QUICKSTART.md")]
        assert [c[0] for c in recorded_runs[:3]] == ["black", "ruff", "ruff"]

    def test_no_markdown_means_no_mdformat_call(self, recorded_runs, monkeypatch):
        monkeypatch.setattr(dev.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
        dev.format_code()
        assert all(c[0] != "mdformat" for c in recorded_runs)


class TestQualityGateContents:
    """The gate must keep enforcing formatting, dead code and docstrings."""

    def test_lint_refuses_unformatted_code(self):
        assert ["ruff", "format", "--check", *dev.CODE_PATHS] in dev.LINT_COMMANDS
        assert ["black", "--check", *dev.CODE_PATHS] in dev.LINT_COMMANDS

    def test_lint_verifies_the_lock_file(self):
        assert ["poetry", "check", "--lock"] in dev.LINT_COMMANDS

    def test_check_includes_dead_code_and_docstrings(self):
        assert dev.DEAD_CODE_COMMAND in dev.CHECK_COMMANDS
        for cmd in dev.DOCSTRINGS_COMMANDS:
            assert cmd in dev.CHECK_COMMANDS

    def test_check_covers_both_packages_with_interrogate(self):
        targets = {cmd[1] for cmd in dev.DOCSTRINGS_COMMANDS}
        assert targets == {"tiny_audio", "scripts"}


class TestRatchetFloors:
    """Thresholds may go up, never down."""

    def test_docstring_floors(self):
        assert int(dev.DOCSTRING_MIN_LIB) >= 95
        assert int(dev.DOCSTRING_MIN_SCRIPTS) >= 65

    def test_dead_code_confidence_floor(self):
        assert int(dev.DEAD_CODE_MIN_CONFIDENCE) <= 80

    def test_coverage_floor(self, pyproject):
        assert pyproject["tool"]["coverage"]["report"]["fail_under"] >= 45
        assert pyproject["tool"]["coverage"]["run"]["branch"] is True

    def test_pytest_is_strict(self, pyproject):
        opts = pyproject["tool"]["pytest"]["ini_options"]
        assert "--strict-markers" in opts["addopts"]
        assert "--strict-config" in opts["addopts"]
        assert opts["xfail_strict"] is True

    def test_python_targets_agree(self, pyproject):
        requires = pyproject["project"]["requires-python"]
        assert requires.startswith(">=3.12")
        assert pyproject["tool"]["ruff"]["target-version"] == "py312"
        assert pyproject["tool"]["black"]["target-version"] == ["py312"]

    def test_ruff_rule_families_are_not_dropped(self, pyproject):
        selected = set(pyproject["tool"]["ruff"]["lint"]["select"])
        assert {"E", "F", "I", "B", "UP", "SIM", "PT", "RUF", "PLW", "PLE", "TRY"} <= selected


class TestPrecommitHookIsTheGate:
    """The git hook must run the same gate as CI."""

    def test_hook_calls_ta_dev_precommit(self):
        hook = (get_project_root() / ".pre-commit-config.yaml").read_text()
        assert "ta dev precommit" in hook

    def test_ci_runs_check_and_test(self):
        ci = (get_project_root() / ".github" / "workflows" / "ci.yml").read_text()
        assert "ta dev check" in ci
        assert "ta dev test" in ci


class TestLazyRegistration:
    """`ta <group>` imports only that group's module."""

    @staticmethod
    def _fresh_cli():
        """Reload scripts.cli so its LazyGroup starts with an empty cache."""
        import importlib

        from scripts import cli

        return importlib.reload(cli)

    def test_root_lists_every_subcommand_without_importing(self, monkeypatch):
        cli = self._fresh_cli()
        for module in cli.SUBCOMMANDS.values():
            monkeypatch.delitem(sys.modules, module, raising=False)
        group = typer.main.get_command(cli.app)
        ctx = typer.Context(group)
        assert group.list_commands(ctx) == list(cli.SUBCOMMANDS)
        assert not any(m in sys.modules for m in cli.SUBCOMMANDS.values())

    def test_resolving_one_command_imports_only_its_module(self, monkeypatch):
        cli = self._fresh_cli()
        for module in cli.SUBCOMMANDS.values():
            monkeypatch.delitem(sys.modules, module, raising=False)
        group = typer.main.get_command(cli.app)
        ctx = typer.Context(group)
        command = group.get_command(ctx, "dev")
        assert command is not None
        assert command.name == "dev"
        assert "scripts.dev" in sys.modules
        assert "demo.app" not in sys.modules
        assert "scripts.eval.cli" not in sys.modules

    def test_resolved_commands_are_cached(self):
        cli = self._fresh_cli()
        group = typer.main.get_command(cli.app)
        ctx = typer.Context(group)
        assert group.get_command(ctx, "dev") is group.get_command(ctx, "dev")

    def test_unknown_command_resolves_to_none(self):
        cli = self._fresh_cli()
        group = typer.main.get_command(cli.app)
        ctx = typer.Context(group)
        assert group.get_command(ctx, "nonsense") is None


def test_project_root_has_pyproject():
    assert (Path(get_project_root()) / "pyproject.toml").is_file()
