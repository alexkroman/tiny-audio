"""Tests for `ta dev`: command wiring plus guards on the quality ratchets.

The ratchet tests pin the *minimum* each threshold may take. Raising a floor
is a one-line edit here too; lowering one fails CI, which is the point.
"""

import importlib
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

from scripts import cli as cli_module
from scripts import dev
from scripts.utils import get_project_root

runner = CliRunner()


@pytest.fixture
def recorded_runs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Replace `dev.run` with a recorder that returns exit code 0."""
    calls: list[tuple[str, ...]] = []

    def fake_run(*args: str) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(dev, "run", fake_run)
    return calls


@pytest.fixture
def pyproject() -> dict[str, Any]:
    return tomllib.loads((get_project_root() / "pyproject.toml").read_text())


class TestRunHelpers:
    """`run` and `run_all` semantics."""

    def test_run_returns_subprocess_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, tuple[str, ...]] = {}

        def fake_call(args: tuple[str, ...]) -> int:
            seen["args"] = args
            return 7

        monkeypatch.setattr("scripts.dev.subprocess.call", fake_call)
        assert dev.run("echo", "hi") == 7
        assert seen["args"] == ("echo", "hi")

    def test_run_all_stops_at_first_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        codes = iter([0, 3, 0])
        executed: list[tuple[str, ...]] = []

        def fake_run(*args: str) -> int:
            executed.append(args)
            return next(codes)

        monkeypatch.setattr(dev, "run", fake_run)
        assert dev.run_all(["a"], ["b"], ["c"]) == 3
        assert executed == [("a",), ("b",)]

    def test_run_all_succeeds_when_everything_passes(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert dev.run_all(["a"], ["b"]) == 0
        assert recorded_runs == [("a",), ("b",)]


class TestCommandWiring:
    """Each subcommand runs exactly the command list it advertises."""

    def test_check_runs_every_check_in_order(self, recorded_runs: list[tuple[str, ...]]) -> None:
        result = runner.invoke(dev.app, ["check"])
        assert result.exit_code == 0
        assert recorded_runs == [tuple(cmd) for cmd in dev.check_commands()]

    def test_lint_runs_lint_commands(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert runner.invoke(dev.app, ["lint"]).exit_code == 0
        assert recorded_runs == [tuple(cmd) for cmd in dev.lint_commands()]

    def test_type_check_runs_both_checkers(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert runner.invoke(dev.app, ["type-check"]).exit_code == 0
        assert [c[0] for c in recorded_runs] == ["mypy", "pyright"]

    def test_test_enforces_coverage(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert runner.invoke(dev.app, ["test"]).exit_code == 0
        cmd, floors = recorded_runs
        assert cmd[0] == "pytest"
        assert "--cov=tiny_audio" in cmd
        assert "--cov=scripts" in cmd
        assert "--cov-report=xml" in cmd
        # The per-file floors read the JSON report the test run just wrote.
        assert "--cov-report=json" in cmd
        assert floors == tuple(dev.COVERAGE_FLOORS_COMMAND)

    def test_coverage_adds_html_report(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert runner.invoke(dev.app, ["coverage"]).exit_code == 0
        (cmd,) = recorded_runs
        assert cmd[: len(dev.TEST_COMMAND)] == tuple(dev.TEST_COMMAND)
        assert cmd[-1] == "--cov-report=html"

    def test_precommit_formats_then_checks_tests_and_builds(self, recorded_runs: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dev, "format_code", lambda: recorded_runs.append(("<format>",)))
        assert runner.invoke(dev.app, ["precommit"]).exit_code == 0
        assert recorded_runs[0] == ("<format>",)
        assert recorded_runs[1:] == [
            *(tuple(cmd) for cmd in dev.check_commands()),
            tuple(dev.TEST_COMMAND),
            tuple(dev.COVERAGE_FLOORS_COMMAND),
            tuple(dev.BUILD_COMMAND),
            *(tuple(cmd) for cmd in dev.dist_check_commands()),
        ]

    def test_build_validates_what_it_built(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert runner.invoke(dev.app, ["build"]).exit_code == 0
        assert recorded_runs == [
            tuple(dev.BUILD_COMMAND),
            *(tuple(cmd) for cmd in dev.dist_check_commands()),
        ]

    def test_failed_build_skips_the_artifact_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, ...]] = []

        def failing_run(*args: str) -> int:
            calls.append(args)
            return 1

        monkeypatch.setattr(dev, "run", failing_run)
        assert runner.invoke(dev.app, ["build"]).exit_code == 1
        assert calls == [tuple(dev.BUILD_COMMAND)]

    def test_docstrings_is_verbose(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert runner.invoke(dev.app, ["docstrings"]).exit_code == 0
        assert all(cmd[0] == "interrogate" and cmd[-1] == "-v" for cmd in recorded_runs)
        assert len(recorded_runs) == len(dev.DOCSTRINGS_COMMANDS)

    def test_dead_code_uses_the_shared_command(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert runner.invoke(dev.app, ["dead-code"]).exit_code == 0
        assert recorded_runs == [tuple(dev.DEAD_CODE_COMMAND)]

    def test_duplication_uses_the_shared_command(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert runner.invoke(dev.app, ["duplication"]).exit_code == 0
        assert recorded_runs == [tuple(dev.DUPLICATION_COMMAND)]

    def test_deps_uses_the_shared_command(self, recorded_runs: list[tuple[str, ...]]) -> None:
        assert runner.invoke(dev.app, ["deps"]).exit_code == 0
        assert recorded_runs == [tuple(dev.DEPS_COMMAND)]

    def test_failure_exit_code_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_run(*args: str) -> int:
            return 5

        monkeypatch.setattr(dev, "run", failing_run)
        assert runner.invoke(dev.app, ["lint"]).exit_code == 5


class TestFormatCode:
    """`ta dev format` rewrites every tracked Markdown and JSON file."""

    LISTINGS: ClassVar[dict[str, str]] = {
        "*.md": "README.md\nMODEL_CARD.md\ndemo/README.md\n",
        "*.json": "quality/file_length.json\n",
    }

    def _fake_ls_files(self, cmd: list[str], **_kw: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=self.LISTINGS[cmd[-1]])

    def test_every_tracked_markdown_and_json_file_is_formatted(self, recorded_runs: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.dev.subprocess.run", self._fake_ls_files)
        dev.format_code()
        assert ("mdformat", "README.md", "MODEL_CARD.md", "demo/README.md") in recorded_runs
        assert ("pretty-format-json", "--autofix", "quality/file_length.json") in recorded_runs
        assert ("taplo", "fmt", *dev.TOML_FILES) in recorded_runs
        assert [c[0] for c in recorded_runs[:3]] == ["black", "ruff", "ruff"]

    def test_no_markdown_means_no_mdformat_call(self, recorded_runs: list[tuple[str, ...]], monkeypatch: pytest.MonkeyPatch) -> None:
        def empty_ls_files(*_a: object, **_kw: object) -> SimpleNamespace:
            return SimpleNamespace(stdout="")

        monkeypatch.setattr("scripts.dev.subprocess.run", empty_ls_files)
        dev.format_code()
        assert all(c[0] not in {"mdformat", "pretty-format-json"} for c in recorded_runs)


class TestQualityGateContents:
    """The gate must keep enforcing formatting, dead code and docstrings."""

    def test_lint_refuses_unformatted_code(self) -> None:
        assert ["ruff", "format", "--check", *dev.CODE_PATHS] in dev.LINT_COMMANDS
        assert ["black", "--check", *dev.CODE_PATHS] in dev.LINT_COMMANDS

    def test_lint_checks_all_tracked_markdown(self) -> None:
        commands = {cmd[0]: cmd for cmd in dev.lint_commands()}
        assert commands["mdformat"][:2] == ["mdformat", "--check"]
        assert {"README.md", "MODEL_CARD.md", "demo/README.md"} <= set(commands["mdformat"])
        assert commands["pymarkdown"][2:] == commands["mdformat"][2:]

    def test_lint_checks_json_and_toml_layout(self) -> None:
        commands = dev.lint_commands()
        json_check = next(cmd for cmd in commands if cmd[0] == "pretty-format-json")
        assert "--autofix" not in json_check
        assert "quality/file_length.json" in json_check
        assert ["taplo", "fmt", "--check", *dev.TOML_FILES] in commands

    def test_lint_verifies_the_lock_file(self) -> None:
        assert ["poetry", "check", "--lock"] in dev.LINT_COMMANDS

    def test_check_includes_static_analysis_gates(self) -> None:
        assert dev.DEAD_CODE_COMMAND in dev.check_commands()
        assert dev.DEPS_COMMAND in dev.check_commands()
        assert dev.DUPLICATION_COMMAND in dev.check_commands()
        assert dev.FILE_LENGTH_COMMAND in dev.check_commands()
        assert dev.TEST_ASSERTIONS_COMMAND in dev.check_commands()
        for cmd in dev.DOCSTRINGS_COMMANDS:
            assert cmd in dev.check_commands()

    def test_check_covers_both_packages_with_interrogate(self) -> None:
        targets = {cmd[1] for cmd in dev.DOCSTRINGS_COMMANDS}
        assert targets == {"tiny_audio", "scripts"}


class TestRatchetFloors:
    """Thresholds may go up, never down."""

    def test_docstring_floors(self) -> None:
        assert int(dev.DOCSTRING_MIN_LIB) >= 95
        assert int(dev.DOCSTRING_MIN_SCRIPTS) >= 65

    def test_dead_code_confidence_floor(self) -> None:
        assert int(dev.DEAD_CODE_MIN_CONFIDENCE) <= 80

    def test_coverage_floor(self, pyproject: dict[str, Any]) -> None:
        assert pyproject["tool"]["coverage"]["report"]["fail_under"] >= 45
        assert pyproject["tool"]["coverage"]["run"]["branch"] is True

    def test_pytest_is_strict(self, pyproject: dict[str, Any]) -> None:
        opts = pyproject["tool"]["pytest"]["ini_options"]
        assert "--strict-markers" in opts["addopts"]
        assert "--strict-config" in opts["addopts"]
        assert opts["xfail_strict"] is True

    def test_python_targets_agree(self, pyproject: dict[str, Any]) -> None:
        requires = pyproject["project"]["requires-python"]
        assert requires.startswith(">=3.12")
        assert pyproject["tool"]["ruff"]["target-version"] == "py312"
        assert pyproject["tool"]["black"]["target-version"] == ["py312"]

    def test_ruff_rule_families_are_not_dropped(self, pyproject: dict[str, Any]) -> None:
        selected = set(pyproject["tool"]["ruff"]["lint"]["select"])
        assert {"E", "F", "I", "B", "UP", "SIM", "PT", "RUF", "PLW", "PLE", "TRY"} <= selected


class TestPrecommitHookIsTheGate:
    """The git hook must run the same gate as CI."""

    def test_hook_calls_ta_dev_precommit(self) -> None:
        hook = (get_project_root() / ".pre-commit-config.yaml").read_text()
        assert "ta dev precommit" in hook

    def test_ci_runs_check_and_test(self) -> None:
        ci = (get_project_root() / ".github" / "workflows" / "ci.yml").read_text()
        assert "ta dev check" in ci
        assert "ta dev test" in ci


class TestLazyRegistration:
    """`ta <group>` imports only that group's module."""

    @staticmethod
    def _fresh_cli() -> None:
        """Reload scripts.cli so its LazyGroup starts with an empty cache."""
        importlib.reload(cli_module)  # re-executes the module in place

    def test_root_lists_every_subcommand_without_importing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fresh_cli()
        for module in cli_module.SUBCOMMANDS.values():
            monkeypatch.delitem(sys.modules, module, raising=False)
        group = typer.main.get_command(cli_module.app)
        assert isinstance(group, cli_module.LazyGroup)
        ctx = typer.Context(group)
        assert group.list_commands(ctx) == list(cli_module.SUBCOMMANDS)
        assert not any(m in sys.modules for m in cli_module.SUBCOMMANDS.values())

    def test_resolving_one_command_imports_only_its_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fresh_cli()
        for module in cli_module.SUBCOMMANDS.values():
            monkeypatch.delitem(sys.modules, module, raising=False)
        group = typer.main.get_command(cli_module.app)
        assert isinstance(group, cli_module.LazyGroup)
        ctx = typer.Context(group)
        command = group.get_command(ctx, "dev")
        assert command is not None
        assert command.name == "dev"
        assert "scripts.dev" in sys.modules
        assert "demo.app" not in sys.modules
        assert "scripts.eval.cli" not in sys.modules

    def test_resolved_commands_are_cached(self) -> None:
        self._fresh_cli()
        group = typer.main.get_command(cli_module.app)
        assert isinstance(group, cli_module.LazyGroup)
        ctx = typer.Context(group)
        assert group.get_command(ctx, "dev") is group.get_command(ctx, "dev")

    def test_unknown_command_resolves_to_none(self) -> None:
        self._fresh_cli()
        group = typer.main.get_command(cli_module.app)
        assert isinstance(group, cli_module.LazyGroup)
        ctx = typer.Context(group)
        assert group.get_command(ctx, "nonsense") is None


def test_project_root_has_pyproject() -> None:
    assert (Path(get_project_root()) / "pyproject.toml").is_file()


class TestDistChecks:
    """`dist_check_commands` points the checkers at exactly what is in dist/."""

    def test_checks_every_artifact_and_the_wheel(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        for name in ("pkg-0.1-py3-none-any.whl", "pkg-0.1.tar.gz"):
            (tmp_path / name).touch()
        monkeypatch.setattr(dev, "DIST_DIR", tmp_path)
        wheel_check, twine_check = dev.dist_check_commands()
        assert wheel_check[0] == "check-wheel-contents"
        assert wheel_check[-1] == str(tmp_path / "pkg-0.1-py3-none-any.whl")
        assert twine_check[:3] == ["twine", "check", "--strict"]
        assert twine_check[3:] == sorted(str(p) for p in tmp_path.iterdir())

    def test_missing_dist_dir_yields_no_artifacts(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(dev, "DIST_DIR", tmp_path / "missing")
        wheel_check, twine_check = dev.dist_check_commands()
        assert not any(arg.endswith(".whl") for arg in wheel_check)
        assert twine_check == ["twine", "check", "--strict"]
