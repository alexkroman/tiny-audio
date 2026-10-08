"""Tests for the unified CLI (tiny-audio / ta).

This file uses parametrized tests to reduce duplication while maintaining
comprehensive coverage of all CLI commands.
"""

import re

import pytest
from typer.testing import CliRunner

from scripts.cli import app

runner = CliRunner()

# Typer/Rich injects ANSI color codes into help output (e.g. "--model" renders
# as `\x1b[36m-\x1b[0m\x1b[36m-model\x1b[0m`), which breaks substring matching
# on CI Linux runners even though it works on macOS where TTY detection
# differs. Strip codes before asserting.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestMainCLI:
    """Tests for the main CLI entry point."""

    def test_help_shows_all_commands(self) -> None:
        """Test that --help shows all registered commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = _clean(result.output)
        expected_commands = ["train", "eval", "deploy", "push", "runpod", "debug", "demo", "dev"]
        for cmd in expected_commands:
            assert cmd in output, f"Expected '{cmd}' in help output"

    def test_no_args_shows_help(self) -> None:
        """Test that running without args shows help."""
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "Usage:" in _clean(result.output)

    def test_invalid_command(self) -> None:
        """Test that invalid command shows error."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0


class TestSubcommandHelp:
    """Parametrized tests for subcommand --help."""

    @pytest.mark.parametrize(
        ("cmd", "expected_keywords"),
        [
            (["eval"], ["--model", "-m"]),
            (["deploy"], ["--repo-id", "-r"]),
            (["push"], ["--repo-id", "-r"]),
            (["runpod"], ["deploy", "train", "attach"]),
            (["debug"], ["check-gradient-flow"]),
            (["demo"], ["--model", "-m", "--port", "-p"]),
            (["dev"], ["lint", "format", "test", "handler"]),
        ],
    )
    def test_subcommand_help(self, cmd: list[str], expected_keywords: list[str]) -> None:
        """Test that subcommand --help works and shows expected keywords."""
        result = runner.invoke(app, [*cmd, "--help"])
        assert result.exit_code == 0, f"'{' '.join(cmd)} --help' failed: {result.output}"
        output = _clean(result.output)
        for keyword in expected_keywords:
            assert keyword in output, f"Expected '{keyword}' in {cmd} help"


class TestNestedCommands:
    """Parametrized tests for nested command accessibility."""

    @pytest.mark.parametrize(
        ("cmd_path", "expected_keyword"),
        [
            # Runpod subcommands (now top-level)
            (["runpod", "deploy"], "host"),
            (["runpod", "train"], "host"),
            (["runpod", "attach"], "host"),
            (["runpod", "checkpoint"], "host"),
            # Debug subcommands
            (["debug", "check-gradient-flow"], "model"),
            # Dev subcommands
            (["dev", "lint"], "linter"),
            (["dev", "format"], "format"),
            (["dev", "test"], "test"),
            (["dev", "build"], "build"),
            (["dev", "handler"], "model"),
        ],
    )
    def test_nested_command_help(self, cmd_path: list[str], expected_keyword: str) -> None:
        """Test that nested commands are accessible and show expected options."""
        result = runner.invoke(app, [*cmd_path, "--help"])
        assert result.exit_code == 0, f"'{' '.join(cmd_path)} --help' failed: {result.output}"
        err = f"Expected '{expected_keyword}' in {cmd_path} help"
        assert expected_keyword.lower() in _clean(result.output).lower(), err


class TestDevCommands:
    """Tests specific to dev command structure."""

    def test_dev_has_all_expected_commands(self) -> None:
        """Test that dev --help lists all expected subcommands."""
        result = runner.invoke(app, ["dev", "--help"])
        assert result.exit_code == 0
        output = _clean(result.output)
        expected_commands = [
            "lint",
            "format",
            "type-check",
            "test",
            "coverage",
            "check",
            "build",
            "precommit",
            "install-hooks",
            "security",
            "dead-code",
            "docstrings",
            "handler",
        ]
        for cmd in expected_commands:
            assert cmd in output, f"Expected '{cmd}' in dev --help output"


class TestEvalCommand:
    """Tests specific to eval command behavior."""

    def test_eval_no_args_shows_error(self) -> None:
        """`--model` is required, so Click reports it before anything runs."""
        result = runner.invoke(app, ["eval"])
        assert result.exit_code == 2
        output = _clean(result.output)
        assert "--model" in output or "-m" in output

    def test_eval_rejects_unknown_dataset(self) -> None:
        """Dataset names are a Click choice built from the registry."""
        result = runner.invoke(app, ["eval", "-m", "x", "-d", "not-a-dataset"])
        assert result.exit_code == 2
        assert "not-a-dataset" in _clean(result.output)

    def test_subcommands_do_not_install_completion(self) -> None:
        """Only the root app owns shell completion."""
        for cmd in (["eval"], ["dev"], ["runpod", "train"]):
            output = _clean(runner.invoke(app, [*cmd, "--help"]).output)
            assert "--install-completion" not in output, cmd
