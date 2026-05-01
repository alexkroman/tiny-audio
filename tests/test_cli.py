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

    def test_help_shows_all_commands(self):
        """Test that --help shows all registered commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = _clean(result.output)
        expected_commands = ["eval", "analysis", "deploy", "push", "runpod", "debug", "demo", "dev"]
        for cmd in expected_commands:
            assert cmd in output, f"Expected '{cmd}' in help output"

    def test_no_args_shows_help(self):
        """Test that running without args shows help."""
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "Usage:" in _clean(result.output)

    def test_invalid_command(self):
        """Test that invalid command shows error."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0


class TestSubcommandHelp:
    """Parametrized tests for subcommand --help."""

    @pytest.mark.parametrize(
        "cmd,expected_keywords",
        [
            (["eval"], ["--model", "-m"]),
            (["analysis"], ["high-wer", "compare"]),
            (["deploy"], ["repo-id"]),  # Now a direct command
            (["push"], ["repo-id"]),  # Now a direct command
            (["runpod"], ["deploy", "train", "attach"]),  # Top-level command
            (["debug"], ["check-mosa", "analyze-lora"]),
            (["demo"], ["model", "port"]),
            (["dev"], ["lint", "format", "test", "handler"]),
        ],
    )
    def test_subcommand_help(self, cmd, expected_keywords):
        """Test that subcommand --help works and shows expected keywords."""
        result = runner.invoke(app, cmd + ["--help"])
        assert result.exit_code == 0, f"'{' '.join(cmd)} --help' failed: {result.output}"
        output = _clean(result.output)
        for keyword in expected_keywords:
            assert keyword in output, f"Expected '{keyword}' in {cmd} help"


class TestNestedCommands:
    """Parametrized tests for nested command accessibility."""

    @pytest.mark.parametrize(
        "cmd_path,expected_keyword",
        [
            # Runpod subcommands (now top-level)
            (["runpod", "deploy"], "host"),
            (["runpod", "train"], "host"),
            (["runpod", "attach"], "host"),
            (["runpod", "checkpoint"], "host"),
            # Debug subcommands
            (["debug", "check-mosa"], "model"),
            (["debug", "analyze-lora"], "repo"),
            # Analysis subcommands
            (["analysis", "high-wer"], "threshold"),
            (["analysis", "compare"], "models"),
            # Dev subcommands
            (["dev", "lint"], "linter"),
            (["dev", "format"], "format"),
            (["dev", "test"], "test"),
            (["dev", "build"], "build"),
            (["dev", "handler"], "model"),
        ],
    )
    def test_nested_command_help(self, cmd_path, expected_keyword):
        """Test that nested commands are accessible and show expected options."""
        result = runner.invoke(app, cmd_path + ["--help"])
        assert result.exit_code == 0, f"'{' '.join(cmd_path)} --help' failed: {result.output}"
        err = f"Expected '{expected_keyword}' in {cmd_path} help"
        assert expected_keyword.lower() in _clean(result.output).lower(), err


class TestDevCommands:
    """Tests specific to dev command structure."""

    def test_dev_has_all_expected_commands(self):
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

    def test_eval_no_args_shows_error(self):
        """Test that eval without model shows helpful error."""
        result = runner.invoke(app, ["eval"])
        output = _clean(result.output)
        assert "--model" in output or "-m" in output


class TestCLIStructure:
    """Tests for overall CLI structure validation."""

    @pytest.mark.parametrize(
        "cmd_path",
        [
            # Top-level commands (now direct)
            ["deploy"],
            ["push"],
            # Runpod subcommands
            ["runpod", "deploy"],
            ["runpod", "train"],
            ["runpod", "attach"],
            ["runpod", "checkpoint"],
            # Debug subcommands
            ["debug", "check-mosa"],
            ["debug", "analyze-lora"],
            # Analysis subcommands
            ["analysis", "high-wer"],
            ["analysis", "compare"],
            # Dev subcommands
            ["dev", "lint"],
            ["dev", "test"],
            ["dev", "handler"],
        ],
    )
    def test_nested_commands_accessible(self, cmd_path):
        """Test that all nested commands are accessible."""
        result = runner.invoke(app, cmd_path + ["--help"])
        assert result.exit_code == 0, f"'{' '.join(cmd_path)} --help' failed: {result.output}"
        assert len(result.output) > 50, f"'{' '.join(cmd_path)}' help output seems too short"
