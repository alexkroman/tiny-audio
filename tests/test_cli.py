"""Tests for the unified CLI (tiny-audio / ta).

This file uses parametrized tests to reduce duplication while maintaining
comprehensive coverage of all CLI commands.
"""

import pytest
from typer.testing import CliRunner

from scripts.cli import app

runner = CliRunner()


class TestMainCLI:
    """Tests for the main CLI entry point."""

    def test_help_shows_all_commands(self):
        """Test that --help shows all registered commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        expected_commands = ["eval", "analysis", "deploy", "hub", "debug", "demo", "dev"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Expected '{cmd}' in help output"

    def test_no_args_shows_help(self):
        """Test that running without args shows help."""
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "Usage:" in result.output

    def test_invalid_command(self):
        """Test that invalid command shows error."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0


class TestSubcommandHelp:
    """Parametrized tests for subcommand --help."""

    @pytest.mark.parametrize(
        "cmd,expected_keywords",
        [
            (["eval"], ["model"]),
            (["analysis"], ["high-wer", "compare"]),
            (["deploy"], ["hf", "handler", "runpod"]),
            (["hub"], ["push"]),
            (["debug"], ["check-mosa", "analyze-lora"]),
            (["demo"], ["model", "port"]),
            (["dev"], ["lint", "format", "test"]),
        ],
    )
    def test_subcommand_help(self, cmd, expected_keywords):
        """Test that subcommand --help works and shows expected keywords."""
        result = runner.invoke(app, cmd + ["--help"])
        assert result.exit_code == 0, f"'{' '.join(cmd)} --help' failed: {result.output}"
        for keyword in expected_keywords:
            assert keyword in result.output, f"Expected '{keyword}' in {cmd} help"


class TestNestedCommands:
    """Parametrized tests for nested command accessibility."""

    @pytest.mark.parametrize(
        "cmd_path,expected_keyword",
        [
            # Deploy subcommands
            (["deploy", "hf"], "repo-id"),
            (["deploy", "handler"], "model"),
            (["deploy", "runpod"], "deploy"),
            (["deploy", "runpod", "train"], "host"),
            # Hub subcommands
            (["hub", "push"], "repo-id"),
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
        ],
    )
    def test_nested_command_help(self, cmd_path, expected_keyword):
        """Test that nested commands are accessible and show expected options."""
        result = runner.invoke(app, cmd_path + ["--help"])
        assert result.exit_code == 0, f"'{' '.join(cmd_path)} --help' failed: {result.output}"
        assert expected_keyword.lower() in result.output.lower(), (
            f"Expected '{expected_keyword}' in {cmd_path} help"
        )


class TestDevCommands:
    """Tests specific to dev command structure."""

    def test_dev_has_all_expected_commands(self):
        """Test that dev --help lists all expected subcommands."""
        result = runner.invoke(app, ["dev", "--help"])
        assert result.exit_code == 0
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
        ]
        for cmd in expected_commands:
            assert cmd in result.output, f"Expected '{cmd}' in dev --help output"


class TestEvalCommand:
    """Tests specific to eval command behavior."""

    def test_eval_no_args_shows_usage(self):
        """Test that eval without args shows usage."""
        result = runner.invoke(app, ["eval"])
        assert "Usage:" in result.output or "Missing argument" in result.output


class TestCLIStructure:
    """Tests for overall CLI structure validation."""

    @pytest.mark.parametrize(
        "cmd_path",
        [
            ["deploy", "hf"],
            ["deploy", "handler"],
            ["deploy", "runpod"],
            ["deploy", "runpod", "deploy"],
            ["deploy", "runpod", "train"],
            ["deploy", "runpod", "attach"],
            ["deploy", "runpod", "checkpoint"],
            ["hub", "push"],
            ["debug", "check-mosa"],
            ["debug", "analyze-lora"],
            ["analysis", "high-wer"],
            ["analysis", "compare"],
            ["dev", "lint"],
            ["dev", "test"],
        ],
    )
    def test_nested_commands_accessible(self, cmd_path):
        """Test that all nested commands are accessible."""
        result = runner.invoke(app, cmd_path + ["--help"])
        assert result.exit_code == 0, f"'{' '.join(cmd_path)} --help' failed: {result.output}"
        assert len(result.output) > 50, f"'{' '.join(cmd_path)}' help output seems too short"
