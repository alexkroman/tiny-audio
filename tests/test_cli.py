"""Tests for the unified CLI (tiny-audio / ta)."""

from typer.testing import CliRunner

from scripts.cli import app

runner = CliRunner()


class TestMainCLI:
    """Tests for the main CLI entry point."""

    def test_help_shows_all_commands(self):
        """Test that --help shows all registered commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "eval" in result.output
        assert "analysis" in result.output
        assert "deploy" in result.output
        assert "hub" in result.output
        assert "debug" in result.output
        assert "demo" in result.output
        assert "dev" in result.output

    def test_no_args_shows_help(self):
        """Test that running without args shows help."""
        result = runner.invoke(app, [])
        # Typer with no_args_is_help=True returns exit code 0 or 2
        assert result.exit_code in (0, 2)
        assert "Usage:" in result.output

    def test_invalid_command(self):
        """Test that invalid command shows error."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0


class TestEvalCommand:
    """Tests for the eval subcommand."""

    def test_eval_help(self):
        """Test eval --help."""
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()
        assert "datasets" in result.output.lower() or "dataset" in result.output.lower()

    def test_eval_no_args_shows_help(self):
        """Test that eval without args shows usage."""
        result = runner.invoke(app, ["eval"])
        # Should show usage/help when missing required arg
        assert "Usage:" in result.output or "Missing argument" in result.output


class TestAnalysisCommand:
    """Tests for the analysis subcommand."""

    def test_analysis_help(self):
        """Test analysis --help."""
        result = runner.invoke(app, ["analysis", "--help"])
        assert result.exit_code == 0
        assert "high-wer" in result.output
        assert "entity-errors" in result.output
        assert "extract-entities" in result.output
        assert "compare" in result.output

    def test_analysis_high_wer_help(self):
        """Test analysis high-wer --help."""
        result = runner.invoke(app, ["analysis", "high-wer", "--help"])
        assert result.exit_code == 0
        assert "threshold" in result.output.lower()

    def test_analysis_compare_help(self):
        """Test analysis compare --help."""
        result = runner.invoke(app, ["analysis", "compare", "--help"])
        assert result.exit_code == 0
        assert "models" in result.output.lower()


class TestDeployCommand:
    """Tests for the deploy subcommand."""

    def test_deploy_help(self):
        """Test deploy --help."""
        result = runner.invoke(app, ["deploy", "--help"])
        assert result.exit_code == 0
        assert "hf" in result.output
        assert "handler" in result.output
        assert "runpod" in result.output

    def test_deploy_hf_help(self):
        """Test deploy hf --help."""
        result = runner.invoke(app, ["deploy", "hf", "--help"])
        assert result.exit_code == 0
        assert "repo-id" in result.output.lower()

    def test_deploy_handler_help(self):
        """Test deploy handler --help."""
        result = runner.invoke(app, ["deploy", "handler", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()

    def test_deploy_runpod_help(self):
        """Test deploy runpod --help."""
        result = runner.invoke(app, ["deploy", "runpod", "--help"])
        assert result.exit_code == 0
        assert "deploy" in result.output
        assert "train" in result.output
        assert "attach" in result.output
        assert "checkpoint" in result.output

    def test_deploy_runpod_train_help(self):
        """Test deploy runpod train --help."""
        result = runner.invoke(app, ["deploy", "runpod", "train", "--help"])
        assert result.exit_code == 0
        assert "host" in result.output.lower()
        assert "port" in result.output.lower()
        assert "experiment" in result.output.lower()


class TestHubCommand:
    """Tests for the hub subcommand."""

    def test_hub_help(self):
        """Test hub --help."""
        result = runner.invoke(app, ["hub", "--help"])
        assert result.exit_code == 0
        assert "push" in result.output

    def test_hub_push_help(self):
        """Test hub push --help."""
        result = runner.invoke(app, ["hub", "push", "--help"])
        assert result.exit_code == 0
        assert "repo-id" in result.output.lower()


class TestDebugCommand:
    """Tests for the debug subcommand."""

    def test_debug_help(self):
        """Test debug --help."""
        result = runner.invoke(app, ["debug", "--help"])
        assert result.exit_code == 0
        assert "check-mosa" in result.output
        assert "analyze-lora" in result.output

    def test_debug_check_mosa_help(self):
        """Test debug check-mosa --help."""
        result = runner.invoke(app, ["debug", "check-mosa", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()

    def test_debug_analyze_lora_help(self):
        """Test debug analyze-lora --help."""
        result = runner.invoke(app, ["debug", "analyze-lora", "--help"])
        assert result.exit_code == 0
        assert "repo" in result.output.lower()


class TestDemoCommand:
    """Tests for the demo subcommand."""

    def test_demo_help(self):
        """Test demo --help."""
        result = runner.invoke(app, ["demo", "--help"])
        assert result.exit_code == 0
        assert "main" in result.output  # Demo has 'main' subcommand

    def test_demo_main_help(self):
        """Test demo main --help."""
        result = runner.invoke(app, ["demo", "main", "--help"])
        assert result.exit_code == 0
        assert "port" in result.output.lower()
        assert "model" in result.output.lower()


class TestDevCommand:
    """Tests for the dev subcommand."""

    def test_dev_help(self):
        """Test dev --help."""
        result = runner.invoke(app, ["dev", "--help"])
        assert result.exit_code == 0
        # Check that all dev commands are listed
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

    def test_dev_lint_help(self):
        """Test dev lint --help."""
        result = runner.invoke(app, ["dev", "lint", "--help"])
        assert result.exit_code == 0
        assert "linter" in result.output.lower()

    def test_dev_format_help(self):
        """Test dev format --help."""
        result = runner.invoke(app, ["dev", "format", "--help"])
        assert result.exit_code == 0
        assert "format" in result.output.lower()

    def test_dev_test_help(self):
        """Test dev test --help."""
        result = runner.invoke(app, ["dev", "test", "--help"])
        assert result.exit_code == 0
        assert "pytest" in result.output.lower() or "test" in result.output.lower()

    def test_dev_build_help(self):
        """Test dev build --help."""
        result = runner.invoke(app, ["dev", "build", "--help"])
        assert result.exit_code == 0
        assert "build" in result.output.lower()


class TestCLIStructure:
    """Tests for CLI structure and organization."""

    def test_all_subcommands_have_help(self):
        """Test that all top-level subcommands have help text."""
        subcommands = ["eval", "analysis", "deploy", "hub", "debug", "demo", "dev"]
        for cmd in subcommands:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0, f"'{cmd} --help' failed with: {result.output}"
            # Should have some description
            assert len(result.output) > 50, f"'{cmd}' help output seems too short"

    def test_nested_commands_accessible(self):
        """Test that nested commands are accessible."""
        nested_commands = [
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
            ["demo", "main"],
        ]
        for cmd_path in nested_commands:
            result = runner.invoke(app, cmd_path + ["--help"])
            assert result.exit_code == 0, f"'{' '.join(cmd_path)} --help' failed: {result.output}"
