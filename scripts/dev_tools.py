#!/usr/bin/env python3
"""Development tools for tiny-audio project."""

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Development tools for tiny-audio project")


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    typer.echo(f"\n{'=' * 60}")
    typer.echo(description)
    typer.echo(f"{'=' * 60}")
    typer.echo(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=get_project_root())
    return result.returncode


def _lint() -> int:
    """Run ruff linter."""
    return run_command(["ruff", "check", "src", "scripts"], "Running Ruff Linter")


def _format_code() -> int:
    """Format code with black, ruff, and mdformat."""
    exit_code = 0

    code = run_command(["black", "src", "scripts"], "Formatting with Black")
    exit_code = max(exit_code, code)

    code = run_command(["ruff", "format", "src", "scripts"], "Formatting with Ruff")
    exit_code = max(exit_code, code)

    code = run_command(["ruff", "check", "--fix", "src", "scripts"], "Fixing with Ruff")
    exit_code = max(exit_code, code)

    # Markdown formatting
    project_root = get_project_root()
    md_files = list(project_root.glob("*.md")) + list(project_root.glob("docs/**/*.md"))
    excluded_files = {"model_card.md", "MODEL_CARD.md"}
    md_files = [f for f in md_files if f.name not in excluded_files]

    if md_files:
        code = run_command(
            ["mdformat"] + [str(f) for f in md_files], "Formatting Markdown Files"
        )
        exit_code = max(exit_code, code)

    return exit_code


def _type_check() -> int:
    """Run mypy and pyright type checkers."""
    mypy_code = run_command(["mypy", "src"], "Running MyPy Type Checker")
    pyright_code = run_command(["pyright", "src"], "Running Pyright Type Checker")
    return max(mypy_code, pyright_code)


def _test() -> int:
    """Run pytest tests."""
    return run_command(["pytest", "-v"], "Running Tests with Pytest")


def _check() -> int:
    """Run all checks: lint and type-check."""
    exit_code = 0

    typer.echo(f"\n{'=' * 60}")
    typer.echo("RUNNING ALL CHECKS")
    typer.echo(f"{'=' * 60}\n")

    exit_code = max(exit_code, _lint())
    exit_code = max(exit_code, _type_check())

    typer.echo(f"\n{'=' * 60}")
    typer.echo("CHECK SUMMARY")
    typer.echo("=" * 60)
    if exit_code == 0:
        typer.echo("✅ All checks passed!")
    else:
        typer.echo("❌ Some checks failed. Please review the output above.")
    typer.echo("=" * 60 + "\n")

    return exit_code


# Typer commands (wrap internal functions)
@app.command()
def lint() -> None:
    """Run ruff linter."""
    raise typer.Exit(_lint())


@app.command("format")
def format_cmd() -> None:
    """Format code with black, ruff, and mdformat."""
    raise typer.Exit(_format_code())


@app.command("type-check")
def type_check() -> None:
    """Run mypy and pyright type checkers."""
    raise typer.Exit(_type_check())


@app.command()
def test() -> None:
    """Run pytest tests."""
    raise typer.Exit(_test())


@app.command()
def check() -> None:
    """Run all checks: lint and type-check."""
    raise typer.Exit(_check())


# Poetry script entry points (must exit with proper code)
def lint_cli() -> None:
    sys.exit(_lint())


def format_cli() -> None:
    sys.exit(_format_code())


def type_check_cli() -> None:
    sys.exit(_type_check())


def test_cli() -> None:
    sys.exit(_test())


def check_cli() -> None:
    sys.exit(_check())


def cli() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    app()
