#!/usr/bin/env python3
"""Development commands for tiny-audio.

Commands:
    lint         Run linters (Poetry + Python + YAML + TOML)
    format       Format code with black, ruff, and mdformat
    type-check   Run type checkers (mypy and pyright)
    test         Run fast pytest tests (skips slow model-loading tests)
    test-slow    Run only slow pytest tests
    test-all     Run all pytest tests
    coverage     Run tests with coverage report
    check        Run all checks (lint + type-check + security + docstrings)
    build        Build package (wheel and sdist)
    precommit    Pre-commit quality gate
    security     Run security checks with bandit
    dead-code    Find dead/unused code with vulture
    docstrings   Check docstring coverage with interrogate
    install-hooks Install pre-commit hooks
    handler      Test inference endpoint handler locally
"""

import subprocess
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="dev",
    help="Development commands (lint, test, format, etc.)",
    no_args_is_help=True,
)
console = Console()


def run(*args: str) -> int:
    """Run a command and return exit code."""
    console.print(f"[dim]$ {' '.join(args)}[/dim]")
    return subprocess.call(args)


def run_all(*commands: list[str]) -> int:
    """Run multiple commands, stopping on first failure."""
    for cmd in commands:
        code = run(*cmd)
        if code != 0:
            return code
    return 0


@app.command()
def lint():
    """Run linters (Poetry + Python + YAML + TOML)."""
    code = run_all(
        ["poetry", "check"],
        ["ruff", "check", "tiny_audio", "scripts", "tests"],
        ["yamllint", "-d", "relaxed", "configs/"],
        ["taplo", "check", "pyproject.toml"],
    )
    raise typer.Exit(code)


@app.command("format")
def format_code():
    """Format code with black, ruff, and mdformat."""
    run("black", "tiny_audio", "scripts", "tests")
    run("ruff", "format", "tiny_audio", "scripts", "tests")
    run("ruff", "check", "--fix", "tiny_audio", "scripts", "tests")

    # Format markdown files
    md_files = [
        str(f)
        for f in Path().rglob("*.md")
        if ".venv" not in str(f) and "docs/course" not in str(f) and f.name != "MODEL_CARD.md"
    ]
    if md_files:
        run("mdformat", *md_files)


@app.command("type-check")
def type_check():
    """Run type checkers (mypy and pyright)."""
    code = run_all(
        ["mypy", "tiny_audio"],
        ["pyright", "tiny_audio"],
    )
    raise typer.Exit(code)


@app.command()
def test():
    """Run pytest tests."""
    raise typer.Exit(run("pytest", "-v"))


@app.command()
def coverage():
    """Run tests with coverage report."""
    raise typer.Exit(
        run(
            "pytest",
            "--cov=tiny_audio",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v",
        )
    )


@app.command()
def check():
    """Run all checks (lint + type-check + security + docstrings)."""
    code = run_all(
        ["poetry", "check"],
        ["ruff", "check", "tiny_audio", "scripts", "tests"],
        ["yamllint", "-d", "relaxed", "configs/"],
        ["taplo", "check", "pyproject.toml"],
        ["mypy", "tiny_audio"],
        ["pyright", "tiny_audio"],
        ["bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"],
        ["interrogate", "tiny_audio", "--fail-under", "50"],
    )
    raise typer.Exit(code)


@app.command()
def build():
    """Build package (wheel and sdist)."""
    raise typer.Exit(run("poetry", "build"))


@app.command()
def precommit():
    """Pre-commit quality gate (format, lint, type-check, security, docstrings, test, build)."""
    # Format first (doesn't fail)
    format_code()

    # Then run all checks
    code = run_all(
        ["poetry", "check"],
        ["ruff", "check", "tiny_audio", "scripts", "tests"],
        ["yamllint", "-d", "relaxed", "configs/"],
        ["taplo", "check", "pyproject.toml"],
        ["mypy", "tiny_audio"],
        ["pyright", "tiny_audio"],
        ["bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"],
        ["interrogate", "tiny_audio", "--fail-under", "50"],
        ["pytest", "-v"],
        ["poetry", "build"],
    )
    raise typer.Exit(code)


@app.command("install-hooks")
def install_hooks():
    """Install pre-commit hooks."""
    raise typer.Exit(run("pre-commit", "install"))


@app.command()
def security():
    """Run security checks with bandit."""
    raise typer.Exit(run("bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"))


@app.command("dead-code")
def dead_code():
    """Find dead/unused code with vulture."""
    raise typer.Exit(run("vulture", "tiny_audio", "scripts", "--min-confidence", "80"))


@app.command()
def docstrings():
    """Check docstring coverage with interrogate."""
    raise typer.Exit(run("interrogate", "tiny_audio", "-v", "--fail-under", "50"))


def _register_handler():
    """Register handler subcommand (lazy import)."""
    from scripts.deploy.handler_local import test as handler_test

    app.command(name="handler", help="Test inference endpoint handler locally")(handler_test)


_register_handler()

if __name__ == "__main__":
    app()
