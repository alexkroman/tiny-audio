#!/usr/bin/env python3
"""Development commands for tiny-audio."""

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

CODE_PATHS = ["tiny_audio", "scripts", "tests"]
LIB_PATH = "tiny_audio"

LINT_COMMANDS = [
    ["poetry", "check"],
    ["ruff", "check", *CODE_PATHS],
    ["yamllint", "configs/"],
    ["taplo", "check", "pyproject.toml"],
]
TYPE_CHECK_COMMANDS = [
    ["mypy", LIB_PATH],
    ["pyright", LIB_PATH],
]
SECURITY_COMMAND = ["bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"]
DOCSTRINGS_COMMAND = ["interrogate", LIB_PATH, "--fail-under", "50"]
CHECK_COMMANDS = [*LINT_COMMANDS, *TYPE_CHECK_COMMANDS, SECURITY_COMMAND, DOCSTRINGS_COMMAND]


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
    raise typer.Exit(run_all(*LINT_COMMANDS))


@app.command("format")
def format_code():
    """Format code with black, ruff, and mdformat."""
    run("black", *CODE_PATHS)
    run("ruff", "format", *CODE_PATHS)
    run("ruff", "check", "--fix", *CODE_PATHS)

    # Use git ls-files so we only format tracked markdown — naturally skips
    # worktrees, build/cache/checkpoint dirs, and vendored docs that gitignore
    # already excludes. (Pre-rglob version mis-handled `.claude/worktrees`,
    # `swift/.build`, etc.)
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    )
    md_excludes = ("docs/course/",)
    md_files = [
        line
        for line in tracked.stdout.splitlines()
        if line
        and not any(line.startswith(p) for p in md_excludes)
        and Path(line).name != "MODEL_CARD.md"
    ]
    if md_files:
        run("mdformat", *md_files)


@app.command("type-check")
def type_check():
    """Run type checkers (mypy and pyright)."""
    raise typer.Exit(run_all(*TYPE_CHECK_COMMANDS))


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
            f"--cov={LIB_PATH}",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v",
        )
    )


@app.command()
def check():
    """Run all checks (lint + type-check + security + docstrings)."""
    raise typer.Exit(run_all(*CHECK_COMMANDS))


@app.command()
def build():
    """Build package (wheel and sdist)."""
    raise typer.Exit(run("poetry", "build"))


@app.command()
def precommit():
    """Pre-commit quality gate (format, lint, type-check, security, docstrings, test, build)."""
    format_code()
    raise typer.Exit(run_all(*CHECK_COMMANDS, ["pytest", "-v"], ["poetry", "build"]))


@app.command("install-hooks")
def install_hooks():
    """Install pre-commit hooks."""
    raise typer.Exit(run("pre-commit", "install"))


@app.command()
def security():
    """Run security checks with bandit."""
    raise typer.Exit(run(*SECURITY_COMMAND))


@app.command("dead-code")
def dead_code():
    """Find dead/unused code with vulture."""
    raise typer.Exit(run("vulture", "tiny_audio", "scripts", "--min-confidence", "80"))


@app.command()
def docstrings():
    """Check docstring coverage with interrogate."""
    raise typer.Exit(run("interrogate", LIB_PATH, "-v", "--fail-under", "50"))


def _register_handler():
    from scripts.deploy.handler_local import test as handler_test

    app.command(name="handler", help="Test inference endpoint handler locally")(handler_test)


_register_handler()

if __name__ == "__main__":
    app()
