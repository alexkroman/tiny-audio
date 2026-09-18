#!/usr/bin/env python3
"""Development commands for tiny-audio."""

import subprocess

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

# Every threshold below is a ratchet: raise it when the codebase clears the
# next rung, never lower it to get a red build green. Coverage's floor lives in
# `[tool.coverage.report] fail_under` (pyproject.toml) and is enforced by
# TEST_COMMAND via pytest-cov.
DOCSTRING_MIN_LIB = "95"  # tiny_audio/: the published model code
DOCSTRING_MIN_SCRIPTS = "65"  # scripts/: CLI + training; probes drag it down
DEAD_CODE_MIN_CONFIDENCE = "80"

LINT_COMMANDS = [
    # `--lock` also fails when poetry.lock is missing or stale relative to
    # pyproject.toml, so a dependency edit can't land without its lock update.
    ["poetry", "check", "--lock"],
    ["ruff", "check", *CODE_PATHS],
    # `ta dev format` rewrites; `check` must refuse anything it would rewrite,
    # otherwise formatting drift only shows up as noise in the next PR.
    ["ruff", "format", "--check", *CODE_PATHS],
    ["black", "--check", *CODE_PATHS],
    ["yamllint", "configs/", ".github/"],
    ["taplo", "check", "pyproject.toml"],
]
TYPE_CHECK_COMMANDS = [
    ["mypy", LIB_PATH],
    ["pyright", LIB_PATH],
]
SECURITY_COMMAND = ["bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"]
DEAD_CODE_COMMAND = [
    "vulture",
    "tiny_audio",
    "scripts",
    "--min-confidence",
    DEAD_CODE_MIN_CONFIDENCE,
]
DOCSTRINGS_COMMANDS = [
    ["interrogate", LIB_PATH, "--fail-under", DOCSTRING_MIN_LIB],
    ["interrogate", "scripts", "--fail-under", DOCSTRING_MIN_SCRIPTS],
]
CHECK_COMMANDS = [
    *LINT_COMMANDS,
    *TYPE_CHECK_COMMANDS,
    SECURITY_COMMAND,
    DEAD_CODE_COMMAND,
    *DOCSTRINGS_COMMANDS,
]
# pytest-cov reads `fail_under` from `[tool.coverage.report]`, so the same
# floor applies locally, in the pre-commit gate and in CI.
TEST_COMMAND = [
    "pytest",
    "-v",
    f"--cov={LIB_PATH}",
    "--cov=scripts",
    "--cov-report=term-missing",
    "--cov-report=xml",
]


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
    md_skip_exact = {"MODEL_CARD.md", "demo/README.md"}
    md_files = [
        line
        for line in tracked.stdout.splitlines()
        if line and not any(line.startswith(p) for p in md_excludes) and line not in md_skip_exact
    ]
    if md_files:
        run("mdformat", *md_files)


@app.command("type-check")
def type_check():
    """Run type checkers (mypy and pyright)."""
    raise typer.Exit(run_all(*TYPE_CHECK_COMMANDS))


@app.command()
def test():
    """Run pytest with the coverage floor enforced."""
    raise typer.Exit(run(*TEST_COMMAND))


@app.command()
def coverage():
    """Run tests with coverage report (adds an HTML report under htmlcov/)."""
    raise typer.Exit(run(*TEST_COMMAND, "--cov-report=html"))


@app.command()
def check():
    """Run all checks (lint + format + type-check + security + dead-code + docstrings)."""
    raise typer.Exit(run_all(*CHECK_COMMANDS))


@app.command()
def build():
    """Build package (wheel and sdist)."""
    raise typer.Exit(run("poetry", "build"))


@app.command()
def precommit():
    """Pre-commit quality gate (format, check, test with coverage floor, build)."""
    format_code()
    raise typer.Exit(run_all(*CHECK_COMMANDS, TEST_COMMAND, ["poetry", "build"]))


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
    raise typer.Exit(run(*DEAD_CODE_COMMAND))


@app.command()
def docstrings():
    """Check docstring coverage with interrogate (verbose, per-file table)."""
    raise typer.Exit(run_all(*[[*cmd, "-v"] for cmd in DOCSTRINGS_COMMANDS]))


def _register_handler():
    from scripts.deploy.handler_local import test as handler_test

    app.command(name="handler", help="Test inference endpoint handler locally")(handler_test)


_register_handler()

if __name__ == "__main__":
    app()
