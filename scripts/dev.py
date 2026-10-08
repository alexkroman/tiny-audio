#!/usr/bin/env python3
"""Development commands for tiny-audio."""

import subprocess
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="dev",
    help="Run the project's lint, format, test and build tasks.",
    no_args_is_help=True,
    add_completion=False,
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
    # Workflow syntax/expressions, then security shapes (template injection,
    # unpinned actions, persisted checkout credentials). `--offline` keeps
    # zizmor off the GitHub API so a local token can't change the verdict.
    ["actionlint"],
    ["zizmor", "--offline", ".github/workflows"],
]
TYPE_CHECK_COMMANDS = [
    ["mypy", LIB_PATH],
    ["pyright", LIB_PATH],
]
SECURITY_COMMAND = ["bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"]
# Unused, missing, transitive-only and misplaced dependencies; configured in
# `[tool.deptry]` (pyproject.toml).
DEPS_COMMAND = ["deptry", "."]
# Copy-pasted blocks of 4+ lines across tiny_audio/ and scripts/ (comments,
# docstrings, imports and signatures ignored). demo/ is left out: it deploys
# to the Space on its own and cannot share code with scripts/.
DUPLICATION_COMMAND = [
    "pylint",
    "--disable=all",
    "--enable=duplicate-code",
    "--min-similarity-lines=4",
    "--score=n",
    LIB_PATH,
    "scripts",
]
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
# Everything `ta dev check` runs after the linters; see `check_commands()`.
ANALYSIS_COMMANDS = [
    *TYPE_CHECK_COMMANDS,
    SECURITY_COMMAND,
    DEAD_CODE_COMMAND,
    DUPLICATION_COMMAND,
    DEPS_COMMAND,
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


# `--clean` empties dist/ first, so the checks below see only this build.
BUILD_COMMAND = ["poetry", "build", "--clean"]
DIST_DIR = Path("dist")


def dist_check_commands() -> list[list[str]]:
    """Checks for the artifacts `BUILD_COMMAND` just wrote to dist/.

    check-wheel-contents compares the wheel against the source packages (a
    module missing from the wheel fails here, not on the pod) and twine checks
    the metadata of both the wheel and the sdist. W005/W009 are ignored: the
    top-level `scripts` package is deliberate, it carries the `ta` entry point.
    """
    wheels = sorted(str(p) for p in DIST_DIR.glob("*.whl"))
    artifacts = sorted(str(p) for p in DIST_DIR.iterdir()) if DIST_DIR.is_dir() else []
    return [
        [
            "check-wheel-contents",
            "--ignore",
            "W005,W009",
            "--package",
            LIB_PATH,
            "--package",
            "scripts",
            *wheels,
        ],
        ["twine", "check", "--strict", *artifacts],
    ]


def build_and_check() -> int:
    """Build the wheel and sdist, then validate them; the first failure wins."""
    return run(*BUILD_COMMAND) or run_all(*dist_check_commands())


# YAML front matter, which mdformat mangles (also excluded in [tool.mdformat]).
MARKDOWN_SKIP = {"MODEL_CARD.md", "demo/README.md"}


def markdown_files() -> list[str]:
    """Tracked Markdown that `ta dev format` rewrites and `ta dev lint` checks.

    git ls-files rather than a glob: it skips worktrees, build/cache dirs and
    anything else gitignore already excludes.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    )
    return [f for f in tracked.stdout.splitlines() if f and f not in MARKDOWN_SKIP]


def lint_commands() -> list[list[str]]:
    """`LINT_COMMANDS` plus the Markdown check, whose file list comes from git."""
    return [*LINT_COMMANDS, ["mdformat", "--check", *markdown_files()]]


def check_commands() -> list[list[str]]:
    """Everything `ta dev check` (and CI's quality step) runs, in order."""
    return [*lint_commands(), *ANALYSIS_COMMANDS]


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
    """Run linters (Poetry + Python + YAML + TOML + GitHub Actions + Markdown)."""
    raise typer.Exit(run_all(*lint_commands()))


@app.command("format")
def format_code():
    """Format code with black, ruff, and mdformat."""
    run("black", *CODE_PATHS)
    run("ruff", "format", *CODE_PATHS)
    run("ruff", "check", "--fix", *CODE_PATHS)

    md_files = markdown_files()
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
    """Run all checks (lint, type-check, security, dead code, duplication, deps, docstrings)."""
    raise typer.Exit(run_all(*check_commands()))


@app.command()
def build():
    """Build package (wheel and sdist) and validate both."""
    raise typer.Exit(build_and_check())


@app.command()
def precommit():
    """Pre-commit quality gate (format, check, test with coverage floor, build)."""
    format_code()
    raise typer.Exit(run_all(*check_commands(), TEST_COMMAND) or build_and_check())


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
def duplication():
    """Find copy-pasted code with pylint's duplicate-code check."""
    raise typer.Exit(run(*DUPLICATION_COMMAND))


@app.command()
def deps():
    """Find unused, missing and transitive-only dependencies with deptry."""
    raise typer.Exit(run(*DEPS_COMMAND))


@app.command()
def docstrings():
    """Check docstring coverage with interrogate (verbose, per-file table)."""
    raise typer.Exit(run_all(*[[*cmd, "-v"] for cmd in DOCSTRINGS_COMMANDS]))


def _register_handler():
    from scripts.deploy.handler_local import test as handler_test

    app.command(name="handler")(handler_test)


_register_handler()

if __name__ == "__main__":
    app()
