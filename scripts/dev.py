#!/usr/bin/env python3
"""Development commands for tiny-audio."""

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from scripts import quality
from scripts.deploy import handler_local

app = typer.Typer(
    name="dev",
    help="Run the project's lint, format, test and build tasks.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(quality.app)
console = Console()

# Every Python tree in the repo: the package, its CLI, the tests, the Space
# demo and the course examples. Linters and type checkers all run over this.
CODE_PATHS = ["tiny_audio", "scripts", "tests", "demo", "docs", "typings"]
LIB_PATH = "tiny_audio"
TOML_FILES = ["pyproject.toml", ".mdformat.toml"]

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
    # Every YAML file; `.yamllint` skips what .gitignore ignores.
    ["yamllint", "--strict", "."],
    # TOML syntax, then layout (`ta dev format` runs `taplo fmt`).
    ["taplo", "check", *TOML_FILES],
    ["taplo", "fmt", "--check", *TOML_FILES],
    # Workflow syntax/expressions, then security shapes (template injection,
    # unpinned actions, persisted checkout credentials). `--offline` keeps
    # zizmor off the GitHub API so a local token can't change the verdict.
    ["actionlint"],
    ["zizmor", "--offline", ".github/workflows"],
]
TYPE_CHECK_COMMANDS = [
    ["mypy", *CODE_PATHS],
    # Paths come from `[tool.pyright] include`.
    ["pyright"],
]
SECURITY_COMMAND = ["bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"]
# Unused, missing, transitive-only and misplaced dependencies; configured in
# `[tool.deptry]` (pyproject.toml).
DEPS_COMMAND = ["deptry", "."]
# Ratchets in scripts/quality.py, against baselines committed under quality/.
QUALITY = [sys.executable, "-m", "scripts.quality"]
FILE_LENGTH_COMMAND = [*QUALITY, "file-length"]
TEST_ASSERTIONS_COMMAND = [*QUALITY, "test-assertions"]
# Reads the coverage.json TEST_COMMAND writes, so it runs after the tests.
COVERAGE_FLOORS_COMMAND = [*QUALITY, "coverage-floors"]
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
# `@overload` stubs are type signatures for the documented implementation
# below them, not separate functions, so they are not counted.
DOCSTRINGS_COMMANDS = [
    [
        "interrogate",
        LIB_PATH,
        "--ignore-overloaded-functions",
        "--fail-under",
        DOCSTRING_MIN_LIB,
    ],
    [
        "interrogate",
        "scripts",
        "--ignore-overloaded-functions",
        "--fail-under",
        DOCSTRING_MIN_SCRIPTS,
    ],
]
# Everything `ta dev check` runs after the linters; see `check_commands()`.
ANALYSIS_COMMANDS = [
    *TYPE_CHECK_COMMANDS,
    SECURITY_COMMAND,
    DEAD_CODE_COMMAND,
    DUPLICATION_COMMAND,
    FILE_LENGTH_COMMAND,
    TEST_ASSERTIONS_COMMAND,
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
    "--cov-report=json",
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


def tracked_files(pattern: str) -> list[str]:
    """Tracked files matching `pattern`.

    git ls-files rather than a glob: it skips worktrees, build/cache dirs and
    anything else gitignore already excludes.
    """
    tracked = subprocess.run(
        ["git", "ls-files", pattern], capture_output=True, text=True, check=True
    )
    return [f for f in tracked.stdout.splitlines() if f]


def markdown_files() -> list[str]:
    """Tracked Markdown that `ta dev format` rewrites and `ta dev lint` checks."""
    return tracked_files("*.md")


def json_files() -> list[str]:
    """Tracked JSON (the quality/ baselines) that `ta dev lint` checks.

    tests/fixtures/ is excluded: it holds Hub files vendored byte-for-byte.
    """
    return [f for f in tracked_files("*.json") if not f.startswith("tests/fixtures/")]


def lint_commands() -> list[list[str]]:
    """`LINT_COMMANDS` plus the Markdown and JSON checks, whose file lists come from git.

    pretty-format-json without `--autofix` fails on invalid JSON and on any
    file that is not sorted, 2-space-indented JSON (the layout
    scripts/quality.py writes).
    """
    markdown = markdown_files()
    return [
        *LINT_COMMANDS,
        ["mdformat", "--check", *markdown],
        ["pymarkdown", "scan", *markdown],
        ["pretty-format-json", *json_files()],
    ]


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
    """Run linters (Poetry + Python + YAML + TOML + GitHub Actions + Markdown + JSON)."""
    raise typer.Exit(run_all(*lint_commands()))


@app.command("format")
def format_code():
    """Format code with black, ruff, mdformat, taplo and pretty-format-json."""
    run("black", *CODE_PATHS)
    run("ruff", "format", *CODE_PATHS)
    run("ruff", "check", "--fix", *CODE_PATHS)
    run("taplo", "fmt", *TOML_FILES)

    md_files = markdown_files()
    if md_files:
        run("mdformat", *md_files)
    json = json_files()
    if json:
        run("pretty-format-json", "--autofix", *json)


@app.command("type-check")
def type_check():
    """Run type checkers (mypy and pyright)."""
    raise typer.Exit(run_all(*TYPE_CHECK_COMMANDS))


@app.command()
def test():
    """Run pytest with the coverage floor and per-file floors enforced."""
    raise typer.Exit(run_all(TEST_COMMAND, COVERAGE_FLOORS_COMMAND))


@app.command()
def coverage():
    """Run tests with coverage report (adds an HTML report under htmlcov/)."""
    raise typer.Exit(run(*TEST_COMMAND, "--cov-report=html"))


@app.command()
def check():
    """Run all checks (lint, types, security, dead code, duplication, ratchets, deps, docs)."""
    raise typer.Exit(run_all(*check_commands()))


@app.command()
def build():
    """Build package (wheel and sdist) and validate both."""
    raise typer.Exit(build_and_check())


@app.command()
def precommit():
    """Pre-commit quality gate (format, check, test with coverage floor, build)."""
    format_code()
    raise typer.Exit(
        run_all(*check_commands(), TEST_COMMAND, COVERAGE_FLOORS_COMMAND) or build_and_check()
    )


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


app.command(name="handler")(handler_local.run_handler)

if __name__ == "__main__":
    app()
