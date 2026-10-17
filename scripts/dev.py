#!/usr/bin/env python3
"""Development commands for tiny-audio."""

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from scripts import mutate, quality

app: typer.Typer = typer.Typer(
    name="dev",
    help="Run the project's lint, format, test and build tasks.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(quality.app)
app.command("mutate")(mutate.main)
console: Console = Console()

# Every Python tree in the repo: the package, its CLI, the tests, the Space
# demo and the course examples. Linters and type checkers all run over this.
CODE_PATHS = ["tiny_audio", "scripts", "tests", "demo", "docs", "typings"]
LIB_PATH = "tiny_audio"
# The three trees every quality gate covers alike: the published package, the
# CLI and training code that builds and ships it, and the tests that vouch
# for both. demo/, docs/ and typings/ get the linters and pyright only.
QUALITY_PATHS = [LIB_PATH, "scripts", "tests"]
TOML_FILES = ["pyproject.toml", ".mdformat.toml"]

# Every threshold below is a ratchet: raise it when the codebase clears the
# next rung, never lower it to get a red build green. Coverage's floor lives in
# `[tool.coverage.report] fail_under` (pyproject.toml) and is enforced by
# TEST_COMMAND via pytest-cov.
DOCSTRING_MIN = "95"  # each of QUALITY_PATHS on its own, not pooled
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
    # Paths come from `[tool.pyright] include`.
    ["pyright"],
]
SECURITY_COMMAND = ["bandit", "-c", "pyproject.toml", "-r", *QUALITY_PATHS, "-ll"]
# Unused, missing, transitive-only and misplaced dependencies; configured in
# `[tool.deptry]` (pyproject.toml).
DEPS_COMMAND = ["deptry", "."]
# Ratchets in scripts/quality.py, against baselines committed under quality/.
QUALITY = [sys.executable, "-m", "scripts.quality"]
FILE_LENGTH_COMMAND = [*QUALITY, "file-length"]
TEST_ASSERTIONS_COMMAND = [*QUALITY, "test-assertions"]
ESCAPE_HATCHES_COMMAND = [*QUALITY, "escape-hatches"]
MODULE_TESTS_COMMAND = [*QUALITY, "module-tests"]
TYPE_COMPLETENESS_COMMAND = [*QUALITY, "type-completeness"]
TRACKED_IGNORED_COMMAND = [*QUALITY, "tracked-ignored"]
INVARIANTS_COMMAND = [*QUALITY, "invariants"]
OWNING_IMPORTS_COMMAND = [*QUALITY, "owning-imports"]
# Reads the coverage.json TEST_COMMAND writes, so it runs after the tests.
COVERAGE_FLOORS_COMMAND = [*QUALITY, "coverage-floors"]
# Copy-pasted blocks of 4+ lines across QUALITY_PATHS, within and between
# them (comments, docstrings, imports and signatures ignored). demo/ is left
# out: it deploys to the Space on its own and cannot share code with scripts/.
DUPLICATION_COMMAND = [
    "pylint",
    "--disable=all",
    "--enable=duplicate-code",
    "--min-similarity-lines=4",
    "--score=n",
    *QUALITY_PATHS,
]
DEAD_CODE_COMMAND = [
    "vulture",
    *QUALITY_PATHS,
    # Protocol / type-stub parameter names; see the file's docstring.
    "quality/vulture_whitelist.py",
    "--min-confidence",
    DEAD_CODE_MIN_CONFIDENCE,
]
# The same rule for each tree, checked separately so one tree's surplus
# cannot cover another's gap. Not counted: `@overload` stubs (signatures for
# the documented implementation below them), nested functions (closures and
# test fakes, documented by the function that holds them), and `test_*`
# functions, whose names are their documentation.
DOCSTRINGS_COMMANDS = [
    [
        "interrogate",
        path,
        "--ignore-overloaded-functions",
        "--ignore-nested-functions",
        "--ignore-regex",
        "^test_",
        "--fail-under",
        DOCSTRING_MIN,
    ]
    for path in QUALITY_PATHS
]
# Everything `ta dev check` runs after the linters; see `check_commands()`.
ANALYSIS_COMMANDS = [
    *TYPE_CHECK_COMMANDS,
    SECURITY_COMMAND,
    DEAD_CODE_COMMAND,
    DUPLICATION_COMMAND,
    FILE_LENGTH_COMMAND,
    TEST_ASSERTIONS_COMMAND,
    ESCAPE_HATCHES_COMMAND,
    MODULE_TESTS_COMMAND,
    TYPE_COMPLETENESS_COMMAND,
    TRACKED_IGNORED_COMMAND,
    INVARIANTS_COMMAND,
    OWNING_IMPORTS_COMMAND,
    DEPS_COMMAND,
    *DOCSTRINGS_COMMANDS,
]
# pytest-cov reads `fail_under` from `[tool.coverage.report]`, so the same
# floor applies locally, in the pre-commit gate and in CI.
TEST_COMMAND = [
    "pytest",
    "-v",
    *(f"--cov={path}" for path in QUALITY_PATHS),
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


def shell_files() -> list[str]:
    """Tracked shell scripts that `ta dev lint` checks with shellcheck."""
    return tracked_files("*.sh")


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
        # Optional checks are switched on in .shellcheckrc.
        *([["shellcheck", *shell]] if (shell := shell_files()) else []),
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


def run_each(*commands: list[str]) -> int:
    """Run every command, then list the failures; returns the first failing exit code.

    For independent read-only checks: stopping at the first failure hides
    what the rest would say, so their findings arrive one red run at a time.
    """
    failed = [(cmd, code) for cmd in commands if (code := run(*cmd)) != 0]
    if failed:
        console.print(f"[red]{len(failed)} of {len(commands)} checks failed:[/red]")
        for cmd, code in failed:
            console.print(f"[red]  exit {code}: {' '.join(cmd)}[/red]")
    return failed[0][1] if failed else 0


def git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git quietly; the caller reads the exit code and output."""
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def prepush_problems() -> list[str]:
    """Why this push would fail review: on main, behind main, or conflicting with it.

    Fails open (no problem) when origin/main cannot be fetched, so an offline
    push is not blocked on a check it cannot run.
    """
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch in {"main", "master"}:
        return [f"direct pushes to {branch} are not allowed; push a branch and open a PR"]
    if git("fetch", "--quiet", "origin", "main").returncode != 0:
        console.print("[yellow]could not fetch origin/main; skipping the freshness checks[/yellow]")
        return []
    # `merge-tree --write-tree` (git >= 2.38) merges in memory: exit 1 means conflicts.
    if git("merge-tree", "--write-tree", "HEAD", "origin/main").returncode == 1:
        return ["this branch conflicts with origin/main; merge it and resolve the conflicts"]
    if git("merge-base", "--is-ancestor", "origin/main", "HEAD").returncode != 0:
        return ["this branch is behind origin/main; run `git merge origin/main` and re-check"]
    return []


@app.command()
def lint() -> None:
    """Run linters (Poetry + Python + YAML + TOML + GitHub Actions + Markdown + JSON)."""
    raise typer.Exit(run_each(*lint_commands()))


@app.command("format")
def format_code() -> None:
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
def type_check() -> None:
    """Run the type checker (pyright, strict)."""
    raise typer.Exit(run_all(*TYPE_CHECK_COMMANDS))


@app.command()
def test() -> None:
    """Run pytest with the coverage floor and per-file floors enforced."""
    raise typer.Exit(run_all(TEST_COMMAND, COVERAGE_FLOORS_COMMAND))


@app.command()
def coverage() -> None:
    """Run tests with coverage report (adds an HTML report under htmlcov/)."""
    raise typer.Exit(run(*TEST_COMMAND, "--cov-report=html"))


@app.command()
def check() -> None:
    """Run all checks (lint, types, security, dead code, duplication, ratchets, deps, docs).

    Every check runs even after one fails; the failures are listed at the end.
    """
    raise typer.Exit(run_each(*check_commands()))


@app.command()
def build() -> None:
    """Build package (wheel and sdist) and validate both."""
    raise typer.Exit(build_and_check())


@app.command()
def precommit() -> None:
    """Pre-commit quality gate (format, check, test with coverage floor, build)."""
    format_code()
    raise typer.Exit(
        run_each(*check_commands())
        or run_all(TEST_COMMAND, COVERAGE_FLOORS_COMMAND)
        or build_and_check()
    )


@app.command()
def prepush() -> None:
    """Pre-push gate: not on main, up to date with origin/main, no conflicts with it."""
    problems = prepush_problems()
    for problem in problems:
        console.print(f"[red]{problem}[/red]")
    raise typer.Exit(1 if problems else 0)


@app.command("install-hooks")
def install_hooks() -> None:
    """Install the git hooks (`default_install_hook_types`: pre-commit and pre-push)."""
    raise typer.Exit(run("pre-commit", "install"))


@app.command()
def security() -> None:
    """Run security checks with bandit."""
    raise typer.Exit(run(*SECURITY_COMMAND))


@app.command("dead-code")
def dead_code() -> None:
    """Find dead/unused code with vulture."""
    raise typer.Exit(run(*DEAD_CODE_COMMAND))


@app.command()
def duplication() -> None:
    """Find copy-pasted code with pylint's duplicate-code check."""
    raise typer.Exit(run(*DUPLICATION_COMMAND))


@app.command()
def deps() -> None:
    """Find unused, missing and transitive-only dependencies with deptry."""
    raise typer.Exit(run(*DEPS_COMMAND))


@app.command()
def docstrings() -> None:
    """Check docstring coverage with interrogate (verbose, per-file table)."""
    raise typer.Exit(run_all(*[[*cmd, "-v"] for cmd in DOCSTRINGS_COMMANDS]))


if __name__ == "__main__":
    app()
