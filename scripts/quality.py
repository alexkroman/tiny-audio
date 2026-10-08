#!/usr/bin/env python3
"""Quality ratchets: file length, per-file coverage floors, tests that assert.

Each check holds a line rather than demanding the codebase already meet it:
files that predate a rule are recorded in a committed baseline under
`quality/` and may only improve. `--update` rewrites a baseline from the
current tree; review its diff like any other change.

    ta dev quality file-length [--update]
    ta dev quality coverage-floors [--update]   # after `ta dev test`
    ta dev quality test-assertions
"""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from scripts.utils import get_project_root

app = typer.Typer(
    name="quality",
    help="Ratchets on file length, per-file coverage and test assertions.",
    no_args_is_help=True,
    add_completion=False,
)

ROOT = get_project_root()
BASELINE_DIR = ROOT / "quality"
FILE_LENGTH_BASELINE = BASELINE_DIR / "file_length.json"
COVERAGE_BASELINE = BASELINE_DIR / "coverage_floors.json"
COVERAGE_REPORT = ROOT / "coverage.json"

# Code lines (non-blank, not comment-only) a Python file may have. Long files
# are where complexity hides and where reviewers stop reading.
MAX_CODE_LINES = 600
# Line coverage a file needs when it has no recorded floor yet.
NEW_FILE_COVERAGE_FLOOR = 50.0

UPDATE_HELP = "Rewrite the baseline from the current tree instead of checking"

# A call to one of these counts as an assertion: the code under test is itself
# a guard, so "it did not raise" is the claim (`_assert_audio_token_counts`,
# `_require_fused_cross_entropy`, `_check_remote_disk`, ...).
_GUARD_NAME = re.compile(r"_*(assert|require|check|ensure|validate|verify)")
_PYTEST_ASSERTIONS = frozenset({"raises", "warns", "fail", "deprecated_call"})


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")


def _tracked_python_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [f for f in out.stdout.splitlines() if f]


# ------------------------------------------------------------------ file length


def code_lines(text: str) -> int:
    """Lines that are neither blank nor comment-only."""
    return sum(
        1 for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )


def file_length_violations(lengths: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Files over the cap that are not grandfathered, or grandfathered and grown."""
    problems = []
    for path, n in sorted(lengths.items()):
        allowed = baseline.get(path, MAX_CODE_LINES)
        if n > max(allowed, MAX_CODE_LINES):
            if path in baseline:
                problems.append(f"{path}: {n} code lines, grew past its recorded {allowed}")
            else:
                problems.append(f"{path}: {n} code lines, over the {MAX_CODE_LINES} cap")
    return problems


@app.command("file-length")
def file_length(
    update: Annotated[bool, typer.Option("--update", help=UPDATE_HELP)] = False,
) -> None:
    """Cap Python files at MAX_CODE_LINES; files already over it may not grow."""
    lengths = {f: code_lines((ROOT / f).read_text()) for f in _tracked_python_files()}
    if update:
        _save(FILE_LENGTH_BASELINE, {f: n for f, n in lengths.items() if n > MAX_CODE_LINES})
        typer.echo(f"Recorded {FILE_LENGTH_BASELINE.relative_to(ROOT)}")
        return
    problems = file_length_violations(lengths, _load(FILE_LENGTH_BASELINE))
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        typer.echo("Split the file, or shrink it; never raise the baseline to pass.", err=True)
        raise typer.Exit(1)
    typer.echo(f"file-length: {len(lengths)} files within limits")


# --------------------------------------------------------------- coverage floors


def file_coverage(report: dict) -> dict[str, float]:
    """Line coverage percent per file from a coverage.py JSON report."""
    out = {}
    for path, data in report.get("files", {}).items():
        summary = data["summary"]
        statements = summary["num_statements"]
        out[path] = 100.0 if statements == 0 else 100.0 * summary["covered_lines"] / statements
    return out


def coverage_violations(coverage: dict[str, float], floors: dict[str, float]) -> list[str]:
    """Files below their recorded floor, or new files below NEW_FILE_COVERAGE_FLOOR."""
    problems = []
    for path, pct in sorted(coverage.items()):
        floor = floors.get(path, NEW_FILE_COVERAGE_FLOOR)
        if pct + 1e-9 < floor:
            kind = "its recorded floor" if path in floors else "the new-file floor"
            problems.append(f"{path}: {pct:.1f}% line coverage, below {kind} of {floor:.1f}%")
    return problems


@app.command("coverage-floors")
def coverage_floors(
    update: Annotated[bool, typer.Option("--update", help=UPDATE_HELP)] = False,
) -> None:
    """Per-file coverage floors; the global `fail_under` cannot see one bare file.

    Reads coverage.json, which `ta dev test` writes. Record floors from a run
    with Hugging Face Hub access, since tests that load models skip offline.
    """
    if not COVERAGE_REPORT.exists():
        typer.echo(f"{COVERAGE_REPORT.name} not found: run `ta dev test` first", err=True)
        raise typer.Exit(1)
    coverage = file_coverage(json.loads(COVERAGE_REPORT.read_text()))
    if update:
        # Rounded down to 0.1 so the run that records a floor also passes it.
        _save(COVERAGE_BASELINE, {f: math.floor(p * 10) / 10 for f, p in coverage.items()})
        typer.echo(f"Recorded {COVERAGE_BASELINE.relative_to(ROOT)}")
        return
    problems = coverage_violations(coverage, _load(COVERAGE_BASELINE))
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        typer.echo("Add tests for the uncovered lines; never lower a floor to pass.", err=True)
        raise typer.Exit(1)
    typer.echo(f"coverage-floors: {len(coverage)} files at or above their floors")


# --------------------------------------------------------------- test assertions


def _callee(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _asserts_directly(func: ast.AST) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            name = _callee(node)
            if _GUARD_NAME.match(name) or name in _PYTEST_ASSERTIONS:
                return True
    return False


def tests_without_assertions(source: str) -> list[tuple[int, str]]:
    """`(line, name)` of each test function that checks nothing.

    A test asserts when its body has an `assert`, a `pytest.raises`-style
    context, a mock `assert_*` call or a call to a guard, or calls a helper in
    the same module that does (followed transitively).
    """
    tree = ast.parse(source)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    asserting = {f.name for f in funcs if _asserts_directly(f)}
    changed = True
    while changed:
        changed = False
        for f in funcs:
            if f.name not in asserting and any(
                isinstance(n, ast.Call) and _callee(n) in asserting for n in ast.walk(f)
            ):
                asserting.add(f.name)
                changed = True
    return [
        (f.lineno, f.name) for f in funcs if f.name.startswith("test") and f.name not in asserting
    ]


@app.command("test-assertions")
def test_assertions() -> None:
    """Every test must assert something; one that only runs code checks nothing."""
    problems = []
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    for path in tests:
        for line, name in tests_without_assertions(path.read_text()):
            problems.append(f"{path.relative_to(ROOT)}:{line}: {name} asserts nothing")
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        raise typer.Exit(1)
    typer.echo(f"test-assertions: every test in {len(tests)} files asserts")


if __name__ == "__main__":
    app()
