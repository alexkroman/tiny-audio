#!/usr/bin/env python3
"""Quality ratchets: file length, coverage, assertions, escape hatches, module tests, types.

Each check holds a line rather than demanding the codebase already meet it:
files that predate a rule are recorded in a committed baseline under
`quality/` and may only improve. `--update` rewrites a baseline from the
current tree; review its diff like any other change.

    ta dev quality file-length [--update]
    ta dev quality coverage-floors [--update]   # after `ta dev test`
    ta dev quality test-assertions
    ta dev quality escape-hatches [--update]    # --update only lowers budgets
    ta dev quality module-tests [--update]      # --update only shrinks the list
    ta dev quality type-completeness [--update]
    ta dev quality tracked-ignored
    ta dev quality invariants                   # scripts/invariants.py
"""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import tokenize
from io import StringIO
from pathlib import Path
from typing import Annotated, Any

import typer

from scripts import invariants
from scripts.utils import get_project_root

app: typer.Typer = typer.Typer(
    name="quality",
    help="Ratchets on file length, coverage, assertions, escape hatches and typing.",
    no_args_is_help=True,
    add_completion=False,
)

ROOT = get_project_root()
BASELINE_DIR = ROOT / "quality"
FILE_LENGTH_BASELINE = BASELINE_DIR / "file_length.json"
COVERAGE_BASELINE = BASELINE_DIR / "coverage_floors.json"
COVERAGE_REPORT = ROOT / "coverage.json"
ESCAPE_HATCH_BASELINE = BASELINE_DIR / "escape_hatches.json"
UNTESTED_MODULES_BASELINE = BASELINE_DIR / "untested_modules.json"
TYPE_COMPLETENESS_BASELINE = BASELINE_DIR / "type_completeness.json"

# Code lines (non-blank, not comment-only) a Python file may have. Long files
# are where complexity hides and where reviewers stop reading.
MAX_CODE_LINES = 600
# Line coverage a file needs when it has no recorded floor yet.
NEW_FILE_COVERAGE_FLOOR = 50.0

UPDATE_HELP = "Rewrite the baseline from the current tree instead of checking"

# Comments that switch a checker off for one line. Each is sometimes right,
# and each is free to add, so the count per file is budgeted and only shrinks.
ESCAPE_HATCHES = {
    "noqa": re.compile(r"#\s*noqa\b"),
    "type: ignore": re.compile(r"#\s*type:\s*ignore\b"),
    "pyright: ignore": re.compile(r"#\s*pyright:\s*ignore\b"),
    "nosec": re.compile(r"#\s*nosec\b"),
    "pragma: no cover": re.compile(r"#\s*pragma:\s*no cover\b"),
}
# The packages whose modules must each be reached by name from tests/.
TESTED_PACKAGES = ("tiny_audio", "scripts")
# The packages `pyright --verifytypes` scores; both ship py.typed in the wheel.
TYPED_PACKAGES = ("tiny_audio", "scripts")

# A call to one of these counts as an assertion: the code under test is itself
# a guard, so "it did not raise" is the claim (`_assert_audio_token_counts`,
# `_require_fused_cross_entropy`, `_check_remote_disk`, ...).
_GUARD_NAME = re.compile(r"_*(assert|require|check|ensure|validate|verify)")
_PYTEST_ASSERTIONS = frozenset({"raises", "warns", "fail", "deprecated_call"})


def _load(path: Path) -> dict[str, Any]:
    """A baseline file's contents, or `{}` when it does not exist yet."""
    data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as sorted, 2-space JSON (the layout `pretty-format-json` checks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _tracked_python_files() -> list[str]:
    """Every tracked `.py` file, relative to the repo root."""
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
    problems: list[str] = []
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


def file_coverage(report: dict[str, Any]) -> dict[str, float]:
    """Line coverage percent per file from a coverage.py JSON report."""
    out: dict[str, float] = {}
    files: dict[str, dict[str, Any]] = report.get("files", {})
    for path, data in files.items():
        summary: dict[str, int] = data["summary"]
        statements = summary["num_statements"]
        out[path] = 100.0 if statements == 0 else 100.0 * summary["covered_lines"] / statements
    return out


def coverage_violations(coverage: dict[str, float], floors: dict[str, float]) -> list[str]:
    """Files below their recorded floor, or new files below NEW_FILE_COVERAGE_FLOOR."""
    problems: list[str] = []
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
    """The called name of `call`: `f` for `f()` and `obj.f()`, else empty."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _asserts_directly(func: ast.AST) -> bool:
    """Whether `func`'s own body asserts, raises-checks or calls a guard."""
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
    problems: list[str] = []
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    for path in tests:
        for line, name in tests_without_assertions(path.read_text()):
            problems.append(f"{path.relative_to(ROOT)}:{line}: {name} asserts nothing")
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        raise typer.Exit(1)
    typer.echo(f"test-assertions: every test in {len(tests)} files asserts")


# ---------------------------------------------------------------- escape hatches


def count_escape_hatches(source: str) -> dict[str, int]:
    """Escape-hatch comments in `source`, by kind; only real comments count, not strings."""
    counts: dict[str, int] = {}
    for token in tokenize.generate_tokens(StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        for kind, pattern in ESCAPE_HATCHES.items():
            if pattern.search(token.string):
                counts[kind] = counts.get(kind, 0) + 1
    return counts


def escape_hatch_counts() -> dict[str, dict[str, int]]:
    """`{kind: {file: count}}` over every tracked Python file."""
    out: dict[str, dict[str, int]] = {}
    for path in _tracked_python_files():
        for kind, n in count_escape_hatches((ROOT / path).read_text()).items():
            out.setdefault(kind, {})[path] = n
    return out


def escape_hatch_violations(
    counts: dict[str, dict[str, int]], budget: dict[str, dict[str, int]]
) -> list[str]:
    """Files holding more of a kind than their budget (no entry means a budget of 0)."""
    problems: list[str] = []
    for kind, files in sorted(counts.items()):
        for path, n in sorted(files.items()):
            allowed = budget.get(kind, {}).get(path, 0)
            if n > allowed:
                problems.append(f"{path}: {n} `# {kind}` comments, budget is {allowed}")
    return problems


@app.command("escape-hatches")
def escape_hatches(
    update: Annotated[
        bool, typer.Option("--update", help="Lower budgets to the current tree")
    ] = False,
) -> None:
    """Per-file budgets for `# noqa`, `# pyright: ignore` and friends; they only go down.

    `--update` lowers budgets and refuses to raise one: an increase is a hand
    edit of quality/escape_hatches.json, so it shows up in review.
    """
    counts = escape_hatch_counts()
    problems = escape_hatch_violations(counts, _load(ESCAPE_HATCH_BASELINE))
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        typer.echo("Fix the finding instead of silencing it.", err=True)
        raise typer.Exit(1)
    if update:
        _save(ESCAPE_HATCH_BASELINE, counts)
        typer.echo(f"Recorded {ESCAPE_HATCH_BASELINE.relative_to(ROOT)}")
        return
    total = sum(sum(files.values()) for files in counts.values())
    typer.echo(f"escape-hatches: {total} within budget")


# ------------------------------------------------------------------ module tests


def _modules_in_scope() -> list[str]:
    """Dotted names of every non-`__init__` module under TESTED_PACKAGES."""
    return sorted(
        f.removesuffix(".py").replace("/", ".")
        for f in _tracked_python_files()
        if f.split("/", 1)[0] in TESTED_PACKAGES and not f.endswith("__init__.py")
    )


def referenced_names(source: str) -> set[str]:
    """Dotted names a test file reaches: its imports and any dotted string literal.

    Strings count because `monkeypatch.setattr("scripts.x.subprocess.call", ...)`
    and `importlib.import_module("scripts.x")` are how a test reaches a module
    that is loaded lazily.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def untested_modules(modules: list[str], references: set[str]) -> list[str]:
    """Modules no test names directly (incidental coverage does not count)."""
    return [m for m in modules if not any(r == m or r.startswith(f"{m}.") for r in references)]


@app.command("module-tests")
def module_tests(
    update: Annotated[
        bool, typer.Option("--update", help="Drop modules that now have tests from the list")
    ] = False,
) -> None:
    """Every module must be imported (or named) by some test, not just run through.

    Per-file coverage can be met by a test written for another module walking
    through this one. A test that names the module is a claim about it.
    """
    references: set[str] = set()
    for path in sorted((ROOT / "tests").rglob("*.py")):
        references |= referenced_names(path.read_text())
    untested = untested_modules(_modules_in_scope(), references)
    allowed = set(json.loads(UNTESTED_MODULES_BASELINE.read_text()))
    new = [m for m in untested if m not in allowed]
    for module in new:
        typer.echo(f"{module}: no test imports it; add tests/test_<name>.py", err=True)
    if new:
        raise typer.Exit(1)
    if update:
        UNTESTED_MODULES_BASELINE.write_text(json.dumps(untested, indent=2) + "\n")
        typer.echo(f"Recorded {UNTESTED_MODULES_BASELINE.relative_to(ROOT)}")
        return
    stale = sorted(allowed - set(untested))
    if stale:
        typer.echo(f"Now tested, run `--update` to drop them: {', '.join(stale)}", err=True)
        raise typer.Exit(1)
    typer.echo(f"module-tests: {len(untested)} grandfathered modules without a test")


# ------------------------------------------------------------- type completeness


def completeness_percent(report: dict[str, Any]) -> float:
    """`pyright --verifytypes --outputjson`'s completeness score, as a percent."""
    return 100.0 * float(report["typeCompleteness"]["completenessScore"])


def verifytypes_score(package: str) -> float:
    """`pyright --verifytypes` completeness of `package`, as a percent."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--verifytypes",
            package,
            "--ignoreexternal",
            "--outputjson",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return completeness_percent(json.loads(proc.stdout))
    except (json.JSONDecodeError, KeyError) as exc:
        msg = f"pyright --verifytypes {package} gave no score: {exc}\n{proc.stderr}"
        raise ValueError(msg) from exc


@app.command("type-completeness")
def type_completeness(
    update: Annotated[bool, typer.Option("--update", help=UPDATE_HELP)] = False,
) -> None:
    """Share of each package's public symbols with a fully known type; it may only rise.

    Both packages ship `py.typed`, so their annotations are an API. Strict
    mode turns off the `reportUnknown*` rules, which this measures instead.
    """
    try:
        scores = {package: verifytypes_score(package) for package in TYPED_PACKAGES}
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if update:
        _save(TYPE_COMPLETENESS_BASELINE, {p: math.floor(s * 10) / 10 for p, s in scores.items()})
        typer.echo(f"Recorded {TYPE_COMPLETENESS_BASELINE.relative_to(ROOT)}")
        return
    floors = _load(TYPE_COMPLETENESS_BASELINE)
    problems = [
        f"{package}: {score:.1f}% type-complete, below {float(floors.get(package, 0.0)):.1f}%"
        for package, score in scores.items()
        if score + 1e-9 < float(floors.get(package, 0.0))
    ]
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        typer.echo("Annotate the new public symbols; never lower the floor.", err=True)
        raise typer.Exit(1)
    summary = ", ".join(f"{p} {s:.1f}%" for p, s in scores.items())
    typer.echo(f"type-completeness: {summary}")


# --------------------------------------------------------------- tracked-ignored


@app.command("tracked-ignored")
def tracked_ignored() -> None:
    """No tracked file may match a .gitignore rule.

    Such a file stays tracked but drops out of `git status`, so later edits to
    it are never noticed or committed.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--ignored", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    shadowed = out.stdout.split()
    for path in shadowed:
        typer.echo(f"{path}: tracked but matched by a .gitignore rule", err=True)
    if shadowed:
        raise typer.Exit(1)
    typer.echo("tracked-ignored: no tracked file is ignored")


app.command("invariants")(invariants.check)


if __name__ == "__main__":
    app()
