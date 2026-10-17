"""Mutation testing: change one operator, re-run the tests, see if any noticed.

Line coverage says a line ran; it cannot say the line's result was checked.
A surviving mutant is a behaviour change no test objected to. Point this at
modules already near 100% coverage, where coverage has nothing left to say.

    ta dev mutate tiny_audio/projectors.py
    ta dev mutate scripts/eval/speaker_metrics.py --tests tests/test_speaker_metrics.py --max 40

Deliberately NOT part of `ta dev check`: a run takes minutes, and some
survivors are equivalent mutants (a change that cannot alter behaviour), so
the output is a to-do list that needs judgement, not a pass/fail gate.

The target file is rewritten in place for each mutant and restored afterwards,
including on Ctrl-C. Bytecode writing is disabled so a stale .pyc can never
stand in for a mutant.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from scripts.utils import get_project_root

ROOT = get_project_root()
TEST_TIMEOUT_S = 600

_SWAPS: dict[type[ast.AST], type[ast.AST]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.And: ast.Or,
    ast.Or: ast.And,
}


@dataclass(frozen=True)
class Mutant:
    """One applicable mutation: where it is and what it does."""

    index: int
    line: int
    description: str


def _skipped(tree: ast.Module) -> set[int]:
    """ids of nodes inside `if __name__ == "__main__":` or TYPE_CHECKING blocks."""
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and (
            "__main__" in ast.unparse(node.test) or "TYPE_CHECKING" in ast.unparse(node.test)
        ):
            skipped.update(id(n) for n in ast.walk(node))
    return skipped


def _sites(tree: ast.Module) -> Iterator[tuple[int, str, Callable[[], None]]]:
    """`(line, description, apply)` for every mutation site, in a fixed order."""
    skipped = _skipped(tree)
    for node in ast.walk(tree):
        if id(node) in skipped:
            continue
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in _SWAPS:
                    yield line, _describe(op), _replace_in(node.ops, i)
        elif isinstance(node, ast.BinOp | ast.AugAssign | ast.BoolOp) and type(node.op) in _SWAPS:
            yield line, _describe(node.op), _set_op(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            yield line, f"{node.value} -> {not node.value}", _set_value(node, not node.value)
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            yield line, f"{node.value} -> {node.value + 1}", _set_value(node, node.value + 1)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield line, "drop `not`", _set_op_to_uadd(node)


def _swap(op: ast.AST) -> Any:
    """The operator `op` swaps to (Any: the AST field types are per-node unions)."""
    return _SWAPS[type(op)]()


def _describe(op: ast.AST) -> str:
    return f"{type(op).__name__} -> {_SWAPS[type(op)].__name__}"


def _replace_in(ops: list[ast.cmpop], i: int) -> Callable[[], None]:
    def apply() -> None:
        ops[i] = _swap(ops[i])

    return apply


def _set_op(node: ast.BinOp | ast.AugAssign | ast.BoolOp) -> Callable[[], None]:
    def apply() -> None:
        node.op = _swap(node.op)

    return apply


def _set_value(node: ast.Constant, value: bool | int) -> Callable[[], None]:
    def apply() -> None:
        node.value = value

    return apply


def _set_op_to_uadd(node: ast.UnaryOp) -> Callable[[], None]:
    def apply() -> None:
        node.op = ast.UAdd()

    return apply


def mutants(source: str) -> list[Mutant]:
    """Every mutant `source` admits."""
    return [Mutant(i, line, desc) for i, (line, desc, _) in enumerate(_sites(ast.parse(source)))]


def mutate(source: str, index: int) -> str:
    """`source` with mutant `index` applied (comments and layout are not preserved)."""
    tree = ast.parse(source)
    for i, (_, _, apply) in enumerate(_sites(tree)):
        if i == index:
            apply()
            return ast.unparse(tree) + "\n"
    raise IndexError(index)


def default_tests(target: Path) -> list[Path]:
    """`tests/test_*<stem>*.py`, the usual home of a module's tests."""
    return sorted((ROOT / "tests").glob(f"test_*{target.stem}*.py"))


def tests_pass(tests: list[Path]) -> bool:
    """Run `tests` (fail fast, no coverage); a timeout counts as caught."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-o", "addopts=", "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(
            [*cmd, *map(str, tests)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            timeout=TEST_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def run(target: Path, tests: list[Path], limit: int | None) -> list[Mutant]:
    """Try each mutant of `target` against `tests`; return the survivors."""
    original = target.read_text()
    found = mutants(original)[:limit]
    survivors: list[Mutant] = []
    try:
        for n, mutant in enumerate(found, 1):
            target.write_text(mutate(original, mutant.index))
            survived = tests_pass(tests)
            verdict = "SURVIVED" if survived else "killed"
            typer.echo(f"[{n}/{len(found)}] line {mutant.line}: {mutant.description}: {verdict}")
            if survived:
                survivors.append(mutant)
    finally:
        target.write_text(original)
    return survivors


def main(
    target: Annotated[Path, typer.Argument(help="Python file to mutate", exists=True)],
    tests: Annotated[
        list[Path] | None,
        typer.Option("--tests", help="Test files to run (default: tests/test_*<stem>*.py)"),
    ] = None,
    limit: Annotated[int | None, typer.Option("--max", help="Stop after this many mutants")] = None,
) -> None:
    """Mutation-test one module; exits 1 if any mutant survives."""
    target = target.resolve()
    chosen = tests or default_tests(target)
    if not chosen:
        typer.echo(f"no tests found for {target.name}; pass --tests", err=True)
        raise typer.Exit(2)
    if not tests_pass(chosen):
        typer.echo("the tests fail before any mutation; fix them first", err=True)
        raise typer.Exit(2)
    survivors = run(target, chosen, limit)
    rel = target.relative_to(ROOT)
    for mutant in survivors:
        typer.echo(f"{rel}:{mutant.line}: survived: {mutant.description}", err=True)
    typer.echo(f"mutate: {len(survivors)} survivors")
    if survivors:
        raise typer.Exit(1)
