"""Reach every repo name through the module that defines it.

`from tiny_audio.asr_pipeline import PreparedChunk` works because
asr_pipeline happens to import PreparedChunk from asr_types; so does
`quality.subprocess.run`. Both couple the caller to another module's imports:
the day that module stops needing the name, an unrelated file breaks. Pyright
reports the same shape as `reportPrivateImportUsage` once a package ships
`py.typed`, but only when it treats the package as a library, which it does
not inside this project.

For the repo's own modules (TESTED_PACKAGES), this flags:

- `from M import N` where M binds N only by importing it;
- `M.N` (read, not assigned) where M is an imported repo module and N is bound
  in M only by import;
- a dotted string such as `"scripts.dev.subprocess.call"` that reaches *through*
  such a name. A string that ends at it (`"tiny_audio.asr_pipeline.ffmpeg_read"`)
  is allowed: that is a monkeypatch target rebinding the consumer's own name,
  which is how you patch where a name is used.

Package `__init__` facades are not exempt: the repo's own code imports from the
owning module even where a facade re-exports the name for outside users.

    ta dev quality owning-imports
"""

from __future__ import annotations

import ast
import subprocess
from functools import cache
from pathlib import Path

import typer

from scripts.utils import get_project_root

ROOT = get_project_root()
# Packages whose modules must be reached through their owners.
OWNED_PACKAGES = ("tiny_audio", "scripts")
# Trees whose code is checked: the quality gates' trees, plus the Space demo
# and the docs examples, which import from both packages too.
CHECKED_TREES = ("tiny_audio", "scripts", "tests", "demo", "docs")


def _tracked(pattern: str) -> list[str]:
    """Tracked files matching `pattern`, relative to the repo root."""
    out = subprocess.run(
        ["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.split()


def module_name(path: str) -> str:
    """`scripts/eval/cli.py` -> `scripts.eval.cli`; a package's `__init__.py` -> the package."""
    name = path.removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


@cache
def repo_modules() -> dict[str, Path]:
    """Every module of OWNED_PACKAGES, by dotted name."""
    return {
        module_name(f): ROOT / f for f in _tracked("*.py") if f.split("/", 1)[0] in OWNED_PACKAGES
    }


def bindings(source: str) -> tuple[set[str], set[str]]:
    """Top-level names `source` defines, and those it binds only by importing.

    Imports under `if TYPE_CHECKING:` and in `try`/`except` fallbacks count as
    imports; a name both imported and assigned counts as defined.
    """
    defined: set[str] = set()
    imported: set[str] = set()

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                defined.update(
                    n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)
                )
            elif isinstance(node, ast.Import):
                imported.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                visit(node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)

    visit(ast.parse(source).body)
    return defined, imported - defined


@cache
def imported_only(module: str) -> frozenset[str]:
    """Names `module` binds only by importing them."""
    return frozenset(bindings(repo_modules()[module].read_text())[1])


def _owned_prefix(dotted: str) -> tuple[str, list[str]] | None:
    """The longest repo module `dotted` starts with, and the remaining parts."""
    parts = dotted.split(".")
    for k in range(len(parts), 0, -1):
        module = ".".join(parts[:k])
        if module in repo_modules():
            return module, parts[k:]
    return None


def _scan_imports(tree: ast.Module, where: str) -> tuple[dict[str, str], list[str]]:
    """Local names bound to repo modules, and `from M import N` lines that skip N's owner."""
    modules = repo_modules()
    aliases: dict[str, str] = {}
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in modules:
                    top = a.name.split(".")[0]
                    aliases[a.asname or top] = a.name if a.asname else top
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                if f"{node.module}.{a.name}" in modules:
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
                elif node.module in modules and a.name in imported_only(node.module):
                    problems.append(
                        f"{where}:{node.lineno}: `from {node.module} import {a.name}`: "
                        f"{node.module} only imports {a.name}; import it from its own module"
                    )
    return aliases, problems


def _attribute_problem(node: ast.Attribute, aliases: dict[str, str], where: str) -> str | None:
    """`M.N` read through a repo module M that only imports N."""
    if not isinstance(node.ctx, ast.Load) or not isinstance(node.value, ast.Name):
        return None
    module = aliases.get(node.value.id)
    if module is None or node.attr not in imported_only(module):
        return None
    return (
        f"{where}:{node.lineno}: `{node.value.id}.{node.attr}` reaches through "
        f"{module}, which only imports {node.attr}"
    )


def _string_problem(node: ast.Constant, where: str) -> str | None:
    """A dotted string that passes *through* a name its module only imports."""
    text = node.value
    if not isinstance(text, str) or not text.startswith(OWNED_PACKAGES):
        return None
    if not text.replace(".", "").isidentifier():
        return None
    found = _owned_prefix(text)
    if found is None or len(found[1]) < 2 or found[1][0] not in imported_only(found[0]):
        return None
    return (
        f"{where}:{node.lineno}: {text!r} reaches through {found[0]}.{found[1][0]}; "
        "name the owner instead"
    )


def violations(source: str, where: str) -> list[str]:
    """Every place `source` reaches a repo name through a module that only imports it."""
    tree = ast.parse(source)
    aliases, problems = _scan_imports(tree, where)
    for node in ast.walk(tree):
        problem = None
        if isinstance(node, ast.Attribute):
            problem = _attribute_problem(node, aliases, where)
        elif isinstance(node, ast.Constant):
            problem = _string_problem(node, where)
        if problem:
            problems.append(problem)
    return problems


def check() -> None:
    """Repo names are reached through the module that defines them."""
    files = [f for f in _tracked("*.py") if f.split("/", 1)[0] in CHECKED_TREES]
    problems = [p for f in files for p in violations((ROOT / f).read_text(), f)]
    for problem in sorted(problems):
        typer.echo(problem, err=True)
    if problems:
        raise typer.Exit(1)
    typer.echo(f"owning-imports: {len(files)} files reach names through their owners")
