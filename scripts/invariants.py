"""Settled decisions and restated facts that must agree across the repo.

Several facts live in more than one file (the transformers commit, the Poetry
version, the torch pin) and several decisions were paid for once and are
recorded only as a comment (the liger-kernel floor, the nltk ceiling, keeping
the hybrid-attention kernels optional). A comment is enforced only while a
reviewer remembers it; each rule here turns one of them into a failure.

Every rule is a function from the repo root to a list of problems, so the
tests can hand each one a broken tree and see it fire: a rule that stopped
matching would otherwise look exactly like a clean tree.

    ta dev quality invariants
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from scripts.utils import get_project_root

Rule = Callable[[Path], list[str]]

# Auto-loaded agent guides: past this size a guide crowds out the task itself.
AGENT_GUIDE_MAX_CHARS = 40_000
AGENT_GUIDES = (".claude/CLAUDE.md", ".claude/skills/*/SKILL.md", ".claude/agents/*.md")
# Kernels that must stay out of the main dependencies (pyproject.toml's
# `hybrid-kernels` group explains: half-installed, fla crashes training).
OPTIONAL_KERNELS = ("causal-conv1d", "flash-linear-attention", "tilelang")

_VERSION = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_REQ_TRANSFORMERS_REV = re.compile(r"^transformers @ git\+\S+@([0-9a-f]{40})\s*$", re.MULTILINE)
_POETRY_VERSION = re.compile(r"""POETRY_VERSION[=:]\s*["']?([\d.]+)""")
_PYTHON_VERSION = re.compile(r"""PYTHON_VERSION:\s*["']?([\d.]+)""")
# A backticked token that names a repo path: has a slash or a known suffix, no
# spaces, globs, placeholders or URLs.
_GUIDE_PATH = re.compile(r"`([\w.\-/]+(?:/[\w.\-]*|\.(?:py|md|json|ya?ml|toml|sh|txt)))`")


def _version(text: str) -> tuple[int, int, int]:
    match = _VERSION.search(text)
    if match is None:
        raise ValueError(text)
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _dependencies(root: Path) -> dict[str, Any]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    deps: dict[str, Any] = pyproject["tool"]["poetry"]["dependencies"]
    return deps


def _spec(dep: Any) -> str:
    """The version constraint of a Poetry dependency (a string or a table)."""
    if isinstance(dep, dict):
        return str(dep.get("version", ""))
    return str(dep)


def transformers_commit_agrees(root: Path) -> list[str]:
    """pyproject.toml and requirements.txt (uploaded to the Hub) pin one transformers commit."""
    pinned = _dependencies(root)["transformers"]["rev"]
    match = _REQ_TRANSFORMERS_REV.search((root / "requirements.txt").read_text())
    if match is None:
        return ["requirements.txt: transformers is not pinned to a git commit"]
    if match[1] != pinned:
        return [f"requirements.txt pins transformers@{match[1][:7]}, pyproject @{pinned[:7]}"]
    return []


def torch_and_torchaudio_agree(root: Path) -> list[str]:
    """torchaudio's major.minor must match torch's, or its extension fails to load."""
    deps = _dependencies(root)
    torch, audio = _version(_spec(deps["torch"])), _version(_spec(deps["torchaudio"]))
    if torch[:2] != audio[:2]:
        return [f"torch {torch[0]}.{torch[1]} and torchaudio {audio[0]}.{audio[1]} differ"]
    return []


def liger_kernel_floor(root: Path) -> list[str]:
    """liger-kernel >= 0.8.0: older releases lack the Qwen3.5 patch (a ~29 GiB OOM)."""
    spec = _spec(_dependencies(root)["liger-kernel"])
    floor = re.search(r">=\s*([\d.]+)", spec)
    if floor is None or _version(floor[1]) < (0, 8, 0):
        return [f"liger-kernel {spec!r} admits releases without apply_liger_kernel_to_qwen3_5"]
    return []


def nltk_ceiling(root: Path) -> list[str]:
    """nltk < 3.10: 3.10 re-glues leading apostrophes ("get' em") in training labels."""
    spec = _spec(_dependencies(root)["nltk"])
    ceiling = re.search(r"<\s*([\d.]+)", spec)
    if ceiling is None or _version(ceiling[1]) > (3, 10, 0):
        return [f"nltk {spec!r} admits 3.10, which breaks label normalization"]
    return []


def hybrid_kernels_stay_optional(root: Path) -> list[str]:
    """The Qwen3.5 fast-path kernels never become main dependencies."""
    deps = _dependencies(root)
    return [
        f"{name} is a main dependency; keep it in `hybrid-kernels`"
        for name in OPTIONAL_KERNELS
        if name in deps
    ]


def poetry_version_agrees(root: Path) -> list[str]:
    """CI and the web-session hook install the same Poetry."""
    versions: dict[str, str] = {}
    for rel in (".github/workflows/ci.yml", ".claude/hooks/session-start.sh"):
        match = _POETRY_VERSION.search((root / rel).read_text())
        if match is None:
            return [f"{rel}: no POETRY_VERSION"]
        versions[rel] = match[1]
    if len(set(versions.values())) > 1:
        return [f"Poetry versions differ: {versions}"]
    return []


def ci_python_matches_requires_python(root: Path) -> list[str]:
    """CI tests on the one Python minor `requires-python` allows."""
    requires = tomllib.loads((root / "pyproject.toml").read_text())["project"]["requires-python"]
    match = _PYTHON_VERSION.search((root / ".github/workflows/ci.yml").read_text())
    if match is None:
        return [".github/workflows/ci.yml: no PYTHON_VERSION"]
    if not requires.startswith(f">={match[1]}"):
        return [f"CI runs Python {match[1]}, requires-python is {requires!r}"]
    return []


def agent_guides_are_accurate(root: Path) -> list[str]:
    """Agent guides exist, stay under the size cap, and name only paths that exist."""
    guides = [p for pattern in AGENT_GUIDES for p in sorted(root.glob(pattern))]
    if not (root / AGENT_GUIDES[0]).exists():
        return [f"{AGENT_GUIDES[0]} is missing"]
    problems: list[str] = []
    for guide in guides:
        text = guide.read_text()
        rel = guide.relative_to(root)
        if len(text) > AGENT_GUIDE_MAX_CHARS:
            problems.append(f"{rel}: {len(text)} chars, over {AGENT_GUIDE_MAX_CHARS}")
        problems.extend(
            f"{rel}: names `{path}`, which does not exist"
            for path in sorted(set(_GUIDE_PATH.findall(text)))
            if not (root / path.split(":", 1)[0]).exists()
        )
    return problems


RULES: list[Rule] = [
    transformers_commit_agrees,
    torch_and_torchaudio_agree,
    liger_kernel_floor,
    nltk_ceiling,
    hybrid_kernels_stay_optional,
    poetry_version_agrees,
    ci_python_matches_requires_python,
    agent_guides_are_accurate,
]


def violations(root: Path) -> list[str]:
    """Every rule's problems, each prefixed with the rule's name."""
    return [f"{rule.__name__}: {problem}" for rule in RULES for problem in rule(root)]


def check() -> None:
    """Restated facts agree and settled decisions hold (see scripts/invariants.py)."""
    problems = violations(get_project_root())
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        typer.echo(
            "Fix the drift; if a decision changed, change its rule in the same diff.", err=True
        )
        raise typer.Exit(1)
    typer.echo(f"invariants: {len(RULES)} rules hold")
