"""scripts/invariants.py: every rule passes on the repo and fires on a broken copy.

The second half is the self-test: a rule whose pattern stopped matching would
pass a broken tree exactly as it passes a clean one.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts import invariants, quality
from scripts.utils import get_project_root

ROOT = get_project_root()
FILES = [
    "pyproject.toml",
    "requirements.txt",
    ".github/workflows/ci.yml",
    ".claude/hooks/session-start.sh",
    ".claude/CLAUDE.md",
]
REV = "91ea1192a00e0ebb684400434375fbd2345c8449"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A copy of the files the rules read."""
    for rel in FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    # The guide names repo paths; give them something to resolve against.
    (tmp_path / ".claude" / "CLAUDE.md").write_text("See `pyproject.toml`.\n")
    return tmp_path


def edit(tree: Path, rel: str, old: str, new: str) -> None:
    path = tree / rel
    text = path.read_text()
    assert old in text, f"{old!r} not in {rel}; the fixture no longer matches the repo"
    path.write_text(text.replace(old, new, 1))


def test_every_rule_holds_on_the_repository() -> None:
    assert invariants.violations(ROOT) == []


def test_every_rule_holds_on_the_fixture(tree: Path) -> None:
    assert invariants.violations(tree) == []


Breakage = Callable[[Path], object]
BREAKAGES: list[tuple[Breakage, Callable[[Path], list[str]]]] = [
    (
        lambda t: edit(t, "requirements.txt", REV, "0" * 40),
        invariants.transformers_commit_agrees,
    ),
    (
        lambda t: edit(t, "requirements.txt", "transformers @ git+https", "transformers>=5 #"),
        invariants.transformers_commit_agrees,
    ),
    (
        lambda t: edit(t, "pyproject.toml", 'torchaudio = "~2.8.0"', 'torchaudio = "~2.9.0"'),
        invariants.torch_and_torchaudio_agree,
    ),
    (
        lambda t: edit(t, "pyproject.toml", 'version = ">=0.8.0"', 'version = ">=0.5.0"'),
        invariants.liger_kernel_floor,
    ),
    (
        lambda t: edit(t, "pyproject.toml", 'nltk = "<3.10"', 'nltk = "*"'),
        invariants.nltk_ceiling,
    ),
    (
        lambda t: edit(t, "pyproject.toml", 'nltk = "<3.10"', 'nltk = "<3.10"\ntilelang = "*"'),
        invariants.hybrid_kernels_stay_optional,
    ),
    (
        lambda t: edit(t, ".claude/hooks/session-start.sh", 'VERSION="2.', 'VERSION="9.'),
        invariants.poetry_version_agrees,
    ),
    (
        lambda t: edit(
            t, ".github/workflows/ci.yml", 'PYTHON_VERSION: "3.12"', 'PYTHON_VERSION: "3.11"'
        ),
        invariants.ci_python_matches_requires_python,
    ),
    (
        lambda t: edit(t, ".claude/CLAUDE.md", "`pyproject.toml`", "`scripts/gone.py`"),
        invariants.agent_guides_are_accurate,
    ),
    (
        lambda t: (t / ".claude" / "CLAUDE.md").write_text("x" * 40_001),
        invariants.agent_guides_are_accurate,
    ),
    (
        lambda t: (t / ".claude" / "CLAUDE.md").unlink(),
        invariants.agent_guides_are_accurate,
    ),
]


@pytest.mark.parametrize(("breakage", "rule"), BREAKAGES, ids=lambda x: getattr(x, "__name__", ""))
def test_each_rule_fires_on_its_breakage(
    tree: Path, breakage: Breakage, rule: Callable[[Path], list[str]]
) -> None:
    breakage(tree)
    assert rule(tree), f"{rule.__name__} missed the breakage"


def test_every_rule_has_a_breakage() -> None:
    assert {rule for _, rule in BREAKAGES} == set(invariants.RULES)


def test_guide_paths_ignore_globs_and_line_numbers(tree: Path) -> None:
    (tree / ".claude" / "CLAUDE.md").write_text(
        "`quality/*.json`, `pyproject.toml:12`, `ta dev check`, `.github/`\n"
    )
    assert invariants.agent_guides_are_accurate(tree) == []


def test_version_needs_a_number() -> None:
    with pytest.raises(ValueError, match="any"):
        invariants._version("any")


def test_command_reports_and_fails(monkeypatch: pytest.MonkeyPatch, tree: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(quality.app, ["invariants"]).exit_code == 0
    edit(tree, "pyproject.toml", 'nltk = "<3.10"', 'nltk = "*"')
    monkeypatch.setattr(invariants, "get_project_root", lambda: tree)
    result = runner.invoke(quality.app, ["invariants"])
    assert result.exit_code == 1
    assert "nltk_ceiling" in result.output
