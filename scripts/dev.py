"""Development commands for tiny-audio."""

import subprocess
import sys
from pathlib import Path


def run(*args: str) -> int:
    """Run a command and return exit code."""
    print(f"$ {' '.join(args)}")
    return subprocess.call(args)


def run_all(*commands: list[str]) -> int:
    """Run multiple commands, stopping on first failure."""
    for cmd in commands:
        code = run(*cmd)
        if code != 0:
            return code
    return 0


def lint() -> None:
    """Run linters (Poetry + Python + YAML + TOML)."""
    code = run_all(
        ["poetry", "check"],
        ["ruff", "check", "tiny_audio", "scripts", "tests"],
        ["yamllint", "-d", "relaxed", "configs/"],
        ["taplo", "check", "pyproject.toml"],
    )
    sys.exit(code)


def format() -> None:
    """Format code with black, ruff, and mdformat."""
    run("black", "tiny_audio", "scripts", "tests")
    run("ruff", "format", "tiny_audio", "scripts", "tests")
    run("ruff", "check", "--fix", "tiny_audio", "scripts", "tests")

    # Format markdown files
    md_files = [
        str(f)
        for f in Path().rglob("*.md")
        if ".venv" not in str(f) and "docs/course" not in str(f) and f.name != "MODEL_CARD.md"
    ]
    if md_files:
        run("mdformat", *md_files)


def type_check() -> None:
    """Run type checkers (mypy and pyright)."""
    code = run_all(
        ["mypy", "tiny_audio"],
        ["pyright", "tiny_audio"],
    )
    sys.exit(code)


def test() -> None:
    """Run fast pytest tests (skips slow model-loading tests)."""
    sys.exit(run("pytest", "-v"))


def test_slow() -> None:
    """Run only slow pytest tests (model-loading tests)."""
    sys.exit(run("pytest", "-v", "-m", "slow"))


def test_all() -> None:
    """Run all pytest tests including slow ones."""
    sys.exit(run("pytest", "-v", "-m", ""))


def coverage() -> None:
    """Run tests with coverage report."""
    sys.exit(
        run(
            "pytest",
            "--cov=tiny_audio",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v",
        )
    )


def check() -> None:
    """Run all checks (lint + type-check + security + docstrings)."""
    code = run_all(
        ["poetry", "check"],
        ["ruff", "check", "tiny_audio", "scripts", "tests"],
        ["yamllint", "-d", "relaxed", "configs/"],
        ["taplo", "check", "pyproject.toml"],
        ["mypy", "tiny_audio"],
        ["pyright", "tiny_audio"],
        ["bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"],
        ["interrogate", "tiny_audio", "--fail-under", "50"],
    )
    sys.exit(code)


def build() -> None:
    """Build package (wheel and sdist)."""
    sys.exit(run("poetry", "build"))


def precommit() -> None:
    """Pre-commit quality gate (format, lint, type-check, security, docstrings, test, build)."""
    # Format first (doesn't fail)
    format()

    # Then run all checks
    code = run_all(
        ["poetry", "check"],
        ["ruff", "check", "tiny_audio", "scripts", "tests"],
        ["yamllint", "-d", "relaxed", "configs/"],
        ["taplo", "check", "pyproject.toml"],
        ["mypy", "tiny_audio"],
        ["pyright", "tiny_audio"],
        ["bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"],
        ["interrogate", "tiny_audio", "--fail-under", "50"],
        ["pytest", "-v"],
        ["poetry", "build"],
    )
    sys.exit(code)


def install_hooks() -> None:
    """Install pre-commit hooks."""
    sys.exit(run("pre-commit", "install"))


def security() -> None:
    """Run security checks with bandit."""
    sys.exit(run("bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"))


def dead_code() -> None:
    """Find dead/unused code with vulture."""
    sys.exit(run("vulture", "tiny_audio", "scripts", "--min-confidence", "80"))


def docstrings() -> None:
    """Check docstring coverage with interrogate."""
    sys.exit(run("interrogate", "tiny_audio", "-v", "--fail-under", "50"))
