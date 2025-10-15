#!/usr/bin/env python3
"""Development tools for tiny-audio project."""

import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=get_project_root())
    return result.returncode


def lint() -> int:
    """Run ruff linter."""
    return run_command(
        ["ruff", "check", "src", "scripts"],
        "Running Ruff Linter"
    )


def format_code() -> int:
    """Format code with black and ruff."""
    exit_code = 0

    # Run black
    code = run_command(
        ["black", "src", "scripts"],
        "Formatting with Black"
    )
    exit_code = max(exit_code, code)

    # Run ruff format
    code = run_command(
        ["ruff", "format", "src", "scripts"],
        "Formatting with Ruff"
    )
    exit_code = max(exit_code, code)

    # Run ruff check --fix
    code = run_command(
        ["ruff", "check", "--fix", "src", "scripts"],
        "Fixing with Ruff"
    )
    exit_code = max(exit_code, code)

    return exit_code


def type_check() -> int:
    """Run mypy type checker."""
    return run_command(
        ["mypy", "src"],
        "Running MyPy Type Checker"
    )


def test() -> int:
    """Run pytest tests."""
    return run_command(
        ["pytest", "-v"],
        "Running Tests with Pytest"
    )


def check_all() -> int:
    """Run all checks: lint, type-check, and test."""
    exit_code = 0

    print("\n" + "="*60)
    print("RUNNING ALL CHECKS")
    print("="*60 + "\n")

    # Run lint
    code = lint()
    exit_code = max(exit_code, code)

    # Run type-check
    code = type_check()
    exit_code = max(exit_code, code)

    # Run tests
    code = test()
    exit_code = max(exit_code, code)

    # Summary
    print("\n" + "="*60)
    print("CHECK SUMMARY")
    print("="*60)
    if exit_code == 0:
        print("✅ All checks passed!")
    else:
        print("❌ Some checks failed. Please review the output above.")
    print("="*60 + "\n")

    return exit_code


if __name__ == "__main__":
    # Allow running individual functions from command line
    if len(sys.argv) > 1:
        func_name = sys.argv[1]
        if func_name in globals() and callable(globals()[func_name]):
            sys.exit(globals()[func_name]())
        else:
            print(f"Unknown function: {func_name}")
            print("Available functions: lint, format_code, type_check, test, check_all")
            sys.exit(1)
    else:
        print("Available functions: lint, format_code, type_check, test, check_all")
        print("Run with: python scripts/dev_tools.py <function_name>")
