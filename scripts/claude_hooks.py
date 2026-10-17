"""Claude Code hooks (wired up in .claude/settings.json).

    python -m scripts.claude_hooks protect     # PreToolUse: refuse edits to generated files
    python -m scripts.claude_hooks lint-edit   # PostToolUse: format + lint the edited file

Each reads the hook payload (JSON) on stdin. Exit code 2 sends stderr back to
Claude: for PreToolUse it blocks the edit, for PostToolUse it is advice the
next turn sees. Anything else this module cannot judge exits 0.
"""

from __future__ import annotations

import json
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# Files only a tool may write. Hand edits to a lock file desync it from the
# resolver; hand edits to a ratchet baseline are how a floor gets lowered.
PROTECTED = {
    "poetry.lock": "regenerate it with `poetry lock`",
    "quality/*.json": (
        "baselines change only through `ta dev quality <check> --update`, which records"
        " improvements; raising a budget is the user's call, so ask them"
    ),
}
BLOCK = 2


def edited_path(payload: dict[str, Any], root: Path) -> str | None:
    """The edited file, relative to `root`, or None for a file outside it."""
    tool_input: dict[str, Any] = payload.get("tool_input") or {}
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not raw:
        return None
    path = Path(raw)
    path = path if path.is_absolute() else root / path
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def protect(payload: dict[str, Any], root: Path) -> tuple[int, str]:
    """Block edits to PROTECTED files."""
    rel = edited_path(payload, root)
    for pattern, how in PROTECTED.items():
        if rel is not None and fnmatch(rel, pattern):
            return BLOCK, f"{rel} is generated: {how}."
    return 0, ""


def lint_edit(payload: dict[str, Any], root: Path) -> tuple[int, str]:
    """Apply ruff's fixes and formatting to an edited Python file, then report what is left.

    Fixes first: a fix can leave a line the formatter would rewrap.
    """
    rel = edited_path(payload, root)
    if rel is None or not rel.endswith(".py") or not (root / rel).exists():
        return 0, ""
    ruff = [sys.executable, "-m", "ruff"]
    subprocess.run([*ruff, "check", "--fix-only", "--quiet", rel], cwd=root, check=False)
    subprocess.run([*ruff, "format", "--quiet", rel], cwd=root, check=False)
    lint = subprocess.run(
        [*ruff, "check", "--quiet", rel],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if lint.returncode != 0:
        return BLOCK, f"ruff check {rel}:\n{lint.stdout}{lint.stderr}"
    return 0, ""


HOOKS = {"protect": protect, "lint-edit": lint_edit}


def main(argv: list[str]) -> int:
    """Run the hook named by `argv[0]` on the payload from stdin."""
    if len(argv) != 1 or argv[0] not in HOOKS:
        print(f"usage: python -m scripts.claude_hooks {{{','.join(HOOKS)}}}", file=sys.stderr)
        return 1
    try:
        payload: dict[str, Any] = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    code, message = HOOKS[argv[0]](payload, Path.cwd())
    if message:
        print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
