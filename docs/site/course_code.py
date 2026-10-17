#!/usr/bin/env python3
"""Check that every code example in the course and the docs still compiles.

Each fenced block in docs/course/*.md, and in every other tracked Markdown
file (README.md, MODEL_CARD.md, CONTRIBUTING.md, docs/*.md, the agent guides),
is checked by its language:

- `python`: compiled with `compile()`.
- `yaml`: parsed with `yaml.safe_load`.
- `bash`: syntax-checked with `bash -n`, after `<PLACEHOLDER>`s like `<HOST>`
  become plain words. Then each command line is checked against the repo:
  `ta ...` is parsed by the real CLI (unknown commands, options, or missing
  arguments fail), `python <file>` needs the file, and `+experiments=<name>`
  needs its config.

Untagged and `text` blocks are prose diagrams and are skipped.

    python docs/site/course_code.py               # everything
    python docs/site/course_code.py --syntax-only # no `ta` parsing, needs only PyYAML
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COURSE_DIR = ROOT / "docs" / "course"

# Configs the course has students write before using them (Class 2, Exercise 3).
COURSE_CREATED_CONFIGS = {"my_run"}

# A fence opens with optional list indent or blockquote markers, then ``` and a
# language tag; it closes with the same prefix and a bare ```.
_FENCE_OPEN = re.compile(r"^(?P<prefix>(?:> ?| {2,})*)```(?P<lang>\w*)\s*$")
_FENCE_CLOSE = re.compile(r"^(?:> ?| {2,})*```\s*$")
_PLACEHOLDER = re.compile(r"<([A-Za-z][\w-]*)>")
# Sample values for placeholders the CLI type-checks; any other `<NAME>` becomes `NAME`.
PLACEHOLDER_VALUES = {"port": "22"}
_EXPERIMENT = re.compile(r"\+experiments=(\S+)")


@dataclass(frozen=True)
class CodeBlock:
    """One fenced code block from a course page."""

    path: Path
    line: int  # 1-based line of the opening fence
    lang: str
    code: str

    @property
    def where(self) -> str:
        """`file:line` of the opening fence, relative to the repo root."""
        return f"{self.path.relative_to(ROOT)}:{self.line}"


def extract_blocks(path: Path) -> list[CodeBlock]:
    """Every fenced code block in a Markdown file, with its prefix stripped."""
    lines = path.read_text().splitlines()
    blocks: list[CodeBlock] = []
    i = 0
    while i < len(lines):
        match = _FENCE_OPEN.match(lines[i])
        if match is None or not match["lang"]:
            i += 1
            continue
        prefix_len = len(match["prefix"])
        end = i + 1
        while end < len(lines) and not _FENCE_CLOSE.match(lines[end]):
            end += 1
        body = [line[prefix_len:] for line in lines[i + 1 : end]]
        blocks.append(CodeBlock(path, i + 1, match["lang"], "\n".join(body) + "\n"))
        i = end + 1
    return blocks


def course_blocks() -> list[CodeBlock]:
    """Every tagged code block across the course pages."""
    return [block for page in sorted(COURSE_DIR.glob("*.md")) for block in extract_blocks(page)]


def doc_blocks() -> list[CodeBlock]:
    """Every tagged code block in tracked Markdown outside the course.

    These are what a reader copies first, and nothing else runs them.
    """
    listed = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    pages = [ROOT / p for p in listed if not (ROOT / p).is_relative_to(COURSE_DIR)]
    return [block for page in pages for block in extract_blocks(page)]


def fill_placeholders(code: str) -> str:
    """Replace each `<NAME>` with a sample value the CLI will accept."""
    return _PLACEHOLDER.sub(lambda m: PLACEHOLDER_VALUES.get(m[1].lower(), m[1]), code)


def shell_commands(code: str) -> list[list[str]]:
    """Each command of a shell block as argv, placeholders filled in and comments dropped."""
    joined = fill_placeholders(code.replace("\\\n", " "))
    commands: list[list[str]] = []
    for line in joined.splitlines():
        argv = shlex.split(line, comments=True)
        if argv:
            commands.append(argv)
    return commands


def check_python(block: CodeBlock) -> list[str]:
    """Compile a Python block."""
    try:
        compile(block.code, block.where, "exec")
    except SyntaxError as e:
        return [f"{block.where}: Python does not compile: {e}"]
    return []


def check_yaml(block: CodeBlock) -> list[str]:
    """Parse a YAML block."""
    try:
        yaml.safe_load(block.code)
    except yaml.YAMLError as e:
        return [f"{block.where}: YAML does not parse: {e}"]
    return []


def check_bash_syntax(block: CodeBlock) -> list[str]:
    """`bash -n` over a shell block."""
    code = fill_placeholders(block.code)
    result = subprocess.run(["bash", "-n"], input=code, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return [f"{block.where}: bash syntax error: {result.stderr.strip()}"]
    return []


def check_repo_references(block: CodeBlock) -> list[str]:
    """Scripts and experiment configs a shell block names must exist."""
    errors: list[str] = []
    for argv in shell_commands(block.code):
        script = argv[3] if argv[:3] == ["poetry", "run", "python"] and len(argv) > 3 else ""
        if script.endswith(".py") and not (ROOT / script).is_file():
            errors.append(f"{block.where}: no such script: {script}")
        for arg in argv:
            match = _EXPERIMENT.fullmatch(arg)
            if match is None:
                continue
            name = match[1]
            config = ROOT / "configs" / "experiments" / f"{name}.yaml"
            if name not in COURSE_CREATED_CONFIGS and not config.is_file():
                errors.append(f"{block.where}: no such experiment config: {name}")
    return errors


def check_ta_command(where: str, args: list[str]) -> list[str]:
    """Parse `ta <args>` with the real CLI, without running it."""
    # Imported here so `--syntax-only` runs with nothing but PyYAML installed.
    import typer  # noqa: PLC0415
    import typer.core  # noqa: PLC0415
    import typer.main  # noqa: PLC0415
    from typer import _click  # noqa: PLC0415

    from scripts.cli import app  # noqa: PLC0415

    command: _click.Command = typer.main.get_command(app)
    ctx = _click.Context(command, info_name="ta")
    rest = list(args)
    try:
        while isinstance(command, typer.core.TyperGroup):
            if not rest or rest[0] == "--help":  # a group's help needs no parsing
                return []
            name = rest.pop(0)
            sub = command.get_command(ctx, name)
            if sub is None:
                return [f"{where}: `ta {' '.join(args)}`: no such command {name!r}"]
            command = sub
            ctx = _click.Context(command, info_name=name, parent=ctx)
        with contextlib.redirect_stdout(io.StringIO()):  # swallow `--help` output
            command.make_context(ctx.info_name, rest, parent=ctx.parent)
    except typer.Exit:
        return []  # --help
    except _click.exceptions.UsageError as e:
        return [f"{where}: `ta {' '.join(args)}`: {e.format_message()}"]
    return []


def check_ta_commands(block: CodeBlock) -> list[str]:
    """Parse every `poetry run ta ...` line of a shell block."""
    errors: list[str] = []
    for argv in shell_commands(block.code):
        if argv[:3] == ["poetry", "run", "ta"]:
            errors += check_ta_command(block.where, argv[3:])
    return errors


def check_block(block: CodeBlock, *, syntax_only: bool = False) -> list[str]:
    """All checks that apply to one block; an empty list means it passes."""
    if block.lang == "python":
        return check_python(block)
    if block.lang == "yaml":
        return check_yaml(block)
    if block.lang == "bash":
        errors = check_bash_syntax(block)
        if not errors:
            errors = check_repo_references(block)
        if not errors and not syntax_only:
            errors = check_ta_commands(block)
        return errors
    return []


def main() -> int:
    """Check every course and doc code block and report failures."""
    parser = argparse.ArgumentParser(
        description="Check that every code example in the course and the docs still compiles."
    )
    parser.add_argument(
        "--syntax-only", action="store_true", help="Skip parsing `ta` commands with the CLI"
    )
    opts = parser.parse_args()
    if not opts.syntax_only:
        sys.path.insert(0, str(ROOT))

    blocks = course_blocks() + doc_blocks()
    errors = [e for block in blocks for e in check_block(block, syntax_only=opts.syntax_only)]
    for error in errors:
        print(error, file=sys.stderr)
    print(f"course and doc code: {len(blocks)} blocks checked, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
