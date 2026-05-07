#!/usr/bin/env python3
"""Development commands for tiny-audio."""

import subprocess
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="dev",
    help="Development commands (lint, test, format, etc.)",
    no_args_is_help=True,
)
console = Console()

CODE_PATHS = ["tiny_audio", "scripts", "tests"]
LIB_PATH = "tiny_audio"

LINT_COMMANDS = [
    ["poetry", "check"],
    ["ruff", "check", *CODE_PATHS],
    ["yamllint", "configs/"],
    ["taplo", "check", "pyproject.toml"],
]
TYPE_CHECK_COMMANDS = [
    ["mypy", LIB_PATH],
    ["pyright", LIB_PATH],
]
SECURITY_COMMAND = ["bandit", "-c", "pyproject.toml", "-r", "tiny_audio", "scripts", "-ll"]
DOCSTRINGS_COMMAND = ["interrogate", LIB_PATH, "--fail-under", "50"]
CHECK_COMMANDS = [*LINT_COMMANDS, *TYPE_CHECK_COMMANDS, SECURITY_COMMAND, DOCSTRINGS_COMMAND]


def run(*args: str) -> int:
    """Run a command and return exit code."""
    console.print(f"[dim]$ {' '.join(args)}[/dim]")
    return subprocess.call(args)


def run_all(*commands: list[str]) -> int:
    """Run multiple commands, stopping on first failure."""
    for cmd in commands:
        code = run(*cmd)
        if code != 0:
            return code
    return 0


@app.command()
def lint():
    """Run linters (Poetry + Python + YAML + TOML)."""
    raise typer.Exit(run_all(*LINT_COMMANDS))


@app.command("format")
def format_code():
    """Format code with black, ruff, and mdformat."""
    run("black", *CODE_PATHS)
    run("ruff", "format", *CODE_PATHS)
    run("ruff", "check", "--fix", *CODE_PATHS)

    # Use git ls-files so we only format tracked markdown — naturally skips
    # worktrees, build/cache/checkpoint dirs, and vendored docs that gitignore
    # already excludes. (Pre-rglob version mis-handled `.claude/worktrees`,
    # `swift/.build`, etc.)
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    )
    md_excludes = ("docs/course/",)
    md_files = [
        line
        for line in tracked.stdout.splitlines()
        if line
        and not any(line.startswith(p) for p in md_excludes)
        and Path(line).name != "MODEL_CARD.md"
    ]
    if md_files:
        run("mdformat", *md_files)


@app.command("type-check")
def type_check():
    """Run type checkers (mypy and pyright)."""
    raise typer.Exit(run_all(*TYPE_CHECK_COMMANDS))


@app.command()
def test():
    """Run pytest tests."""
    raise typer.Exit(run("pytest", "-v"))


@app.command()
def coverage():
    """Run tests with coverage report."""
    raise typer.Exit(
        run(
            "pytest",
            f"--cov={LIB_PATH}",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v",
        )
    )


@app.command()
def check():
    """Run all checks (lint + type-check + security + docstrings)."""
    raise typer.Exit(run_all(*CHECK_COMMANDS))


@app.command()
def build():
    """Build package (wheel and sdist)."""
    raise typer.Exit(run("poetry", "build"))


@app.command()
def precommit():
    """Pre-commit quality gate (format, lint, type-check, security, docstrings, test, build)."""
    format_code()
    raise typer.Exit(run_all(*CHECK_COMMANDS, ["pytest", "-v"], ["poetry", "build"]))


@app.command("install-hooks")
def install_hooks():
    """Install pre-commit hooks."""
    raise typer.Exit(run("pre-commit", "install"))


@app.command()
def security():
    """Run security checks with bandit."""
    raise typer.Exit(run(*SECURITY_COMMAND))


@app.command("dead-code")
def dead_code():
    """Find dead/unused code with vulture."""
    raise typer.Exit(run("vulture", "tiny_audio", "scripts", "--min-confidence", "80"))


@app.command()
def docstrings():
    """Check docstring coverage with interrogate."""
    raise typer.Exit(run("interrogate", LIB_PATH, "-v", "--fail-under", "50"))


SWIFT_RESOURCES_MODEL_DIR = Path("swift/Sources/TinyAudio/Resources/Model")


@app.command("build-swift-weights")
def build_swift_weights(
    source_repo: str = typer.Option(
        "mazesmazes/tiny-audio-embedded",
        "--source-repo",
        help="Tiny-audio embedded checkpoint to load encoder + projector from.",
    ),
    dest: Path = typer.Option(
        SWIFT_RESOURCES_MODEL_DIR,
        "--dest",
        help="Where to install the bundle. Default: the Swift package's Resources/Model.",
    ),
    push: bool = typer.Option(
        False, "--push", help="Also push the bundle to a HuggingFace Hub repo."
    ),
    target_repo: str = typer.Option(
        "mazesmazes/tiny-audio-mlx",
        "--target-repo",
        help="Hub repo to push to when --push is set.",
    ),
    unquantized: bool = typer.Option(
        False,
        "--unquantized",
        help=(
            "Ship fp16 encoder + decoder weights with no quantization. "
            "For Swift↔PyTorch equivalence testing — Swift's loader skips "
            "quantize() when the bundle config has no quantization block."
        ),
    ),
):
    """Build the MLX bundle (encoder + projector + decoder + manifest) and install
    it into the Swift package so `Transcriber.load()` finds it.

    One command for the full Swift-weights build: re-quantizes the encoder,
    re-saves the projector, extracts and mlx_lm-quantizes the decoder LM
    from the source checkpoint at 8-bit / group_size=64 (same recipe for
    both frozen-LM and full-decoder-fine-tune checkpoints), writes
    config.json + manifest.json, and copies everything into
    ``swift/Sources/TinyAudio/Resources/Model``. Pass --push to also publish
    to the Hub.
    """
    import shutil
    import tempfile

    from scripts.mlx.convert_decoder import convert_decoder
    from scripts.publish_mlx_bundle import build_bundle

    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tiny-audio-mlx-") as work_str:
        work = Path(work_str)

        decoder_dir = work / "decoder-mlx"
        if unquantized:
            console.print(
                f"[bold yellow]Converting decoder[/bold yellow] from {source_repo} "
                f"(fp16, no quantization — equivalence-test mode)"
            )
        else:
            console.print(f"[bold]Converting decoder[/bold] from {source_repo} (8-bit / group=64)")
        convert_decoder(
            checkpoint=source_repo,
            out_dir=decoder_dir,
            q_bits=8,
            q_group_size=64,
            quantize=not unquantized,
        )

        console.print(f"[bold]Building bundle[/bold] in {work}")
        build_bundle(work, source_repo, str(decoder_dir), quantize=not unquantized)

        console.print(f"[bold]Installing[/bold] bundle into {dest}")
        for src in sorted(work.iterdir()):
            if src.is_file():
                shutil.copy(src, dest / src.name)

        if push:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(target_repo, exist_ok=True)
            console.print(f"[bold]Uploading[/bold] to {target_repo}")
            api.upload_folder(folder_path=str(work), repo_id=target_repo)
            console.print(f"[green]Pushed[/green] to https://huggingface.co/{target_repo}")

    installed = sorted(p.name for p in dest.iterdir() if p.is_file())
    console.print(f"[green]Done.[/green] {len(installed)} files at {dest}: {installed}")


def _register_handler():
    from scripts.deploy.handler_local import test as handler_test

    app.command(name="handler", help="Test inference endpoint handler locally")(handler_test)


_register_handler()

if __name__ == "__main__":
    app()
