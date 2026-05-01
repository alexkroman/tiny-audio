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

    md_files = [
        str(f)
        for f in Path().rglob("*.md")
        if ".venv" not in str(f) and "docs/course" not in str(f) and f.name != "MODEL_CARD.md"
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


OPENSLR28_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
OPENSLR28_DEFAULT_DIR = Path.home() / ".cache" / "openslr-28"

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_DEFAULT_DIR = Path.home() / ".cache" / "musan"


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    import tarfile
    import zipfile

    name = archive_path.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {name}")


def _http_download(url: str, dst: Path) -> None:
    """Download ``url`` to ``dst``. Prefers aria2c (16-way parallel chunks) if
    available — openslr.org throttles per-connection, so single-stream urllib
    is much slower. Falls back to urllib when aria2c isn't installed.
    """
    import shutil
    import subprocess
    import urllib.request

    if shutil.which("aria2c"):
        result = subprocess.run(
            [
                "aria2c",
                "-x",
                "16",
                "-s",
                "16",
                "-k",
                "1M",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--summary-interval=10",
                "-d",
                str(dst.parent),
                "-o",
                dst.name,
                url,
            ],
            check=False,
        )
        if result.returncode == 0 and dst.exists():
            return
        console.print("[yellow]aria2c failed; falling back to urllib.[/yellow]")
    with urllib.request.urlopen(url) as resp, dst.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _download_corpus(
    url: str,
    target_dir: Path,
    archive_name: str,
    sentinel_subdir: str,
    asset_label: str,
    config_field: str,
    force: bool,
    post_extract=None,
) -> None:
    target_dir = target_dir.expanduser()
    sentinel = target_dir / sentinel_subdir
    if sentinel.exists() and not force:
        wav_count = sum(1 for _ in sentinel.rglob("*.wav"))
        console.print(f"[green]Already present:[/green] {sentinel} ({wav_count} .wav files)")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / archive_name

    console.print(f"[bold]Downloading[/bold] {url} -> {archive_path}")
    _http_download(url, archive_path)

    console.print(f"[bold]Extracting[/bold] {archive_path} -> {target_dir}")
    _extract_archive(archive_path, target_dir)
    if post_extract is not None:
        post_extract(target_dir)
    archive_path.unlink(missing_ok=True)

    wav_count = sum(1 for _ in sentinel.rglob("*.wav"))
    console.print(
        f"[green]Done.[/green] {wav_count} {asset_label} at {sentinel}. "
        f"Set {config_field} to this path."
    )


def _flatten_openslr28(target_dir: Path) -> None:
    import shutil

    extracted_root = target_dir / "RIRS_NOISES"
    if not extracted_root.exists():
        return
    for sub in ("real_rirs_isotropic_noises", "pointsource_noises", "simulated_rirs"):
        src = extracted_root / sub
        dst = target_dir / sub
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
    shutil.rmtree(extracted_root, ignore_errors=True)


@app.command("download-rirs")
def download_rirs(
    target_dir: Path = typer.Option(
        OPENSLR28_DEFAULT_DIR,
        "--target-dir",
        "-t",
        help="Directory to extract OpenSLR-28 into (default: ~/.cache/openslr-28).",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if already present."),
):
    """Download OpenSLR-28 (real RIRs + point-source noise, ~1 GB).

    Used by RIRAugmentation when ``corpus_path`` points at the extracted
    real_rirs_isotropic_noises subset.
    """
    _download_corpus(
        url=OPENSLR28_URL,
        target_dir=target_dir,
        archive_name="rirs_noises.zip",
        sentinel_subdir="real_rirs_isotropic_noises",
        asset_label="real RIRs",
        config_field="rir_augmentation.corpus_path",
        force=force,
        post_extract=_flatten_openslr28,
    )


@app.command("download-musan")
def download_musan(
    target_dir: Path = typer.Option(
        MUSAN_DEFAULT_DIR,
        "--target-dir",
        "-t",
        help="Directory to extract MUSAN into (default: ~/.cache/musan).",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if already present."),
):
    """Download MUSAN (real music + speech + noise, ~11 GB).

    Used by NoiseAugmentation when ``corpus_path`` points at the extracted
    musan/ directory (with ``music/``, ``speech/``, ``noise/`` subdirs).
    """
    _download_corpus(
        url=MUSAN_URL,
        target_dir=target_dir,
        archive_name="musan.tar.gz",
        sentinel_subdir="musan",
        asset_label="audio files",
        config_field="noise_augmentation.corpus_path",
        force=force,
    )


def _register_handler():
    from scripts.deploy.handler_local import test as handler_test

    app.command(name="handler", help="Test inference endpoint handler locally")(handler_test)


_register_handler()

if __name__ == "__main__":
    app()
