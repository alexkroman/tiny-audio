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
    md_skip_exact = {"MODEL_CARD.md", "demo/README.md"}
    md_files = [
        line
        for line in tracked.stdout.splitlines()
        if line and not any(line.startswith(p) for p in md_excludes) and line not in md_skip_exact
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

# FSD50K eval split — sound-event clips for AddShortNoises augmentation.
# Hosted as a split-zip on Zenodo: a .zip "header" part + .z01 continuation
# part that must be reassembled with `zip -s-` before unzip can read it.
# Eval (~7 GB unpacked, ~10K clips) is plenty for transient augmentation;
# the dev split is 5x larger and split across 6 parts.
FSD50K_EVAL_PARTS = [
    "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01",
    "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip",
]
FSD50K_DEFAULT_DIR = Path.home() / ".cache" / "fsd50k"


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    # Shell out to `unzip` / `tar`: 2-5x faster than Python's pure-Python
    # zipfile / tarfile on archives with many small files (OpenSLR-28's ~60K
    # simulated RIRs), and per-file stdout output makes long extractions
    # visibly progressing instead of looking hung.
    import shutil
    import subprocess

    name = archive_path.name
    if name.endswith(".zip"):
        # 7z extracts ZIP entries in parallel across cores, which is the win
        # for OpenSLR-28's ~60K-file simulated_rirs payload. Falls back to
        # single-threaded unzip when p7zip isn't installed.
        if shutil.which("7z"):
            subprocess.run(
                ["7z", "x", "-y", f"-o{target_dir}", str(archive_path)],
                check=True,
            )
        else:
            subprocess.run(
                ["unzip", "-o", str(archive_path), "-d", str(target_dir)],
                check=True,
            )
    elif name.endswith((".tar.gz", ".tgz")):
        # pigz parallelizes gzip across cores; cuts MUSAN extract from ~15 min to ~3 min on RunPod.
        # --no-same-owner: MUSAN's tar entries carry uid 60706:gid 21 (the maintainer's uid);
        # restoring ownership fails in containers without that uid mapping. We don't need it.
        decomp = "pigz" if shutil.which("pigz") else "gzip"
        subprocess.run(
            [
                "tar",
                f"--use-compress-program={decomp}",
                "--no-same-owner",
                "-xvf",
                str(archive_path),
                "-C",
                str(target_dir),
            ],
            check=True,
        )
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
    # urllib URL is a hardcoded https constant — bandit B310 false positive.
    with urllib.request.urlopen(url) as resp, dst.open("wb") as out:  # nosec B310
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

    # RunPod pins ~/.cache/<corpus> to /workspace/.cache/<corpus> via symlink.
    # If /workspace got wiped between runs the symlink dangles, and
    # `Path.mkdir(exist_ok=True)` raises FileExistsError on it (pathlib only
    # swallows EEXIST when is_dir() is true, which broken symlinks fail).
    # Resolve through and create the real target.
    if target_dir.is_symlink() and not target_dir.exists():
        target_dir.resolve().mkdir(parents=True, exist_ok=True)
    else:
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
    real_rirs_isotropic_noises / simulated_rirs subdirs.
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


@app.command("download-fsd50k")
def download_fsd50k(
    target_dir: Path = typer.Option(
        FSD50K_DEFAULT_DIR,
        "--target-dir",
        "-t",
        help="Directory to extract FSD50K eval split into (default: ~/.cache/fsd50k).",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if already present."),
):
    """Download FSD50K eval split (~7 GB, ~10K sound-event clips).

    Used by NoiseAugmentation's AddShortNoises step when
    ``short_noises_corpus_path`` points at the extracted
    FSD50K.eval_audio/ directory. FSD50K is a sound-event corpus
    (transients) — the right pool for short-noise augmentation, vs MUSAN
    which is stationary background noise.
    """
    import shutil
    import subprocess

    target_dir = target_dir.expanduser()
    sentinel = target_dir / "FSD50K.eval_audio"
    if sentinel.exists() and not force:
        wav_count = sum(1 for _ in sentinel.rglob("*.wav"))
        console.print(f"[green]Already present:[/green] {sentinel} ({wav_count} .wav files)")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # Zenodo split-zip: download both parts in parallel (aria2c parallelizes
    # within a file but not across files), then reassemble with `zip -s-`
    # before unzip can read the result.
    from concurrent.futures import ThreadPoolExecutor

    def _fetch(url: str) -> Path:
        part_path = target_dir / url.rsplit("/", 1)[-1]
        if not part_path.exists() or force:
            console.print(f"[bold]Downloading[/bold] {url} -> {part_path}")
            _http_download(url, part_path)
        return part_path

    with ThreadPoolExecutor(max_workers=len(FSD50K_EVAL_PARTS)) as ex:
        parts: list[Path] = list(ex.map(_fetch, FSD50K_EVAL_PARTS))

    main_archive = target_dir / "FSD50K.eval_audio.zip"
    combined = target_dir / "FSD50K.eval_audio.combined.zip"
    if not shutil.which("zip"):
        raise RuntimeError("`zip` CLI is required to reassemble FSD50K split archive.")
    console.print(f"[bold]Reassembling[/bold] split archive -> {combined}")
    subprocess.run(["zip", "-s-", str(main_archive), "-O", str(combined)], check=True)

    console.print(f"[bold]Extracting[/bold] {combined} -> {target_dir}")
    _extract_archive(combined, target_dir)

    for p in parts:
        p.unlink(missing_ok=True)
    combined.unlink(missing_ok=True)

    wav_count = sum(1 for _ in sentinel.rglob("*.wav"))
    console.print(
        f"[green]Done.[/green] {wav_count} sound-event clips at {sentinel}. "
        f"Set noise_augmentation.short_noises_corpus_path to this path."
    )


def _register_handler():
    from scripts.deploy.handler_local import test as handler_test

    app.command(name="handler", help="Test inference endpoint handler locally")(handler_test)


_register_handler()

if __name__ == "__main__":
    app()
