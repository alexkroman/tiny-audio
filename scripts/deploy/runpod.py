#!/usr/bin/env python3
"""Unified CLI for RunPod operations."""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from fabric import Connection
from invoke import UnexpectedExit

app = typer.Typer(help="RunPod remote operations CLI")

SSH_KEY_PATH = "~/.ssh/id_ed25519"


def _auto_session_name(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    return f"{prefix}_{timestamp}"


def _start_remote_tmux_script(
    conn: Connection,
    host: str,
    port: int,
    session_name: str,
    script_content: str,
    script_path: str,
    no_attach: bool,
) -> None:
    """Upload a script to /tmp, start it in a tmux session, optionally attach."""
    conn.run(f"cat > {script_path} << 'EOF'\n{script_content}\nEOF", hide=True)
    conn.run(f"chmod +x {script_path}", hide=True)
    result = conn.run(f"tmux new-session -d -s {session_name} {script_path}", warn=True)
    if not result.ok:
        print(f"Failed to start tmux session: {result.stderr}")
        sys.exit(1)
    print(f"\nSession '{session_name}' started.")
    print(f"To re-attach later: ssh -p {port} root@{host} -t 'tmux attach -t {session_name}'")
    if not no_attach:
        time.sleep(2)
        attach_tmux_session(host, port, session_name)


def get_connection(host: str, port: int) -> Connection:
    """Create a Fabric connection with standard SSH settings."""
    return Connection(
        host=host,
        user="root",
        port=port,
        connect_kwargs={
            "key_filename": str(Path(SSH_KEY_PATH).expanduser()),
            "look_for_keys": False,
            "allow_agent": False,
        },
        connect_timeout=10,
    )


def test_connection(conn: Connection) -> bool:
    """Test SSH connection to the remote host."""
    print(f"Testing SSH connection to {conn.host}:{conn.port}...")
    try:
        conn.run("echo Connected", hide=True)
        print("SSH connection successful!")
        return True
    except Exception as e:
        print(f"Failed to connect via SSH: {e}")
        return False


def list_tmux_sessions(conn: Connection) -> list[str]:
    """Get list of tmux session names on remote host."""
    try:
        result = conn.run('tmux list-sessions -F "#S" 2>/dev/null', hide=True, warn=True)
        if result.ok and result.stdout.strip():
            return result.stdout.strip().split("\n")
    except UnexpectedExit:
        pass
    return []


def kill_tmux_session(conn: Connection, session_name: str) -> bool:
    """Kill a tmux session by name. Returns True if killed, False if not found."""
    result = conn.run(f"tmux kill-session -t {session_name}", hide=True, warn=True)
    return result.ok


def get_tmux_logs(conn: Connection, session_name: str, lines: int = 100) -> str | None:
    """Capture recent output from a tmux session."""
    try:
        result = conn.run(
            f"tmux capture-pane -t '{session_name}' -p -S -{lines}",
            hide=True,
            warn=True,
        )
        if result.ok:
            return result.stdout
    except UnexpectedExit:
        pass
    return None


def attach_tmux_session(host: str, port: int, session_name: str) -> None:
    """Attach to a tmux session interactively (requires subprocess for TTY)."""
    print(f"\nAttaching to session '{session_name}'...")
    print("=" * 50)
    print("TMUX CONTROLS:")
    print("  - Detach (and leave running): Ctrl+B then D")
    print("  - Scroll Mode:              Ctrl+B then [ (use arrows, q to exit)")
    print("=" * 50)

    cmd = (
        f"ssh -i {SSH_KEY_PATH} -p {port} -o StrictHostKeyChecking=no "
        f"-t root@{host} \"tmux attach-session -t '{session_name}'\""
    )
    subprocess.run(cmd, shell=True)
    print(f"\nDetached from session '{session_name}'.")


# Suffixes filtered out of the rsync file set on top of gitignore. Belt-and-
# suspenders against accidentally syncing checkpoint weights into a RunPod
# workspace.
RSYNC_SUFFIX_BLOCKLIST = (".safetensors",)


def _gitignore_aware_file_list(project_root: Path) -> str:
    """Return newline-separated repo-relative paths git would track or add.

    Uses ``git ls-files --cached --others --exclude-standard`` so the rsync
    file set has exact gitignore semantics (including ``!`` un-ignore lines,
    which rsync's own ``:- .gitignore`` filter mishandles). Suffixes in
    ``RSYNC_SUFFIX_BLOCKLIST`` are then dropped on top.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    # Drop tracked-but-deleted paths — they're in --cached but missing on disk,
    # so rsync --files-from would skip them and exit 23.
    deleted = subprocess.run(
        ["git", "ls-files", "--deleted"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    deleted_set = set(deleted.stdout.splitlines())
    lines = [
        line
        for line in result.stdout.splitlines()
        if line and line not in deleted_set and not line.endswith(RSYNC_SUFFIX_BLOCKLIST)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def setup_remote_environment(conn: Connection) -> None:
    """Install system dependencies the base RunPod image is missing.

    apt-get install is idempotent, so this is a no-op when packages are already
    present in the image. Anything the image already ships (PyTorch, CUDA,
    Python, common libs) we leave alone.
    """
    print("\nSetting up remote environment...")
    conn.run("apt-get update -qq || true")
    # Required packages — training and corpus extraction need these.
    conn.run(
        "apt-get install -y -qq ffmpeg tmux rsync libsndfile1 unzip",
    )
    # Optional perf adds — fall back gracefully on pods where these aren't
    # available (e.g. minimal images, universe repo disabled, partial apt
    # cache). aria2 → faster RIRs/MUSAN download (urllib fallback exists);
    # pigz → parallel gzip for MUSAN tar (single-threaded fallback);
    # p7zip-full → parallel zip extract for OpenSLR-28 (unzip fallback);
    # zip → `zip -s-` reassembly of FSD50K's split archive;
    # portaudio19-dev → only needed for pyaudio runtime, never training.
    conn.run(
        "apt-get install -y aria2 pigz p7zip-full zip portaudio19-dev || true",
    )
    # Pin augmentation corpora onto the persistent /workspace volume so they
    # survive container restarts. ~/.cache/<name> becomes a symlink, which
    # keeps `corpus_path: ~/.cache/...` in production.yaml portable across
    # local dev and RunPod without per-environment config branches.
    conn.run(
        "mkdir -p /workspace/.cache/openslr-28 /workspace/.cache/musan "
        "/workspace/.cache/fsd50k /root/.cache && "
        "ln -sfn /workspace/.cache/openslr-28 /root/.cache/openslr-28 && "
        "ln -sfn /workspace/.cache/musan /root/.cache/musan && "
        "ln -sfn /workspace/.cache/fsd50k /root/.cache/fsd50k",
    )
    print("Remote environment setup complete!")


def sync_project(conn: Connection, project_root: Path) -> None:
    """Sync project files to the RunPod instance using rsync.

    File set comes from git so gitignore is honored exactly. ``--delete`` is
    omitted (it doesn't compose cleanly with ``--files-from`` and would
    require a parallel exclude set anyway); stale files on the remote are
    harmless for training. Wipe ``/workspace`` manually if needed.
    """
    print(f"\nSyncing project from {project_root}...")

    file_list = _gitignore_aware_file_list(project_root)
    if not file_list.strip():
        raise RuntimeError(f"git ls-files returned no files under {project_root}")

    rsync_cmd = (
        f"rsync -avz --no-owner --no-group --files-from=- "
        f'-e "ssh -i ~/.ssh/id_ed25519 -p {conn.port} -o StrictHostKeyChecking=no" '
        f"{project_root}/ root@{conn.host}:/workspace/"
    )

    subprocess.run(rsync_cmd, shell=True, check=True, input=file_list, text=True)
    print("Project synced successfully!")


def install_dependencies(conn: Connection) -> None:
    """Install Python dependencies on top of the base RunPod image.

    Trusts the base image to provide a CUDA-enabled PyTorch and Python; we only
    fill in the gaps (Poetry tooling + project deps).
    """
    print("\nInstalling Python dependencies...")

    setup_script = """\
#!/bin/bash
# `pipefail` ensures `pip ... | grep ...` fails when pip fails — without it,
# pip errors are masked by grep's exit code.
set -eo pipefail

export PATH="/root/.local/bin:$PATH"
export PIP_ROOT_USER_ACTION=ignore
export POETRY_VIRTUALENVS_CREATE=false
export PIP_BREAK_SYSTEM_PACKAGES=1

# Configure pip
mkdir -p /root/.config/pip
echo -e "[global]\\nbreak-system-packages = true" > /root/.config/pip/pip.conf

# Verify the base image actually ships a CUDA-enabled PyTorch — fail loudly
# rather than silently reinstalling a different version on top.
python -c "import torch; assert torch.cuda.is_available()" || {
    echo "ERROR: base image is missing a CUDA-enabled PyTorch. Pick a runpod/pytorch:* image." >&2
    exit 1
}

# Poetry tooling — only install what's missing
command -v poetry >/dev/null 2>&1 || pip install --user poetry
python -c "import poetry_plugin_export" 2>/dev/null || pip install --user poetry-plugin-export
poetry config virtualenvs.create false
poetry config installer.max-workers 10

# Project deps — pip skips packages already satisfied by the base image.
# `poetry export` fails fast when poetry.lock is out of sync with pyproject.toml;
# its stderr is the actionable error in that case ("Run `poetry lock` to fix").
cd /workspace
poetry export --only main --without-hashes | grep -v "^torch==" > /tmp/requirements.txt
pip install --user -r /tmp/requirements.txt

# Install project in editable mode
pip install --user -e . --no-deps

# Verify torch is available (from base image) — we never pin or replace torch.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Verify the user-site install actually landed where `ta` will look.
# `/root/.local/bin/ta` is a generated console script with a shebang pinned
# to the python pip used; if its interpreter can't import typer, then
# pip --user installed to a different python's user-site than the one ta
# runs under (e.g., python3.10 vs python3.11 in the base image), and
# every subsequent `ta dev <cmd>` will fail with `ModuleNotFoundError`.
TA_PYTHON=$(head -1 /root/.local/bin/ta | sed 's|^#!||')
if ! "$TA_PYTHON" -c "import typer, hydra, omegaconf, datasets, transformers" 2>/tmp/tiny_audio_import_check.err; then
    echo "ERROR: deps did not install into the python that /root/.local/bin/ta uses." >&2
    echo "  ta interpreter: $TA_PYTHON" >&2
    echo "  pip used:        $(which pip) ($(pip --version))" >&2
    echo "  python --user site: $(python -m site --user-site)" >&2
    echo "  ta-python --user site: $("$TA_PYTHON" -m site --user-site 2>&1)" >&2
    cat /tmp/tiny_audio_import_check.err >&2
    exit 1
fi
echo "Dependencies verified for $TA_PYTHON"
"""

    # Upload the script via a single-quoted heredoc so apostrophes, dollar
    # signs, and other shell metachars in the body are preserved verbatim.
    # This avoids the entire class of `bash -c '...'` quoting bugs.
    script_path = "/tmp/tiny_audio_install_deps.sh"
    log_path = "/tmp/tiny_audio_install.log"
    conn.run(
        f"cat > {script_path} << 'INSTALL_DEPS_EOF'\n{setup_script}\nINSTALL_DEPS_EOF",
        hide=True,
    )
    # Capture all output to a log file silently rather than streaming live.
    # Pip's progress bars + ANSI color codes corrupt the local TTY when piped
    # through Fabric. On failure we fetch the tail and print it as plain text.
    print(f"  (silent; remote log: {log_path})")
    try:
        conn.run(f"bash {script_path} > {log_path} 2>&1", hide=True)
    except UnexpectedExit:
        print(f"\n[install_dependencies] FAILED. Last 80 lines of {log_path}:\n")
        tail = conn.run(f"tail -n 80 {log_path}", hide=True, warn=True)
        sys.stdout.write(tail.stdout)
        sys.stdout.flush()
        raise
    print(f"Dependencies installed successfully! Full log: {log_path}")


def _download_corpus(conn: Connection, label: str, ta_subcommand: str) -> None:
    """Run ``ta dev <ta_subcommand>`` on the remote to fetch a corpus (idempotent).

    Output is captured silently to ``/tmp/tiny_audio_<subcommand>.log`` on
    the remote rather than streamed live — aria2c progress bars and rich
    Console color codes corrupt the local TTY when piped through Fabric.
    On failure, the tail of the log is fetched and printed as plain text.
    ``NO_COLOR``/``TERM=dumb`` belt-and-suspenders the suppression in case
    a future remote command does stream output.
    """
    log_path = f"/tmp/tiny_audio_{ta_subcommand}.log"
    print(f"\nDownloading {label}... (silent; remote log: {log_path})")
    cmd = (
        f'bash -lc "cd /workspace && '
        f"export PATH=/root/.local/bin:$PATH && "
        f"export NO_COLOR=1 TERM=dumb PYTHONUNBUFFERED=1 && "
        f'ta dev {ta_subcommand} > {log_path} 2>&1"'
    )
    try:
        conn.run(cmd, hide=True)
    except UnexpectedExit:
        print(f"\n[{label}] FAILED. Last 80 lines of {log_path}:\n")
        tail = conn.run(f"tail -n 80 {log_path}", hide=True, warn=True)
        sys.stdout.write(tail.stdout)
        sys.stdout.flush()
        raise
    print(f"{label} ready.")


def download_rirs(conn: Connection) -> None:
    """Download OpenSLR-28 to ~/.cache/openslr-28 on the remote."""
    _download_corpus(conn, "RIR corpus (OpenSLR-28)", "download-rirs")


def download_musan(conn: Connection) -> None:
    """Download MUSAN to ~/.cache/musan on the remote."""
    _download_corpus(conn, "noise corpus (MUSAN)", "download-musan")


def download_fsd50k(conn: Connection) -> None:
    """Download FSD50K eval split to ~/.cache/fsd50k on the remote."""
    _download_corpus(conn, "sound-event corpus (FSD50K)", "download-fsd50k")


def resample_fsd50k(conn: Connection) -> None:
    """Resample FSD50K .wav files to 16 kHz mono in-place on the remote.

    FSD50K ships at 44.1 kHz; the training pipeline runs at 16 kHz, so
    audiomentations resamples on every load — wasted CPU on every batch
    and a flood of "had to be resampled" warnings. Pre-resampling once
    here trades disk for runtime cost (the files get smaller too).

    Idempotent via a sentinel file: subsequent deploys see the sentinel
    and skip the work. ``ffprobe`` per-file lets the worker also skip any
    individual clip that's already 16 kHz.
    """
    target = "/root/.cache/fsd50k/FSD50K.eval_audio"
    sentinel = f"{target}.16k.done"
    script_path = "/tmp/tiny_audio_resample_fsd50k.sh"
    log_path = "/tmp/tiny_audio_resample-fsd50k.log"

    # Heredoc-uploaded script avoids double-quote escaping hell (find/xargs
    # plus per-file bash -c is hostile to single-line ssh quoting).
    script = f"""#!/bin/bash
set -eo pipefail

if [ -f "{sentinel}" ]; then
    echo "FSD50K already resampled (sentinel: {sentinel}). Skipping."
    exit 0
fi
if [ ! -d "{target}" ]; then
    echo "ERROR: FSD50K not found at {target}." >&2
    echo "Run 'ta dev download-fsd50k' (or remove --skip-fsd50k) first." >&2
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "ERROR: ffmpeg/ffprobe required. apt-get install ffmpeg." >&2
    exit 1
fi

total=$(find "{target}" -name '*.wav' | wc -l)
echo "Resampling $total FSD50K clips to 16 kHz mono in-place..."
export NO_COLOR=1 TERM=dumb

# `xargs -P $(nproc)` parallelizes per-clip ffmpeg across all cores.
# Per-clip: probe sample rate, skip if already 16 kHz, else re-encode
# to a sibling .tmp.wav and atomic-rename. -ac 1 forces mono (matches
# our pipeline; FSD50K mixes mono/stereo).
find "{target}" -name '*.wav' -print0 | \\
  xargs -0 -n 1 -P "$(nproc)" bash -c '
    f="$0"
    sr=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate \\
           -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null || echo unknown)
    if [ "$sr" = "16000" ]; then exit 0; fi
    ffmpeg -hide_banner -loglevel error -y -i "$f" -ar 16000 -ac 1 "$f.tmp.wav"
    mv "$f.tmp.wav" "$f"
  '

touch "{sentinel}"
echo "Done. Sentinel: {sentinel}"
"""

    print(f"\nResampling FSD50K to 16 kHz... (silent; remote log: {log_path})")
    conn.run(
        f"cat > {script_path} << 'RESAMPLE_FSD50K_EOF'\n{script}\nRESAMPLE_FSD50K_EOF",
        hide=True,
    )
    try:
        conn.run(f"bash {script_path} > {log_path} 2>&1", hide=True)
    except UnexpectedExit:
        print(f"\n[resample_fsd50k] FAILED. Last 80 lines of {log_path}:\n")
        tail = conn.run(f"tail -n 80 {log_path}", hide=True, warn=True)
        sys.stdout.write(tail.stdout)
        sys.stdout.flush()
        raise
    print(f"FSD50K resample complete. Sentinel: {sentinel}")


@app.command()
def deploy(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    skip_setup: bool = typer.Option(False, "--skip-setup", help="Skip remote environment setup"),
    skip_sync: bool = typer.Option(False, "--skip-sync", help="Skip project file sync"),
    skip_deps: bool = typer.Option(
        False, "--skip-deps", help="Skip Python dependency installation"
    ),
    skip_rirs: bool = typer.Option(
        False, "--skip-rirs", help="Skip OpenSLR-28 RIR corpus download"
    ),
    skip_musan: bool = typer.Option(False, "--skip-musan", help="Skip MUSAN noise corpus download"),
    skip_fsd50k: bool = typer.Option(
        False, "--skip-fsd50k", help="Skip FSD50K sound-event corpus download"
    ),
    skip_resample_fsd50k: bool = typer.Option(
        False,
        "--skip-resample-fsd50k",
        help="Skip resampling FSD50K to 16 kHz (skip if pre-resampled or already done)",
    ),
):
    """Deploy ASR project to a RunPod instance."""
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    project_root = Path(__file__).parent.parent.parent.absolute()

    if not skip_setup:
        setup_remote_environment(conn)

    if not skip_sync:
        sync_project(conn, project_root)

    if not skip_deps:
        install_dependencies(conn)

    if not skip_rirs:
        download_rirs(conn)

    if not skip_musan:
        download_musan(conn)

    if not skip_fsd50k:
        download_fsd50k(conn)

    if not skip_resample_fsd50k:
        resample_fsd50k(conn)

    print("\nDeployment finished!")
    print(f"To connect: ssh -i ~/.ssh/id_ed25519 -p {port} root@{host}")


def build_training_script(
    experiment: str,
    hf_token: str,
    wandb_run_id: str | None,
    wandb_resume: str | None,
    extra_args: list[str],
) -> str:
    """Generate the training script content."""
    wandb_exports = ""
    if wandb_run_id:
        wandb_exports += f'export WANDB_RUN_ID="{wandb_run_id}"\n'
    if wandb_resume:
        wandb_exports += f'export WANDB_RESUME="{wandb_resume}"\n'

    extra_args_str = " ".join(extra_args) if extra_args else ""

    return f"""#!/bin/bash
# NOTE: "set -e" intentionally removed so session stays active on crash for debugging

ulimit -n 65536
pip install hf_transfer --quiet --root-user-action=ignore
export PATH="/root/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_AUDIO_DECODER="soundfile"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="{hf_token}"
{wandb_exports}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_BENCHMARK=1
export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCH_DYNAMO_ALLOW_UNSPEC_INT_ON_NN_MODULE=1
export TORCH_CUDA_GRAPHS_ENABLED=0

cd /workspace
python -m scripts.train +experiments={experiment} {extra_args_str}
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "===== Training Completed Successfully ====="
else
    echo "===== Training Failed with exit code: $EXIT_CODE ====="
fi

echo "Training script finished. Session will remain active for inspection."
sleep infinity
"""


@app.command()
def train(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    experiment: str = typer.Option("mlp", "--experiment", "-e", help="Experiment config to run"),
    session_name: str | None = typer.Option(
        None, "--session-name", "-s", help="Custom tmux session name"
    ),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start session but don't attach"),
    force: bool = typer.Option(False, "--force", "-f", help="Kill existing session with same name"),
    wandb_run_id: str | None = typer.Option(None, "--wandb-run-id", help="W&B run ID to resume"),
    wandb_resume: Annotated[
        str | None,
        typer.Option("--wandb-resume", help="W&B resume mode: must, allow, or never"),
    ] = None,
    extra_args: Annotated[
        list[str] | None,
        typer.Argument(help="Extra Hydra overrides passed to training script"),
    ] = None,
):
    """Start training on a remote RunPod instance in a tmux session."""
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    if session_name is None:
        session_name = _auto_session_name(f"train_{experiment}")

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    hf_token = os.environ.get("HF_TOKEN", "")
    wandb_run_id = wandb_run_id or os.environ.get("WANDB_RUN_ID")
    wandb_resume = wandb_resume or os.environ.get("WANDB_RESUME")

    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")

    print(f"\nStarting training session '{session_name}' with experiment '{experiment}'...")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")

    _start_remote_tmux_script(
        conn,
        host,
        port,
        session_name,
        build_training_script(experiment, hf_token, wandb_run_id, wandb_resume, extra_args or []),
        f"/tmp/train_{session_name}.sh",
        no_attach,
    )


@app.command()
def attach(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    session_name: str | None = typer.Option(None, "--session-name", "-s", help="Tmux session name"),
    list_sessions: bool = typer.Option(False, "--list", "-l", help="List all sessions and exit"),
    logs: bool = typer.Option(False, "--logs", help="Show recent logs instead of attaching"),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of log lines to show"),
):
    """Attach to, list, or view logs from a tmux session on a remote RunPod instance."""
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    sessions = list_tmux_sessions(conn)

    if list_sessions:
        print("\nAvailable tmux sessions:")
        if not sessions:
            print("  No active sessions found.")
        else:
            for session in sessions:
                print(f"  - {session}")
        return

    if not session_name:
        if not sessions:
            print("\nNo active tmux sessions found. Start a training session first.")
            sys.exit(1)
        elif len(sessions) == 1:
            session_name = sessions[0]
            print(f"\nFound one active session: '{session_name}'. Proceeding automatically.")
        else:
            print("\nMultiple active sessions found. Please choose one:")
            for i, session in enumerate(sessions, 1):
                print(f"  {i}. {session}")
            try:
                choice = input(f"Enter number (1-{len(sessions)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    session_name = sessions[idx]
                else:
                    print("Invalid selection. Exiting.")
                    sys.exit(1)
            except (KeyboardInterrupt, ValueError):
                print("\nSelection cancelled. Exiting.")
                sys.exit(0)

    if logs:
        log_content = get_tmux_logs(conn, session_name, lines)
        if log_content:
            print("\n" + "=" * 50)
            print(log_content)
            print("=" * 50)
        else:
            print(f"Session '{session_name}' not found or an error occurred.")
    else:
        attach_tmux_session(host, port, session_name)


def build_sift_script(
    hf_token: str,
    output_repo: str,
    batch_size: int,
    max_samples: int | None,
    use_compile: bool,
    datasets: list[str] | None,
) -> str:
    """Generate the SIFT dataset generation script content."""
    max_samples_arg = f"--max-samples {max_samples}" if max_samples else ""
    compile_arg = "--compile" if use_compile else ""
    datasets_arg = f"--datasets {' '.join(datasets)}" if datasets else ""

    return f"""#!/bin/bash
# NOTE: "set -e" intentionally removed so session stays active on crash for debugging

ulimit -n 65536
pip install hf_transfer --quiet --root-user-action=ignore
export PATH="/root/.local/bin:$PATH"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="{hf_token}"

# A40 GPU optimizations (48GB VRAM)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_BENCHMARK=1

cd /workspace

python -m scripts.generate_sift_dataset \\
    --output-repo {output_repo} \\
    --batch-size {batch_size} \\
    {compile_arg} \\
    {max_samples_arg} \\
    {datasets_arg}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "===== SIFT Dataset Generation Completed Successfully ====="
else
    echo "===== SIFT Dataset Generation Failed with exit code: $EXIT_CODE ====="
fi

echo "Script finished. Session will remain active for inspection."
sleep infinity
"""


@app.command()
def sift(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    output_repo: str = typer.Option(
        "mazesmazes/sift-audio", "--output-repo", "-o", help="HuggingFace repo for output"
    ),
    session_name: str | None = typer.Option(
        None, "--session-name", "-s", help="Custom tmux session name"
    ),
    batch_size: int = typer.Option(128, "--batch-size", "-b", help="Batch size for generation"),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Max samples per dataset"
    ),
    datasets: Annotated[
        list[str] | None,
        typer.Option("--datasets", "-d", help="Specific datasets to process"),
    ] = None,
    use_compile: bool = typer.Option(
        False, "--compile", help="Use torch.compile for faster inference"
    ),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start session but don't attach"),
    force: bool = typer.Option(False, "--force", "-f", help="Kill existing session with same name"),
):
    """Generate SIFT datasets on a remote RunPod instance.

    Available datasets: crema-d, ravdess, tess, meld, loquacious
    """
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    if session_name is None:
        session_name = _auto_session_name("sift")

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")

    print(f"\nStarting SIFT generation session '{session_name}'...")
    print(f"Output repo: {output_repo}")
    print(f"Batch size: {batch_size}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    if datasets:
        print(f"Datasets: {', '.join(datasets)}")
    if use_compile:
        print("Using torch.compile for faster inference")

    _start_remote_tmux_script(
        conn,
        host,
        port,
        session_name,
        build_sift_script(hf_token, output_repo, batch_size, max_samples, use_compile, datasets),
        f"/tmp/sift_{session_name}.sh",
        no_attach,
    )


def build_eval_script(
    hf_token: str,
    model: str,
    datasets: list[str],
    max_samples: int | None,
    assemblyai_api_key: str | None,
    assemblyai_model: str,
    num_workers: int,
    streaming: bool,
    extra_args: list[str] | None,
) -> str:
    """Generate the eval script content."""
    max_samples_arg = f"--max-samples {max_samples}" if max_samples else ""
    datasets_arg = f"--datasets {' '.join(datasets)}" if datasets else ""
    streaming_arg = "--streaming" if streaming else ""
    workers_arg = f"--num-workers {num_workers}" if num_workers > 1 else ""
    assemblyai_model_arg = f"--assemblyai-model {assemblyai_model}"
    extra_args_str = " ".join(extra_args) if extra_args else ""

    assemblyai_export = ""
    if assemblyai_api_key:
        assemblyai_export = f'export ASSEMBLYAI_API_KEY="{assemblyai_api_key}"'

    return f"""#!/bin/bash
# NOTE: "set -e" intentionally removed so session stays active on crash for debugging

ulimit -n 65536
pip install hf_transfer modelscope --quiet --root-user-action=ignore
export PATH="/root/.local/bin:$PATH"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="{hf_token}"
{assemblyai_export}

# GPU optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace

python -m scripts.eval.cli \\
    --model {model} \\
    {datasets_arg} \\
    {max_samples_arg} \\
    {assemblyai_model_arg} \\
    {workers_arg} \\
    {streaming_arg} \\
    --output-dir /workspace/outputs \\
    {extra_args_str}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "===== Evaluation Completed Successfully ====="
else
    echo "===== Evaluation Failed with exit code: $EXIT_CODE ====="
fi

echo "Eval script finished. Session will remain active for inspection."
sleep infinity
"""


@app.command()
def eval(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    model: str = typer.Option(..., "--model", "-m", help="Model path/ID or 'assemblyai'"),
    datasets: Annotated[
        list[str] | None,
        typer.Option("--datasets", "-d", help="Datasets to evaluate on"),
    ] = None,
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Max samples per dataset"
    ),
    assemblyai_model: str = typer.Option(
        "universal-3-pro",
        "--assemblyai-model",
        help="AssemblyAI model (best, universal, universal-3-pro)",
    ),
    num_workers: int = typer.Option(
        1, "--num-workers", "-w", help="Number of parallel workers for API evaluations"
    ),
    streaming: bool = typer.Option(False, "--streaming", "-s", help="Use streaming evaluation"),
    session_name: str | None = typer.Option(
        None, "--session-name", help="Custom tmux session name"
    ),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start session but don't attach"),
    force: bool = typer.Option(False, "--force", "-f", help="Kill existing session with same name"),
    extra_args: Annotated[
        list[str] | None,
        typer.Argument(help="Extra arguments passed to eval script"),
    ] = None,
):
    """Run ASR evaluation on a remote RunPod instance.

    Examples:
        runpod eval host port -m mazesmazes/tiny-audio -d loquacious
        runpod eval host port -m assemblyai --assemblyai-model universal-3-pro -d loquacious -w 4
    """
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    if session_name is None:
        model_short = model.split("/")[-1] if "/" in model else model
        session_name = _auto_session_name(f"eval_{model_short}")

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    hf_token = os.environ.get("HF_TOKEN", "")
    assemblyai_api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")

    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")
    if model == "assemblyai" and not assemblyai_api_key:
        print(
            "Warning: ASSEMBLYAI_API_KEY environment variable not set (required for assemblyai model)."
        )

    if datasets is None:
        datasets = ["loquacious"]

    print(f"\nStarting eval session '{session_name}'...")
    print(f"Model: {model}")
    print(f"Datasets: {', '.join(datasets)}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    if num_workers > 1:
        print(f"Workers: {num_workers}")
    if streaming:
        print("Streaming mode enabled")

    _start_remote_tmux_script(
        conn,
        host,
        port,
        session_name,
        build_eval_script(
            hf_token,
            model,
            datasets,
            max_samples,
            assemblyai_api_key,
            assemblyai_model,
            num_workers,
            streaming,
            extra_args,
        ),
        f"/tmp/eval_{session_name}.sh",
        no_attach,
    )


@app.command()
def checkpoint(
    host: str = typer.Argument(..., help="Remote server IP or hostname"),
    port: int = typer.Argument(22, help="SSH port"),
):
    """Find the latest checkpoint on a remote training server."""
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    result = conn.run(
        "find /workspace/outputs -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -1",
        hide=True,
        warn=True,
    )

    if not result.ok:
        print(f"Error: {result.stderr}", file=sys.stderr)
        raise typer.Exit(1)

    ckpt = result.stdout.strip()
    if ckpt:
        print(ckpt)
    else:
        print("No checkpoints found", file=sys.stderr)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
