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


def _gitignore_aware_file_list(project_root: Path) -> str:
    """Return newline-separated repo-relative paths git would track or add.

    Uses ``git ls-files --cached --others --exclude-standard`` so the rsync
    file set has exact gitignore semantics (including ``!`` un-ignore lines,
    which rsync's own ``:- .gitignore`` filter mishandles).
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def setup_remote_environment(conn: Connection) -> None:
    """Install system dependencies the base RunPod image is missing.

    apt-get install is idempotent, so this is a no-op when packages are already
    present in the image. Anything the image already ships (PyTorch, CUDA,
    Python, common libs) we leave alone.
    """
    print("\nSetting up remote environment...")
    conn.run("apt-get update -qq || true", hide=True)
    conn.run(
        "apt-get install -y -qq ffmpeg tmux rsync libsndfile1 portaudio19-dev aria2 unzip pigz",
        hide=True,
    )
    # Pin augmentation corpora onto the persistent /workspace volume so they
    # survive container restarts. ~/.cache/<name> becomes a symlink, which
    # keeps `corpus_path: ~/.cache/...` in production.yaml portable across
    # local dev and RunPod without per-environment config branches.
    conn.run(
        "mkdir -p /workspace/.cache/openslr-28 /workspace/.cache/musan /root/.cache && "
        "ln -sfn /workspace/.cache/openslr-28 /root/.cache/openslr-28 && "
        "ln -sfn /workspace/.cache/musan /root/.cache/musan",
        hide=True,
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
set -e

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

# Project deps — pip skips packages already satisfied by the base image
cd /workspace
poetry export --only main --without-hashes | grep -v "^torch==" > /tmp/requirements.txt
pip install --user -r /tmp/requirements.txt 2>&1 | grep -v "already satisfied" || true

# Install project in editable mode
pip install --user -e . --no-deps

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
"""

    # Upload the script via a single-quoted heredoc so apostrophes, dollar
    # signs, and other shell metachars in the body are preserved verbatim.
    # This avoids the entire class of `bash -c '...'` quoting bugs.
    script_path = "/tmp/tiny_audio_install_deps.sh"
    conn.run(
        f"cat > {script_path} << 'INSTALL_DEPS_EOF'\n{setup_script}\nINSTALL_DEPS_EOF",
        hide=True,
    )
    conn.run(f"bash {script_path}", hide=False)
    print("Dependencies installed successfully!")


def _download_corpus(conn: Connection, label: str, ta_subcommand: str) -> None:
    """Run ``ta dev <ta_subcommand>`` on the remote to fetch a corpus (idempotent)."""
    print(f"\nDownloading {label}...")
    conn.run(
        f'bash -lc "cd /workspace && export PATH=/root/.local/bin:$PATH && ta dev {ta_subcommand}"',
        hide=False,
    )
    print(f"{label} ready.")


def download_rirs(conn: Connection) -> None:
    """Download OpenSLR-28 to ~/.cache/openslr-28 on the remote."""
    _download_corpus(conn, "RIR corpus (OpenSLR-28)", "download-rirs")


def download_musan(conn: Connection) -> None:
    """Download MUSAN to ~/.cache/musan on the remote."""
    _download_corpus(conn, "noise corpus (MUSAN)", "download-musan")


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
