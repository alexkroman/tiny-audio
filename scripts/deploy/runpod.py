#!/usr/bin/env python3
"""
Unified CLI for RunPod operations.

Consolidates deployment, training, evaluation, session management, and checkpoint discovery.

Usage:
    runpod deploy <host> <port>          # Deploy code to remote
    runpod train <host> <port>           # Start training in tmux
    runpod eval <host> <port> -m <model> # Run evaluation in tmux
    runpod attach <host> <port>          # Attach to running session
    runpod checkpoint <host> <port>      # Find latest checkpoint
"""

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

# =============================================================================
# SSH/Tmux Utilities
# =============================================================================

SSH_KEY_PATH = "~/.ssh/id_ed25519"


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


# =============================================================================
# Deploy Command
# =============================================================================

RSYNC_EXCLUDES = [
    "__pycache__",
    "*.pyc",
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "./data/",
    "datasets_cache/",
    "outputs/",
    "logs/",
    "runs/",
    "wandb/",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
    "dist/",
    "build/",
    ".DS_Store",
    ".idea/",
    ".vscode/",
    "node_modules/",
    ".cache/",
    "datasets/",
    "checkpoints/",
    "*.ckpt",
    "*.pth",
    "*.pt",
]


def setup_remote_environment(conn: Connection) -> None:
    """Install system dependencies on the remote."""
    print("\nSetting up remote environment...")
    conn.run("apt-get update -qq || true", hide=True)
    conn.run(
        "apt-get install -y -qq ffmpeg tmux rsync libsndfile1 ninja-build portaudio19-dev",
        hide=True,
    )
    print("Remote environment setup complete!")


def sync_project(conn: Connection, project_root: Path) -> None:
    """Sync project files to the RunPod instance using rsync."""
    print(f"\nSyncing project from {project_root}...")

    excludes = " ".join(f"--exclude={e}" for e in RSYNC_EXCLUDES)
    rsync_cmd = (
        f"rsync -avz --delete --no-owner --no-group {excludes} "
        f'-e "ssh -i ~/.ssh/id_ed25519 -p {conn.port} -o StrictHostKeyChecking=no" '
        f"{project_root}/ root@{conn.host}:/workspace/"
    )

    subprocess.run(rsync_cmd, shell=True, check=True)
    print("Project synced successfully!")


def install_dependencies(conn: Connection) -> None:
    """Install Python dependencies from poetry.lock."""
    print("\nInstalling Python dependencies...")

    setup_commands = """
        export PATH="/root/.local/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
        export PIP_ROOT_USER_ACTION=ignore
        export POETRY_VIRTUALENVS_CREATE=false
        export PIP_BREAK_SYSTEM_PACKAGES=1

        # Configure pip
        mkdir -p /root/.config/pip
        echo -e "[global]\\nbreak-system-packages = true" > /root/.config/pip/pip.conf

        # Install Poetry if needed
        command -v poetry >/dev/null 2>&1 || python3 -m pip install --user poetry
        python3 -m pip install --user poetry-plugin-export

        # Configure Poetry
        poetry config virtualenvs.create false
        poetry config installer.max-workers 10

        # Install PyTorch with CUDA 12.8 if needed
        python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
            python3 -m pip install --user torch~=2.8.0 --index-url=https://download.pytorch.org/whl/cu128

        # Install cusparseLt library required by torch (may already be satisfied)
        python3 -m pip install --user nvidia-cusparselt-cu12 2>&1 | grep -v "already satisfied" || true

        # Fix torchvision version mismatch if present (torch 2.8 needs torchvision 0.23)
        python3 -m pip uninstall torchvision -y 2>/dev/null || true
        python3 -m pip install --user torchvision~=0.23.0 --index-url=https://download.pytorch.org/whl/cu128 2>&1 | grep -v "already satisfied" || true

        # Export and install dependencies (excluding torch to preserve CUDA-specific version)
        cd /workspace
        poetry export --only main --without-hashes | grep -v "^torch==" > /tmp/requirements.txt
        python3 -m pip install --user -r /tmp/requirements.txt 2>&1 | grep -v "already satisfied" || true

        # Install project in editable mode
        python3 -m pip install --user -e . --no-deps

        # Verify (warn on failure but do not abort)
        python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" || echo "WARNING: torch verification failed"
    """

    conn.run(f"bash -c '{setup_commands}'", hide=False)
    print("Dependencies installed successfully!")


@app.command()
def deploy(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    skip_setup: bool = typer.Option(False, "--skip-setup", help="Skip remote environment setup"),
    skip_sync: bool = typer.Option(False, "--skip-sync", help="Skip project file sync"),
    skip_deps: bool = typer.Option(
        False, "--skip-deps", help="Skip Python dependency installation"
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

    print("\nDeployment finished!")
    print(f"To connect: ssh -i ~/.ssh/id_ed25519 -p {port} root@{host}")


# =============================================================================
# Train Command
# =============================================================================


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
export PATH="/root/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:${{LD_LIBRARY_PATH:-}}"
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

    # Generate session name if not provided
    if session_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        session_name = f"train_{experiment}_{timestamp}"

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    # Get environment variables
    hf_token = os.environ.get("HF_TOKEN", "")
    wandb_run_id = wandb_run_id or os.environ.get("WANDB_RUN_ID")
    wandb_resume = wandb_resume or os.environ.get("WANDB_RESUME")

    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")

    # Build and upload training script
    script_content = build_training_script(
        experiment, hf_token, wandb_run_id, wandb_resume, extra_args or []
    )
    script_path = f"/tmp/train_{session_name}.sh"

    print(f"\nStarting training session '{session_name}' with experiment '{experiment}'...")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")

    # Write script to remote
    conn.run(f"cat > {script_path} << 'EOF'\n{script_content}\nEOF", hide=True)
    conn.run(f"chmod +x {script_path}", hide=True)

    # Start tmux session
    result = conn.run(f"tmux new-session -d -s {session_name} {script_path}", warn=True)
    if not result.ok:
        print(f"Failed to start tmux session: {result.stderr}")
        sys.exit(1)

    print(f"\nTraining started in session '{session_name}'.")
    print(f"To re-attach later: ssh -p {port} root@{host} -t 'tmux attach -t {session_name}'")

    if not no_attach:
        time.sleep(2)
        attach_tmux_session(host, port, session_name)


# =============================================================================
# Attach Command
# =============================================================================


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

    # Auto-select session if not specified
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


# =============================================================================
# SIFT Dataset Generation Command
# =============================================================================


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
        "mazesmazes/sift-audio-2", "--output-repo", "-o", help="HuggingFace repo for output"
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

    # Generate session name if not provided
    if session_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        session_name = f"sift_{timestamp}"

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    # Get environment variables
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")

    # Build and upload script
    script_content = build_sift_script(
        hf_token, output_repo, batch_size, max_samples, use_compile, datasets
    )
    script_path = f"/tmp/sift_{session_name}.sh"

    print(f"\nStarting SIFT generation session '{session_name}'...")
    print(f"Output repo: {output_repo}")
    print(f"Batch size: {batch_size}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    if datasets:
        print(f"Datasets: {', '.join(datasets)}")
    if use_compile:
        print("Using torch.compile for faster inference")

    # Write script to remote
    conn.run(f"cat > {script_path} << 'EOF'\n{script_content}\nEOF", hide=True)
    conn.run(f"chmod +x {script_path}", hide=True)

    # Start tmux session
    result = conn.run(f"tmux new-session -d -s {session_name} {script_path}", warn=True)
    if not result.ok:
        print(f"Failed to start tmux session: {result.stderr}")
        sys.exit(1)

    print(f"\nSIFT generation started in session '{session_name}'.")
    print(f"To re-attach later: ssh -p {port} root@{host} -t 'tmux attach -t {session_name}'")

    if not no_attach:
        time.sleep(2)
        attach_tmux_session(host, port, session_name)


# =============================================================================
# ClothoAQA Dataset Generation Command
# =============================================================================


def build_clothoaqa_script(
    hf_token: str,
    output_repo: str,
    batch_size: int,
    max_samples: int | None,
) -> str:
    """Generate the ClothoAQA dataset generation script content."""
    max_samples_arg = f"--max-samples {max_samples}" if max_samples else ""

    return f"""#!/bin/bash
# NOTE: "set -e" intentionally removed so session stays active on crash for debugging

ulimit -n 65536
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

python -m scripts.generate_clothoaqa_dataset \\
    --output-repo {output_repo} \\
    --batch-size {batch_size} \\
    {max_samples_arg}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "===== ClothoAQA Dataset Generation Completed Successfully ====="
else
    echo "===== ClothoAQA Dataset Generation Failed with exit code: $EXIT_CODE ====="
fi

echo "Script finished. Session will remain active for inspection."
sleep infinity
"""


@app.command()
def clothoaqa(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    output_repo: str = typer.Option(
        "mazesmazes/clothoaqa", "--output-repo", "-o", help="HuggingFace repo for output"
    ),
    session_name: str | None = typer.Option(
        None, "--session-name", "-s", help="Custom tmux session name"
    ),
    batch_size: int = typer.Option(1024, "--batch-size", "-b", help="Batch size for generation"),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Max samples per split"
    ),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start session but don't attach"),
    force: bool = typer.Option(False, "--force", "-f", help="Kill existing session with same name"),
):
    """Generate enhanced ClothoAQA dataset on a remote RunPod instance.

    Takes the CLAPv2/ClothoAQA dataset and generates LLM-augmented answers
    that are more detailed than simple yes/no responses.
    """
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    # Generate session name if not provided
    if session_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        session_name = f"clothoaqa_{timestamp}"

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    # Get environment variables
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")

    # Build and upload script
    script_content = build_clothoaqa_script(hf_token, output_repo, batch_size, max_samples)
    script_path = f"/tmp/clothoaqa_{session_name}.sh"

    print(f"\nStarting ClothoAQA generation session '{session_name}'...")
    print(f"Output repo: {output_repo}")
    print(f"Batch size: {batch_size}")
    if max_samples:
        print(f"Max samples: {max_samples}")

    # Write script to remote
    conn.run(f"cat > {script_path} << 'EOF'\n{script_content}\nEOF", hide=True)
    conn.run(f"chmod +x {script_path}", hide=True)

    # Start tmux session
    result = conn.run(f"tmux new-session -d -s {session_name} {script_path}", warn=True)
    if not result.ok:
        print(f"Failed to start tmux session: {result.stderr}")
        sys.exit(1)

    print(f"\nClothoAQA generation started in session '{session_name}'.")
    print(f"To re-attach later: ssh -p {port} root@{host} -t 'tmux attach -t {session_name}'")

    if not no_attach:
        time.sleep(2)
        attach_tmux_session(host, port, session_name)


# =============================================================================
# Jenny MIMI Dataset Generation Command
# =============================================================================


def build_jenny_mimi_script(
    hf_token: str,
    output_repo: str,
    batch_size: int,
    max_samples: int | None,
) -> str:
    """Generate the Jenny MIMI dataset generation script content."""
    max_samples_arg = f"--max-samples {max_samples}" if max_samples else ""

    return f"""#!/bin/bash
# NOTE: "set -e" intentionally removed so session stays active on crash for debugging

ulimit -n 65536
export PATH="/root/.local/bin:$PATH"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="{hf_token}"

# GPU optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_BENCHMARK=1

cd /workspace

python -m scripts.generate_jenny_mimi \\
    --output-repo {output_repo} \\
    --batch-size {batch_size} \\
    {max_samples_arg}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "===== Jenny MIMI Dataset Generation Completed Successfully ====="
else
    echo "===== Jenny MIMI Dataset Generation Failed with exit code: $EXIT_CODE ====="
fi

echo "Script finished. Session will remain active for inspection."
sleep infinity
"""


@app.command("jenny-mimi")
def jenny_mimi(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    output_repo: str = typer.Option(
        "mazesmazes/jenny-mimi", "--output-repo", "-o", help="HuggingFace repo for output"
    ),
    session_name: str | None = typer.Option(
        None, "--session-name", "-s", help="Custom tmux session name"
    ),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size for encoding"),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Max samples (for testing)"
    ),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start session but don't attach"),
    force: bool = typer.Option(False, "--force", "-f", help="Kill existing session with same name"),
):
    """Generate Jenny TTS dataset with Mimi codec codes.

    Takes reach-vb/jenny_tts_dataset, resamples audio to 24kHz,
    encodes with Mimi codec, and pushes to HuggingFace Hub.

    Example:
        ta runpod jenny-mimi host port
        ta runpod jenny-mimi host port --max-samples 100
    """
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    # Generate session name if not provided
    if session_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        session_name = f"jenny_mimi_{timestamp}"

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    # Get environment variables
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")

    # Build and upload script
    script_content = build_jenny_mimi_script(hf_token, output_repo, batch_size, max_samples)
    script_path = f"/tmp/jenny_mimi_{session_name}.sh"

    print(f"\nStarting Jenny MIMI generation session '{session_name}'...")
    print(f"Output repo: {output_repo}")
    print(f"Batch size: {batch_size}")
    if max_samples:
        print(f"Max samples: {max_samples}")

    # Write script to remote
    conn.run(f"cat > {script_path} << 'EOF'\n{script_content}\nEOF", hide=True)
    conn.run(f"chmod +x {script_path}", hide=True)

    # Start tmux session
    result = conn.run(f"tmux new-session -d -s {session_name} {script_path}", warn=True)
    if not result.ok:
        print(f"Failed to start tmux session: {result.stderr}")
        sys.exit(1)

    print(f"\nJenny MIMI generation started in session '{session_name}'.")
    print(f"To re-attach later: ssh -p {port} root@{host} -t 'tmux attach -t {session_name}'")

    if not no_attach:
        time.sleep(2)
        attach_tmux_session(host, port, session_name)


# =============================================================================
# LibriTTS MIMI Dataset Generation Command
# =============================================================================


def build_libritts_mimi_script(
    hf_token: str,
    output_repo: str,
    max_samples: int | None,
) -> str:
    """Generate the LibriTTS MIMI dataset generation script content."""
    max_samples_arg = f"--max-samples {max_samples}" if max_samples else ""

    return f"""#!/bin/bash
# NOTE: "set -e" intentionally removed so session stays active on crash for debugging

ulimit -n 65536
export PATH="/root/.local/bin:$PATH"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="{hf_token}"

# GPU optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_BENCHMARK=1

cd /workspace

python -m scripts.generate_libritts_mimi \\
    --output-repo {output_repo} \\
    {max_samples_arg}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "===== LibriTTS MIMI Dataset Generation Completed Successfully ====="
else
    echo "===== LibriTTS MIMI Dataset Generation Failed with exit code: $EXIT_CODE ====="
fi

echo "Script finished. Session will remain active for inspection."
sleep infinity
"""


@app.command("libritts-mimi")
def libritts_mimi(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    output_repo: str = typer.Option(
        "mazesmazes/libritts-mimi",
        "--output-repo",
        "-o",
        help="HuggingFace repo for output",
    ),
    session_name: str | None = typer.Option(
        None, "--session-name", "-s", help="Custom tmux session name"
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Max samples (for testing)"
    ),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start session but don't attach"),
    force: bool = typer.Option(False, "--force", "-f", help="Kill existing session with same name"),
):
    """Generate LibriTTS dataset with Mimi codec codes.

    Takes parler-tts/libritts_r_filtered train.clean.100, encodes audio with Mimi codec,
    and pushes to HuggingFace Hub.

    Examples:
        ta runpod libritts-mimi host port
        ta runpod libritts-mimi host port --max-samples 100
    """
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    # Generate session name if not provided
    if session_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        session_name = f"libritts_mimi_{timestamp}"

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    # Get environment variables
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")

    # Build and upload script
    script_content = build_libritts_mimi_script(hf_token, output_repo, max_samples)
    script_path = f"/tmp/libritts_mimi_{session_name}.sh"

    print(f"\nStarting LibriTTS MIMI generation session '{session_name}'...")
    print(f"Output repo: {output_repo}")
    if max_samples:
        print(f"Max samples: {max_samples}")

    # Write script to remote
    conn.run(f"cat > {script_path} << 'EOF'\n{script_content}\nEOF", hide=True)
    conn.run(f"chmod +x {script_path}", hide=True)

    # Start tmux session
    result = conn.run(f"tmux new-session -d -s {session_name} {script_path}", warn=True)
    if not result.ok:
        print(f"Failed to start tmux session: {result.stderr}")
        sys.exit(1)

    print(f"\nLibriTTS MIMI generation started in session '{session_name}'.")
    print(f"To re-attach later: ssh -p {port} root@{host} -t 'tmux attach -t {session_name}'")

    if not no_attach:
        time.sleep(2)
        attach_tmux_session(host, port, session_name)


# =============================================================================
# Eval Command
# =============================================================================


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
        "slam_1", "--assemblyai-model", help="AssemblyAI model (best, universal, slam_1, nano)"
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
        runpod eval host port -m assemblyai --assemblyai-model slam_1 -d loquacious -w 4
    """
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    # Generate session name if not provided
    if session_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        model_short = model.split("/")[-1] if "/" in model else model
        session_name = f"eval_{model_short}_{timestamp}"

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    # Get environment variables
    hf_token = os.environ.get("HF_TOKEN", "")
    assemblyai_api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")

    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")
    if model == "assemblyai" and not assemblyai_api_key:
        print(
            "Warning: ASSEMBLYAI_API_KEY environment variable not set (required for assemblyai model)."
        )

    # Default datasets
    if datasets is None:
        datasets = ["loquacious"]

    # Build and upload script
    script_content = build_eval_script(
        hf_token,
        model,
        datasets,
        max_samples,
        assemblyai_api_key,
        assemblyai_model,
        num_workers,
        streaming,
        extra_args,
    )
    script_path = f"/tmp/eval_{session_name}.sh"

    print(f"\nStarting eval session '{session_name}'...")
    print(f"Model: {model}")
    print(f"Datasets: {', '.join(datasets)}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    if num_workers > 1:
        print(f"Workers: {num_workers}")
    if streaming:
        print("Streaming mode enabled")

    # Write script to remote
    conn.run(f"cat > {script_path} << 'EOF'\n{script_content}\nEOF", hide=True)
    conn.run(f"chmod +x {script_path}", hide=True)

    # Start tmux session
    result = conn.run(f"tmux new-session -d -s {session_name} {script_path}", warn=True)
    if not result.ok:
        print(f"Failed to start tmux session: {result.stderr}")
        sys.exit(1)

    print(f"\nEvaluation started in session '{session_name}'.")
    print(f"To re-attach later: ssh -p {port} root@{host} -t 'tmux attach -t {session_name}'")

    if not no_attach:
        time.sleep(2)
        attach_tmux_session(host, port, session_name)


# =============================================================================
# Checkpoint Command
# =============================================================================


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


# =============================================================================
# CLI Entry Point
# =============================================================================


def cli():
    """Entry point for pyproject.toml scripts."""
    app()


if __name__ == "__main__":
    app()
