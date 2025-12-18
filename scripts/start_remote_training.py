#!/usr/bin/env python3
"""
Start training on a remote RunPod instance in a tmux session using Fabric.
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Annotated

import typer

from scripts.remote import (
    attach_tmux_session,
    get_connection,
    kill_tmux_session,
    test_connection,
)

app = typer.Typer(help="Start and manage training on a remote RunPod instance")


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
accelerate launch --config_file configs/accelerate/a40.yaml -m scripts.train +experiments={experiment} {extra_args_str}
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
def main(
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


def cli():
    """Entry point for pyproject.toml scripts."""
    app()


if __name__ == "__main__":
    app()
