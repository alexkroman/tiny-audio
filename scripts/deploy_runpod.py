#!/usr/bin/env python3
"""
Deploy and sync ASR project to a RunPod instance using Fabric.
"""

import sys
from pathlib import Path

import typer
from fabric import Connection

from scripts.remote import get_connection, test_connection

app = typer.Typer(help="Deploy ASR project to RunPod instance")

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
    conn.run("apt-get install -y -qq ffmpeg tmux rsync libsndfile1 ninja-build portaudio19-dev", hide=True)
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

    # Run rsync locally (it connects to remote)
    import subprocess

    subprocess.run(rsync_cmd, shell=True, check=True)
    print("Project synced successfully!")


def install_dependencies(conn: Connection) -> None:
    """Install Python dependencies from poetry.lock."""
    print("\nInstalling Python dependencies...")

    # Setup script that preserves system PyTorch
    setup_commands = """
        export PATH="/root/.local/bin:$PATH"
        export PIP_ROOT_USER_ACTION=ignore
        export POETRY_VIRTUALENVS_CREATE=false
        export PIP_BREAK_SYSTEM_PACKAGES=1

        # Configure pip
        mkdir -p /root/.config/pip
        echo -e "[global]\\nbreak-system-packages = true" > /root/.config/pip/pip.conf

        # Install Poetry if needed
        command -v poetry >/dev/null 2>&1 || pip install --user poetry
        pip install --user poetry-plugin-export

        # Configure Poetry
        poetry config virtualenvs.create false
        poetry config installer.max-workers 10

        # Install flash-attn if needed
        python -c "import flash_attn" 2>/dev/null || pip install --user flash-attn --no-build-isolation

        # Install PyTorch with CUDA 12.8 if needed
        python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
            pip install --user torch~=2.8.0 --index-url=https://download.pytorch.org/whl/cu128

        # Install accelerate and peft
        pip install --user accelerate peft

        # Export and install dependencies (excluding torch to preserve system version)
        cd /workspace
        poetry export --only main --without-hashes | grep -v "^torch==" > /tmp/requirements.txt
        pip install --user -r /tmp/requirements.txt 2>&1 | grep -v "already satisfied" || true

        # Install project in editable mode
        pip install --user -e . --no-deps

        # Verify
        which accelerate
        python -c "import accelerate; print('accelerate', accelerate.__version__)"
    """

    conn.run(f"bash -c '{setup_commands}'", hide=False)
    print("Dependencies installed successfully!")


@app.command()
def main(
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

    project_root = Path(__file__).parent.parent.absolute()

    if not skip_setup:
        setup_remote_environment(conn)

    if not skip_sync:
        sync_project(conn, project_root)

    if not skip_deps:
        install_dependencies(conn)

    print("\nDeployment finished!")
    print(f"To connect: ssh -i ~/.ssh/id_ed25519 -p {port} root@{host}")


def cli():
    """Entry point for pyproject.toml scripts."""
    app()


if __name__ == "__main__":
    app()
