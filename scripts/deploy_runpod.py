#!/usr/bin/env python3
"""
Deploy and sync ASR project to a RunPod instance. This script uses the
pre-installed PyTorch toolchain and installs project dependencies globally.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=capture_output, text=True, check=check
    )
    if capture_output:
        return result.stdout.strip()
    return result.returncode


def test_ssh_connection(host, port):
    """Test SSH connection to the RunPod instance."""
    print(f"Testing SSH connection to {host}:{port}...")
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@{host} 'echo Connected successfully'"
    try:
        run_command(cmd, capture_output=True)
        print("SSH connection successful!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to connect via SSH. Please check pod status and SSH key.")
        return False


def setup_remote_environment(host, port):
    """
    # CHANGED: We now only ensure system dependencies and tools are installed,
    # without creating a new virtual environment.
    Install system dependencies and helper tools on the remote.
    """
    print("\nSetting up remote environment...")
    setup_script = """#!/bin/bash
set -e
apt-get update -qq || true
apt-get install -y -qq ffmpeg tmux rsync libsndfile1 ninja-build
# Install uv, our package installer
curl -LsSf https://astral.sh/uv/install.sh | sh
# Ensure uv is in the PATH for the session and future sessions
cat >> /root/.bashrc << 'BASHRC_EOF'
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
BASHRC_EOF
export PATH="/root/.local/bin:$PATH"
# REMOVED: The command to create a virtual environment is gone.
# We will use the global Python environment provided by the RunPod image.
"""
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} << 'EOF'\n{setup_script}\nEOF"
    run_command(cmd)
    print("Remote environment setup successful!")
    return True


def sync_project(host, port, project_root):
    """Sync the project files to the RunPod instance."""
    print(f"\nSyncing project from {project_root} to {host}:{port}...")
    exclusions = [
        "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=.git", "--exclude=.venv",
        "--exclude=venv", "--exclude=env", "--exclude=.env", "--exclude=./data/",
        "--exclude=datasets_cache/", "--exclude=outputs/", "--exclude=logs/",
        "--exclude=runs/", "--exclude=wandb/", "--exclude=.mypy_cache",
        "--exclude=.pytest_cache", "--exclude=.ruff_cache", "--exclude=*.egg-info",
        "--exclude=dist/", "--exclude=build/", "--exclude=.DS_Store", "--exclude=.idea/",
        "--exclude=.vscode/", "--exclude=node_modules/", "--exclude=.cache/",
        "--exclude=datasets/", "--exclude=checkpoints/", "--exclude=*.ckpt",
        "--exclude=*.pth", "--exclude=*.pt",
    ]
    exclusion_str = " ".join(exclusions)
    rsync_cmd = f"""rsync -avz --delete --no-owner --no-group {exclusion_str} \
        -e "ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no" \
        {project_root}/ root@{host}:/workspace/"""
    run_command(rsync_cmd)
    print("Project synced successfully!")
    return True


def install_python_dependencies(host, port):
    """
    # CHANGED: Installs ONLY the project-specific dependencies into the global environment.
    """
    print("\nInstalling Python dependencies...")
    
    # **THE FIX IS HERE**:
    # 1. We REMOVED the step that installs torch, torchvision, etc.
    # 2. We now ONLY install the dependencies defined in your project's setup file (e.g., pyproject.toml).
    #    These packages will be installed into the main system Python environment, alongside
    #    the pre-installed PyTorch.
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'cd /workspace && \
         export PATH="/root/.local/bin:$PATH" && \
         echo "--- Installing project-specific dependencies into the global environment ---" && \
         uv pip install \
            --no-cache-dir \
            -e .'
    """
    run_command(cmd)
    print("Python dependencies installed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy ASR project to RunPod instance")
    parser.add_argument("host", help="RunPod instance IP address or hostname")
    parser.add_argument("port", type=int, help="SSH port for the RunPod instance")
    parser.add_argument("--skip-setup", action="store_true", help="Skip remote environment setup")
    parser.add_argument("--skip-sync", action="store_true", help="Skip project file sync")
    parser.add_argument("--skip-deps", action="store_true", help="Skip Python dependency installation")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.absolute()

    if not test_ssh_connection(args.host, args.port):
        sys.exit(1)

    if not args.skip_setup:
        setup_remote_environment(args.host, args.port)

    if not args.skip_sync:
        sync_project(args.host, args.port, project_root)

    if not args.skip_deps:
        install_python_dependencies(args.host, args.port)

    print("\n🚀 Deployment finished!")
    print(f"To connect: ssh -i ~/.ssh/id_ed25519 -p {args.port} root@{args.host}")


if __name__ == "__main__":
    main()