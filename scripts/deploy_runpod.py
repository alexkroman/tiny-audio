#!/usr/bin/env python3
"""Deploy and sync ASR project to RunPod instance."""

import argparse
import subprocess
import sys
from pathlib import Path

import tomllib


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, check=check)
    if capture_output:
        return result.stdout.strip()
    return result.returncode


def test_ssh_connection(host, port):
    """Test SSH connection to the RunPod instance."""
    print(f"Testing SSH connection to {host}:{port}...")
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@{host} 'echo Connected successfully'"
    try:
        run_command(cmd)
        print("SSH connection successful!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to connect via SSH. Please check:")
        print("  - The host and port are correct")
        print("  - The pod is running")
        print("  - Your SSH key ~/.ssh/id_ed25519 is added to the pod")
        return False


def setup_remote_dependencies(host, port):
    """Install required system dependencies on the remote instance."""
    print("\nInstalling system dependencies on remote...")

    # Create setup script content
    setup_script = """#!/bin/bash
set -e

echo "Updating package lists..."
apt-get update

echo "Installing system dependencies..."
apt-get install -y ffmpeg tmux rsync curl ninja-build build-essential wget gnupg2

echo "Installing CUDA 12.8..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-8 libcudnn9-cuda-12

echo "Installing Python..."
apt-get install -y python3 python3-dev python3-pip python3-venv

echo "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Setting up CUDA environment..."
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"

echo "System setup complete!"
"""

    # Execute setup script on remote
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} << 'SETUP_EOF'
cat > /tmp/setup.sh << 'EOF'
{setup_script}
EOF
chmod +x /tmp/setup.sh
bash /tmp/setup.sh
SETUP_EOF"""

    run_command(cmd)
    print("Remote dependencies installed successfully!")
    return True


def sync_project(host, port, project_root):
    """Sync the project files to the RunPod instance."""
    print(f"\nSyncing project from {project_root} to {host}:{port}...")

    # Define exclusions for rsync
    exclusions = [
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=venv",
        "--exclude=env",
        "--exclude=.env",
        "--exclude=.claude",  # Don't sync Claude settings
        "--exclude=./data/",  # Don't sync root data directory but allow configs/hydra/data/
        "--exclude=datasets_cache/",  # Don't sync dataset cache
        "--exclude=outputs",
        "--exclude=logs",
        "--exclude=runs",
        "--exclude=wandb",
        "--exclude=.mypy_cache",
        "--exclude=.pytest_cache",
        "--exclude=.ruff_cache",
        "--exclude=*.egg-info",
        "--exclude=dist",
        "--exclude=build",
        "--exclude=.DS_Store",
        "--exclude=.idea",
        "--exclude=.vscode",
        "--exclude=node_modules",
        "--exclude=.cache",
        "--exclude=datasets",  # Don't sync large dataset caches
        "--exclude=checkpoints",  # Don't sync model checkpoints
        "--exclude=*.ckpt",
        "--exclude=*.pth",
        "--exclude=*.pt",
    ]

    exclusion_str = " ".join(exclusions)

    # Rsync the entire project to /workspace
    rsync_cmd = f"""rsync -avz --delete --no-owner --no-group {exclusion_str} \
        -e "ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no" \
        {project_root}/ root@{host}:/workspace/"""

    run_command(rsync_cmd)
    print("Project synced successfully!")
    return True


def get_project_dependencies(project_root):
    """Load dependencies from pyproject.toml."""
    pyproject_path = project_root / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    # Get base dependencies
    dependencies = pyproject.get("project", {}).get("dependencies", [])

    # Add cuda optional dependencies (bitsandbytes for GPU)
    cuda_deps = pyproject.get("project", {}).get("optional-dependencies", {}).get("cuda", [])
    dependencies.extend(cuda_deps)

    return dependencies


def install_python_dependencies(host, port, project_root):
    """Install Python dependencies using uv on RunPod."""
    print("\nInstalling Python dependencies...")

    # First sync base dependencies (including torch)
    print("Installing base dependencies (including PyTorch)...")
    cmd_base = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'cd /workspace && export PATH="/root/.local/bin:/root/.cargo/bin:$PATH" && \
         export PATH="/usr/local/cuda-12.8/bin:$PATH" && \
         export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH" && \
         uv sync 2>&1'"""
    run_command(cmd_base)

    # Then install CUDA extras (flash-attn needs torch to build)
    print("Installing CUDA extras (flash-attn, bitsandbytes)...")
    cmd_cuda = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'cd /workspace && export PATH="/root/.local/bin:/root/.cargo/bin:$PATH" && \
         export PATH="/usr/local/cuda-12.8/bin:$PATH" && \
         export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH" && \
         uv sync --extra cuda 2>&1'"""
    run_command(cmd_cuda)

    print("Python dependencies installed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy ASR project to RunPod instance")
    parser.add_argument(
        "host", help="RunPod instance IP address (e.g., 192.168.1.100 or pod.runpod.io)"
    )
    parser.add_argument("port", type=int, help="SSH port for the RunPod instance (e.g., 22222)")
    parser.add_argument(
        "--skip-setup", action="store_true", help="Skip system dependency installation"
    )
    parser.add_argument("--skip-sync", action="store_true", help="Skip project file sync")
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip Python dependency installation"
    )

    args = parser.parse_args()

    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent.absolute()

    # Test SSH connection
    if not test_ssh_connection(args.host, args.port):
        sys.exit(1)

    # Setup remote dependencies
    if not args.skip_setup:
        setup_remote_dependencies(args.host, args.port)

    # Sync project files
    if not args.skip_sync:
        sync_project(args.host, args.port, project_root)

    # Install Python dependencies
    if not args.skip_deps:
        install_python_dependencies(args.host, args.port, project_root)

if __name__ == "__main__":
    main()
