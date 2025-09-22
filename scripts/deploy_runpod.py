#!/usr/bin/env python3
"""Deploy and sync ASR project to RunPod instance."""

import argparse
import subprocess
import sys
from pathlib import Path


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
apt-get install -y ffmpeg tmux rsync curl

echo "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "System setup complete!"
"""

    # Execute setup script on remote
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} 'cat > /tmp/setup.sh && chmod +x /tmp/setup.sh && bash /tmp/setup.sh' <<'EOF'
{setup_script}
EOF"""

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


def install_python_dependencies(host, port):
    """Install Python dependencies using uv on RunPod."""
    print("\nInstalling Python dependencies...")

    # Display PyTorch version
    print("Checking PyTorch installation...")
    cmd_check = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'python3 -c "import torch; print(\\"PyTorch \\" + torch.__version__ + \\" with CUDA \\" + str(torch.cuda.is_available()))" 2>&1'"""
    result = run_command(cmd_check, capture_output=True)
    print(f"Found: {result}")

    # Install only the packages not provided by RunPod
    required_packages = [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "bitsandbytes",
        "datasets==3.6.0",
        "peft>=0.6.0",
        "evaluate>=0.4.0",
        "jiwer>=3.0.0",  # For WER (Word Error Rate) calculation
        "tensorboard>=2.14.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "hf-transfer",  # For fast HuggingFace downloads
        "ninja",  # For faster CUDA kernel compilation
        "soundfile",
        "librosa",  # Required by datasets for audio decoding
    ]

    packages_str = " ".join(f'"{pkg}"' for pkg in required_packages)

    print(f"Installing: {', '.join(required_packages)}")

    # Use uv with --system to work with system Python and respect existing packages
    # The --system flag tells uv to use the system Python instead of creating a venv
    # This preserves the pre-installed CUDA-optimized PyTorch while installing our dependencies
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'export PATH="/root/.local/bin:/root/.cargo/bin:$PATH" && \
         echo "Using uv to install packages while preserving system PyTorch..." && \
         uv pip install --system {packages_str} 2>&1'"""

    run_command(cmd)
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
        install_python_dependencies(args.host, args.port)

    print("\n" + "=" * 50)
    print("Deployment complete!")
    print("You can now SSH into your RunPod instance:")
    print(f"  ssh -i ~/.ssh/id_ed25519 -p {args.port} root@{args.host}")
    print("\nTo start training:")
    print("  cd /workspace")
    print("  python3 src/train.py +experiments=production")
    print("\nOr use the training script:")
    print(f"  python scripts/start_remote_training.py {args.host} {args.port}")
    print("=" * 50)


if __name__ == "__main__":
    main()
