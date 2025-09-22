#!/usr/bin/env python3
"""Start training on remote RunPod instance with tmux session management."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, check=check)
    if capture_output:
        return result.stdout.strip()
    return result.returncode


def check_ssh_connection(host, port):
    """Test SSH connection to the RunPod instance."""
    print(f"Testing SSH connection to {host}:{port}...")
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@{host} 'echo Connected'"
    try:
        run_command(cmd, capture_output=True)
        print("SSH connection successful!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to connect via SSH. Please check:")
        print("  - The host and port are correct")
        print("  - The pod is running")
        print("  - Your SSH key ~/.ssh/id_ed25519 is added to the pod")
        return False


def kill_existing_session(host, port, session_name):
    """Kill existing tmux session if it exists."""
    print(f"Checking for existing tmux session '{session_name}'...")
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'tmux kill-session -t {session_name} 2>/dev/null || echo "No existing session"'"""
    try:
        result = run_command(cmd, capture_output=True)
        if "No existing session" not in result:
            print(f"Killed existing session '{session_name}'")
    except subprocess.CalledProcessError:
        pass  # Session doesn't exist, which is fine


def start_training(host, port, experiment, session_name, env_vars=None):
    """Start training in a tmux session on the remote instance."""

    # Build environment variables string
    env_string = ""
    if env_vars:
        env_string = " ".join([f"{k}={v}" for k, v in env_vars.items()]) + " "

    print(f"\nStarting training session '{session_name}' with experiment '{experiment}'...")

    # Get the token from environment
    import os

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")

    # Create a shell script to run on the remote
    training_script = f"""#!/bin/bash
# Don't exit on error immediately, so we can see what happened
cd /workspace
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/datasets
export TORCH_HOME=/workspace/.cache/torch
export HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGING_FACE_HUB_TOKEN="{hf_token}"  # Pass through from host
export DATASETS_PARALLEL=1  # Enable parallel dataset processing
export HF_DATASETS_MULTIPROCESSING_MAX_WORKERS=8  # Optimized for 9 vCPUs
export HF_DATASETS_DOWNLOAD_MANAGER_MAX_WORKERS=4  # Parallel downloads within each dataset
export HF_DATASETS_DOWNLOAD_BATCH_SIZE=50  # Download multiple files in parallel
export DATASETS_MAX_CONCURRENT_DOWNLOADS=4  # Max concurrent file downloads

export TOKENIZERS_PARALLELISM=false  # Disable parallel tokenization to avoid deadlocks
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Better memory management
export NCCL_P2P_DISABLE=0  # Enable GPU peer-to-peer communication
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
{env_string}
echo "===== Training Starting ====="
echo "Experiment: {experiment}"
echo "Python: $(python3 --version)"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo "============================="


# Function to cleanup on exit
cleanup() {{
    echo "Cleaning up..."
    exit
}}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Run the training command and capture any errors
echo "Launching training with experiment: {experiment}"

# Start TensorBoard in background after we're in the right directory
echo "Starting TensorBoard on port 6006..."
(cd /workspace && nohup tensorboard --logdir=/workspace/outputs --port=6006 --bind_all > /tmp/tensorboard.log 2>&1 &)
echo "TensorBoard started in background"
echo "You can access TensorBoard by port-forwarding: ssh -L 6006:localhost:6006 -p {port} root@{host}"
sleep 2  # Give TensorBoard a moment to start

# Now run the training
cd /workspace && accelerate launch --config_file configs/accelerate/a40.yaml src/train.py +experiments={experiment} 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "===== Training Failed with exit code: $EXIT_CODE ====="
    echo "Check the logs above for errors"
else
    echo "===== Training Completed Successfully ====="
fi

# Keep the session alive
echo "Training session complete. TensorBoard is running on port 6006"
echo "Press Ctrl+C to exit."
sleep infinity
"""

    # First, create the script on the remote
    create_script_cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'cat > /tmp/train_{session_name}.sh && chmod +x /tmp/train_{session_name}.sh' <<'EOF'
{training_script}
EOF"""

    try:
        run_command(create_script_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create training script: {e}")
        return False

    # Start tmux session and run the script
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'tmux new-session -d -s {session_name} /tmp/train_{session_name}.sh'"""

    try:
        run_command(cmd)
        print(f"Training started in tmux session '{session_name}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to start training: {e}")
        return False


def attach_to_session(host, port, session_name):
    """Attach to the tmux session to see live output."""
    print(f"\nAttaching to session '{session_name}'...")
    print("=" * 50)
    print("TMUX CONTROLS:")
    print("  - Detach (leave running): Ctrl+B then D")
    print("  - Scroll up/down: Ctrl+B then [ (then q to exit scroll mode)")
    print("  - Kill session: Ctrl+B then : then type 'kill-session'")
    print("=" * 50)
    print("")

    # SSH and attach to tmux session
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no -t root@{host} 'tmux attach-session -t {session_name}'"

    # Use subprocess.call for interactive session
    subprocess.call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(description="Start training on remote RunPod instance")
    parser.add_argument("host", help="RunPod instance IP address or hostname")
    parser.add_argument("port", type=int, help="SSH port for the RunPod instance")
    parser.add_argument(
        "--experiment", default="production", help="Experiment config to use (default: production)"
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Name for the tmux session (default: train_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--no-attach", action="store_true", help="Don't attach to session after starting"
    )
    parser.add_argument(
        "--force", action="store_true", help="Kill existing session with same name if it exists"
    )

    args = parser.parse_args()

    # Generate session name if not provided
    if args.session_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.session_name = f"train_{timestamp}"

    # Test SSH connection
    if not check_ssh_connection(args.host, args.port):
        sys.exit(1)

    # Kill existing session if force flag is set
    if args.force:
        kill_existing_session(args.host, args.port, args.session_name)

    # Start training
    if not start_training(args.host, args.port, args.experiment, args.session_name):
        sys.exit(1)

    print(f"\nTraining started successfully in session '{args.session_name}'!")
    print("To attach to the session manually, run:")
    print(
        f"  python scripts/attach_remote_session.py {args.host} {args.port} --session-name {args.session_name}"
    )

    # Attach to session unless --no-attach is specified
    if not args.no_attach:
        print("\nAttaching to session in 2 seconds...")
        import time

        time.sleep(2)
        attach_to_session(args.host, args.port, args.session_name)


if __name__ == "__main__":
    main()
