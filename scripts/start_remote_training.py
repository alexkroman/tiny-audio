#!/usr/bin/env python3
"""
Start training on a remote RunPod instance, ensuring execution within the
project's virtual environment using a managed tmux session.
"""

import argparse
import subprocess
import sys
import textwrap
from datetime import datetime, timezone


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command, printing it first."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, check=check)
    if capture_output:
        return result.stdout.strip()
    return result.returncode


def check_ssh_connection(host, port):
    """Test the SSH connection to the RunPod instance."""
    print(f"Testing SSH connection to {host}:{port}...")
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@{host} 'echo Connected'"
    try:
        run_command(cmd, capture_output=True)
        print("SSH connection successful!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to connect via SSH. Please check pod status and SSH key.")
        return False


def kill_existing_session(host, port, session_name):
    """Kill an existing tmux session on the remote host if it exists."""
    print(f"Checking for and removing existing tmux session '{session_name}'...")
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        "tmux kill-session -t {session_name}"
    """
    # We don't check for errors, as the command will fail if the session doesn't exist, which is fine.
    subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"Ensured no session named '{session_name}' is running.")


def start_training(host, port, experiment, session_name):
    """
    Creates and executes a training script inside a new tmux session on the remote host.
    """
    print(f"\nStarting training session '{session_name}' with experiment '{experiment}'...")

    import os

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set. This may cause issues.")

    # This script will be created and executed on the remote machine.
    # It activates the virtual environment to ensure all commands use the correct packages.
    training_script_content = textwrap.dedent(
        f"""\
        #!/bin/bash
        # NOTE: "set -e" is intentionally removed.
        # This ensures that if the training script crashes, the tmux session
        # remains active so you can attach and debug the error message.


        echo "--- Setting up environment variables ---"
        export TOKENIZERS_PARALLELISM=false
        export HF_DATASETS_AUDIO_DECODER="torchcodec"
        export HF_HOME=/workspace/.cache/huggingface
        export HF_DATASETS_CACHE=/workspace/datasets
        export HF_HUB_ENABLE_HF_TRANSFER=1
        export HF_TOKEN="{hf_token}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        # Enable TF32 for A40 (2x matmul speedup with no accuracy loss)
        export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
        # Enable cudnn benchmarking for faster convolutions
        export TORCH_CUDNN_BENCHMARK=1
        export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_cache  # Cache compiled kernels
        export TORCHINDUCTOR_FX_GRAPH_CACHE=1  # Cache graph transformations
        # Allow dynamic layer_idx to avoid recompilations for each layer
        export TORCH_DYNAMO_ALLOW_UNSPEC_INT_ON_NN_MODULE=1
        # Disable CUDA graphs (incompatible with CPU-GPU sync in flash attention)
        export TORCH_CUDA_GRAPHS_ENABLED=0
        echo "--- Starting TensorBoard ---"
        pkill -f tensorboard || true
        tensorboard --logdir /workspace/outputs --host 0.0.0.0 --port 6006 &
        echo "TensorBoard started on port 6006"

        echo "--- Verifying environment ---"
        echo "Experiment: {experiment}"
        echo "Python executable: $(which python)"
        echo "Python version: $(python --version)"
        echo "Accelerate executable: $(which accelerate)"
        echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
        echo "Audio decoder: $HF_DATASETS_AUDIO_DECODER"
        echo "============================="

        cd /workspace

        echo "--- Launching Training ---"
        accelerate launch --config_file configs/accelerate/a40.yaml -m src.train +experiments={experiment}
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "===== Training Completed Successfully ====="
        else
            echo "===== Training Failed with exit code: $EXIT_CODE ====="
        fi

        echo "Training script finished. Session will remain active for inspection."
        sleep infinity
    """
    )

    # Use a heredoc to safely write the script to a temporary file on the remote
    create_script_cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} bash << 'EOF'
        cat > /tmp/train_{session_name}.sh << 'INNER_EOF'
{training_script_content}
INNER_EOF
        chmod +x /tmp/train_{session_name}.sh
EOF"""

    try:
        run_command(create_script_cmd)
        print("Remote training script created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to create training script on remote host. {e}")
        return False

    # Start a new detached tmux session running the script
    start_session_cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        "tmux new-session -d -s {session_name} /tmp/train_{session_name}.sh"
    """

    try:
        run_command(start_session_cmd)
        print(f"Successfully started tmux session '{session_name}'.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to start tmux session. {e}")
        return False


def attach_to_session(host, port, session_name):
    """Attach to the running tmux session for live monitoring."""
    print(f"\nAttaching to tmux session '{session_name}'...")
    print("Use 'Ctrl+B' then 'D' to detach from the session without killing it.")
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no -t root@{host} 'tmux attach-session -t {session_name}'"
    subprocess.run(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(
        description="Start and manage training on a remote RunPod instance."
    )
    parser.add_argument("host", help="RunPod instance IP address or hostname.")
    parser.add_argument("port", type=int, help="SSH port for the RunPod instance.")
    parser.add_argument("--experiment", default="stage1", help="Experiment config to run.")
    parser.add_argument("--session-name", default=None, help="Custom name for the tmux session.")
    parser.add_argument(
        "--no-attach", action="store_true", help="Start the session but do not attach to it."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Kill any existing session with the same name before starting.",
    )

    args = parser.parse_args()

    if args.session_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        args.session_name = f"train_{args.experiment}_{timestamp}"

    if not check_ssh_connection(args.host, args.port):
        sys.exit(1)

    if args.force:
        kill_existing_session(args.host, args.port, args.session_name)

    if not start_training(args.host, args.port, args.experiment, args.session_name):
        print("Exiting due to failure in starting the training session.")
        sys.exit(1)

    print(f"\n✅ Training started in session '{args.session_name}'.")
    print(
        f"To re-attach later: ssh -p {args.port} root@{args.host} -t 'tmux attach -t {args.session_name}'"
    )

    if not args.no_attach:
        import time

        time.sleep(2)
        attach_to_session(args.host, args.port, args.session_name)


if __name__ == "__main__":
    main()
