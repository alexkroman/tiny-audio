# Class 3: Training on RunPod

**Duration**: 45 min workshop

**Goal**: Get your first training run started on RunPod using the deploy and remote-train scripts

## Prerequisites

Before starting this workshop, ensure you have:

- Completed Classes 1 and 2
- A RunPod account with credit ($20-30 recommended)
- An SSH key set up at `~/.ssh/id_ed25519`
- HuggingFace token (get from https://huggingface.co/settings/tokens)

______________________________________________________________________

# WORKSHOP

## Exercise 1: Create a RunPod Instance (10 min)

### Goal

Launch a GPU instance on RunPod that we'll use for training.

### Instructions

**Step 1: Log into RunPod**

- Go to [runpod.io](https://runpod.io) and log in
- Navigate to "Pods" in the sidebar

**Step 2: Deploy a new pod**

- Click "Deploy"
- Select **NVIDIA A40** (40GB VRAM) - best price/performance for our training
- Choose the **RunPod PyTorch** template
- Click "Deploy"
- Wait for the pod to start (~2 min)

**Step 3: Get your connection details**

Once the pod is running, find the SSH connection info:

- Click on the pod name to expand details
- Look for the SSH connection command
- Note the **host** (e.g., `ssh.runpod.io`) and **port** (e.g., `22115`)

**Step 4: Test SSH connection**

```bash
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST> 'echo Connected!'
```

Replace `<HOST>` and `<PORT>` with your values.

### Success Checkpoint

- [ ] Pod is running on RunPod
- [ ] Found SSH host and port
- [ ] SSH connection test works

______________________________________________________________________

## Exercise 2: Deploy the Project (15 min)

### Goal

Use the deploy script to set up the training environment on your RunPod instance.

### Instructions

**Step 1: Review what the deploy script does**

The `scripts/deploy_runpod.py` script automates:

1. Testing SSH connection
2. Installing system dependencies (ffmpeg, tmux, rsync, etc.)
3. Syncing project files to `/workspace/`
4. Installing Python dependencies from `poetry.lock`

**Step 2: Run the deploy script**

```bash
python scripts/deploy_runpod.py <HOST> <PORT>
```

Example:

```bash
python scripts/deploy_runpod.py ssh.runpod.io 22115
```

**What to expect:**

```
Testing SSH connection to ssh.runpod.io:22115...
SSH connection successful!

Setting up remote environment...
Remote environment setup successful!

Syncing project from /path/to/tiny-audio to ssh.runpod.io:22115...
Project synced successfully!

Installing Python dependencies from poetry.lock (excluding dev dependencies)...
Python dependencies installed successfully!

🚀 Deployment finished!
To connect: ssh -i ~/.ssh/id_ed25519 -p 22115 root@ssh.runpod.io
```

This takes 5-10 minutes depending on network speed.

**Step 3: Verify deployment (optional)**

SSH into the pod and check:

```bash
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST>

# On the pod:
cd /workspace
ls -la  # Should see project files
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Troubleshooting

**SSH connection fails:**

- Verify pod is running in RunPod dashboard
- Check your SSH key exists: `ls ~/.ssh/id_ed25519`
- Try regenerating connection details in RunPod

**Dependency installation fails:**

- The script can be re-run safely
- Use `--skip-setup` to skip system dependencies if already done
- Use `--skip-sync` to skip file sync if already done

### Success Checkpoint

- [ ] Deploy script completed without errors
- [ ] Project files are at `/workspace/` on the pod
- [ ] PyTorch and CUDA are working

______________________________________________________________________

## Exercise 3: Start Training (15 min)

### Goal

Launch a training run using the remote-train script.

### Instructions

**Step 1: Set your HuggingFace token**

The training script needs this to download datasets and upload models:

```bash
export HF_TOKEN='hf_your_token_here'
```

**Step 2: Review what the remote-train script does**

The `scripts/start_remote_training.py` script:

1. Tests SSH connection
2. Creates a training script on the remote machine
3. Starts a tmux session with the training process
4. Optionally attaches you to watch progress

**Step 3: Start training**

```bash
python scripts/start_remote_training.py <HOST> <PORT> --experiment transcribe
```

Available experiments:

- `transcribe` - Default transcription training

Example:

```bash
python scripts/start_remote_training.py ssh.runpod.io 22115 --experiment transcribe
```

**Step 4: Monitor training**

The script automatically attaches to the tmux session. You'll see:

```
--- Setting up environment variables ---
--- Verifying environment ---
Experiment: transcribe
CUDA available: True
--- Launching Training ---
...
Step 1/10000 | Loss: 8.3456
Step 2/10000 | Loss: 7.9123
...
```

**Detach from the session** (training continues in background):

- Press `Ctrl+B`, then `D`

**Re-attach later:**

```bash
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST> -t 'tmux attach -t <SESSION_NAME>'
```

The session name is shown when you start training (e.g., `train_transcribe_20241126_1430`).

### Script Options

```bash
# Start without attaching (useful for scripting)
python scripts/start_remote_training.py <HOST> <PORT> --no-attach

# Use a custom session name
python scripts/start_remote_training.py <HOST> <PORT> --session-name my-training

# Force restart (kills existing session with same name)
python scripts/start_remote_training.py <HOST> <PORT> --force
```

### Success Checkpoint

- [ ] Training started successfully
- [ ] Seeing loss values in the output
- [ ] Know how to detach and re-attach

______________________________________________________________________

## Exercise 4: Monitor and Manage (5 min)

### Goal

Learn how to check on your training run and manage the pod.

### Check Training Status

**Re-attach to session:**

```bash
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST> -t 'tmux attach -t <SESSION_NAME>'
```

**List all tmux sessions:**

```bash
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST> 'tmux list-sessions'
```

**Check GPU usage:**

```bash
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST> 'nvidia-smi'
```

### Managing Your Pod

**Stop the pod** (pauses billing, preserves data):

- In RunPod dashboard, click "Stop" on your pod

**Terminate the pod** (stops billing, deletes data):

- In RunPod dashboard, click "Terminate"

**Resume training after pod restart:**

1. Start the pod again
2. Re-run deploy (with `--skip-deps` if dependencies are still installed)
3. Training should auto-resume from the latest checkpoint

### Expected Training Time

- **A40 GPU**: ~24 hours for full training (10,000 steps)
- **Cost**: ~$12-15 total

### Success Checkpoint

- [ ] Can re-attach to training session
- [ ] Know how to check GPU usage
- [ ] Understand pod management options

[Previous: Class 2: The End-to-End ASR Architecture](./2-end-to-end-architecture.md) | [Next: Class 4: Evaluation and Deployment](./4-evaluation-and-deployment.md)
