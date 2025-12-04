# Class 2: Training

**Duration**: 1 hour (15 min intro + 45 min hands-on)

**Goal**: Set up a cloud training environment, configure an experiment, and launch your first training run.

## Learning Objectives

By the end of this class, you will:

- Set up RunPod for cloud GPU training
- Understand training metrics (loss, gradient norm, learning rate)
- Configure experiments using Hydra YAML files
- Launch and monitor a training run
- Know when to cancel and restart training
- **Terminate unused instances to avoid charges**

## Prerequisites

Before starting, ensure you have:

- Completed Class 1 (environment setup)
- RunPod account with SSH key configured
- Hugging Face token with **write** permissions
- Weights & Biases account

______________________________________________________________________

## PART A: LECTURE (15 min)

### 1. Why RunPod? (5 min)

Tiny Audio trains locally on a MacBook (MPS driver), but it's slow. I typically:

1. Test changes locally to verify they don't break anything
1. Deploy to RunPod for real training

RunPod provides:

- Easy access to powerful GPUs (A40 with 48GB VRAM)
- Simple UI for managing instances
- Pay-as-you-go (~$0.40/hour for A40)
- Ability to scale to multiple GPUs

**Cost**: A full training run takes ~20 hours and costs ~$8-12. For this class, even a few hours produces a working model.

### 2. How Training Works (5 min)

When you start a training run:

1. **Downloads models**: Whisper encoder (~3GB) and SmolLM3 decoder (~6GB) from Hugging Face
1. **Streams data**: Training data streams from Hugging Face (no terabyte downloads needed)
1. **Trains MoE projector**: Only the MoE projector trains; encoder and decoder stay frozen
1. **Saves checkpoints**: Every 500 steps, model saves to Hugging Face (resume if something crashes)

### 3. Key Metrics (5 min)

During training, you'll see:

| Metric | What it means | What to look for |
|--------|---------------|------------------|
| **Training Loss** | How well the model fits training data | Should decrease over time |
| **Eval Loss** | Performance on unseen data (every 1,000 steps) | Should also decrease; if it rises while training loss falls, you're overfitting |
| **Gradient Norm** | Size of updates the optimizer wants to make | Starts high (40+), should drop to \<10 |
| **Learning Rate** | How big updates are allowed to be | Usually constant or scheduled decay |

**The "Cliff" Phenomenon**

With multimodal models, you often see a dramatic cliff in training loss:

- **Steps 0-1500**: Model outputs complete gibberish
- **Steps 1500-1600**: Something clicks—within ~20 steps, it goes from nothing to decent ASR
- **Steps 1600+**: Gradual improvement

This is normal! Don't panic if your model seems broken for the first hour.

______________________________________________________________________

## PART B: HANDS-ON WORKSHOP (45 min)

## Exercise 1: Set Up RunPod (10 min)

### Goal

Create a RunPod account and deploy an A40 GPU instance.

### Instructions

**Step 1: Sign up for RunPod**

Go to [runpod.io](https://runpod.io) and create an account (or use an invite link if provided).

**Step 2: Add your SSH key**

Go to **Settings → SSH Public Keys**.

If you don't have an SSH key:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Copy your public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Paste into RunPod and click "Update".

**Step 3: Deploy an A40 instance**

1. Click **Deploy**
1. Select **NVIDIA A40** (Featured GPUs)
   - Cheapest with good availability
   - 48GB VRAM
   - ~$0.40/hour
1. Choose **RunPod PyTorch 2.x** template
1. GPU count: **1**
1. Uncheck "Start Jupyter Notebook" (unless you want it)
1. Ensure "SSH Terminal Access" is checked
1. Click **Deploy**

**Important**: Don't create "Savings Plans"—use on-demand only.

**Step 4: Get connection details**

Once running, click the pod to expand details. Note the **host** and **port** (e.g., `ssh.runpod.io` and `22115`).

### Success Checkpoint

- [ ] RunPod account with SSH key
- [ ] A40 pod deployed and running
- [ ] Have host and port for SSH

______________________________________________________________________

## Exercise 2: Deploy Your Code (10 min)

### Goal

Sync your local project to RunPod and install dependencies.

### Instructions

**Step 1: Get latest code**

```bash
cd tiny-audio
git pull origin main
poetry install  # If you haven't already
```

**Step 2: Run the deploy script**

```bash
poetry run python scripts/deploy_runpod.py <HOST> <PORT>
```

Example:

```bash
poetry run python scripts/deploy_runpod.py ssh.runpod.io 22115
```

**What this script does** (I spent more time on this than the model code!):

1. Tests SSH connection
1. Syncs project files with `rsync`
1. Installs system dependencies
1. Installs Flash Attention (NVIDIA optimization)
1. Installs Hugging Face Accelerate (multi-GPU support)
1. Installs Python dependencies

Takes 5-10 minutes. Output:

```
Testing SSH connection...
SSH connection successful!

Syncing project files...
Project synced successfully!

Installing dependencies...
Python dependencies installed successfully!

🚀 Deployment finished!
```

**Re-run this script** every time you make local changes.

### Troubleshooting

- **SSH fails**: Verify your public key is in RunPod settings
- **Dependency errors**: Script is safe to re-run; it picks up where it left off

### Success Checkpoint

- [ ] Deploy script completed
- [ ] Files synced to `/workspace/` on the pod

______________________________________________________________________

## Exercise 3: Configure Your Experiment (10 min)

### Goal

Create your own experiment configuration.

### Instructions

**Step 1: Understand the config system**

Most changes are **config changes, not code changes**. Configs are YAML files in `configs/hydra/experiments/`.

**Step 2: Create your config**

```bash
cp configs/hydra/experiments/moe.yaml configs/hydra/experiments/my_experiment.yaml
```

Edit `my_experiment.yaml`:

```yaml
# REQUIRED: Change to YOUR Hugging Face username
training:
  hub_model_id: "your-username/tiny-audio-yourname"

# Optional: Experiment with these
# training:
#   max_steps: 10000       # Fewer steps = faster (default: 15000)
#   per_device_train_batch_size: 8  # Higher = faster, but uses more VRAM
#   learning_rate: 1e-3    # How aggressive updates are
```

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_steps` | 15,000 | Training duration (5,000 is enough to test) |
| `per_device_train_batch_size` | 14 | Samples per step (higher = faster, more VRAM) |
| `learning_rate` | 1e-3 | Update aggressiveness |
| `projector_type` | moe | Projector architecture (moe, swiglu, residual) |

**Step 3: Re-deploy**

```bash
poetry run python scripts/deploy_runpod.py <HOST> <PORT>
```

### Success Checkpoint

- [ ] Created your experiment config
- [ ] Changed `hub_model_id` to your Hugging Face account
- [ ] Re-deployed to sync

______________________________________________________________________

## Exercise 4: Start Training (10 min)

### Goal

Launch training and understand the output.

### Instructions

**Step 1: Set your Hugging Face token**

Get a token with **write** permissions from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):

```bash
export HF_TOKEN='hf_your_token_here'
```

**Step 2: Start training**

```bash
poetry run python scripts/start_remote_training.py <HOST> <PORT> --experiment my_experiment
```

**Step 3: Set up Weights & Biases**

When prompted:

1. Press **2** to log in with API key
1. Paste your key from [wandb.ai/authorize](https://wandb.ai/authorize)
1. It creates a project and starts logging

**Step 4: Watch training start**

You'll see:

```
Downloading Whisper encoder... (1.6GB)
Downloading SmolLM3 decoder... (10GB)
Streaming training data...

Training 62,000,000 parameters
Step 25/20000 | Loss: 8.34 | Grad Norm: 45.2 | LR: 1e-4 | Time: 2min | ETA: 20hr
```

**Understanding output:**

- `Loss`: Should decrease (high at first is normal)
- `Grad Norm`: Starts 40+, should drop to \<10 in a few hundred steps
- `ETA`: Estimated time remaining

### Success Checkpoint

- [ ] Training started
- [ ] Loss values appearing
- [ ] W&B logging working

______________________________________________________________________

## Exercise 5: Monitor and Manage (5 min)

### Goal

Learn to monitor training and manage costs.

### Detach/Reattach

Training runs in `tmux`, so you can disconnect safely:

```bash
# Detach: Press Ctrl+B, then D

# Reattach later:
poetry run python scripts/attach_remote.py <HOST> <PORT>
```

### Monitor in RunPod

Click your pod to see telemetry:

| Metric | Target | Problem if exceeded |
|--------|--------|---------------------|
| VRAM | 80-90% | 100% = CUDA OOM crash |
| GPU Util | 100% | \<100% = bottleneck somewhere |
| RAM | \<100% | 100% = crash |

### Monitor in Weights & Biases

At [wandb.ai](https://wandb.ai):

- **train/loss**: Should steadily decrease
- **eval/loss**: Every 1,000 steps; both should decrease

**Warning sign**: Eval loss goes up while training loss goes down = overfitting. Cancel and adjust.

### When to Cancel

Press `Ctrl+C` in tmux to stop. Cancel if:

- Loss isn't decreasing after several hundred steps
- Eval loss starts going up
- You want to change hyperparameters

**After canceling:**

1. **Start fresh**: Change config, restart
1. **Resume from checkpoint**: Uncomment `pretrained_model_path` in your config

### CRITICAL: Terminate Unused Instances

**RunPod charges by the hour whether you're using the GPU or not!**

When done:

| Action | Effect |
|--------|--------|
| **Stop** | Pauses billing, keeps data |
| **Terminate** | Stops billing, deletes data |

**If you close your laptop and forget, you keep getting charged.**

Always terminate when you're done. You can re-deploy in 5-10 minutes.

### Success Checkpoint

- [ ] Can detach/reattach to training
- [ ] Can view metrics in W&B
- [ ] Know when to cancel
- [ ] **Pod terminated or stopped when done**

______________________________________________________________________

## What's Next

During training (several hours to a day):

1. **Check in periodically**: View loss in W&B, reattach to see output
1. **Run evaluation**: Test WER locally (Class 3)
1. **Run the demo**: Load your checkpoint to "vibe test" it

Every 1,000 steps, your model saves to Hugging Face. You can evaluate or demo any checkpoint.

## Experimentation Ideas

Once you have a working pipeline:

**Easy wins:**

- Try different projector types (MoE, SwiGLU, Residual)
- Adjust learning rate and batch size

**Interesting experiments:**

- Swap decoder to a larger LLM
- Multilingual datasets
- Other tasks: emotion detection, audio description

**Advanced:**

- Fine-tune from checkpoint on domain-specific data (medical, legal)
- Multi-task training

______________________________________________________________________

[← Class 1: Introduction](./1-introduction-and-setup.md) | [Class 3: Evaluation →](./3-evaluation-and-deployment.md)
