# Class 4: Training

**Duration**: 1 hour (20 min lecture + 40 min hands-on)

**Goal**: Understand parameter-efficient training and start your first training run

## Learning Objectives

By the end of this class, you will:

- Understand LoRA (Low-Rank Adaptation) and why it works

- Know how to configure training with Hydra

- Set up cloud GPU infrastructure

- Start a training run and monitor progress

- Understand key training hyperparameters

______________________________________________________________________

# PART A: LECTURE (20 minutes)

## 1. The Training Marathon (5 min)

Before we dive into the specifics of LoRA and Hydra, let's talk about what it means to train a model. Training isn't a sprint; it's a **marathon**. It's not just about hitting "run" and waiting for it to finish. It involves:

- **Preparation**: Setting up your environment, data, and configuration correctly.

- **Monitoring**: Keeping an eye on your training run to make sure it's progressing as expected.

- **Debugging**: Being prepared to diagnose and fix problems when they inevitably arise.

- **Patience**: Long training runs can take hours or even days. You need to be patient and methodical.

This chapter will guide you through the first steps of this marathon: preparing for and starting your training run.

______________________________________________________________________

## 2. Why Parameter-Efficient Training? (5 min)

### The Full Fine-Tuning Problem

**Traditional approach**: Update all model parameters

- HuBERT/Whisper encoder: 1.3-1.5B params

- Qwen3-8B/SmolLM3-3B decoder: 3-8B params

- **Total**: 4.3-9.5B+ parameters to train

**Problems**:

- Requires massive GPU memory (80GB+ per GPU)

- Very slow (weeks of training)

- Expensive ($1000s in compute)

- Risk of catastrophic forgetting

- Hard to reproduce

### The Solution: Parameter-Efficient Fine-Tuning (PEFT)

**Key insight**: You don't need to update all parameters!

Instead, we:

1. **Freeze** most parameters (keep pre-trained knowledge)
1. **Train projector fully** (~13-138M params, bridges audio and text)
1. **Add small LoRA adapters** (optional, for encoder and/or decoder)
1. **Train only trainable components** (~1.6% of total params with full PEFT)

**Training Modes**:

- **Full PEFT**: Projector + Encoder LoRA + Decoder LoRA (~150M params)
- **Projector + Decoder**: Frozen encoder, trainable projector + decoder LoRA (~142M params)
- **Projector Only**: Frozen encoder and decoder, only projector trains (~138M params)

**Results**:

- Faster training (24 hours vs weeks)

- Cheaper (~$12 vs $1000s)

- Less memory (40GB vs 80GB+)

- Better generalization

- Easy to share (only save adapter weights + projector)

- Flexible configuration (enable/disable LoRA components)

______________________________________________________________________

## 2. Understanding LoRA (10 min)

### What is LoRA?

**LoRA** = **Lo**w-**R**ank **A**daptation

**Core idea**: Large weight matrices can be approximated by low-rank decompositions.

### The Math (Intuitive)

Instead of updating a full weight matrix W (millions of params):

```
Normal: W_new = W_old + ΔW   (update all 4.2M parameters)
LoRA:   W_new = W_old + B×A  (update only 32K parameters!)
```

**The Key Insight**: Most updates are low-rank - they don't need millions of parameters.

**Example**: For a 2048×2048 weight matrix with rank r=8:

- Original: 2048 × 2048 = **4.2M parameters**
- LoRA: (2048×8) + (8×2048) = **32K parameters** (0.76%!)

**Visual intuition:**

```
[2048 × 2048]  =  [2048 × 8]  ×  [8 × 2048]
   4.2M params      16K          16K
   Full update      Matrix B     Matrix A
```

The "rank" (r=8) is how much information we allow the update to carry.

### How LoRA Works in Practice

```python
# Original forward pass
output = linear_layer(input)  # Uses W

# With LoRA
output = linear_layer(input) + lora_B(lora_A(input))
         └─────frozen──────┘   └────────trainable────────┘


```

**During training**:

- W stays frozen

- Only B and A get gradient updates

- Much less memory and computation

**During inference**:

- Can merge: W' = W + B×A

- No speed penalty!

- Same inference cost as original model

### LoRA Hyperparameters

**Rank (r)**:

- Controls adapter capacity

- Encoder: r=0 (disabled, frozen) or r=16 (when enabled)

- Decoder: r=0 (disabled, frozen) or r=8 (when enabled)

**Alpha (lora_alpha)**:

- Scaling factor: `scale = alpha / r`

- Encoder: alpha=16 (scale=1.0 when r=16)

- Decoder: alpha=8 (scale=1.0 when r=8)

- Controls magnitude of adapter contribution

**Target Modules**:

- Which layers get adapters

- Encoder: `attention.q_proj`, `attention.k_proj` (query and key in HuBERT attention)

- Decoder: `q_proj`, `v_proj` (query and value in decoder attention)

- More modules = more parameters but more capacity

**Dropout**:

- Regularization for adapters

- Encoder: 0.05 (light dropout)

- Decoder: 0.05 (light dropout)

- Helps prevent overfitting

### Why These Specific Configurations?

**Encoder LoRA (r=16 when enabled, or r=0 disabled)**:

- **r=0 (disabled)**: Fastest training, relies on pre-trained features
- **r=16 (enabled)**: Light adaptation for task-specific features
- Already well pre-trained on speech
- ~4M parameters when enabled

**Decoder LoRA (r=8 when enabled, or r=0 disabled)**:

- **r=0 (disabled)**: Frozen decoder, only projector adapts
- **r=8 (enabled)**: Adaptation for speech-aware text generation
- More conservative rank than encoder
- ~4M parameters when enabled

**Projector (no LoRA)**:

- Brand new component (no pre-training)

- Train fully from scratch

- ~13M parameters (simple linear) to ~138M parameters (varies by encoder/decoder dimensions)

**Configuration Examples**:

```bash
# Full PEFT (projector + encoder LoRA + decoder LoRA)
poetry run python src/train.py encoder_lora.r=16 peft.r=8

# Projector + Decoder LoRA only (frozen encoder)
poetry run python src/train.py encoder_lora.r=0 peft.r=8

# Projector only (fastest, both models frozen)
poetry run python src/train.py encoder_lora.r=0 peft.r=0
```

______________________________________________________________________

## 3. Training Configuration with Hydra (5 min)

### What is Hydra?

**Hydra**: Configuration management framework

- Compose configs from multiple files

- Override via command line

- Experiment tracking

- Clean, maintainable configs

### Tiny Audio Config Structure

```
configs/hydra/
├── config.yaml                    # Main config (ties everything together)
├── model/
│   ├── whisper_turbo.yaml        # Default: Whisper Turbo encoder
│   ├── large.yaml                # HuBERT-XLarge encoder
│   └── hubert_large.yaml         # HuBERT-Large encoder
├── data/
│   ├── loquacious_clean.yaml     # Clean LoquaciousSet
│   ├── loquacious_large.yaml     # Large LoquaciousSet
│   └── multi_task_complete.yaml  # Multi-task dataset
├── training/
│   ├── base.yaml                 # Base training hyperparameters
│   ├── production.yaml           # Production settings
│   ├── mac.yaml                  # Mac-specific settings
│   └── mac_override.yaml         # Mac overrides
├── peft/
│   └── default.yaml              # Decoder LoRA config
├── encoder_lora/
│   └── default.yaml              # Encoder LoRA config
└── experiments/
    ├── transcribe_hubert.yaml    # HuBERT transcription
    ├── transcribe_whisper.yaml   # Whisper transcription
    └── multi_task.yaml           # Multi-task training


```

### Key Training Hyperparameters

These are the most important knobs to turn when training a model. Understanding them is key to successful training.

**Learning Rate**: `1e-4`

- **What it is**: How big of a step the optimizer takes with each update.

- **Why this value?**: It's a safe, standard starting point for fine-tuning with the AdamW optimizer. The optimal learning rate is often found through experimentation (sweeps), but `1e-4` is a solid default.

- **Trade-offs**: Too high, and the training can become unstable and diverge. Too low, and the model will learn too slowly.

**Learning Rate Schedule**: `cosine`

- **What it is**: A plan for changing the learning rate over time. We don't use a fixed learning rate throughout the entire training. Instead, we use a schedule that includes:

  - **Warmup**: We start with a very low learning rate and gradually increase it to the peak value (`1e-4`) over the first `500` steps. This prevents the model from making large, destabilizing updates at the beginning of training.
  - **Decay**: After the warmup, we gradually decrease the learning rate, following a cosine curve. This allows the model to settle into a good minimum.

- **Why this is important**: A good learning rate schedule is crucial for stable and efficient training.

**Batch Size**: `8` (per device)

- **What it is**: The number of training examples processed in a single forward/backward pass.

- **Why this value?**: It's a balance between memory usage and gradient quality. A larger batch size provides a more accurate estimate of the gradient, but it also requires more GPU memory.

- **Effective Batch Size**: With `gradient_accumulation_steps=4`, our effective batch size is `8 * 4 = 32`. This means we accumulate gradients over 4 small batches before updating the model, simulating a larger batch size without the memory overhead.

**Max Steps**: `10,000`

- **What it is**: The total number of training iterations.

- **Why this value?**: This is chosen to be long enough for the model to converge on the Loquacious dataset, which takes about 24 hours on an A40 GPU.

**Mixed Precision**: `bf16`

- **What it is**: Using a 16-bit floating-point format (bfloat16) for training instead of the standard 32-bit format.

- **Why?**: It dramatically reduces memory usage and speeds up training on modern GPUs (like the A40 and H100) with minimal impact on accuracy.

### Pre-flight Checklist

Before launching a long training run, it's a good practice to go through a pre-flight checklist:

- **[ ] Infrastructure Readiness**: Is your GPU available and working correctly? (We'll do a local test run to verify this).

- **[ ] Evaluation Setup**: Are your evaluation metrics and scripts ready to go? (We'll cover this in the next chapter).

- **[ ] Checkpoint & Auto-resume**: Is your training script set up to save checkpoints periodically and resume from the latest one if it gets interrupted? (The `transformers` Trainer does this for us automatically!).

- **[ ] Logging**: Are you logging all the important metrics (loss, learning rate, etc.) to a tool like Weights & Biases? (We'll set this up in the workshop).

### A Glimpse into Scaling Laws

How do researchers at large labs decide how big of a model to train and for how long? They use **scaling laws**.

Scaling laws are empirical formulas that predict how a model's performance will improve as you increase:

- **Model size** (number of parameters)

- **Training data** (number of tokens)

- **Compute** (total FLOPs)

These laws allow researchers to make informed decisions about how to allocate their massive compute budgets. For example, the "Chinchilla" scaling laws from DeepMind suggested that for a given amount of compute, it's often better to train a smaller model on more data.

While we won't be deriving our own scaling laws in this course, it's a fascinating area of research that drives many of the decisions behind the models we use every day.

______________________________________________________________________

# PART B: HANDS-ON WORKSHOP (40 minutes)

>

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Explore configs and experiment with hyperparameters

- **Exercise 2**: Set up cloud GPU (RunPod or similar)

- **Exercise 3**: Start training and test different configurations

- **Exercise 4**: Monitor progress and experiment with metrics

By the end, you'll have a model training in the cloud and understand how to optimize it!

______________________________________________________________________

## Workshop Exercise 1: Explore Training Configs (10 min)

### Goal

Understand the training configuration files.

### Your Task

Read and understand the configuration structure.

### Instructions

**Step 1: Examine the main config**

```bash
cat configs/hydra/config.yaml


```

Look for:

- Which sub-configs are imported (model, data, training, etc.)

- Default values

- How components are composed

**Step 2: Check the experiment config**

```bash
cat configs/hydra/experiments/stage1.yaml


```

This shows the full production training setup:

- Which configs it overrides

- LoRA settings

- Dataset configuration

- Training hyperparameters

**Step 3: Compare with minimal config**

```bash
cat configs/hydra/experiments/mac_minimal.yaml


```

This is for quick local testing:

- Small dataset samples

- Fewer steps

- Same architecture

**Step 4: Create your own experiment config**

Create `configs/hydra/experiments/my_experiment.yaml`:

```yaml
# @package _global_

# Inherit from stage1 but with modifications
defaults:
  - /model: default
  - /data: loquacious
  - /training: production
  - /peft: lora
  - /encoder_lora: r8

# Custom modifications
training:
  run_name: "my-first-training"
  output_dir: "./outputs/my_experiment"
  learning_rate: 1e-4
  per_device_train_batch_size: 8
  max_steps: 10000

data:
  max_train_samples: 10000  # Use subset for faster iteration
  max_eval_samples: 500

# Your name for the model card!
model:
  system_prompt: "/no_think /system_override"

# Experiment notes
# TODO: Describe what you're testing here


```

### Success Checkpoint

- [ ] Examined config.yaml

- [ ] Understood stage1.yaml

- [ ] Created my_experiment.yaml

- [ ] Ready to customize training settings

______________________________________________________________________

## Workshop Exercise 2: Local Test Run (15 min)

### Goal

Run a quick training test on your local machine.

### Your Task

Start a minimal training run to verify everything works.

### Instructions

**Step 1: Check available compute**

```bash
# Check if you have a GPU
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}')"

# Or for Mac
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"


```

**Step 2: Run minimal test**

```bash
# Quick local test with manual overrides (10 steps)
poetry run python src/train.py \
  training.max_steps=10 \
  data.max_train_samples=50 \
  training.per_device_train_batch_size=2 \
  encoder_lora.r=0 \
  peft.r=0


```

**What happens:**

1. Downloads/loads dataset samples
1. Initializes model (encoder + projector + decoder)
1. Trains projector only (LoRA disabled for speed)
1. Trains for 10 steps
1. Saves checkpoint

**Expected output:**

```
Loading model...
✓ Loaded encoder (Whisper Turbo, frozen)
✓ Loaded projector (~13M params, trainable)
✓ Loaded decoder (SmolLM3-3B, frozen)

Loading dataset...
✓ Train samples: 50

Training...
Step 1/10 | Loss: 8.3456
Step 5/10 | Loss: 5.2345
Step 10/10 | Loss: 3.4567

✓ Training complete!
✓ Saved checkpoint to outputs/[timestamp]/


```

**Step 3: Check the output**

```bash
ls outputs/*/  # Check most recent output


```

You should see:

- `config.json` - Model configuration

- `projector.safetensors` - Projector weights (always present)

- `encoder.safetensors` - Encoder LoRA (only if encoder_lora.r > 0)

- `decoder.safetensors` - Decoder LoRA (only if peft.r > 0)

- `trainer_state.json` - Training state

### Success Checkpoint

- [ ] Training started successfully

- [ ] Saw loss decreasing over steps

- [ ] Checkpoint saved to outputs/

- [ ] No errors or crashes

**Note**: This is a minimal test! The model won't be good yet - we need full training.

### Training Experiments

**Experiment 1: Monitor loss curves**

Create a script to visualize training progress:

```python
# monitor_training.py
import json
import matplotlib.pyplot as plt

# Load trainer state
with open('outputs/mac_minimal/trainer_state.json') as f:
    state = json.load(f)

# Extract losses
steps = [h['step'] for h in state['log_history'] if 'loss' in h]
losses = [h['loss'] for h in state['log_history'] if 'loss' in h]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, marker='o')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.savefig('loss_curve.png')
print("✓ Saved loss curve to loss_curve.png")

# Analyze convergence
if len(losses) > 5:
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    print(f"Loss reduction: {reduction:.1f}%")
    print(f"Average loss per step: {sum(losses)/len(losses):.3f}")


```

**Experiment 2: Test different datasets**

```bash
# Try with different data configs
poetry run python src/train.py \
  +experiments=mac_minimal \
  data.max_train_samples=50 \
  training.max_steps=10 \
  training.run_name="tiny-test"

# Compare with more data
poetry run python src/train.py \
  +experiments=mac_minimal \
  data.max_train_samples=200 \
  training.max_steps=40 \
  training.run_name="medium-test"


```

**Experiment 3: Learning rate warmup test**

```python
# Test different warmup strategies
warmup_configs = [
    (0, "no-warmup"),
    (100, "quick-warmup"),
    (500, "standard-warmup"),
    (1000, "slow-warmup")
]

for warmup_steps, name in warmup_configs:
    print(f"\nTesting {name} ({warmup_steps} steps)...")
    # Run training with different warmup
    # Compare convergence speed


```

______________________________________________________________________

## Workshop Exercise 3: Set Up Cloud Training (15 min)

### Goal

Prepare for full-scale training on cloud GPU.

### Your Task

Set up a cloud GPU and prepare to train.

### Instructions

**Option A: Using RunPod (Recommended)**

**Step 1: Create RunPod account**

- Go to [runpod.io](https://runpod.io)

- Sign up and add credit ($20-30 recommended)

**Step 2: Launch a pod**

- Click "Deploy"

- Select "NVIDIA A40" or "A40" (40GB VRAM)

- Choose "RunPod Pytorch" template

- Click "Deploy"

- Wait for pod to start (~2 min)

**Step 3: Connect via SSH**

```bash
# Get SSH command from RunPod dashboard (looks like this)
ssh root@<pod-id>.runpod.io -p 22115 -i ~/.ssh/id_ed25519


```

**Step 4: Set up environment on pod**

```bash
# Once connected to pod
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
pip install poetry
poetry install

# Set up HuggingFace token (for pushing model)
export HF_TOKEN='your_token_here'  # Get from https://huggingface.co/settings/tokens


```

**Step 5: Start training**

```bash
# Full production training with HuBERT
poetry run python src/train.py +experiments=transcribe_hubert

# Or with Whisper encoder
poetry run python src/train.py +experiments=transcribe_whisper


```

**Step 6: Monitor (optional)**

```bash
# In a separate terminal, watch logs
ssh root@<pod-id>.runpod.io -p 22115 -i ~/.ssh/id_ed25519
tail -f tiny-audio/outputs/stage1/trainer_log.txt


```

**Option B: Using Local GPU**

If you have NVIDIA RTX 3090/4090 or better:

```bash
# Make sure CUDA is available
nvidia-smi

# Set up W&B for monitoring
export WANDB_API_KEY='your_key'  # From https://wandb.ai/settings

# Start training
poetry run python src/train.py +experiments=stage1


```

**Option C: Using Google Colab (Not Recommended for 24hr run)**

Good for testing but may disconnect:

1. Go to [colab.research.google.com](https://colab.research.google.com)
1. Upload notebook with training code
1. Use GPU runtime
1. Run training cells

### Success Checkpoint

- [ ] Cloud GPU pod is running

- [ ] SSH connection works

- [ ] Code is cloned and dependencies installed

- [ ] Ready to start full training

**Important**: Training takes ~24 hours. Don't close the connection! Use `tmux` or `screen` to keep it running:

```bash
# Start a persistent session
tmux new -s training

# Run training
poetry run python src/train.py +experiments=stage1

# Detach: Ctrl+B, then D
# Re-attach later: tmux attach -t training


```

______________________________________________________________________

# CLASS SUMMARY

## What We Covered Today

**Lecture (20 min):**

- Why parameter-efficient training matters

- LoRA's low-rank adaptation explained

- Training configuration with Hydra

- Key hyperparameters

**Workshop (40 min):**

- Explored training configs

- Ran local test training

- Set up cloud GPU infrastructure

______________________________________________________________________

## Further Reading (Optional)

### Papers

- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

- [QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314)

- [Parameter-Efficient Transfer Learning](https://arxiv.org/abs/1902.00751)

### Tools

- [Hydra documentation](https://hydra.cc/)

- [Weights & Biases](https://wandb.ai/)

- [RunPod guides](https://docs.runpod.io/)

### Code

- [PEFT library](https://github.com/huggingface/peft)

- [Training script](../../src/train.py)

[Previous: Class 3: Language Models and Projectors](./3-language-models-and-projectors.md) | [Next: Class 5: Evaluation and Debugging](./5-evaluation-and-debugging.md)

**See you next time!**
