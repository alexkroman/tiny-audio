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

---

# PART A: LECTURE (20 minutes)

> **Instructor**: Present these concepts. Students should just listen.

## 1. Why Parameter-Efficient Training? (5 min)

### The Full Fine-Tuning Problem

**Traditional approach**: Update all model parameters

- HuBERT encoder: 1.3B params
- SmolLM3 decoder: 3B params
- **Total**: 4.3B+ parameters to train

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
2. **Add small adapters** that learn the specific task
3. **Train only adapters** (~3% of total params)

**Results**:

- Faster training (24 hours vs weeks)
- Cheaper (~$12 vs $1000s)
- Less memory (40GB vs 80GB+)
- Better generalization
- Easy to share (only save adapter weights)

---

## 2. Understanding LoRA (10 min)

### What is LoRA?

**LoRA** = **Lo**w-**R**ank **A**daptation

**Core idea**: Large weight matrices can be approximated by low-rank decompositions.

### The Math (Simplified)

Normal training updates weight matrix W:

```
W_new = W_old + ΔW
```

LoRA approximates ΔW with two small matrices:

```
ΔW ≈ B × A
```

Where:

- W is large (e.g., 2048 × 2048 = 4.2M params)
- B is tall and thin (2048 × 8 = 16K params)
- A is short and wide (8 × 2048 = 16K params)
- **Total**: 32K params instead of 4.2M! (0.76%)

**Rank (r)**: The middle dimension (8 in this example)

- Lower rank = fewer parameters, less capacity
- Higher rank = more parameters, more capacity

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
- Encoder: r=8 (conservative)
- Decoder: r=64 (more capacity for language task)

**Alpha (lora_alpha)**:

- Scaling factor: `scale = alpha / r`
- Encoder: alpha=8 (scale=1.0)
- Decoder: alpha=32 (scale=0.5)
- Controls magnitude of adapter contribution

**Target Modules**:

- Which layers get adapters
- Encoder: q_proj, k_proj (query and key in attention)
- Decoder: q_proj, v_proj (query and value in attention)
- More modules = more parameters but more capacity

**Dropout**:

- Regularization for adapters
- We use 0.0 (no dropout)
- Pre-trained models already well-regularized

### Why These Specific Configurations?

**Encoder (r=8, small)**:

- Already well pre-trained on speech
- Just needs small adjustments
- ~2M parameters

**Decoder (r=64, larger)**:

- Bigger adaptation needed (text → speech-aware text)
- More capacity for language generation
- ~15M parameters

**Projector (no LoRA)**:

- Brand new component (no pre-training)
- Train fully from scratch
- ~122M parameters

---

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
├── config.yaml              # Main config (ties everything together)
├── model/
│   ├── default.yaml         # Model architecture
│   └── hubert_xlarge.yaml   # Encoder variant
├── data/
│   └── loquacious.yaml      # Dataset config
├── training/
│   ├── default.yaml         # Training hyperparameters
│   └── production.yaml      # Production settings
├── peft/
│   └── lora.yaml           # Decoder LoRA config
├── encoder_lora/
│   └── r8.yaml             # Encoder LoRA config
└── experiments/
    ├── stage1.yaml         # Full training preset
    └── mac_minimal.yaml    # Local testing preset
```

### Key Training Hyperparameters

**Learning Rate**: `1e-4`

- How fast the model learns
- Too high = unstable, diverges
- Too low = slow convergence

**Batch Size**: `8` (per device)

- How many samples per gradient update
- Larger = more stable, faster, more memory
- Smaller = less memory, noisier gradients

**Gradient Accumulation**: `4` steps

- Effective batch size = 8 × 4 = 32
- Simulates larger batches with less memory

**Max Steps**: `10,000`

- Total training iterations
- ~24 hours on A40

**Warmup Steps**: `500`

- Gradually increase learning rate
- Prevents early instability

**Mixed Precision**: `bf16`

- Brain floating point 16-bit
- Faster, less memory, similar accuracy

---

# PART B: HANDS-ON WORKSHOP (40 minutes)

> **Students**: Follow these instructions step-by-step.
>
> **Instructor**: Circulate and help students.

---

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

---

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
# This will train for just 20 steps (~5-10 minutes)
poetry run python src/train.py +experiments=mac_minimal
```

**What happens:**

1. Downloads/loads dataset samples
2. Initializes model with LoRA adapters
3. Trains for 20 steps
4. Saves checkpoint

**Expected output:**

```
Loading model...
✓ Loaded encoder (HuBERT-XLarge + LoRA r=8)
✓ Loaded decoder (SmolLM3-3B + LoRA r=64)
✓ Loaded projector (122M params)

Loading dataset...
✓ Train samples: 100
✓ Eval samples: 50

Training...
Step 1/20 | Loss: 12.3456
Step 5/20 | Loss: 8.2345
Step 10/20 | Loss: 5.1234
...
Step 20/20 | Loss: 3.4567

✓ Training complete!
✓ Saved checkpoint to outputs/mac_minimal/
```

**Step 3: Check the output**

```bash
ls outputs/mac_minimal/
```

You should see:

- `config.json` - Model configuration
- `model.safetensors` - Trained weights (projector + LoRA)
- `training_args.bin` - Training settings
- `trainer_state.json` - Training state

### Success Checkpoint

- [ ] Training started successfully
- [ ] Saw loss decreasing over steps
- [ ] Checkpoint saved to outputs/
- [ ] No errors or crashes

**Note**: This is a minimal test! The model won't be good yet - we need full training.

---

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
# Full production training
poetry run python src/train.py +experiments=stage1
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
2. Upload notebook with training code
3. Use GPU runtime
4. Run training cells

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

---

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

## Key Takeaways

✅ LoRA trains only ~3% of parameters (139M / 4.3B)
✅ Low-rank decomposition: ΔW ≈ B × A
✅ Encoder uses r=8, decoder uses r=64
✅ Hydra manages complex configurations
✅ Training takes ~24 hours on A40 (~$12)

## Homework

**Required** (Do this before next class!):

1. Start your full training run on cloud GPU
2. Check back every few hours to ensure it's running
3. Training should complete before Class 5

**Optional**:

1. Set up Weights & Biases for training monitoring
2. Experiment with different hyperparameters
3. Read about LoRA paper (link below)

## Check Your Understanding

1. **What problem does LoRA solve?**
   - Makes fine-tuning feasible with limited compute
   - Trains small adapters instead of full model
   - Reduces memory, time, and cost

2. **How does LoRA work?**
   - Approximates weight updates with low-rank matrices
   - ΔW ≈ B × A where B and A are small
   - Only B and A are trainable

3. **Why different ranks for encoder vs decoder?**
   - Encoder (r=8): Already well pre-trained, needs small adjustments
   - Decoder (r=64): Needs more capacity for text generation task

4. **What's the effective batch size?**
   - per_device_batch_size × gradient_accumulation_steps
   - 8 × 4 = 32 in our config

---

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

---

## Next Class

In [Class 5: Evaluation and Debugging](./5-evaluation-and-debugging.md), we'll:

- Evaluate your trained model
- Calculate Word Error Rate (WER)
- Debug common training issues
- Improve model performance

**Prerequisites**: Have your training run completed or nearly done!

[Previous: Class 3: Language Models and Projectors](./3-language-models-and-projectors.md) | [Next: Class 5: Evaluation and Debugging](./5-evaluation-and-debugging.md)

**See you next time!**
