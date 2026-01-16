# Class 2: Training

*1 hour (15 min intro + 45 min hands-on)*

**Goal**: Set up cloud training, configure an experiment, and launch training.

---

## Part A: Lecture (15 min)

### How Training Works

1. **Downloads models**: GLM-ASR encoder + Qwen3 decoder from Hugging Face
2. **Streams data**: Training data streams (no terabyte downloads)
3. **Trains projector**: Only the MLP projector trains; encoder and decoder frozen
4. **Saves checkpoints**: Every 1,000 steps to Hugging Face

### Key Metrics

| Metric | What to look for |
|--------|------------------|
| **Training Loss** | Should decrease over time |
| **Eval Loss** | Should decrease; rising = overfitting |
| **Gradient Norm** | Starts high (40+), drops to <10 |

### The "Cliff" Phenomenon

- Steps 0-1500: Model outputs gibberish
- Steps 1500-1600: Suddenly clicks—goes from nothing to decent ASR
- Steps 1600+: Gradual improvement

This is normal. Don't panic if it seems broken for the first hour.

### Multi-Stage Training

For best results, training happens in stages:

| Stage | What trains | Config | Purpose |
|-------|-------------|--------|---------|
| **Stage 1** | Projector only | `+experiments=mlp` | Learn audio→text mapping |
| **Stage 2** | LoRA adapters only | `+experiments=mlp_lora` | Fine-tune language model |
| **Stage 3** | Projector + LoRA | `+experiments=mlp_fine_tune` | Joint optimization |

Most users only need Stage 1. Stages 2 and 3 can improve accuracy but require more time.

---

## Part B: Hands-On (45 min)

### Exercise 1: Set Up RunPod (10 min)

1. Sign up at [runpod.io](https://runpod.io)
2. Add SSH key (Settings → SSH Public Keys):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   cat ~/.ssh/id_ed25519.pub
   ```
3. Deploy an **NVIDIA A40** with **RunPod PyTorch 2.x** template
4. Note the **host** and **port** from pod details

### Exercise 2: Deploy Your Code (10 min)

```bash
cd tiny-audio
git pull origin main

# Deploy to RunPod
poetry run ta runpod deploy <HOST> <PORT>
```

Takes 5-10 minutes. Re-run whenever you make local changes.

### Exercise 3: Configure Experiment (10 min)

```bash
cp configs/experiments/mlp.yaml configs/experiments/my_experiment.yaml
```

Edit `my_experiment.yaml`:

```yaml
model:
  projector_type: mlp

training:
  hub_model_id: "your-username/tiny-audio-yourname"  # CHANGE THIS
```

Re-deploy to sync:

```bash
poetry run ta runpod deploy <HOST> <PORT> --skip-setup
```

### Exercise 4: Start Training (10 min)

```bash
export HF_TOKEN='hf_your_token_here'
poetry run ta runpod train <HOST> <PORT> --experiment my_experiment
```

When prompted for W&B, press **2** and paste your key from [wandb.ai/authorize](https://wandb.ai/authorize).

**Output:**

```
Step 25/20000 | Loss: 8.34 | Grad Norm: 45.2 | LR: 1e-4 | ETA: 20hr
```

### Exercise 5: Monitor and Manage (5 min)

**Detach/Reattach:**

```bash
# Detach: Ctrl+B, then D
# Reattach:
poetry run ta runpod attach <HOST> <PORT>
```

**Monitor in W&B** at [wandb.ai](https://wandb.ai):
- `train/loss` should decrease
- `eval/loss` should also decrease (rising = overfitting)

**When to cancel** (Ctrl+C):
- Loss not decreasing after several hundred steps
- Eval loss rising while training loss falls

### CRITICAL: Terminate Unused Instances

**RunPod charges by the hour whether you're using it or not!**

| Action | Effect |
|--------|--------|
| **Stop** | Pauses billing, keeps data |
| **Terminate** | Stops billing, deletes data |

Always terminate when done. Re-deploy takes 5-10 minutes.

---

## Local Training (Optional)

If you have a local GPU (24GB+ VRAM):

```bash
# Quick test (10 steps)
poetry run python scripts/train.py +experiments=mlp training.max_steps=10

# Full training
poetry run python scripts/train.py +experiments=mlp

# Override settings
poetry run python scripts/train.py +experiments=mlp training.learning_rate=1e-4

# Resume from checkpoint
poetry run python scripts/train.py +experiments=mlp training.resume_from_checkpoint=/path/to/checkpoint-XXXX
```

---

## Projector Types

| Type | Description | Speed | VRAM |
|------|-------------|-------|------|
| `mlp` | 2-layer MLP with frame stacking | Fast | Low |
| `mosa` | Dense MoE, all experts contribute | Slow | High |
| `moe` | Shared + sparse routed experts | Medium | Medium |
| `qformer` | QFormer with learnable queries | Slow | High |

Start with `mlp`. Try others after you have a baseline.

```bash
# Different projectors
poetry run python scripts/train.py +experiments=mlp
poetry run python scripts/train.py +experiments=mosa
poetry run python scripts/train.py +experiments=moe
poetry run python scripts/train.py +experiments=qformer
```

---

## Advanced: Multi-Stage Training with LoRA

After Stage 1 training, you can optionally fine-tune the language model with LoRA:

**Stage 2: Train LoRA adapters (freeze projector)**

```bash
poetry run python scripts/train.py +experiments=mlp_lora \
    training.resume_from_checkpoint=/path/to/stage1-checkpoint
```

**Stage 3: Fine-tune both projector and LoRA**

```bash
poetry run python scripts/train.py +experiments=mlp_fine_tune \
    training.resume_from_checkpoint=/path/to/stage2-checkpoint
```

LoRA adds ~1-2M trainable parameters to the language model without full fine-tuning.

---

## Configuration Reference

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.max_steps` | 50000 | Total training steps |
| `training.learning_rate` | 1e-4 | Learning rate |
| `training.per_device_train_batch_size` | 4 | Batch size per GPU |
| `training.gradient_accumulation_steps` | 4 | Effective batch = batch_size * accumulation |
| `training.warmup_steps` | 1000 | LR warmup steps |
| `training.save_steps` | 1000 | Checkpoint frequency |

### Key Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.audio_model_id` | `zai-org/GLM-ASR-Nano-2512` | Audio encoder |
| `model.text_model_id` | `Qwen/Qwen3-0.6B` | Language model |
| `model.projector_type` | `mlp` | Projector architecture |
| `model.projector_pool_stride` | 5 | Frame stacking factor |

---

## Key Takeaways

1. Only the projector trains (~12M params)
2. Loss drops dramatically around step 1500 ("the cliff")
3. Monitor eval loss for overfitting
4. Multi-stage training with LoRA can improve accuracy
5. **Terminate RunPod instances when done**

---

[← Class 1](./1-introduction-and-setup.md) | [Class 3: Evaluation →](./3-evaluation-and-deployment.md)
