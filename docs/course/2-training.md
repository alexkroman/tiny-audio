# Class 2: Training

*1 hour (15 min intro + 45 min hands-on)*

**Goal**: Set up cloud training, configure an experiment, and launch training.

---

## Part A: Lecture (15 min)

### How Training Works

1. **Downloads models**: GLM-ASR encoder + Qwen decoder from Hugging Face
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

## Projector Types

| Type | Description | Speed |
|------|-------------|-------|
| `mlp` | 2-layer MLP with 4x downsampling | Fast |
| `mosa` | Dense MoE, all 4 experts contribute | Slow |
| `moe` | Shared + sparse routed experts | Medium |

Start with `mlp`. Try others after you have a baseline.

---

## Key Takeaways

1. Only the projector trains (~12M params)
2. Loss drops dramatically around step 1500 ("the cliff")
3. Monitor eval loss for overfitting
4. **Terminate RunPod instances when done**

---

[← Class 1](./1-introduction-and-setup.md) | [Class 3: Evaluation →](./3-evaluation-and-deployment.md)
