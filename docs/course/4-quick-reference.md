# Quick Reference Guide

A one-page reference for the most common commands and concepts.

______________________________________________________________________

## Essential Commands

### Development

```bash
# Install dependencies
poetry install

# Run the demo
poetry run ta demo --model mazesmazes/tiny-audio

# Evaluate a model
poetry run ta eval -m your-username/your-model -d loquacious -n 100

# Run all checks (lint, type-check, test)
poetry run ta dev check

# Run tests
poetry run ta dev test
```

### Training (Cloud GPU)

```bash
# Deploy to RunPod
poetry run ta runpod deploy <HOST> <PORT>

# Start training
export HF_TOKEN='hf_your_token'
poetry run ta runpod train <HOST> <PORT> --experiment my_experiment

# Attach to running session
poetry run ta runpod attach <HOST> <PORT>

# Find latest checkpoint
poetry run ta runpod checkpoint <HOST> <PORT>
```

### Training (Local)

```bash
# Train with default MLP projector
poetry run python scripts/train.py +experiments=mlp

# Train with different projector
poetry run python scripts/train.py +experiments=mosa
poetry run python scripts/train.py +experiments=moe
poetry run python scripts/train.py +experiments=qformer

# Override config values
poetry run python scripts/train.py +experiments=mlp training.learning_rate=1e-4

# Resume from checkpoint
poetry run python scripts/train.py +experiments=mlp training.resume_from_checkpoint=/path/to/checkpoint

# Multi-stage training with LoRA
poetry run python scripts/train.py +experiments=mlp_lora       # Stage 2
poetry run python scripts/train.py +experiments=mlp_fine_tune  # Stage 3
```

### Evaluation & Analysis

```bash
# Basic evaluation
poetry run ta eval -m your-model -n 100

# Evaluate on specific dataset
poetry run ta eval -m your-model -d earnings22 -n 100

# Find high-error samples
poetry run ta analysis high-wer your-model --threshold 30

# Compare models
poetry run ta analysis compare model1 model2

# Find entity errors
poetry run ta analysis entity-errors your-model

# Debug model health
poetry run ta debug check-mosa your-model
poetry run ta debug analyze-lora your-model
```

### Deployment

```bash
# Push model to HuggingFace Hub
poetry run ta push --repo-id your-username/your-model

# Deploy to HuggingFace Spaces
poetry run ta deploy --repo-id your-username/your-space
```

______________________________________________________________________

## Architecture

```
Audio → GLM-ASR Encoder (frozen) → Projector (trained) → Qwen3 (frozen) → Text
```

Only the projector (~12M params) is trained. Encoder and decoder remain frozen.

| Component | Model | Parameters | Status |
|-----------|-------|------------|--------|
| Audio Encoder | GLM-ASR-Nano-2512 | ~600M | Frozen |
| Projector | MLP (2-layer) | ~12M | **Trained** |
| Language Model | Qwen3-0.6B | ~600M | Frozen |

______________________________________________________________________

## Projector Types

| Type | Description | Use Case |
|------|-------------|----------|
| `mlp` | 2-layer MLP with frame stacking | Default, fast training |
| `mosa` | Dense mixture of experts | Better accuracy, more VRAM |
| `moe` | Shared + sparse routed experts | Balance of speed and accuracy |
| `qformer` | QFormer with learnable queries | Advanced, BLIP-2 style |

______________________________________________________________________

## Multi-Stage Training

| Stage | Config | What trains | Purpose |
|-------|--------|-------------|---------|
| 1 | `+experiments=mlp` | Projector only | Learn audio→text mapping |
| 2 | `+experiments=mlp_lora` | LoRA adapters only | Fine-tune LLM |
| 3 | `+experiments=mlp_fine_tune` | Projector + LoRA | Joint optimization |

______________________________________________________________________

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.max_steps` | 50,000 | Training duration |
| `training.per_device_train_batch_size` | 4 | Samples per step |
| `training.learning_rate` | 1e-4 | Update aggressiveness |
| `training.warmup_steps` | 1000 | LR warmup period |
| `model.projector_pool_stride` | 5 | Frame stacking factor |

______________________________________________________________________

## Evaluation Datasets

| Dataset | Command | What it tests |
|---------|---------|---------------|
| LoquaciousSet | `-d loquacious` | General benchmark (default) |
| Earnings22 | `-d earnings22` | Financial domain |
| AMI | `-d ami` | Meeting transcription |

______________________________________________________________________

## Training Metrics

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| Training Loss | Decreasing | Stuck or increasing |
| Eval Loss | Decreasing | Rising (overfitting) |
| Gradient Norm | 1-10 | >100 (instability) |

**The "Cliff"**: Training loss often plateaus for 1000-1500 steps, then drops suddenly. This is normal!

______________________________________________________________________

## Config File Structure

```
configs/
├── config.yaml          # Main config (model defaults)
├── data/
│   └── loquacious.yaml  # Dataset configuration
├── training/
│   └── production.yaml  # Training hyperparameters
└── experiments/         # Projector presets
    ├── mlp.yaml         # Stage 1: MLP projector
    ├── mosa.yaml        # MOSA projector
    ├── moe.yaml         # MoE projector
    ├── qformer.yaml     # QFormer projector
    ├── mlp_lora.yaml    # Stage 2: LoRA only
    └── mlp_fine_tune.yaml  # Stage 3: Projector + LoRA
```

______________________________________________________________________

## CLI Reference

| Command | Description |
|---------|-------------|
| `ta eval` | Evaluate ASR models |
| `ta analysis` | WER analysis (high-wer, entity-errors, compare) |
| `ta demo` | Launch Gradio demo |
| `ta deploy` | Deploy to HF Spaces |
| `ta push` | Push model to HF Hub |
| `ta debug` | Debug utilities |
| `ta runpod` | Remote training |
| `ta dev` | Development tools |

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | Model ID |
| `--datasets` | `-d` | Datasets |
| `--max-samples` | `-n` | Sample limit |
| `--output-dir` | `-o` | Output path |
| `--num-workers` | `-w` | Parallel workers |

______________________________________________________________________

## Tmux Cheat Sheet

| Action | Keys |
|--------|------|
| Detach | `Ctrl+B`, then `D` |
| Scroll up | `Ctrl+B`, then `[` |
| Exit scroll | `q` |
| Stop training | `Ctrl+C` |

______________________________________________________________________

## Common Issues

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce batch size |
| SSH fails | Check RunPod SSH key |
| Slow training | Check GPU utilization |
| Loss not decreasing | Try different learning rate |
| Model outputs gibberish | Wait for "cliff" (~1500 steps) |
| Import errors | Run `poetry install` |
| Hydra config error | Use `key=value` not `--key value` |

______________________________________________________________________

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace authentication |
| `WANDB_API_KEY` | Weights & Biases |
| `ASSEMBLYAI_API_KEY` | AssemblyAI evaluation |
| `DEEPGRAM_API_KEY` | Deepgram evaluation |

______________________________________________________________________

## Account URLs

- GitHub: [github.com](https://github.com)
- Hugging Face: [huggingface.co](https://huggingface.co)
- Weights & Biases: [wandb.ai](https://wandb.ai)
- RunPod: [runpod.io](https://runpod.io)

______________________________________________________________________

## Key Formulas

**Frame stacking**:
```
output_length = (input_length - k) // k + 1
```
Where `k` is the pooling stride (default: 5).

**Word Error Rate**:
```
WER = (Substitutions + Insertions + Deletions) / Total Reference Words
```

______________________________________________________________________

[← Class 3: Evaluation](./3-evaluation-and-deployment.md) | [Glossary →](./5-glossary.md)
