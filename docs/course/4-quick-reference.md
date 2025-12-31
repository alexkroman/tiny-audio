# Quick Reference Guide

A one-page reference for the most common commands and concepts.

______________________________________________________________________

## Essential Commands

### Development

```bash
# Install dependencies
poetry install

# Run the demo
poetry run python demo/gradio/app.py --model mazesmazes/tiny-audio

# Evaluate a model
poetry run eval your-username/your-model --dataset loquacious --max-samples 100

# Visualize data flow
poetry run python docs/course/examples/trace_data.py

# Run all checks (lint, type-check, test)
poetry run check
```

### Training (Cloud GPU)

```bash
# Deploy to RunPod
poetry run deploy-runpod <HOST> <PORT>

# Start training
export HF_TOKEN='hf_your_token'
poetry run remote-train <HOST> <PORT> --experiment my_experiment

# Attach to running session
poetry run attach-remote <HOST> <PORT>

# Find latest checkpoint
poetry run find-checkpoint <HOST> <PORT>
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
poetry run python scripts/train.py training.learning_rate=1e-4

# Resume from checkpoint
poetry run python scripts/train.py +experiments=mlp training.resume_from_checkpoint=/path/to/checkpoint
```

______________________________________________________________________

## Architecture

```
Audio → Whisper Encoder (frozen) → Projector (trained) → SmolLM3 (frozen) → Text
```

Only the projector (~12M params) is trained. See [README](../../README.md#how-it-works-the-tiny-audio-architecture) for details.

______________________________________________________________________

## Projector Types

| Type | Description | Use Case |
|------|-------------|----------|
| `mlp` | 2-layer MLP with frame stacking | Default, fast training |
| `mosa` | Dense mixture of experts | Better accuracy, more VRAM |
| `moe` | Shared + sparse routed experts | Balance of speed and accuracy |
| `qformer` | QFormer with learnable queries | Advanced, BLIP-2 style |

______________________________________________________________________

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 15,000 | Training duration |
| `per_device_train_batch_size` | 14 | Samples per step |
| `learning_rate` | 1e-3 | Update aggressiveness |
| `projector_pool_stride` | 4 | Temporal downsampling factor |

______________________________________________________________________

## Evaluation Datasets

| Dataset | Command | What it tests |
|---------|---------|---------------|
| LoquaciousSet | `--dataset loquacious` | General benchmark |
| Earnings22 | `--dataset earnings22` | Financial domain |
| AMI | `--dataset ami` | Meeting transcription |

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
    ├── mlp.yaml
    ├── mosa.yaml
    ├── moe.yaml
    └── qformer.yaml
```

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
| Slow training | Check GPU utilization in RunPod |
| Loss not decreasing | Try different learning rate |

______________________________________________________________________

## Account URLs

- GitHub: [github.com](https://github.com)
- Hugging Face: [huggingface.co](https://huggingface.co)
- Weights & Biases: [wandb.ai](https://wandb.ai)
- RunPod: [runpod.io](https://runpod.io)

______________________________________________________________________

[← Class 3: Evaluation](./3-evaluation-and-deployment.md) | [Glossary →](./5-glossary.md)
