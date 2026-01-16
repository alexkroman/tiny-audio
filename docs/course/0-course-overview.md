# Build Your Own Speech Recognition Model

A hands-on course: build, train, and deploy your own ASR model from scratch.

| | |
|---|---|
| **Duration** | 3.5 hours |
| **Cost** | ~$12 for GPU training |
| **Prerequisites** | Python, command line, git |

## What You'll Build

```
Audio → GLM-ASR Encoder (frozen) → MLP Projector (trained) → Qwen3 (frozen) → Text
```

Train only ~12M parameters instead of billions. Publish to Hugging Face. Deploy a live demo.

---

## Course Structure

### [Class 1: Introduction and Setup](./1-introduction-and-setup.md)
*1.5 hours*

- Encoder-projector-decoder architecture
- How audio becomes text
- Environment setup
- Running inference

### [Class 2: Training](./2-training.md)
*1 hour*

- Cloud GPU setup (RunPod)
- Training metrics and monitoring
- Hydra configuration
- Multi-stage training with LoRA
- Cost management

### [Class 3: Evaluation and Deployment](./3-evaluation-and-deployment.md)
*1 hour*

- Word Error Rate (WER)
- Multi-dataset evaluation
- Error analysis and debugging
- Deploy to Hugging Face Spaces

### [Quick Reference](./4-quick-reference.md)

Commands, hyperparameters, troubleshooting.

### [Glossary](./5-glossary.md)

Key terms defined.

---

## Requirements

**Training (Class 2):**
- Cloud: NVIDIA A40 on RunPod (~$0.40/hr)
- Local: RTX 3090/4090 24GB+ or Apple Silicon 32GB+

**Development (Classes 1 & 3):**
- Any modern laptop, 8GB RAM, 20GB disk

---

## Accounts Needed

1. [GitHub](https://github.com)
2. [Hugging Face](https://huggingface.co)
3. [Weights & Biases](https://wandb.ai)
4. [RunPod](https://runpod.io) (for Class 2)

---

[Start Class 1 →](./1-introduction-and-setup.md)
