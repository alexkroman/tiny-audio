# Build Your Own Speech Recognition Model

## Course Overview

Welcome to a hands-on course on building modern AI. This isn't just about learning how speech recognition works—it's about building, training, and deploying your own ASR model from scratch. By the end, you'll have a real, working model with your name on it, and you'll understand exactly how it works.

**What you'll accomplish:**

- Understand the encoder-projector-decoder architecture used in modern ASR
- Train your own model by keeping large models frozen and only training the projector
- Work with ~13M trainable parameters instead of 4.3+ billion
- Publish your model to Hugging Face Hub
- Deploy a live demo anyone can use
- Add your results to the community leaderboard

**Course Details:**

| | |
|---|---|
| **Duration** | 3.5 hours (3 classes) |
| **Cost** | ~$8-12 for cloud GPU training |
| **Prerequisites** | Basic Python, command line, git |
| **Architecture** | Whisper encoder + MLP projector + SmolLM3 decoder |

**Format options:**

- **Self-paced** — Work through materials on your own
- **Study group** — Learn with friends
- **Workshop** — Great for teaching ASR and ML engineering

______________________________________________________________________

## Course Structure

### [Class 1: Introduction, Architecture, and Setup](./1-introduction-and-setup.md)

**Duration**: 1.5 hours (40 min lecture + 50 min hands-on)

- What is automatic speech recognition?
- The encoder-projector-decoder architecture
- How audio becomes embeddings, then text
- Setting up your development environment
- Running inference with a pre-trained model
- Visualizing data flow through the pipeline

### [Class 2: Training](./2-training.md)

**Duration**: 1 hour (15 min intro + 45 min hands-on)

- Setting up RunPod for cloud GPU training
- Understanding training metrics (loss, gradient norm)
- Configuring experiments with Hydra
- Launching and monitoring training runs
- Managing costs (terminate unused instances!)

### [Class 3: Evaluation and Deployment](./3-evaluation-and-deployment.md)

**Duration**: 1 hour (15 min lecture + 45 min hands-on)

- Understanding Word Error Rate (WER)
- Evaluating on multiple datasets (LoquaciousSet, Earnings22, AMI)
- Deploying a live demo to Hugging Face Spaces
- Setting up Hugging Face Inference Endpoints
- Adding your results to the leaderboard

______________________________________________________________________

## Prerequisites

### Required

- Python programming basics
- Command line / terminal usage
- Git basics

### Helpful but not required

- PyTorch fundamentals
- Hugging Face Transformers experience
- Machine learning concepts (embeddings, attention)

Don't worry if you're missing some prerequisites—the course teaches what you need as you go.

______________________________________________________________________

## Hardware Requirements

### For Training (Class 2)

- **Cloud GPU (recommended)**: NVIDIA A40 48GB on RunPod (~$0.40/hour)
- **Local GPU**: NVIDIA RTX 3090/4090 24GB+
- **Apple Silicon**: M1/M2/M3 Max/Ultra with 32GB+ RAM (slower)

### For Development (Classes 1 & 3)

- Any modern laptop
- 8GB RAM minimum
- 20GB free disk space

______________________________________________________________________

## Accounts You'll Need

Create these free accounts before starting:

1. **GitHub** — [github.com](https://github.com) (for code)
1. **Hugging Face** — [huggingface.co](https://huggingface.co) (for models and demos)
1. **Weights & Biases** — [wandb.ai](https://wandb.ai) (for training monitoring)
1. **RunPod** — [runpod.io](https://runpod.io) (for cloud GPUs, Class 2)

______________________________________________________________________

## Support

- **GitHub Issues**: Report bugs or ask questions
- **Discussions**: Share your results and learnings
- **Leaderboard**: See how your model compares to others

______________________________________________________________________

[Next: Class 1 →](./1-introduction-and-setup.md)
