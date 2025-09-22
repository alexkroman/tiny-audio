# Tiny Audio - Learn ASR by Building One

A minimal (~300 line) speech recognition model that combines Whisper's audio understanding with Hugging Face SmolLM2's text generation. Perfect for learning how modern ASR systems work by building and training your own.

## Why This Project

- **Actually Tiny**: The core model is just 300 lines of readable Python - small enough to understand completely
- **Modern Architecture**: Combines a frozen Whisper encoder (for audio) with Hugging Face SmolLM2 decoder (for text) using LoRA adapters and a custom projection layer
- **Trains on your Laptop**: Works on your laptop (even M1/M2 Macs!) or scale up to GPUs
- **Real Datasets**: Train on actual speech data from LibriSpeech, GigaSpeech, and Common Voice
- **See Progress Live**: Watch your model improve in real-time with TensorBoard
- **Experiment Freely**: Simple config files let you try different ideas without touching code

## Installation

```bash
# Clone the repository
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio

# Install with uv
uv sync
```

## Quick Start

### Training

```bash
# Quick test run (20 steps, ~2 minutes)
python src/train.py

# Production training with larger datasets
python src/train.py +experiments=production
```

### Demo Usage

```bash
# Run with default settings (uses untrained model)
python demo/gradio_app.py

# Run with a trained model
python demo/gradio_app.py --model outputs/2025-09-22/12-51-14/outputs/mac_minimal_model

# Specify custom outputs directory
python demo/gradio_app.py --outputs-dir path/to/audio/files

# Create public link for sharing
python demo/gradio_app.py --share
```

## Model Architecture

The system combines three key components:

1. **Whisper Encoder**: Frozen `whisper-small` model for audio feature extraction (39M parameters, not trained)
2. **Audio Projector**: Lightweight projection layer with LLamaRMSNorm and GELU activation to map audio features to text space
3. **Hugging Face SmolLM2 Decoder**: Efficient language model with LoRA adapters (360M or 1.7B parameters, ~2% trained with LoRA)

```python
Audio Input → Whisper Encoder → Audio Projector → Hugging Face SmolLM2 + LoRA → Text Output
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/hydra/config.yaml` - Base configuration
- `configs/hydra/model/` - Model architecture settings
- `configs/hydra/data/` - Dataset configurations
- `configs/hydra/training/` - Training hyperparameters
- `configs/hydra/experiments/` - Pre-configured experiment setups

### Example: Override Configuration

```bash
# Change model size
python src/train.py model=large

# Use different dataset
python src/train.py data=production_streaming

# Modify training parameters
python src/train.py training.max_steps=10000 training.eval_steps=500
```

## Datasets

Supports multiple ASR datasets through Hugging Face datasets:

- **LibriSpeech**: Clean and other subsets for English ASR
- **GigaSpeech**: Large-scale English speech recognition
- **Common Voice**: Multilingual community-collected speech data

Datasets are automatically downloaded and cached locally.

## Cloud Training with RunPod

The `scripts/` directory contains utilities for deploying and managing training on RunPod GPUs:

### Scripts Overview

- **`deploy_runpod.py`**: Syncs your code to a RunPod instance and sets up dependencies
- **`start_remote_training.py`**: Starts a training session in a tmux window on the remote instance
- **`attach_remote_session.py`**: Attaches to an existing tmux training session for monitoring

### Quick RunPod Setup

1. Create a RunPod account and launch a GPU pod with PyTorch template
2. Add your SSH key to the pod
3. Deploy and start training:

```bash
# Deploy code and setup environment (replace with your pod's host and port)
python scripts/deploy_runpod.py --host <your-pod-id>.runpod.io --port 22

# Start training in a tmux session
python scripts/start_remote_training.py --host <your-pod-id>.runpod.io --port 22 --config production

# Later, attach to monitor progress
python scripts/attach_remote_session.py --host <your-pod-id>.runpod.io --port 22
```

The scripts handle:

- Code synchronization via rsync
- Dependency installation with uv
- Session management with tmux (so training continues if you disconnect)
- Automatic model checkpointing

## Monitoring

Training progress is logged to TensorBoard:

```bash
# View training metrics
tensorboard --logdir outputs/

# Metrics tracked:
# - Training/validation loss
# - Word Error Rate (WER)
# - Learning rate schedule
# - Sample predictions
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
