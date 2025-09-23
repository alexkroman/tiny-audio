# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Automatic Speech Recognition (ASR) training pipeline that combines a Whisper encoder with a SmolLM2 decoder using LoRA for parameter-efficient fine-tuning. The project uses Hydra for configuration management and supports both local and remote training on RunPod.

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies with uv (preferred)
uv sync

# Or install with pip
pip install -e .
```

### Training
```bash
# Run training with default config (Mac settings)
uv run python src/train.py

# Run with specific experiment config
uv run python src/train.py +experiments=mac_minimal
uv run python src/train.py +experiments=production

# Override specific parameters
uv run python src/train.py training.max_steps=100 model.lora_r=64

# Resume from checkpoint
uv run python src/train.py resume_from_checkpoint=outputs/2025-09-22/12-51-14/checkpoint-500
```

### Code Quality
```bash
# Run linter
uv run ruff check src/

# Format code
uv run ruff format src/

# Type checking
uv run mypy src/

# Run all checks (lint + format check)
uv run ruff check src/ && uv run ruff format --check src/
```

### Demo Application
```bash
# Run the Gradio demo with a trained model
uv run python demo/gradio_app.py --model outputs/2025-09-22/12-51-14/outputs/mac_minimal_model

# Run demo with default model search (looks for latest model)
uv run python demo/gradio_app.py
```

### Remote Training (RunPod)
```bash
# Deploy to RunPod (requires host and port)
uv run python scripts/deploy_runpod.py --host <pod-id>.runpod.io --port 22

# Start remote training
uv run python scripts/start_remote_training.py --host <pod-id>.runpod.io --port 22 --config production

# Attach to remote session
uv run python scripts/attach_remote_session.py --host <pod-id>.runpod.io --port 22
```

### Monitoring
```bash
# Start TensorBoard to monitor training
uv run tensorboard --logdir outputs/
```

## Architecture

### Core Components

1. **ASRModel** (`src/modeling.py:151`): Main model combining:
   - `WhisperEncoder`: Frozen Whisper-small encoder for audio feature extraction (39M params)
   - `AudioProjector` (`src/modeling.py:60`): Projects audio features to text embedding space using RMSNorm and GELU activation with scaled initialization
   - `LLMDecoder`: SmolLM2 decoder (360M or 1.7B params) with LoRA adapters for text generation

2. **Training Pipeline** (`src/train.py:248`):
   - Uses Hydra for configuration management (`@hydra.main` decorator)
   - Supports streaming datasets (LibriSpeech, GigaSpeech, Common Voice)
   - `ASRDataCollator` (`src/train.py:68`): Handles audio preprocessing and tokenization
   - `PredictionLoggingCallback` (`src/train.py:177`): Logs predictions and WER metrics to TensorBoard

3. **Configuration System** (Hydra-based):
   - Base config: `configs/hydra/config.yaml` - defines defaults and output structure
   - Model configs: `small.yaml` (r=32), `large.yaml` (r=64) - LoRA rank parameters
   - Data configs: `tiny.yaml` (100 samples), `production_streaming.yaml` (full datasets)
   - Training configs: `mac.yaml` (local), `production.yaml` (GPU optimized)
   - Experiment configs: Pre-configured combinations for common use cases

### Key Design Decisions

- **Frozen Encoder**: Whisper encoder remains frozen to preserve pre-trained audio representations
- **LoRA Fine-tuning**: Only ~2% of parameters are trained via LoRA adapters (rank 32-64)
- **Audio Projection**: Custom projection layer with:
  - RMSNorm for stability
  - GELU activation for non-linearity
  - Scaled initialization (0.01x) to prevent gradient explosion
- **Streaming Datasets**: Uses HuggingFace streaming mode to handle TB-scale datasets
- **Mixed Precision**: BF16 training via accelerate for 2x speedup and memory efficiency
- **Cross-Entropy Loss**: Applied only to decoder outputs, ignoring audio tokens in loss calculation

## Configuration Structure

The project uses Hydra's composition pattern:
- **Defaults**: Specified in `configs/hydra/config.yaml`
- **Experiment presets**: `+experiments=mac_minimal`, `+experiments=production`
- **Parameter overrides**: `model.lora_r=64 training.batch_size=8`
- **Output structure**: `outputs/{date}/{time}/` with automatic timestamping
- **Checkpoint resumption**: `resume_from_checkpoint=path/to/checkpoint`

### Available Experiments
- `mac_minimal`: Quick training for testing (20 steps, tiny dataset)
- `production`: Full training with streaming datasets and optimized settings

## Environment Variables

For Mac with MPS acceleration issues:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

For debugging gradients:
```bash
export DEBUG_GRADIENTS=1
```

For faster downloads with HuggingFace:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## Data Flow

1. **Audio Input**: Raw audio (16kHz) → Whisper feature extractor → Log-mel spectrogram
2. **Encoder**: Spectrogram → Whisper encoder → Audio embeddings (1500 dim)
3. **Projection**: Audio embeddings → RMSNorm → Linear → GELU → Linear → Text space (2048/4096 dim)
4. **Decoder**: Projected features + text prompt → SmolLM2 + LoRA → Generated transcription
5. **Loss Calculation**: Cross-entropy on text tokens only (audio tokens masked with -100)