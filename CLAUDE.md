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
python src/train.py

# Run with specific experiment config
python src/train.py +experiments=mac_minimal
python src/train.py +experiments=production

# Override specific parameters
python src/train.py training.max_steps=100 model.lora_r=64
```

### Code Quality
```bash
# Run linter
ruff check src/

# Format code
ruff format src/

# Type checking
mypy src/
```

### Remote Training (RunPod)
```bash
# Deploy to RunPod
python scripts/deploy_runpod.py

# Start remote training
python scripts/start_remote_training.py

# Attach to remote session
python scripts/attach_remote_session.py
```

## Architecture

### Core Components

1. **ASRModel** (`src/modeling.py`): Main model combining:
   - `WhisperEncoder`: Frozen Whisper-small encoder for audio feature extraction
   - `AudioProjector`: Projects audio features to text embedding space using RMSNorm and GELU activation
   - `LLMDecoder`: SmolLM2 decoder with LoRA adapters for text generation

2. **Training Pipeline** (`src/train.py`):
   - Uses Hydra for configuration management (@hydra.main decorator)
   - Supports streaming datasets (LibriSpeech, GigaSpeech, Common Voice)
   - Custom DataCollator handles audio preprocessing and tokenization
   - PredictionLoggingCallback for monitoring training progress with WER metrics

3. **Configuration System** (Hydra-based):
   - Base config: `configs/hydra/config.yaml`
   - Model configs: `small.yaml`, `large.yaml` (LoRA parameters)
   - Data configs: `tiny.yaml`, `production_streaming.yaml`
   - Training configs: `mac.yaml`, `production.yaml`
   - Experiment configs combine the above for specific scenarios

### Key Design Decisions

- **Frozen Encoder**: Whisper encoder is frozen to preserve pre-trained audio understanding
- **LoRA Fine-tuning**: Only trains LoRA adapters on decoder for efficiency
- **Audio Projection**: Uses scaled initialization (0.01x) for stable training
- **Streaming Datasets**: Supports large-scale training without loading entire datasets into memory
- **Mixed Precision**: Automatically handled via accelerate for faster training

## Configuration Structure

The project uses Hydra's composition pattern:
- Defaults are specified in `configs/hydra/config.yaml`
- Override with experiments: `+experiments=production`
- Override individual params: `model.lora_r=64 training.batch_size=8`
- Output directories are auto-created with timestamps

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