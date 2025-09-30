# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Automatic Speech Recognition (ASR) training pipeline that combines a W2V-BERT 2.0 encoder with SmolLM3 or Qwen decoder using LoRA for parameter-efficient fine-tuning. The project uses Hydra for configuration management and supports both local and remote training on RunPod.

**📦 Pre-trained Model**: Available on [Hugging Face Hub](https://huggingface.co/mazesmazes/tiny-audio)
**🚀 Demo**: Try it at [Hugging Face Spaces](https://huggingface.co/spaces/mazesmazes/tiny-audio)

## Development Commands

### Setup and Dependencies

```bash
# Set up Hugging Face token (required for GigaSpeech, Common Voice datasets)
export HF_TOKEN='your-hugging-face-token'
# Get a token from https://huggingface.co/settings/tokens

# Install dependencies with uv (preferred)
uv sync

# Or install with pip
pip install -e .
```

### Training

```bash
# Run training with default config (Mac settings)
uv run src/train.py

# Run with specific experiment config
uv run src/train.py +experiments=mac_minimal
uv run src/train.py +experiments=production

# Override specific parameters
uv run src/train.py training.max_steps=100 model.lora_r=64

# Resume from checkpoint
uv run src/train.py resume_from_checkpoint=outputs/2025-09-22/12-51-14/checkpoint-500
```

### Code Quality

**IMPORTANT: Always run these after making code changes:**

```bash
# Format code with ruff (replaces black)
uv run ruff format src/ demo/ tests/

# Run linter and auto-fix issues
uv run ruff check src/ demo/ tests/ --fix

# Type checking
uv run mypy src/

# Run all checks together (recommended after every change)
uv run ruff format src/ demo/ tests/ && uv run ruff check src/ demo/ tests/ && uv run mypy src/
```

### Testing

```bash
# Run all tests (end-to-end test that trains a model and tests transcription)
uv run pytest

# Run specific test file
uv run pytest tests/test_e2e.py

# Run tests with specific pattern
uv run pytest -k transcribe

# Run tests with coverage
uv run pytest --cov=src

# IMPORTANT: Always run tests after making changes to modeling.py or train.py
uv run pytest tests/test_e2e.py -v
```

**Note**: The end-to-end test trains a minimal model with 20 steps (from mac_minimal config) and tests transcription functionality. It takes about 2-3 minutes to complete.

### Demo Application

```bash
# Run the Gradio demo with a trained model
uv run demo/app.py --model outputs/2025-09-22/12-51-14/outputs/mac_minimal_model

# Run demo with default model (from HuggingFace Hub)
uv run demo/app.py

# Deploy demo to Hugging Face Spaces (public demo)
# Note: Only deploys demo files (app.py, requirements.txt, README.md, wav_outputs/)
# Models are loaded from HuggingFace Hub at runtime, not included in Space
uv run scripts/deploy_to_hf_space.py --force

# Deploy to a custom Hugging Face Space
uv run scripts/deploy_to_hf_space.py --space-url https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE --force
```

**Important**: The deployment script only copies demo-related files (~3MB). Model weights are downloaded from HuggingFace Hub when the Space runs, which may use 40-50GB of Docker storage for model caching. This is normal for HF Spaces.

### Model Publishing

```bash
# Push updated modeling.py and MODEL_CARD.md to HuggingFace Hub
# Requires HF_TOKEN to be set
uv run scripts/push_to_hub.py

# Push to a custom repository
uv run scripts/push_to_hub.py --repo-id YOUR_USERNAME/YOUR_MODEL --branch main
```

### Remote Training (RunPod)

```bash
# Deploy to RunPod (requires host and port)
uv run scripts/deploy_runpod.py --host <pod-id>.runpod.io --port 22

# Start remote training
uv run scripts/start_remote_training.py --host <pod-id>.runpod.io --port 22 --config production

# Attach to remote session
uv run scripts/attach_remote_session.py --host <pod-id>.runpod.io --port 22
```

### Monitoring

```bash
# Login to W&B (first time only)
wandb login

# View metrics in the W&B dashboard
# Your runs will appear at https://wandb.ai/YOUR_USERNAME/tiny-audio
```

## Project Structure

```
├── configs/hydra/      # Hydra configuration files
│   ├── config.yaml     # Base configuration with defaults
│   ├── model/          # Model configs (small.yaml, large.yaml)
│   ├── data/           # Dataset configs (tiny.yaml, production_streaming.yaml)
│   ├── training/       # Training configs (mac.yaml, production.yaml)
│   └── experiments/    # Pre-configured experiment combinations
├── demo/               # Gradio demo application
│   └── app.py          # Interactive ASR demo with mic/file support
├── scripts/            # Deployment and remote training utilities
│   ├── deploy_runpod.py
│   ├── start_remote_training.py
│   ├── attach_remote_session.py
│   ├── deploy_to_hf_space.py
│   └── push_to_hub.py  # Push modeling.py updates to HuggingFace
├── src/                # Core source code
│   ├── modeling.py     # ASR model implementation (~300 lines)
│   └── train.py        # Training pipeline with Hydra integration
├── tests/              # Test suite
│   └── test_e2e.py     # End-to-end training and transcription test
├── pyproject.toml      # Project dependencies and tool configurations
└── README.md           # User-facing documentation
```

## Architecture

### Core Components

1. **ASRModel** (`src/modeling.py:95`): Main model combining:

   - `W2V-BERT Encoder`: Frozen W2V-BERT 2.0 encoder for audio feature extraction
     (600M params, trained on 4.5M hours of audio across 143+ languages)
   - `AudioProjector`: Projects audio features to text embedding space with:
     • AttentionPoolingHead with learnable probes for feature compression
     • Pre-norm architecture with dual RMSNorm layers for stability
     • SwiGLU activation with residual connections
     • Positional embeddings for pooled features
   - `LLMDecoder`: SmolLM3 (3B) or Qwen3 (1.7B) decoder with LoRA adapters for
     text generation

1. **Training Pipeline** (`src/train.py:248`):

   - Uses Hydra for configuration management (`@hydra.main` decorator)
   - Supports streaming datasets (LibriSpeech, GigaSpeech, Common Voice)
   - `ASRDataCollator` (`src/train.py:68`): Handles audio preprocessing and
     tokenization
   - `PredictionLoggingCallback` (`src/train.py:177`): Logs predictions and WER
     metrics to W&B

1. **Configuration System** (Hydra-based):

   - Base config: `configs/hydra/config.yaml` - defines defaults and output
     structure
   - Model configs: `small.yaml` (SmolLM3-3B, r=8, alpha=16), `large.yaml` (Qwen3-1.7B, r=16, alpha=32) - LoRA
     parameters
   - Data configs: `tiny.yaml` (100 samples), `production_streaming.yaml` (full
     datasets)
   - Training configs: `mac.yaml` (local), `production.yaml` (GPU optimized)
   - Experiment configs: Pre-configured combinations for common use cases

### Key Design Decisions

- **Frozen Encoder**: W2V-BERT encoder remains frozen to preserve pre-trained
  audio representations from 4.5M hours of multilingual training
- **LoRA Fine-tuning**: Only ~2% of parameters are trained via LoRA adapters
  (rank 32-64)
- **Audio Projection**: Modern projection architecture with:
  - AttentionPoolingHead using learnable probes to compress features
  - Pre-norm transformer architecture for stable training
  - Dual layer normalization (before attention and MLP)
  - SwiGLU activation with proper residual connections
  - Positional embeddings for temporal information
- **Streaming Datasets**: Uses HuggingFace streaming mode to handle TB-scale
  datasets
- **Mixed Precision**: BF16 training via accelerate for 2x speedup and memory
  efficiency
- **Cross-Entropy Loss**: Applied only to decoder outputs, ignoring audio tokens
  in loss calculation

### Important Code Guidelines

- **NEVER use `torch_dtype` parameter** - It's deprecated. Don't use `dtype` either for `AutoModelForCausalLM.from_pretrained()` - let it auto-detect
- **Always resize embeddings** when loading models to match tokenizer vocabulary size (`model.resize_token_embeddings(len(tokenizer))`)
- **Run tests after changes** to modeling.py or train.py: `uv run pytest tests/test_e2e.py -v`
- **Frozen W2V-BERT encoder** - The W2V-BERT encoder MUST remain frozen (`requires_grad_(False)`) to preserve audio representations
- **Audio projection architecture** - The projector uses 2x downsampling, SwiGLU with residual connections for improved performance
- **ASR Pipeline Support** - The model's `forward` method detects inference mode (when only `input_features` is provided) and redirects to the `generate` method for compatibility with HuggingFace's ASR pipeline
- **Fixed test output directory** - E2E tests now use a fixed output path `outputs/test_e2e_model` specified via `+output_dir=` override
- **Model Loading** - When loading pretrained models, use `low_cpu_mem_usage=False` to ensure proper initialization of audio special tokens
- **Audio Special Tokens** - The model automatically adds `<|audio_chunk|>` token during initialization to handle audio-text alignment
- **Vectorized Forward Pass** - The forward method uses efficient tensor operations with `torch.cat` and `pad_sequence` for better GPU utilization and compatibility with distributed training
- **W2V-BERT Integration** - Uses `SeamlessM4TFeatureExtractor` for audio preprocessing, which returns `input_features` for the encoder

## Configuration Structure

The project uses Hydra's composition pattern:

- **Defaults**: Specified in `configs/hydra/config.yaml`
- **Experiment presets**: `+experiments=mac_minimal`, `+experiments=production`
- **Parameter overrides**: `model.lora_r=64 training.batch_size=8`
- **Output structure**: `outputs/{date}/{time}/` with automatic timestamping
- **Custom output directory**: Use `+output_dir=path/to/dir` to specify a custom output location
- **Checkpoint resumption**: `resume_from_checkpoint=path/to/checkpoint`

### Available Experiments

- `mac_minimal`: Quick training for testing (20 steps, tiny dataset)
- `production`: Full training with streaming datasets and optimized settings

## Environment Variables

For accessing gated datasets (GigaSpeech, Common Voice):

```bash
export HF_TOKEN='your-hugging-face-token'
```

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

### Data Flow

1. **Audio Input**: Raw audio (16kHz) → SeamlessM4TFeatureExtractor → Log-mel spectrogram features
2. **Encoder**: Spectrogram → W2V-BERT encoder → Audio embeddings (1024 dim)
3. **Projection**: Audio embeddings → RMSNorm → Linear projection → AttentionPoolingHead with learnable probes → Compressed features with positional embeddings → Text space (2048/4096 dim)
4. **Decoder**: Projected features + text prompt → SmolLM3/Qwen3 + LoRA → Generated transcription
5. **Loss Calculation**: Cross-entropy on text tokens only (audio tokens masked with -100)

## Common Pitfalls to Avoid

- **Model dtype issues**: Let transformers auto-detect dtype, don't specify it manually
- **Gradient explosion**: The SwiGLU architecture with residual connections provides stable gradient flow
- **Memory issues on Mac**: Use environment variables `PYTORCH_ENABLE_MPS_FALLBACK=1` and `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- **Dataset access**: Ensure `HF_TOKEN` is set for gated datasets (GigaSpeech, Common Voice)
- **Checkpoint compatibility**: When resuming training, ensure optimizer states are compatible
- **HF Spaces storage**: Space repository should be <10MB (only demo files). The 40-50GB usage is Docker cache for downloaded models, which is managed by HuggingFace
