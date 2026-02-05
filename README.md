<div align="center">
  <img src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/logo.png" alt="Tiny Audio Logo" />
</div>

# Tiny Audio

**Train your own speech recognition model in 24 hours for $12**

A minimal, hackable ASR codebase. Connect a frozen audio encoder to a frozen LLM via a trainable projector (~12M params). That's it.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-mazesmazes%2Ftiny--audio-yellow)](https://huggingface.co/mazesmazes/tiny-audio)

## Try It

**[Live Demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)**

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)
print(pipe("audio.wav")["text"])
```

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio
poetry install

# Or install from PyPI (inference only)
pip install tiny-audio
```

### Basic Inference

```python
from transformers import pipeline

# Load model
pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)

# Transcribe audio file
result = pipe("audio.wav")
print(result["text"])

# Transcribe from URL
result = pipe("https://example.com/audio.mp3")

# Transcribe numpy array (16kHz)
import numpy as np
audio = np.random.randn(16000)  # 1 second of audio
result = pipe(audio)
```

### Word-Level Timestamps

```python
# Enable timestamps in pipeline
result = pipe("audio.wav", return_timestamps="word")

# Returns: {"text": "hello world", "chunks": [{"text": "hello", "start": 0.0, "end": 0.5}, ...]}
```

## Train Your Own

### Quick Test

```bash
# Verify setup (~5 min)
poetry run python scripts/train.py +experiments=transcription data.max_train_samples=100 training.max_steps=10
```

### Full Training

```bash
# Standard training (~24 hours on A40)
poetry run python scripts/train.py +experiments=transcription

# With custom learning rate
poetry run python scripts/train.py +experiments=transcription training.learning_rate=1e-4

# Resume from checkpoint
poetry run python scripts/train.py +experiments=transcription training.resume_from_checkpoint=/path/to/checkpoint-XXXX
```

### Projector Types

```bash
poetry run python scripts/train.py +experiments=transcription  # Simple MLP (~12M params)
poetry run python scripts/train.py +experiments=mosa           # Dense mixture of experts
poetry run python scripts/train.py +experiments=moe            # Sparse routed experts
poetry run python scripts/train.py +experiments=qformer        # Transformer with learnable queries
```

### Multi-Stage Training with LoRA

```bash
# Stage 1: Train projector only (default)
poetry run python scripts/train.py +experiments=transcription

# Stage 2: Freeze projector, train LoRA adapters on LLM
poetry run python scripts/train.py +experiments=mlp_lora

# Stage 3: Fine-tune both projector and LoRA
poetry run python scripts/train.py +experiments=mlp_fine_tune
```

## Architecture

```
Audio (16kHz) → GLM-ASR Encoder (frozen) → MLP Projector (trained) → Qwen3 (frozen) → Text
```

Only the projector trains. The encoder and decoder stay frozen, leveraging their pretrained knowledge.

| Component | Params | Status |
|-----------|--------|--------|
| GLM-ASR Encoder | ~600M | Frozen |
| MLP Projector | ~12M | **Trained** |
| Qwen3-0.6B | ~600M | Frozen |

### How It Works

1. **Audio Encoder**: GLM-ASR converts raw audio to frame-level embeddings
1. **Projector**: Bridges the modality gap by projecting audio embeddings to text embedding space
1. **Language Model**: Qwen3 generates text conditioned on the projected audio

The projector uses frame stacking to reduce sequence length: `output_len = (input_len - k) // k + 1` where k is the pooling stride.

## Evaluation

```bash
# Evaluate on default dataset
poetry run ta eval -m mazesmazes/tiny-audio -n 100

# Evaluate on specific dataset
poetry run ta eval -m mazesmazes/tiny-audio -d loquacious -n 1000

# Compare with other models
poetry run ta eval -m assemblyai --assemblyai-model universal -d loquacious -n 100

# WER analysis
poetry run ta analysis high-wer mazesmazes/tiny-audio --threshold 30
poetry run ta analysis compare model1 model2
```

## CLI Reference

All commands available via `tiny-audio` (or `ta` for short):

```bash
poetry run ta --help  # Show all commands
```

| Command | Description |
|---------|-------------|
| `ta eval` | Evaluate ASR models on datasets |
| `ta analysis` | WER analysis (high-wer, entity-errors, compare) |
| `ta deploy` | Deploy demo to HuggingFace Space |
| `ta push` | Push model to HuggingFace Hub |
| `ta demo` | Launch local Gradio demo |
| `ta debug` | Debug utilities (check-mosa, analyze-lora) |
| `ta runpod` | Remote training on RunPod |
| `ta dev` | Development tools (lint, format, test, etc.) |

### Common Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | HuggingFace model ID |
| `--datasets` | `-d` | Datasets to evaluate |
| `--max-samples` | `-n` | Maximum samples |
| `--output-dir` | `-o` | Output directory |
| `--num-workers` | `-w` | Parallel workers |

## Configuration

Configuration uses [Hydra](https://hydra.cc/). Override any value with `key=value` syntax:

```bash
# Override model settings
poetry run python scripts/train.py model.projector_type=moe

# Override training settings
poetry run python scripts/train.py training.learning_rate=1e-4 training.batch_size=8

# Override data settings
poetry run python scripts/train.py data.max_train_samples=10000
```

### Config Files

```
configs/
├── config.yaml           # Main config (imports data + training)
├── experiments/          # Projector presets
│   ├── transcription.yaml # Simple MLP (Stage 1)
│   ├── mosa.yaml         # Dense MoE
│   ├── moe.yaml          # Sparse MoE
│   ├── qformer.yaml      # Transformer
│   ├── mlp_lora.yaml     # Stage 2: LoRA only
│   └── mlp_fine_tune.yaml # Stage 3: Projector + LoRA
├── data/
│   └── multiasr.yaml     # Multi-ASR dataset config
└── training/
    └── production.yaml   # Training hyperparameters
```

### Key Config Parameters

```yaml
model:
  audio_model_id: "zai-org/GLM-ASR-Nano-2512"  # Audio encoder
  text_model_id: "Qwen/Qwen3-0.6B"              # Language model
  projector_type: mlp                           # mlp, mosa, moe, qformer

training:
  learning_rate: 1e-4
  batch_size: 4
  max_steps: 50000
  warmup_steps: 1000
```

## Project Structure

```
tiny-audio/
├── tiny_audio/              # Core library
│   ├── asr_modeling.py      # ASRModel: encoder + projector + decoder
│   ├── asr_config.py        # ASRConfig: all model settings
│   ├── asr_pipeline.py      # HuggingFace pipeline for inference
│   ├── asr_processing.py    # ASRProcessor: audio/text preprocessing
│   ├── projectors.py        # All projector architectures
│   ├── handler.py           # HF Inference Endpoints handler
│   └── integrations/        # Voice agent integrations
├── scripts/
│   ├── train.py             # Training script (Hydra)
│   ├── cli.py               # Unified CLI entry point
│   ├── dev.py               # Development utilities
│   ├── analysis.py          # WER analysis tools
│   ├── eval/                # Evaluation framework
│   │   ├── evaluators/      # ASR, diarization, alignment
│   │   └── datasets.py      # Dataset loading
│   ├── deploy/              # HF Space deployment
│   ├── hub/                 # HF Hub integration
│   └── debug/               # Debug utilities
├── configs/                 # Hydra configuration
├── tests/                   # Test suite
└── docs/                    # Documentation and course
```

## Development

### Setup

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
poetry install
```

### Running Tests

```bash
poetry run ta dev test                    # Run all tests
poetry run pytest tests/test_projectors.py -v  # Single file
poetry run pytest -k "test_forward" -v    # By name pattern
```

### Code Quality

```bash
poetry run ta dev format      # Format code (black, isort)
poetry run ta dev lint        # Lint (ruff, black --check)
poetry run ta dev type-check  # Type check (mypy, pyright)
poetry run ta dev precommit   # Full quality gate
```

### Adding a New Projector

1. Add your projector class to `tiny_audio/projectors.py`:

```python
class MyProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config):
        super().__init__()
        # Your architecture here

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Return projected features
        return x

    def get_output_length(self, input_length: int) -> int:
        # Return output sequence length given input length
        return input_length
```

2. Register it in the `PROJECTOR_CLASSES` dict in `projectors.py`:

```python
PROJECTOR_CLASSES = {
    "mlp": MLPAudioProjector,
    "mosa": MOSAProjector,
    "moe": MoEAudioProjector,
    "qformer": QFormerAudioProjector,
    "my_projector": MyProjector,  # Add here
}
```

3. Create an experiment config `configs/experiments/my_projector.yaml`:

```yaml
model:
  projector_type: my_projector
```

4. Train: `poetry run python scripts/train.py +experiments=my_projector`

### Adding a New Dataset

1. Add a config file `configs/data/my_dataset.yaml`:

```yaml
dataset_name: "your-org/your-dataset"
dataset_split: "train"
audio_column: "audio"
text_column: "text"
```

2. Train with your dataset: `poetry run python scripts/train.py data=my_dataset`

### Key Files to Understand

| File | Purpose | When to Modify |
|------|---------|----------------|
| `asr_modeling.py` | Core model class | Adding model features, changing forward pass |
| `asr_config.py` | Configuration | Adding new config parameters |
| `projectors.py` | Projector architectures | Adding new projector types |
| `asr_processing.py` | Audio/text preprocessing | Changing input processing |
| `train.py` | Training loop | Modifying training behavior |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token (for private models/pushing) |
| `WANDB_API_KEY` | Weights & Biases API key |
| `WANDB_RUN_ID` | Resume a specific W&B run |
| `ASSEMBLYAI_API_KEY` | For AssemblyAI evaluation comparison |

## Learn More

- **[Free 3.5-hour course](docs/course/0-course-overview.md)** — Build ASR from scratch
- **[Quick Start Guide](docs/QUICKSTART.md)** — Detailed setup instructions
- **[Model Card](MODEL_CARD.md)** — Model documentation template

## Acknowledgments

- [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) for audio encoding
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) for language modeling
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) for training data

## License

MIT
