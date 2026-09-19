<div align="center">
  <img src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/logo.png" alt="Tiny Audio Logo" />
</div>

# Tiny Audio

**Train your own speech recognition model on a single GPU**

A minimal, hackable ASR codebase. Connect a frozen audio encoder to a small LLM through a trainable MLP projector (~6.3M params), then fine-tune the projector and decoder together. That's it.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
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

### Streaming Inference

```python
from tiny_audio import ASRModel
from tiny_audio.asr_processing import ASRProcessor

model = ASRModel.from_pretrained("mazesmazes/tiny-audio")
processor = ASRProcessor.from_pretrained("mazesmazes/tiny-audio")

inputs = processor(audio=audio, return_tensors="pt")  # audio: 16kHz float array

# Yields the partial transcript as each token is decoded
for partial in model.generate_streaming(inputs["input_features"], inputs["audio_attention_mask"]):
    print(partial, end="\r", flush=True)
```

### Word-Level Timestamps and Speaker Labels

```python
# Word timestamps via forced alignment
result = pipe("audio.wav", return_timestamps=True)
# {"text": "hello world", "words": [{"word": "hello", "start": 0.0, "end": 0.5}, ...]}

# Speaker diarization (implies return_timestamps=True)
result = pipe("meeting.wav", return_speakers=True, num_speakers=2)
```

## Train Your Own

### Smoke Test

```bash
# 10 steps on a 73-clip LibriSpeech sample; runs on CPU or Apple Silicon (~5 min)
poetry run python scripts/train.py +experiments=mps_smoke
```

### Full Training

The default recipe is `stage_1`: frozen GLM-ASR encoder, fresh MLP projector, and Qwen3-0.6B decoder trained jointly on the `multiasr` mix (about 3M utterances across ten corpora, over a terabyte on disk). It runs at batch size 100 and needs an 80 GB GPU. Size it before renting hardware:

```bash
# Estimate VRAM and disk for a config and print a pod command
poetry run ta runpod plan -e stage_1

# Standard training
poetry run python scripts/train.py +experiments=stage_1

# With a custom projector learning rate
poetry run python scripts/train.py +experiments=stage_1 training.learning_rate=5e-4

# Resume from checkpoint
poetry run python scripts/train.py +experiments=stage_1 training.resume_from_checkpoint=/path/to/checkpoint-XXXX
```

### Training on RunPod

```bash
# Optional (needs runpodctl + RunPod API key): create a pod on the first GPU
# type with capacity for this config, then block until it exposes SSH
poetry run ta runpod up -e stage_1
poetry run ta runpod wait <POD_ID>   # prints <HOST> <PORT>

# Sync the project and install dependencies on the pod
poetry run ta runpod deploy <HOST> <PORT>

# Start training in a tmux session, then attach to it
export HF_TOKEN='hf_your_token'
poetry run ta runpod train <HOST> <PORT> -e stage_1
poetry run ta runpod attach <HOST> <PORT>

# Find the latest checkpoint, or evaluate on the pod
poetry run ta runpod checkpoint <HOST> <PORT>
poetry run ta runpod eval <HOST> <PORT> -m mazesmazes/tiny-audio -n 100
```

### Multi-Stage Training with LoRA

```bash
# Stage 1: Train the projector (and decoder) with the base recipe
poetry run python scripts/train.py +experiments=stage_1

# Stage 2: Freeze the projector, train LoRA adapters on the LLM
poetry run python scripts/train.py +experiments=stage_1 \
  training.use_lora=true training.freeze_projector=true

# Stage 3: Fine-tune both projector and LoRA
poetry run python scripts/train.py +experiments=stage_1 training.use_lora=true
```

## Architecture

```
Audio (16kHz) → GLM-ASR Encoder (frozen) → MLP Projector (trained) → Qwen3-0.6B (fine-tuned) → Text
```

The encoder stays frozen. The projector trains from scratch at a high learning rate while the decoder fine-tunes at a much lower one, so the two learn together without the fresh projector destabilizing the pretrained LLM.

| Component | Params | Status | Learning rate |
|-----------|--------|--------|---------------|
| GLM-ASR-Nano-2512 encoder | ~635M | Frozen | — |
| MLP Projector | ~6.3M | **Trained** | 1e-3 |
| Qwen3-0.6B decoder | ~596M | **Fine-tuned** | 2e-5 |

### How It Works

1. **Audio Encoder**: GLM-ASR converts raw audio to frame-level embeddings (dim 1280)
1. **Projector**: Stacks 4 adjacent frames (5120 dims), normalizes, then maps through a 1024-wide hidden layer into Qwen3's 1024-dim embedding space
1. **Language Model**: Qwen3 generates text conditioned on the projected audio

Frame stacking reduces sequence length: `output_len = (input_len - k) // k + 1` where k is `projector_pool_stride` (default 4).

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

### Apple SFSpeechRecognizer (macOS, on-device)

```bash
poetry run ta eval -m apple-speech -d loquacious -n 1000
poetry run ta eval -m apple-speech --locale es-ES -d loquacious -n 100
```

Calls Apple's on-device `SFSpeechRecognizer` via PyObjC. macOS-only —
`pyobjc-framework-Speech` is auto-installed on macOS via the Poetry
`sys_platform == 'darwin'` marker. First run prompts for Speech Recognition
consent (System Settings → Privacy & Security → Speech Recognition).

## CLI Reference

All commands available via `tiny-audio` (or `ta` for short):

```bash
poetry run ta --help  # Show all commands
```

| Command | Description |
|---------|-------------|
| `ta eval` | Evaluate ASR models on datasets |
| `ta analysis` | WER analysis (high-wer, entity-errors, extract-entities, compare) |
| `ta deploy` | Deploy demo to HuggingFace Space |
| `ta push` | Push model to HuggingFace Hub |
| `ta demo` | Launch local Gradio demo |
| `ta debug` | Debug utilities (analyze-weights, analyze-lora, compare-to-base, check-gradient-flow) |
| `ta runpod` | Remote training on RunPod (plan, up, wait, deploy, train, attach, eval, checkpoint) |
| `ta dev` | Development tools (lint, format, type-check, test, check, precommit, ...) |

### CLI conventions

Every command follows the same rules, and `tests/test_cli_conventions.py` checks them against
the built command tree:

- Commands that *run* something (`eval`, `demo`, `runpod eval`, `dev handler`) take the model as
  `--model/-m`. Commands that *inspect* a model or its eval runs (`analysis *`, `debug *`) take it
  as the first positional argument.
- `ta runpod` commands that talk to a pod take `<HOST> <PORT>` as their first two arguments.
- Every option has an explicit `--long-name` and help text. Secrets and IDs that usually come from
  the environment (`HF_TOKEN`, `ASSEMBLYAI_API_KEY`, `WANDB_RUN_ID`, `MODEL_ID`, ...) are options
  with an `[env var: ...]` fallback, so `--help` shows where each value comes from.
- Choices (`--datasets`, `--assemblyai-model`, `--component`, `--dtype`) are validated by the CLI
  and listed in `--help`; paths passed with `--output-dir`, `--demo-dir`, `--checkpoint-dir` and
  `--audio` are checked before the command runs.
- A short flag means one thing everywhere, and a long flag keeps the same short flag everywhere:

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | Model path or Hub ID |
| `--datasets` | `-d` | Datasets to evaluate (repeatable, `all` expands) |
| `--max-samples` | `-n` | Maximum samples per dataset |
| `--output-dir` | `-o` | Directory eval results are written to / read from |
| `--num-workers` | `-w` | Parallel workers for API evaluations |
| `--streaming` | `-s` | Streaming evaluation |
| `--config` | `-c` | Dataset config override |
| `--experiment` | `-e` | Experiment config (`ta runpod`) |
| `--force` | `-f` | Kill an existing tmux session first (`ta runpod`) |
| `--repo-id` | `-r` | Hub repo to push or deploy to |
| `--branch` | `-b` | Hub branch (`ta push`) |
| `--threshold` | `-t` | WER threshold (`ta analysis high-wer`) |
| `--top-k` | `-k` | Most-drifted tensors (`ta debug compare-to-base`) |
| `--list` | `-l` | List tmux sessions (`ta runpod attach`) |
| `--port` | `-p` | Server port (`ta demo`) |
| `--audio` | `-a` | Audio file (`ta dev handler`) |
| `--verbose` | `-v` | Verbose output (`ta debug analyze-weights`) |

## Configuration

Configuration uses [Hydra](https://hydra.cc/). Override any value with `key=value` syntax:

```bash
# Override model settings
poetry run python scripts/train.py +experiments=stage_1 model.projector_hidden_dim=2048

# Override training settings
poetry run python scripts/train.py +experiments=stage_1 \
  training.decoder_learning_rate=1e-5 training.per_device_train_batch_size=50

# Swap the dataset config
poetry run python scripts/train.py +experiments=stage_1 data=librispeech_dummy
```

### Config Files

```
configs/
├── config.yaml                  # Main config (model defaults; imports data + training)
├── experiments/                 # Training recipes
│   ├── stage_1.yaml             # Default: frozen encoder + projector + decoder trained jointly
│   ├── encoder_train.yaml       # Trainable Whisper encoder + frozen decoder
│   ├── granite_qwen.yaml        # Granite Speech encoder + Qwen3.5-2B decoder
│   ├── granite_gemma.yaml       # Granite Speech encoder + Gemma 4 decoder
│   ├── granite_gemma_smoke.yaml # granite_gemma on the LibriSpeech dummy set
│   └── mps_smoke.yaml           # Local CPU/MPS smoke test
├── data/
│   ├── multiasr.yaml            # Ten-corpus training mix (default)
│   ├── loquacious_medium.yaml   # LoquaciousSet medium
│   └── librispeech_dummy.yaml   # 73-clip sample for smoke tests
└── training/
    └── production.yaml          # Training hyperparameters
```

### Key Config Parameters

Defaults from `configs/config.yaml` and `configs/training/production.yaml`:

```yaml
model:
  audio_model_id: zai-org/GLM-ASR-Nano-2512   # Audio encoder
  text_model_id: Qwen/Qwen3-0.6B              # Language model
  projector_type: mlp
  projector_pool_stride: 4                    # Frames stacked per projector input
  projector_hidden_dim: 1024

training:
  freeze_language_model: false                # Decoder is fine-tuned
  freeze_projector: false
  learning_rate: 1e-3                         # Projector
  decoder_learning_rate: 2e-5                 # Decoder
  per_device_train_batch_size: 100
  max_steps: -1                               # Train by epochs instead
  num_train_epochs: 2                         # stage_1 overrides this to 1
  warmup_steps: 2000
  lr_scheduler_type: cosine_with_min_lr
```

The encoder is frozen by the `freeze_audio_encoder` default in `ASRConfig`; `encoder_train.yaml` is the recipe that unfreezes it.

## Project Structure

```
tiny-audio/
├── tiny_audio/              # Core library
│   ├── asr_modeling.py      # ASRModel: encoder + projector + decoder
│   ├── asr_config.py        # ASRConfig: all model settings
│   ├── asr_pipeline.py      # HuggingFace pipeline for inference
│   ├── asr_processing.py    # ASRProcessor: audio/text preprocessing
│   ├── projectors.py        # Projector architectures
│   ├── alignment.py         # Forced alignment for word timestamps
│   ├── diarization.py       # Speaker diarization
│   ├── handler.py           # HF Inference Endpoints handler
│   └── integrations/        # Voice agent integrations (Pipecat)
├── scripts/
│   ├── train.py             # Training script (Hydra)
│   ├── cli.py               # Unified CLI entry point
│   ├── dev.py               # Development utilities
│   ├── analysis.py          # WER analysis tools
│   ├── itn.py               # Inverse text normalization for eval
│   ├── eval/                # Evaluation framework
│   │   ├── evaluators/      # ASR evaluators
│   │   └── datasets.py      # Dataset loading
│   ├── deploy/              # RunPod, run planning, HF Space deployment
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
poetry run ta dev test                    # Run all tests (enforces the coverage floor)
poetry run pytest tests/test_projectors.py -v  # Single file
poetry run pytest -k "test_forward" -v    # By name pattern
```

### Code Quality

```bash
poetry run ta dev format      # Format code (black, ruff, mdformat)
poetry run ta dev lint        # Lint + format check (poetry check --lock, ruff, black, yamllint, taplo)
poetry run ta dev type-check  # Type check (mypy, pyright)
poetry run ta dev check       # Lint + type-check + security + dead code + docstrings
poetry run ta dev precommit   # Full quality gate
```

### Adding a New Projector

1. Add your projector class to `tiny_audio/projectors.py`:

```python
class MyProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config.encoder_dim, config.llm_dim, config.projector_pool_stride are available
        # Your architecture here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, encoder_dim] -> [batch, out_len, llm_dim]
        return x

    def get_output_length(self, input_length: int) -> int:
        # Return output sequence length given input length
        return input_length
```

2. Register it in the `PROJECTOR_CLASSES` dict in `projectors.py`:

```python
PROJECTOR_CLASSES = {
    "mlp": MLPAudioProjector,
    "my_projector": MyProjector,  # Add here
}
```

3. Create an experiment config `configs/experiments/my_projector.yaml`:

```yaml
# @package _global_
model:
  projector_type: my_projector
```

4. Train: `poetry run python scripts/train.py +experiments=my_projector`

### Adding a New Dataset

1. Add a config file `configs/data/my_dataset.yaml`. Each entry in `datasets:` is one Hub dataset (or one config of it); see `configs/data/librispeech_dummy.yaml` and `configs/data/multiasr.yaml` for the full set of fields:

```yaml
datasets:
  - path: your-org/your-dataset
    name: en                  # Optional dataset config name
    audio_column: audio
    text_column: text
    task: transcribe
    text_case: cased          # or `mono` for lowercase / ALL-CAPS labels
    text_punct: true
    train_splits: [train]
    eval_splits: [validation]
    target_samples: 600000    # Optional cap on the train split

sample_rate: 16000
dataset_cache_dir: ${hydra:runtime.cwd}/datasets_cache
max_eval_samples_per_dataset: 500
```

2. Train with your dataset: `poetry run python scripts/train.py +experiments=stage_1 data=my_dataset`

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
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) for the default evaluation set

## License

MIT
