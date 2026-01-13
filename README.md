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

## Train Your Own

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio
poetry install

# Quick test (~5 min)
poetry run python scripts/train.py +experiments=mlp data.max_train_samples=100 training.max_steps=10

# Full training (~24 hours on A40)
poetry run python scripts/train.py +experiments=mlp
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

## Configuration

```bash
# Projector types
poetry run python scripts/train.py +experiments=mlp      # MLP (default)
poetry run python scripts/train.py +experiments=mosa     # MOSA
poetry run python scripts/train.py +experiments=moe      # MoE

# Override any config
poetry run python scripts/train.py training.learning_rate=1e-4
```

## Evaluation

```bash
poetry run ta eval -m mazesmazes/tiny-audio -n 100
```

## Project Structure

```
tiny-audio/
├── tiny_audio/           # Core library
│   ├── asr_modeling.py   # Model architecture
│   ├── asr_config.py     # Configuration
│   ├── asr_pipeline.py   # HF pipeline
│   └── projectors.py     # Projector architectures
├── scripts/
│   ├── train.py          # Training (Hydra)
│   └── eval/             # Evaluation CLI
├── configs/              # Hydra configs
│   ├── config.yaml
│   └── experiments/      # mlp, mosa, moe
└── tests/
```

## Development

```bash
poetry run ta dev format      # Format code
poetry run ta dev lint        # Lint
poetry run ta dev type-check  # Type check
poetry run ta dev test        # Run tests
poetry run ta dev precommit   # Full quality gate
```

## Learn More

**[Free 3.5-hour course](docs/course/0-course-overview.md)** — Build ASR from scratch

## Acknowledgments

- [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) for audio encoding
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) for language modeling
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) for training data

## License

MIT
