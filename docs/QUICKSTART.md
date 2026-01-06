# Quick Start Guide

Get started with Tiny Audio in 5 minutes.

## Option 1: Use the Pre-trained Model

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
                model="mazesmazes/tiny-audio",
                trust_remote_code=True)

result = pipe("path/to/audio.wav")
print(result["text"])
```

## Option 2: Train Your Own Model

See the [full course](course/0-course-overview.md) for hands-on training (3.5 hours, ~$12).

Quick version:

```bash
# Clone and install
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
poetry install

# Test locally (runs for ~2 minutes)
poetry run python scripts/train.py +experiments=mlp data.max_train_samples=100 training.max_steps=10

# Full training (requires A40 GPU, ~24 hours)
export HF_TOKEN='your-token'
poetry run python scripts/train.py +experiments=mlp
```

## Projector Types

| Type | Command | Description |
|------|---------|-------------|
| `mlp` | `+experiments=mlp` | Default, fastest training |
| `mosa` | `+experiments=mosa` | Dense MoE, better accuracy |
| `moe` | `+experiments=moe` | Shared + sparse experts |
| `qformer` | `+experiments=qformer` | QFormer with learnable queries |

## Next Steps

- [Course Overview](course/0-course-overview.md) - Full 3.5-hour curriculum
- [Quick Reference](course/4-quick-reference.md) - Command cheat sheet
- [README](../README.md) - Project overview and architecture
- [Glossary](course/5-glossary.md) - Key terminology explained
