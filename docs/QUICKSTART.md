# Quick Start

## Use the Model

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

## Projector Types

| Type | Command | Notes |
|------|---------|-------|
| MLP | `+experiments=mlp` | Default, fastest |
| MOSA | `+experiments=mosa` | Dense MoE |
| MoE | `+experiments=moe` | Sparse experts |

## Evaluate

```bash
poetry run ta eval -m mazesmazes/tiny-audio -n 100
```

## Next Steps

- [Full Course](course/0-course-overview.md) — 3.5 hours, build ASR from scratch
- [Quick Reference](course/4-quick-reference.md) — Command cheat sheet
