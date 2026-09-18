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

# Smoke test: 10 steps on a 73-clip LibriSpeech sample, runs on CPU or Apple Silicon
poetry run python scripts/train.py +experiments=mps_smoke

# Size the full run (VRAM, disk, pod command) before renting a GPU
poetry run ta runpod plan -e stage_1

# Full training: frozen GLM-ASR encoder + MLP projector + Qwen3-0.6B decoder, jointly
poetry run python scripts/train.py +experiments=stage_1
```

The `stage_1` recipe trains on the `multiasr` mix (about 3M utterances across ten corpora,
over a terabyte on disk) at batch size 100, which needs an 80 GB GPU. See
[Training on RunPod](../README.md#training-on-runpod) for the remote workflow.

## Evaluate

```bash
poetry run ta eval -m mazesmazes/tiny-audio -n 100
```

## Next Steps

- [Full Course](course/0-course-overview.md) — 3.5 hours, build ASR from scratch
- [Quick Reference](course/4-quick-reference.md) — Command cheat sheet
