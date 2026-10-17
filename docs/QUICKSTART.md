# Quick Start

## Use the Model

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True
)
print(pipe("audio.wav")["text"])
```

## Train Your Own

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio
poetry install

# Size the full run (VRAM, disk, pod command) before renting a GPU
poetry run ta runpod plan

# Full training (the default config): frozen Granite Speech 470M encoder + MLP projector
# + frozen Qwen3.5-2B decoder with LoRA adapters
poetry run python scripts/train.py
```

The default recipe (`configs/config.yaml` + `configs/training/production.yaml`) trains on the
`multiasr` mix (about 3M utterances across ten corpora, over a terabyte on disk) at batch 64, sized
for one 80 GB GPU. There is no laptop-sized recipe; run `ta runpod plan` to size any config before
renting hardware. See [Training on RunPod](../README.md#training-on-runpod) for the remote workflow.

## Evaluate

```bash
poetry run ta eval -m mazesmazes/tiny-audio -n 100
```

## Next Steps

- [Full Course](course/0-course-overview.md) — 3.5 hours, build ASR from scratch
- [Quick Reference](course/4-quick-reference.md) — Command cheat sheet
