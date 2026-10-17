# Quick Start

## Use the Model

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True
)
print(pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")["text"])
# I have a dream that one day this nation will rise up and live out the true meaning of its creed.
print(pipe("my_recording.wav")["text"])  # or a local file
```

## Train Your Own

```bash
git clone https://github.com/alexkroman/tiny-audio.git && cd tiny-audio
poetry install

# Free: the default recipe with a smaller decoder, ~30 minutes on your own machine
poetry run ta train asr -e tiny

# Size the full run (VRAM, disk, pod command) before renting a GPU
poetry run ta runpod plan

# Full training (the default config): frozen Granite Speech 470M encoder + MLP projector
# + frozen Qwen3.5-2B decoder with LoRA adapters
poetry run python scripts/train.py
```

The default recipe (`configs/config.yaml` + `configs/training/production.yaml`) trains on the
`multiasr` mix (about 3M utterances across ten corpora, over a terabyte on disk) at batch 64, sized
for one 80 GB GPU. `-e tiny` (`configs/experiments/tiny.yaml`) is the same recipe with the smaller
Qwen3.5-0.8B decoder on LibriSpeech dev-clean (5.4 hours): it learns to transcribe (~16% WER on
test-clean) in about 30 minutes on a 16 GB Mac, before you rent anything. Run `ta runpod plan` to
size any config before renting hardware. See [Training on one GPU](course/2-training.md) for the
remote workflow.

## Evaluate

```bash
poetry run ta eval -m mazesmazes/tiny-audio -n 100
```

## Next Steps

- [Full Course](course/0-course-overview.md) — 3.5 hours, build ASR from scratch
- [Quick Reference](course/4-quick-reference.md) — Command cheat sheet
