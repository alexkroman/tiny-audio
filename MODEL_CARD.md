---
license: mit
language:
- en
datasets:
- speechbrain/LoquaciousSet
base_model:
- openai/whisper-large-v3-turbo
- HuggingFaceTB/SmolLM3-3B
pipeline_tag: automatic-speech-recognition
tags:
- asr
- speech-recognition
- audio
- smollm
- whisper
- mlp
---

# Tiny Audio

A speech recognition model trained in 24 hours on a single GPU for ~$12. Built with the [Tiny Audio](https://github.com/alexkroman/tiny-audio) codebase—a minimal, hackable framework for training ASR models.

## Architecture

```
Audio (16kHz) → Whisper Encoder (frozen) → MLP Projector (trained) → SmolLM3-3B (frozen) → Text
```

**MLP Projector:**
- Convolutional downsampling: 4x sequence compression via two stride-2 conv layers
- Linear (1280 → 2048) → GELU → Linear (2048 → 2048)
- Output normalization: RMSNorm

## Training Details

| | |
|---|---|
| **Dataset** | LoquaciousSet (25,000 hours) |
| **Hardware** | Single NVIDIA A40 40GB |
| **Training Time** | ~24 hours |
| **Cost** | ~$12 |
| **Trainable Parameters** | ~12M (projector only) |

## Performance

**Word Error Rate (WER): 12.14%** on LoquaciousSet test set.

See the [community leaderboard](https://github.com/alexkroman/tiny-audio#leaderboard) for comparisons.

## Usage

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)

result = pipe("path/to/audio.wav")
print(result["text"])
```

## Limitations

- English only
- Optimized for 16kHz audio; other sample rates are resampled automatically
- Performance may degrade on heavily accented speech, noisy environments, or domain-specific jargon
- Maximum audio length limited by context window

## Learn More

- **[Train your own model](https://github.com/alexkroman/tiny-audio)** — The full codebase with training scripts
- **[Free 3.5-hour course](https://github.com/alexkroman/tiny-audio/blob/main/docs/course/0-course-overview.md)** — Build your own ASR system from scratch
- **[Submit to leaderboard](https://github.com/alexkroman/tiny-audio#leaderboard)** — Share your trained model
