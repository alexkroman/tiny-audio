---
license: mit
language:
- en
datasets:
- speechbrain/LoquaciousSet
base_model:
- zai-org/GLM-ASR-Nano-2512
- Qwen/Qwen3-0.6B
pipeline_tag: automatic-speech-recognition
tags:
- asr
- speech-recognition
- audio
- qwen
- glm-asr
---

# Tiny Audio

A speech recognition model trained in 24 hours on a single GPU for ~$12. Built with [Tiny Audio](https://github.com/alexkroman/tiny-audio)—a minimal, hackable ASR framework.

## Architecture

```
Audio (16kHz) → GLM-ASR Encoder (frozen) → MLP Projector (trained) → Qwen3 (frozen) → Text
```

Only the projector is trained (~12M params). The encoder and decoder remain frozen.

## Training

| | |
|---|---|
| **Dataset** | LoquaciousSet (25,000 hours) |
| **Hardware** | Single NVIDIA A40 |
| **Time** | ~24 hours |
| **Cost** | ~$12 |

## Usage

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)
result = pipe("audio.wav")
print(result["text"])
```

## Limitations

- English only
- 16kHz audio (other sample rates resampled automatically)
- May degrade on accented speech, noisy audio, or domain-specific terms

## Links

- [Train your own](https://github.com/alexkroman/tiny-audio)
- [Free 3.5-hour course](https://github.com/alexkroman/tiny-audio/blob/main/docs/course/0-course-overview.md)
