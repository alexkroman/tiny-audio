---
license: mit
datasets:
- mozilla-foundation/common_voice_17_0
- speechcolab/gigaspeech
- openslr/librispeech_asr
- speechbrain/LoquaciousSet
language:
- en
metrics:
- wer
base_model:
- facebook/w2v-bert-2.0
- HuggingFaceTB/SmolLM3-3B-Base
pipeline_tag: automatic-speech-recognition
tags:
- w2v-bert
- smollm3
- asr
- speech-recognition
- lora
model-index:
- name: tiny-audio
  results: []
---

# Tiny Audio

A tiny speech recognition model.

## Quick Start

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
transcription = model.transcribe("audio.wav")
```

## Links

- **Code**: [github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)
- **Demo**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/mazesmazes/tiny-audio)
- **License**: MIT
