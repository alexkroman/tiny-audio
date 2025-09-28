---
license: mit
datasets:
- mozilla-foundation/common_voice_17_0
- speechcolab/gigaspeech
- openslr/librispeech_asr
language:
- en
metrics:
- wer
base_model:
- facebook/w2v-bert-2.0
- HuggingFaceTB/SmolLM2-360M-Instruct
- Qwen/Qwen3-1.7B
pipeline_tag: automatic-speech-recognition
tags:
- w2v-bert
- asr
- speech-recognition
- lora
- multilingual
model-index:
- name: tiny-audio
  results: []
---

# Tiny Audio

A tiny speech recognition model.

## Quick Start

```python
from transformers import AutoModel
import torch

# Load model
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float32  # MPS works better with float32
else:
    device = torch.device("cpu")
    dtype = torch.float32

model = AutoModel.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
model = model.to(device, dtype=dtype)
model.eval()

# Transcribe audio
transcription = model.transcribe("audio.wav", max_new_tokens=64)
print(transcription)
```

## Links

- **Code**: [github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)
- **Demo**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/mazesmazes/tiny-audio)
- **License**: MIT
