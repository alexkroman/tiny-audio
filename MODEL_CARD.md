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

Speech recognition combining Whisper encoder with SmolLM3 decoder.

## Usage

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
transcription = model.transcribe("audio.wav")
```

## Architecture

- **Encoder**: Whisper-small (frozen)
- **Projector**: Linear layer with RMSNorm and 2x downsampling
- **Decoder**: SmolLM3 with LoRA (rank 16-32)

## Training

Datasets: LibriSpeech, GigaSpeech, Common Voice, LoquaciousSet

- BF16 mixed precision
- Streaming datasets
- Frozen encoder, LoRA fine-tuning on decoder

## Links

- [Code](https://github.com/alexkroman/tiny-audio)
- [Demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)
- License: MIT
