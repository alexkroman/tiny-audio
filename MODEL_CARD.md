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
- openai/whisper-small
- HuggingFaceTB/SmolLM2-360M-Instruct
pipeline_tag: automatic-speech-recognition
---

# Tiny Audio - Whisper-SmolLM2 ASR Model

A lightweight ASR model combining Whisper-small encoder with SmolLM2 decoder, trained with LoRA for parameter-efficient fine-tuning (~300 lines of code).

## Quick Start

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
transcription = model.transcribe("audio.wav")
```

## Architecture

- **Encoder**: Frozen Whisper-small (39M params) 
- **Decoder**: SmolLM2-360M with LoRA adapters (7.3M trainable params)
- **Total**: ~400M parameters, only 2% trained

## Training

- **Data**: LibriSpeech, GigaSpeech, Common Voice (English)
- **Method**: LoRA rank 32-64, BF16 mixed precision
- **Performance**: ~15-20% WER on LibriSpeech test-clean

## Links

- **Code**: [github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)
- **Demo**: `demo/app.py` in repository
- **License**: MIT
