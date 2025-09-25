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
tags:
- whisper
- asr
- speech-recognition
- lora
model-index:
- name: tiny-audio
  results: []
---

# Tiny Audio - Whisper-SmolLM2 ASR Model

A lightweight ASR model combining Whisper-small encoder with SmolLM2 decoder, trained with LoRA for parameter-efficient fine-tuning (~300 lines of code).

## Quick Start

```python
from transformers import AutoModelForSpeechSeq2Seq
import torch

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "mazesmazes/tiny-audio",
    dtype=dtype,
    trust_remote_code=True
)
model = model.to(device)
model.eval()

# Transcribe audio
transcription = model.transcribe("audio.wav")
print(transcription)
```

## Architecture

- **Encoder**: Frozen Whisper-small (39M params) 
- **Decoder**: SmolLM2-360M-Instruct with LoRA adapters
- **LoRA Config**: rank=16, alpha=32, targets=[q_proj, v_proj, k_proj, o_proj]
- **Trainable Parameters**: 4.94M (only 1.2% of total)
- **Total**: ~400M parameters

## Training

- **Data**: LibriSpeech, GigaSpeech, Common Voice (streaming mode)
- **Optimizer**: AdamW (fused), lr=2e-4, cosine schedule with restarts
- **Batch Size**: 4 per device, gradient accumulation=4 (effective batch size: 16)
- **Training**: 50,000 steps, BF16 mixed precision
- **Hardware**: NVIDIA A40 GPU (1x), 128 vCPUs, 503 GB RAM

## Performance

- **Evaluation**: Performance measured across validation splits of all training datasets
- **Metric**: Word Error Rate (WER)
- **Note**: For specific performance metrics, please see the training logs in the repository

## Links

- **Code**: [github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)
- **Demo**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/mazesmazes/tiny-audio)
- **Local Demo**: `demo/app.py` in repository
- **License**: MIT
