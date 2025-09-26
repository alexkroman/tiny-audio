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

# Tiny Audio - W2V-BERT-LM ASR Model

A lightweight ASR model combining W2V-BERT 2.0 encoder with modern language model decoders (SmolLM2 or Qwen3), trained with LoRA for parameter-efficient fine-tuning (~300 lines of code).

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

- **Encoder**: Frozen W2V-BERT 2.0 (600M params, trained on 4.5M hours across 143+ languages)
- **Decoder**: SmolLM2-360M or Qwen3-1.7B with LoRA adapters
- **Audio Projector**: Advanced architecture with:
  - AttentionPoolingHead with learnable probes (128 tokens)
  - Pre-norm transformer architecture with dual RMSNorm layers
  - SwiGLU activation with proper residual connections
  - Positional embeddings for temporal information
- **LoRA Config**: 
  - Small model: rank=8, alpha=16
  - Large model: rank=16, alpha=32
  - Target modules: [q_proj, k_proj, v_proj, out_proj]
- **Trainable Parameters**: ~2% of total (varies by model size)
- **Total**: ~960M (small) or ~2.3B (large) parameters

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
