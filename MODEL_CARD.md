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
library_name: transformers
---

# Tiny Audio

A speech recognition model trained in 24 hours on a single GPU for ~$12. Built with [Tiny Audio](https://github.com/alexkroman/tiny-audio)—a minimal, hackable ASR framework.

## Quick Start

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)
result = pipe("audio.wav")
print(result["text"])
```

## Usage Examples

### Basic Transcription

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)

# From file
result = pipe("audio.wav")
print(result["text"])

# From URL
result = pipe("https://example.com/audio.mp3")

# From numpy array (must be 16kHz)
import numpy as np
audio = np.random.randn(16000).astype(np.float32)  # 1 second
result = pipe(audio)
```

### Batch Processing

```python
# Process multiple files
files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = pipe(files, batch_size=4)
for r in results:
    print(r["text"])
```

### Word-Level Timestamps

```python
result = pipe("audio.wav", return_timestamps="word")
# Returns:
# {
#   "text": "hello world",
#   "chunks": [
#     {"text": "hello", "timestamp": (0.0, 0.5)},
#     {"text": "world", "timestamp": (0.6, 1.0)}
#   ]
# }
```

### Using with torch directly

```python
from tiny_audio import ASRModel, ASRProcessor
import torch
import librosa

# Load model and processor
model = ASRModel.from_pretrained("mazesmazes/tiny-audio")
processor = ASRProcessor.from_pretrained("mazesmazes/tiny-audio")

# Load audio (16kHz)
audio, sr = librosa.load("audio.wav", sr=16000)

# Process
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

# Generate
with torch.no_grad():
    output = model.generate(
        input_features=inputs["input_features"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256
    )

# Decode
text = processor.batch_decode(output, skip_special_tokens=True)[0]
print(text)
```

### GPU Inference

```python
import torch

pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True,
    device="cuda"  # or device=0
)
```

### Half Precision

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device="cuda"
)
```

## Architecture

```
Audio (16kHz) → GLM-ASR Encoder (frozen) → MLP Projector (trained) → Qwen3 (frozen) → Text
```

Only the projector is trained (~12M params). The encoder and decoder remain frozen, leveraging their pretrained knowledge.

| Component | Model | Parameters | Status |
|-----------|-------|------------|--------|
| Audio Encoder | GLM-ASR-Nano-2512 | ~600M | Frozen |
| Projector | 2-layer MLP | ~12M | Trained |
| Language Model | Qwen3-0.6B | ~600M | Frozen |

### How It Works

1. **Audio Encoder**: GLM-ASR converts 16kHz audio into frame-level embeddings (768-dim)
2. **Projector**: A 2-layer MLP with frame stacking bridges the audio and text embedding spaces
3. **Language Model**: Qwen3 generates text autoregressively, conditioned on the projected audio

The projector reduces sequence length via frame stacking: `output_len = (input_len - 5) // 5 + 1`

## Model Specifications

| Specification | Value |
|---------------|-------|
| Input | Audio (16kHz mono) |
| Output | Text transcription |
| Max Audio Length | ~30 seconds (limited by encoder) |
| Vocabulary | Qwen3 tokenizer |
| Languages | English only |
| Generation | Greedy decoding (num_beams=1, do_sample=False) |

## Training Details

| | |
|---|---|
| **Dataset** | LoquaciousSet (25,000 hours) |
| **Hardware** | Single NVIDIA A40 |
| **Time** | ~24 hours |
| **Cost** | ~$12 |
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Batch Size** | 4 |
| **Steps** | 50,000 |

## Limitations

- **English only**: Not trained on other languages
- **Sample rate**: Expects 16kHz audio (other rates resampled automatically)
- **Audio length**: Best for clips under 30 seconds
- **Accuracy**: May degrade on:
  - Heavily accented speech
  - Noisy or low-quality audio
  - Domain-specific terminology
  - Overlapping speakers
- **No punctuation**: Output is lowercase without punctuation by default

## Requirements

```
transformers>=4.40.0
torch>=2.0.0
torchaudio>=2.0.0
```

Optional for streaming:
```
librosa
soundfile
```

## Files

| File | Description |
|------|-------------|
| `config.json` | Model configuration |
| `model.safetensors` | Projector weights (~48MB) |
| `preprocessor_config.json` | Audio preprocessing config |
| `tokenizer.json` | Tokenizer |
| `tokenizer_config.json` | Tokenizer config |
| `special_tokens_map.json` | Special tokens |

Note: Only the projector weights are stored. The encoder (GLM-ASR) and decoder (Qwen3) are loaded from their respective HuggingFace repos.

## Citation

If you use this model, please cite:

```bibtex
@misc{tinyaudio2024,
  author = {Alex Kroman},
  title = {Tiny Audio: Minimal ASR Training},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/alexkroman/tiny-audio}
}
```

## Links

- [GitHub Repository](https://github.com/alexkroman/tiny-audio) - Train your own model
- [Free 3.5-hour Course](https://github.com/alexkroman/tiny-audio/blob/main/docs/course/0-course-overview.md) - Learn ASR from scratch
- [Live Demo](https://huggingface.co/spaces/mazesmazes/tiny-audio) - Try it in your browser

## Acknowledgments

- [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) for the audio encoder
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) for the language model
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) for training data

## License

MIT
