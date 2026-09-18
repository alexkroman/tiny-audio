---
title: Tiny Audio Demo
emoji: 🎤
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: ASR with a GLM-ASR encoder and Qwen3-0.6B decoder
models:
  - mazesmazes/tiny-audio
tags:
  - audio
  - automatic-speech-recognition
  - glm-asr
  - qwen3
  - mlp
suggested_hardware: cpu-basic
preload_from_hub:
  - mazesmazes/tiny-audio
---

## Demo Overview

This Space demonstrates an Automatic Speech Recognition (ASR) model that combines:

- **GLM-ASR-Nano-2512 encoder** for audio feature extraction
- **Qwen3-0.6B decoder** for text generation

## Features

- 🎙️ **Record from microphone** or upload audio files
- ⏱️ **Word-level timestamps** via forced alignment
- 🗣️ **Speaker diarization** to label who said what
- 🎯 **English transcription** with punctuation and casing

## Model Architecture

The model bridges audio and text with a small trained projector:

1. **Audio Encoder**: GLM-ASR-Nano-2512 encoder (frozen)
2. **Projection Layer**: 2-layer MLP with frame stacking, mapping audio features into the decoder's embedding space (~6.3M params)
3. **Text Decoder**: Qwen3-0.6B, fine-tuned jointly with the projector

## Usage

1. **Upload an audio file** (WAV, MP3, etc.) or **record directly** using your microphone
2. Click **"Transcribe"** to convert speech to text
3. The transcription will appear in the output box

## Limitations

- Best results on clips under 30 seconds
- Optimized for English language
- Best performance with clear speech and minimal background noise

## Links

- 📦 [Model on Hugging Face](https://huggingface.co/mazesmazes/tiny-audio)
- 💻 [GitHub Repository](https://github.com/alexkroman/tiny-audio)
- 📄 [Technical Details](https://github.com/alexkroman/tiny-audio/blob/main/MODEL_CARD.md)

## Citation

If you use this model in your research, please cite:

```bibtex
@software{kroman2024tinyaudio,
  author = {Kroman, Alex},
  title = {Tiny Audio: A minimal, hackable speech recognition codebase},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/alexkroman/tiny-audio}
}
```
