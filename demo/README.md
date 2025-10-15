---
title: Tiny Audio Demo
emoji: 🎤
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Efficient ASR with HuBERT encoder and SmolLM3 decoder
models:
  - mazesmazes/tiny-audio
tags:
  - audio
  - automatic-speech-recognition
  - wav2vec2
  - smollm3
  - lora
suggested_hardware: cpu-upgrade
preload_from_hub:
  - mazesmazes/tiny-audio
---

## Demo Overview

This Space demonstrates an Automatic Speech Recognition (ASR) model that combines:

- **HuBERT-large encoder** for audio feature extraction
- **SmolLM3 decoder** for efficient text generation

## Features

- 🎙️ **Record from microphone** or upload audio files
- ⚡ **Fast inference** with only ~2% trainable parameters
- 🎯 **English transcription** optimized for speech-to-text
- 📊 **Lightweight model** suitable for edge deployment

## Model Architecture

The model uses a novel architecture that bridges audio and text modalities:

1. **Audio Encoder**: Frozen HuBERT-large encoder (317M params)
2. **Projection Layer**: Custom audio-to-text space mapping
3. **Text Decoder**: SmolLM3-3B (frozen)

## Usage

1. **Upload an audio file** (WAV, MP3, etc.) or **record directly** using your microphone
2. Click **"Transcribe"** to convert speech to text
3. The transcription will appear in the output box

## Limitations

- Maximum audio length: 30 seconds
- Optimized for English language
- Best performance with clear speech and minimal background noise

## Links

- 📦 [Model on Hugging Face](https://huggingface.co/mazesmazes/tiny-audio)
- 💻 [GitHub Repository](https://github.com/alexkroman/tiny-audio)
- 📄 [Technical Details](https://github.com/alexkroman/tiny-audio/blob/main/MODEL_CARD.md)

## Citation

If you use this model in your research, please cite:

```bibtex
@software{tiny_audio_2024,
  author = {Kroman, Alex},
  title = {Tiny Audio: Efficient ASR with HuBERT and SmolLM3},
  year = {2024},
  url = {https://github.com/alexkroman/tiny-audio}
}
```
