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
short_description: Efficient ASR with Whisper encoder and SmolLM2 decoder
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

- **Whisper-small encoder** for audio feature extraction
- **SmolLM2 decoder** with LoRA adapters for efficient text generation

## Features

- 🎙️ **Record from microphone** or upload audio files
- ⚡ **Fast inference** with only ~2% trainable parameters
- 🎯 **English transcription** optimized for speech-to-text
- 📊 **Lightweight model** suitable for edge deployment

## Model Architecture

The model uses a novel architecture that bridges audio and text modalities:

1. **Audio Encoder**: Frozen Whisper-small encoder (39M params)
2. **Projection Layer**: Custom audio-to-text space mapping with RMSNorm
3. **Text Decoder**: SmolLM2-135M with LoRA rank-32 adapters

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
  title = {Tiny Audio: Efficient ASR with Whisper and SmolLM2},
  year = {2024},
  url = {https://github.com/alexkroman/tiny-audio}
}
```
