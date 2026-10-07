---
title: Tiny Audio Demo
emoji: 🎤
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.26.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: ASR with a Granite Speech encoder and Qwen3.5-2B decoder
models:
  - mazesmazes/tiny-audio
tags:
  - audio
  - automatic-speech-recognition
  - granite-speech
  - qwen3.5
  - lora
  - mlp
suggested_hardware: cpu-basic
preload_from_hub:
  - mazesmazes/tiny-audio
  - ibm-granite/granite-speech-5.0-470m-turboctc
  - Qwen/Qwen3.5-2B
  - speechbrain/spkrec-ecapa-voxceleb
---

## Demo Overview

This Space demonstrates an Automatic Speech Recognition (ASR) model that combines:

- **Granite Speech 5.0 470M encoder** for audio feature extraction
- **Qwen3.5-2B decoder** with LoRA adapters for text generation

## Features

- 🎙️ **Record from microphone** or upload audio files
- ⏱️ **Word-level timestamps** via forced alignment
- 🗣️ **Speaker diarization** to label who said what
- 🎯 **English transcription** with punctuation and casing

## Model Architecture

The model bridges audio and text with a trained projector and LoRA adapters:

1. **Audio Encoder**: Granite Speech 5.0 470M TurboCTC encoder (frozen)
2. **Projection Layer**: 2-layer MLP mapping audio features into the decoder's embedding space (~12.6M params)
3. **Text Decoder**: Qwen3.5-2B (frozen) with rank-64 LoRA adapters (~67M params), trained jointly with the projector

## Usage

1. **Upload an audio file** (WAV, MP3, etc.) or **record directly** using your microphone
2. Click **"Transcribe"** to convert speech to text
3. The transcription will appear in the output box
4. Optionally tick **Word Timestamps** or **Speaker Diarization**. If you know how many people are speaking, set **Number of Speakers**: auto-detection can split one voice into several speakers on short clips

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
