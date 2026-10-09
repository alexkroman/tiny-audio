---
title: Tiny Audio Demo
emoji: 🎤
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.26.0"
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
short_description: Fast English ASR with word timestamps and speaker labels
thumbnail: https://huggingface.co/spaces/mazesmazes/tiny-audio/resolve/main/thumbnail.png
models:
  - mazesmazes/tiny-audio
tags:
  - audio
  - automatic-speech-recognition
  - speech-recognition
  - speech-to-text
  - transcription
  - word-timestamps
  - speaker-diarization
  - granite-speech
  - qwen3.5
  - lora
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
1. **Projection Layer**: 2-layer MLP mapping audio features into the decoder's embedding space
   (~12.6M params)
1. **Text Decoder**: Qwen3.5-2B (frozen) with rank-64 LoRA adapters (~67M params), trained jointly
   with the projector

## Usage

1. **Upload an audio file** (WAV, MP3, etc.) or **record directly** using your microphone
1. Click **"Transcribe"** to convert speech to text
1. The transcription will appear in the output box
1. Optionally tick **Word Timestamps** or **Speaker Diarization**. If you know how many people are
   speaking, set **Number of Speakers**: it caps how many speakers are kept

## Limitations

- Plain transcription works best on clips up to about 19 seconds (the training length); with
  timestamps or diarization on, longer audio is transcribed in chunks automatically
- Requests run on a GPU server on RunPod; while it is starting up or offline, requests fail with a
  "try again" message
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
