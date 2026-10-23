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
license: cc-by-nc-sa-4.0
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
  - qwen3
suggested_hardware: zero-a10g
preload_from_hub:
  - mazesmazes/tiny-audio
  - ibm-granite/granite-speech-5.0-470m-turboctc-nc
  - Qwen/Qwen3-0.6B
  - Qwen/Qwen3-ForcedAligner-0.6B-hf
  - nvidia/Nemotron-3-Diarization
---

## Demo Overview

This Space demonstrates an Automatic Speech Recognition (ASR) model that combines:

- **Granite Speech 5.0 470M non-causal encoder** for audio feature extraction
- **Qwen3-0.6B decoder**, fully fine-tuned, for text generation

## Features

- 🎙️ **Record from microphone** or upload audio files
- ⏱️ **Word-level timestamps** via forced alignment
- 🗣️ **Speaker diarization** to label who said what
- 🎯 **English transcription** with punctuation and casing

## Model Architecture

The model bridges audio and text with a trained projector and a fine-tuned decoder:

1. **Audio Encoder**: Granite Speech 5.0 470M TurboCTC non-causal encoder (frozen)
1. **Projection Layer**: 2-layer MLP mapping audio features into the decoder's embedding space
   (~8.4M params)
1. **Text Decoder**: Qwen3-0.6B (~596M params), fully fine-tuned jointly with the projector

## Usage

1. **Upload an audio file** (WAV, MP3, etc.) or **record directly** using your microphone
1. Click **"Transcribe"** to convert speech to text
1. The transcription will appear in the output box
1. Optionally tick **Word Timestamps** or **Speaker Diarization**. If you know how many people are
   speaking, set **Number of Speakers**: it caps how many speakers are kept

## Limitations

- Audio longer than 28 seconds is transcribed in chunks cut at pauses
- Requests run on ZeroGPU, which reserves GPU time from your daily quota in proportion to the
  audio's length (20-120 s), so roughly up to five minutes of speech per request
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
