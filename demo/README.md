______________________________________________________________________

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
short_description: Efficient ASR with Whisper encoder and SmolLM3 decoder
models:

- mazesmazes/tiny-audio
  tags:
- audio
- automatic-speech-recognition
- whisper
- smollm
- mlp
  suggested_hardware: cpu-upgrade
  preload_from_hub:
- mazesmazes/tiny-audio

______________________________________________________________________

## Demo Overview

This Space demonstrates an Automatic Speech Recognition (ASR) model that combines:

- **Whisper encoder** for audio feature extraction
- **SmolLM3 decoder** for efficient text generation

## Features

- 🎙️ **Record from microphone** or upload audio files
- ⚡ **Fast inference** with a small number of trainable parameters
- 🎯 **English transcription** optimized for speech-to-text
- 📊 **Lightweight model** suitable for edge deployment

## Model Architecture

The model uses a novel architecture that bridges audio and text modalities:

1. **Audio Encoder**: Frozen Whisper encoder
1. **Projection Layer**: Custom audio-to-text space mapping
1. **Text Decoder**: SmolLM3 (frozen)

## Usage

1. **Upload an audio file** (WAV, MP3, etc.) or **record directly** using your microphone
1. Click **"Transcribe"** to convert speech to text
1. The transcription will appear in the output box

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
@software{kroman2024tinyaudio,
  author = {Kroman, Alex},
  title = {Tiny Audio: Train your own speech recognition model in 24 hours},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/alexkroman/tiny-audio}
}
```
