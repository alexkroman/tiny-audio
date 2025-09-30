# Tiny Audio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-mazesmazes%2Ftiny--audio-yellow)](https://huggingface.co/mazesmazes/tiny-audio)

Speech recognition combining Whisper encoder with SmolLM3 decoder via LoRA.

- [Demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)
- [Model](https://huggingface.co/mazesmazes/tiny-audio)

## Usage

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
transcription = model.transcribe("audio.wav")
```

## Training

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
uv sync

# Quick test (20 steps)
uv run src/train.py

# Full training
export HF_TOKEN='your-token'  # From https://huggingface.co/settings/tokens
uv run src/train.py +experiments=production
```

## Architecture

- **Whisper-small encoder** (frozen)
- **Linear projector** with RMSNorm and 2x downsampling
- **SmolLM3 decoder** with LoRA (rank 16-32)

Audio → Whisper → Linear+Downsample → SmolLM3+LoRA → Text

## Configuration

Hydra-based config in `configs/hydra/`. Override from command line:

```bash
uv run src/train.py model=large training.max_steps=10000
```

## Remote Training

```bash
uv run scripts/deploy_runpod.py --host <pod-id>.runpod.io --port 22
uv run scripts/start_remote_training.py --host <pod-id>.runpod.io --port 22 --config production
```

## Development

```bash
uv run ruff format src/ && uv run ruff check src/ --fix && uv run mypy src/
uv run pytest
```

## License

MIT
