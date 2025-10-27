<div align="center">
  <img src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/logo.png" alt="Tiny Audio Logo" />
</div>

# Tiny Audio

**Train your own speech recognition model in 24 hours for $12**

This repo is a minimal, hackable implementation of an ASR (Automatic Speech Recognition) model that you can train from scratch on a single GPU in 24 hours. Tiny Audio combines a frozen HuBERT-XLarge encoder (1.3B params) with a frozen SmolLM3-3B decoder (3B params), connected by a small trainable audio projector (~13M params). The result is a speech-to-text system you can train in a few hours, deploy to HuggingFace, and use just like Whisper or any other ASR model.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-mazesmazes%2Ftiny--audio-yellow)](https://huggingface.co/mazesmazes/tiny-audio)

## Talk to it

You can try the model right now at [huggingface.co/spaces/mazesmazes/tiny-audio](https://huggingface.co/spaces/mazesmazes/tiny-audio). Upload an audio file and watch it transcribe. It's not perfect - you'll notice it makes mistakes, especially with background noise or heavy accents - but what makes it unique is that **it's fully yours**: fully configurable, tweakable, hackable, and trained by you from start to end.

Or use it via the transformers library:

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
                model="mazesmazes/tiny-audio",
                trust_remote_code=True)

result = pipe("path/to/audio.wav")
print(result["text"])
```

The model handles various audio formats (WAV, MP3, FLAC, etc.) and automatically resamples to 16kHz.

## Quick start

The fastest way to feel the magic is to train your own model. Boot up an A40 GPU box from your favorite provider (I like RunPod), clone this repo, and kick off training:

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
poetry install

# Quick test run (20 steps, ~5 minutes)
poetry run python src/train.py

# Full production training (~24 hours on A40)
export HF_TOKEN='your-token'  # Get from https://huggingface.co/settings/tokens
poetry run python src/train.py +experiments=production
```

If you want to run it on a remote GPU like RunPod, there are deployment scripts to make your life easier:

```bash
# Deploy code to remote GPU
poetry run deploy-runpod --host <pod-id>.runpod.io --port 22

# Start training remotely
poetry run remote-train \
  --host <pod-id>.runpod.io \
  --port 22 \
  --config production
```

Now wait ~24 hours. Once it's done, your model will be pushed to HuggingFace Hub automatically (if you set `HF_TOKEN`), and you can use it just like in the example above!

## How it works

Tiny Audio uses a simple three-component architecture:

```text
Audio Waveform → HuBERT-XLarge → Audio Projector → SmolLM3-3B → Text
                 (1.3B, frozen)  (~13M, trainable)  (3B, frozen)
```

1. **Audio Encoder (Frozen)**: HuBERT-XLarge extracts acoustic features from your audio
2. **Audio Projector (Trainable)**: A SwiGLU MLP that downsamples 5x and maps audio features to language model space - **this is the only part you train**
3. **Language Decoder (Frozen)**: SmolLM3-3B generates the text transcription autoregressively with Flash Attention 2

The projector uses a **SwiGLU** architecture (like Llama):
- `gate_proj`: Linear(6400 → 2048, no bias)
- `up_proj`: Linear(6400 → 2048, no bias)
- `down_proj`: Linear(2048 → 2048, no bias)
- Activation: `silu(gate) * up` → `down`

Why freeze the encoder and decoder? Because:
- **You only train ~13M parameters** instead of 4+ billion
- **Training is fast** (~24 hours on A40)
- **It's cheap** (~$12 for a full run)
- **You leverage pretrained knowledge** from both audio and language domains
- **Memory efficient** - runs on a single A40 40GB

## Training details

**Dataset**: LoquaciousSet (25,000 hours) - a diverse corpus combining CommonVoice, VoxPopuli, Libriheavy, People's Speech, and YODAS. Hundreds of thousands of speakers with varied accents, speech types, and acoustic conditions.

**Hardware**: Single NVIDIA A40 40GB works great.

**Time & Cost**: ~24 hours on A40 = ~$12 depending on your provider

**Configuration**: The repo uses Hydra for configs, so you can easily tweak things:

```bash
# Try different encoders
poetry run python src/train.py model.audio_model_id=facebook/hubert-large-ls960-ft

# Adjust learning rate
poetry run python src/train.py training.learning_rate=5e-5

# Change batch size (if you're running out of memory)
poetry run python src/train.py training.batch_size=16
```

The training script automatically logs to Weights & Biases (wandb), saves checkpoints to `outputs/`, and pushes the final model to HuggingFace.

## Evaluation

Evaluate your model (or any other ASR model) on the LoquaciousSet benchmark:

```bash
# Evaluate on 100 samples (quick check)
poetry run eval mazesmazes/tiny-audio --max-samples 100

# Full evaluation
poetry run eval mazesmazes/tiny-audio
```

Results are measured in Word Error Rate (WER) - lower is better. Detailed predictions are saved to `outputs/eval_*/results.txt` so you can see exactly where your model succeeds or fails.

## Leaderboard

Contributors who have trained and evaluated Tiny Audio models:

| Rank | Contributor | WER | Git Hash | Date |
|------|------------|-----|----------|------|
| 🥇 | [@alexkroman](https://github.com/alexkroman) | **10.14** | [`5a5f3a0`](https://github.com/alexkroman/tiny-audio/commit/5a5f3a055d2e5722d9473f3a1c2fb883eab7ad9c) | 2025-10-23 |

Want to see your name here? Train a model, evaluate it on LoquaciousSet, and submit a PR with your results!

**To reproduce or generate your own WER score:**
```bash
# Evaluate on 500 samples (default)
poetry run eval mazesmazes/tiny-audio

# Or evaluate your own model
poetry run eval your-username/your-model-name

# Quick test on 100 samples
poetry run eval mazesmazes/tiny-audio --max-samples 100
```

## What makes this repo different?

Tiny Audio is not a SOTA ASR model. It's a **single, cohesive, minimal, readable, hackable codebase** designed to train an ASR model start to end and produce a working model you can actually use and learn from.

- **~1000 lines of core code** across 7 Python files in `src/`
- **Dependency-lite**: Just PyTorch, transformers, datasets, and a few other essentials via Poetry
- **No magic**: Read the code and understand exactly what's happening
- **Fully yours**: Train it, modify it, deploy it however you want

The entire codebase is small enough to read in an afternoon and understand deeply.

## Project structure

```text
tiny-audio/
├── src/                      # Core code (~1,200 lines)
│   ├── asr_modeling.py      # Model architecture
│   ├── asr_config.py        # Model configuration
│   ├── asr_pipeline.py      # HuggingFace pipeline integration
│   ├── asr_processing.py    # Audio/text processing
│   ├── train.py             # Training script
│   └── handler.py           # Inference handler
├── configs/                 # Hydra configurations
│   └── hydra/
│       ├── config.yaml      # Main config
│       ├── model/           # Model variants
│       ├── training/        # Training hyperparameters
│       └── experiments/     # Full experiment presets
├── scripts/                 # Utility scripts
│   ├── eval.py             # Evaluation
│   ├── deploy_runpod.py    # Remote deployment
│   └── ...
├── demo/                    # Gradio web interface
└── tests/                   # Test suite
```

## Development

```bash
# Format code
poetry run format

# Run linter
poetry run lint

# Type checking
poetry run type-check

# Run tests
poetry run test

# Run all checks
poetry run check
```

## Contributing

Tiny Audio is nowhere finished. The goal is to make ASR training accessible on budgets < $100 while keeping the codebase small, readable, and hackable. Contributions that align with this philosophy are welcome - please open an issue to discuss major changes.

## Acknowledgments

This project builds upon:
- [HuBERT](https://huggingface.co/docs/transformers/model_doc/hubert) by Facebook AI for audio encoding
- [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) by HuggingFace for language modeling
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) by SpeechBrain for training data

## Citation

If you use Tiny Audio in your research, please cite:

```bibtex
@software{kroman2024tinyaudio,
  author = {Kroman, Alex},
  title = {Tiny Audio: Train your own speech recognition model in 24 hours},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/alexkroman/tiny-audio}
}
```

## License

MIT
