<div align="center">
  <img src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/logo.png" alt="Tiny Audio Logo" />
</div>

# Tiny Audio

## Train Your Own Speech Recognition Model in 24 Hours for $12

This isn't just another ASR model. This is a launchpad. A minimal, hackable, and deeply understandable codebase that empowers you to build, train, and deploy your own speech recognition system from scratch. In a single day, on a single GPU, for the price of a few coffees.

Tiny Audio combines the power of massive pretrained models (Whisper encoder + SmolLM3 decoder) with an efficient projector-only training approach, allowing you to create a high-quality, custom ASR model that is truly yours.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-mazesmazes%2Ftiny--audio-yellow)](https://huggingface.co/mazesmazes/tiny-audio)

## Talk to It

Experience the magic firsthand. Try the live demo on Hugging Face Spaces, or run it yourself with just a few lines of Python.

**[🚀 Live Demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)**

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
                model="mazesmazes/tiny-audio",
                trust_remote_code=True)

result = pipe("path/to/audio.wav")
print(result["text"])
```

## 🎓 Learn by Building: A Free, Hands-On Course

This repository is also a free, 6-hour course designed to teach you the art and science of building modern ASR systems. No black boxes. No magic. Just clean, understandable code and a clear path from raw audio to a deployed model.

**[📚 Start the Course](docs/QUICKSTART.md)** | **[📖 See the Full Curriculum](docs/course/0-course-overview.md)**

In just six hours, you will:

- **Understand the Architecture:** Go deep on the encoder-projector-decoder model that powers modern ASR.
- **Master Efficient Training:** Learn how to train just the projector while leveraging frozen pretrained models.
- **Train Your Own Model:** Get your hands dirty and train a model from scratch on a real-world dataset.
- **Deploy and Share:** Push your model to the Hugging Face Hub, write a professional model card, and share your work with the world.

## Quick Start: Train Your Own Model

Ready to build? You can train your own model in just a few steps.

```bash
# 1. Clone the repo and install dependencies
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
poetry install

# 2. Run a quick test to make sure everything is working (~5 minutes)
poetry run python scripts/train.py +experiments=mlp data.max_train_samples=100 training.max_steps=10

# 3. Start the full training (~24 hours on an A40 GPU)
export HF_TOKEN='your-hugging-face-token' # Get from hf.co/settings/tokens
poetry run python scripts/train.py +experiments=mlp

# 4. Resume training from a checkpoint (if training was interrupted)
poetry run python scripts/train.py +experiments=mlp training.resume_from_checkpoint=/path/to/checkpoint-XXXX

# For remote training, find the latest checkpoint and resume:
poetry run find-checkpoint <host> <port>  # prints latest checkpoint path
poetry run remote-train <host> <port> --experiment mlp --wandb-run-id <run-id> --wandb-resume must training.resume_from_checkpoint=/path/to/checkpoint-XXXX
```

## How It Works: The Tiny Audio Architecture

Tiny Audio is built on a simple, powerful idea: combine the best pretrained models for audio and language, and efficiently teach them to work together.

1. **The Ear (Audio Encoder):** We start with `openai/whisper-large-v3-turbo`, a state-of-the-art model that has already learned to understand the nuances of human speech. We keep this frozen, leveraging its pretrained knowledge.
1. **The Bridge (MLP Audio Projector):** This is a simple Multi-Layer Perceptron (MLP) projector that acts as a translator, converting the audio features from the encoder into a format the language model can understand. It uses convolutional downsampling (4x compression) followed by two linear layers. This is the only component we train.
1. **The Brain (Language Model):** We use `HuggingFaceTB/SmolLM3-3B`, a powerful yet efficient language model that already knows how to generate coherent text. We keep this frozen as well, relying on its language understanding capabilities.

By keeping both the encoder and decoder frozen and only training the projector, we achieve efficiency without sacrificing quality.

### Key Concepts

- **Convolutional Downsampling**: Two 1D conv layers with stride 2 each (4x compression) reduce the audio sequence length while preserving temporal information
- **16kHz Audio**: Industry standard sample rate that captures human speech frequency range while being computationally efficient
- **Projector-only Training**: The encoder and decoder are already excellent at their jobs from pretraining—we only need to teach them how to communicate

### MLP Projector Architecture

```
Whisper embeddings (1280-dim)
    ↓
Conv1D (stride 2) → Conv1D (stride 2)  [4x compression]
    ↓
Linear (1280 → 2048) → GELU → Linear (2048 → 2048)
    ↓
RMSNorm → SmolLM3-3B embeddings (2048-dim)
```

**Why this works:**
- Fast training (~24 hours on A40) with no gradients for frozen models
- Cheap (~$12 for a full run)
- Leverages pretrained knowledge from both audio and language domains
- Memory efficient—runs on a single A40 40GB GPU

## Training details

**Dataset**: LoquaciousSet (25,000 hours) - a diverse corpus combining CommonVoice, VoxPopuli, Libriheavy, People's Speech, and YODAS. Hundreds of thousands of speakers with varied accents, speech types, and acoustic conditions.

**Hardware**: Single NVIDIA A40 40GB works great.

**Time & Cost**: ~24 hours on A40 = ~$12 depending on your provider

**Configuration**: Hydra configs with easy overrides:

```bash
# Projector experiments (see configs/experiments/)
poetry run python scripts/train.py +experiments=mlp       # MLP (default)
poetry run python scripts/train.py +experiments=moe       # MoE
poetry run python scripts/train.py +experiments=swiglu    # SwiGLU
poetry run python scripts/train.py +experiments=residual  # Residual

# Override any config value
poetry run python scripts/train.py training.learning_rate=1e-4
poetry run python scripts/train.py training.per_device_train_batch_size=5
```

Training logs to Weights & Biases, saves checkpoints to `outputs/`, and pushes the final model to HuggingFace.

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
| 🥇 | [@alexkroman](https://github.com/alexkroman) | **12.14** | [`5a5f3a0`](https://github.com/alexkroman/tiny-audio/commit/5a5f3a055d2e5722d9473f3a1c2fb883eab7ad9c) | 2025-10-23 |

Want to see your name here? Train a model, evaluate it on LoquaciousSet, and submit a PR with your results!

**To reproduce or generate your own WER score:**

```bash
# Evaluate on 500 samples
poetry run eval mazesmazes/tiny-audio --max-samples 500

# Or evaluate your own model
poetry run eval your-username/your-model-name

# Quick test on 100 samples
poetry run eval mazesmazes/tiny-audio --max-samples 100
```

## Training Curves

Watch the model learn in real-time. Training loss decreases steadily while evaluation loss shows strong generalization:

<div align="center">
  <img src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/train_loss.png" alt="Training Loss" width="45%" />
  <img src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/eval_loss.png" alt="Evaluation Loss" width="45%" />
</div>

*Left: Training loss over 50k steps. Right: Evaluation loss showing consistent improvement without overfitting.*

All training runs are logged to [Weights & Biases](https://wandb.ai) for detailed monitoring and experiment tracking.

## What makes this repo different?

Tiny Audio is not a SOTA ASR model. It's a **minimal, readable, hackable codebase** designed to train ASR models start to end.

- **~1000 lines of core code** across 7 files in `src/`
- **Projector-only training** keeps encoder and decoder frozen
- **Dependency-lite**: PyTorch, transformers, datasets, and a few essentials
- **No magic**: Read the code and understand exactly what's happening
- **Multiple projector types**: MLP (default), MoE, SwiGLU, Residual

## Project structure

```text
tiny-audio/
├── src/                      # Core library code
│   ├── asr_modeling.py      # Model architecture
│   ├── asr_config.py        # Model configuration
│   ├── asr_pipeline.py      # HuggingFace pipeline integration
│   ├── asr_processing.py    # Audio/text processing
│   ├── projectors.py        # All projector architectures
│   └── handler.py           # Inference handler
├── scripts/                 # Training and utility scripts
│   ├── train.py             # Training script (Hydra-based)
│   ├── eval.py              # Evaluation
│   ├── deploy_runpod.py     # Remote deployment
│   └── ...
├── configs/                 # Hydra configurations
│   ├── config.yaml          # Main config
│   ├── experiments/         # Projector presets
│   ├── data/                # Dataset configs
│   └── training/            # Training hyperparameters
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

- [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) by OpenAI for audio encoding
- [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) by Hugging Face for language modeling
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) by SpeechBrain for training data

## Citation

If you use Tiny Audio in your research, please cite:

```bibtex
@software{kroman2025tinyaudio,
  author = {Kroman, Alex},
  title = {Tiny Audio: Train your own speech recognition model in 24 hours},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/alexkroman/tiny-audio}
}
```

## License

MIT
