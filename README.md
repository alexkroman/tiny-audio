<div align="center">
  <img src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/logo.png" alt="Tiny Audio Logo" />
</div>

# Tiny Audio

## Train Your Own Speech Recognition Model in 24 Hours for $12

This isn't just another ASR model. This is a launchpad. A minimal, hackable, and deeply understandable codebase that empowers you to build, train, and deploy your own speech recognition system from scratch. In a single day, on a single GPU, for the price of a few coffees.

Tiny Audio combines the power of massive pretrained models like HuBERT and Qwen-3 8B with the efficiency of LoRA, allowing you to create a high-quality, custom ASR model that is truly yours.

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
- **Master PEFT:** Learn the theory and practice of parameter-efficient fine-tuning with LoRA.
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
poetry run python src/train.py +experiments=mac_minimal

# 3. Start the full training (~24 hours on an A40 GPU)
export HF_TOKEN='your-hugging-face-token' # Get from hf.co/settings/tokens
poetry run python src/train.py +experiments=stage1
```

## How It Works: The Tiny Audio Architecture

Tiny Audio is built on a simple, powerful idea: combine the best pretrained models for audio and language, and efficiently teach them to work together.

1. **The Ear (Audio Encoder):** We start with `facebook/hubert-xlarge-ls960-ft`, a massive model that has already learned to understand the nuances of human speech. We use LoRA to fine-tune it, teaching it to focus on the specific sounds of our dataset.
1. **The Bridge (Audio Projector):** This is a small, trainable neural network that acts as a translator, converting the audio features from the encoder into a format the language model can understand.
1. **The Brain (Language Model):** We use `Qwen/Qwen-3 8B`, a powerful language model that already knows how to generate coherent text. We use LoRA to teach it to generate transcriptions instead of just general-purpose text.

By freezing the vast majority of the parameters in the encoder and decoder and only training the small LoRA adapters and the projector, we can achieve incredible efficiency without sacrificing performance.

### Key Concepts Explained

**What is LoRA?** Low-Rank Adaptation adds small trainable "adapter" matrices to frozen model weights. Instead of updating a 2048×2048 weight matrix (4.2M params), LoRA uses two small matrices: 2048×8 and 8×2048 (32K params). This is 0.76% the size but captures most of the adaptation needed!

**What is SwiGLU?** Swish Gated Linear Unit - a modern activation function that gates information flow. Used in Llama and other state-of-the-art models. Formula: `SwiGLU(x) = Swish(Wx) ⊗ (Vx)` where `Swish(x) = x × sigmoid(x)`.

**What is RMSNorm?** Root Mean Square Normalization - a simpler, faster alternative to LayerNorm that just normalizes by the RMS value: `x / sqrt(mean(x²))`. Used in modern transformers like Llama.

**Why 16kHz audio?** This sample rate captures human speech frequency range (85-255 Hz fundamental + harmonics up to ~8kHz) while being computationally efficient. Industry standard for ASR.

**Why 5x downsampling?** Reduces computational cost in the decoder while maintaining ~100ms temporal resolution per frame - detailed enough for accurate transcription.

### LoRA Configuration

**Encoder LoRA:**

- Rank: 16
- Alpha: 16 (scaling factor)
- Target modules: q_proj, k_proj in HuBERT attention layers
- Adds ~4M trainable parameters (3,932,160)

**Decoder LoRA:**

- Rank: 8
- Alpha: 32 (scaling factor = 4.0)
- Target modules: q_proj, v_proj in Qwen-3 attention layers
- Adds ~4M trainable parameters (3,833,856)

Why use parameter-efficient training with LoRA on both encoder and decoder? Because:

- **You train ~146M parameters** instead of 9.3+ billion (projector: ~138M + encoder LoRA: ~4M + decoder LoRA: ~4M)
- **Training is fast** (~24 hours on A40) thanks to reduced gradient computations
- **It's cheap** (~$12 for a full run)
- **You leverage pretrained knowledge** from both audio and language domains
- **Memory efficient** - runs on a single A40 40GB with LoRA adapters
- **LoRA enables targeted adaptation** of both encoder and decoder without full fine-tuning
- **Flexible configuration** - easily adjust LoRA rank, target modules, and alpha for both models

## Training details

**Dataset**: LoquaciousSet (25,000 hours) - a diverse corpus combining CommonVoice, VoxPopuli, Libriheavy, People's Speech, and YODAS. Hundreds of thousands of speakers with varied accents, speech types, and acoustic conditions.

**Hardware**: Single NVIDIA A40 40GB works great.

**Time & Cost**: ~24 hours on A40 = ~$12 depending on your provider

**Configuration**: The repo uses Hydra for configs, so you can easily tweak things:

```bash
# Try different encoders
poetry run python src/train.py model.encoder_model_name=facebook/hubert-large-ls960-ft

# Adjust encoder LoRA rank (0 = frozen encoder, higher = more adaptation)
poetry run python src/train.py encoder_lora.r=8

# Adjust decoder LoRA rank (0 = frozen decoder, higher = more adaptation)
poetry run python src/train.py peft.r=64

# Adjust learning rate
poetry run python src/train.py training.learning_rate=1e-4

# Change batch size (if you're running out of memory)
poetry run python src/train.py training.per_device_train_batch_size=5

# Disable encoder LoRA (train only projector + decoder LoRA)
poetry run python src/train.py encoder_lora.r=0

# Disable decoder LoRA (train only projector + encoder LoRA)
poetry run python src/train.py peft.peft_method=null

# Train only projector (no LoRA on encoder or decoder)
poetry run python src/train.py encoder_lora.r=0 peft.peft_method=null
```

The training script automatically logs to Weights & Biases (wandb), saves checkpoints to `outputs/`, and pushes the final model to HuggingFace.

**Training Stages**: The repo includes pre-configured experiment files:

```bash
# Stage 1: Full PEFT training (projector + encoder LoRA + decoder LoRA)
poetry run python src/train.py +experiments=stage1

# Mac minimal: Quick local testing
poetry run python src/train.py +experiments=mac_minimal

# Decoder LoRA only: Use archived config (projector + decoder LoRA, frozen encoder)
poetry run python src/train.py +experiments=archive/lora_decoder
```

Each experiment combines model, data, and training configs. Check `configs/hydra/experiments/` for all available presets.

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

## What makes this repo different?

Tiny Audio is not a SOTA ASR model. It's a **single, cohesive, minimal, readable, hackable codebase** designed to train an ASR model start to end and produce a working model you can actually use and learn from.

- **~1000 lines of core code** across 7 Python files in `src/`
- **Parameter-efficient training**: Train only ~146M parameters (projector + encoder LoRA + decoder LoRA) instead of 9.3B+
- **Dependency-lite**: Just PyTorch, transformers, datasets, PEFT, and a few other essentials via Poetry
- **No magic**: Read the code and understand exactly what's happening
- **Fully yours**: Train it, modify it, deploy it however you want
- **Flexible training**: Easily toggle encoder/decoder LoRA on/off, adjust rank, change target modules

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
- [Qwen-3 8B](https://huggingface.co/Qwen/Qwen3-8B) by Alibaba for language modeling
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
