<div align="center">
  <img src="https://raw.githubusercontent.com/alexkroman/tiny-audio/main/public/logo.png" alt="Tiny Audio Logo" />
</div>

# Tiny Audio

## Train your own speech recognition model in 24 hours for $12

This repo is a minimal, hackable implementation of an ASR (Automatic Speech Recognition) model that you can train from scratch on a single GPU in 24 hours. Tiny Audio combines a HuBERT-XLarge encoder (1.3B params) with a SmolLM3-3B decoder (3B params), connected by a trainable audio projector (~122M params). Both the encoder and decoder use **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning. The result is a speech-to-text system you can train in under 24 hours for ~$12, deploy to HuggingFace, and use just like Whisper or any other ASR model.

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

## 🎓 Learn by Building: Free ASR Course

Want to deeply understand how speech recognition works? We've created a **6-hour hands-on course** that teaches you ASR by building your own model from scratch.

**[📚 Start the Course](docs/QUICKSTART.md)** | **[📖 Course Overview](docs/course/0-course-overview.md)**

**What you'll learn:**
- How audio becomes numbers and how encoders process them
- Language model architectures and multimodal bridging
- Parameter-efficient training with LoRA
- Model evaluation, debugging, and deployment

**What you'll build:**
- Your own trained ASR model
- Published to HuggingFace Hub with your name on it
- Results added to the community leaderboard

**Time**: 6 hours (6 one-hour classes) | **Cost**: ~$12 for GPU training

Each class has a 20-minute lecture and 40 minutes of hands-on coding. All exercise scripts are pre-created for you!

## Quick start

The fastest way to feel the magic is to train your own model. Boot up an A40 GPU box from your favorite provider (I like RunPod), clone this repo, and kick off training:

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
poetry install

# Quick test run (20 steps, ~5 minutes)
poetry run python src/train.py

# Full production training with encoder LoRA (~24 hours on A40)
export HF_TOKEN='your-token'  # Get from https://huggingface.co/settings/tokens
poetry run python src/train.py +experiments=stage1
```

If you want to run it on a remote GPU like RunPod, there are deployment scripts to make your life easier:

```bash
# Deploy code to remote GPU
poetry run deploy-runpod --host <pod-id>.runpod.io --port 22

# Start training remotely
poetry run remote-train \
  --host <pod-id>.runpod.io \
  --port 22 \
  --config stage1
```

Now wait ~24 hours. Once it's done, your model will be pushed to HuggingFace Hub automatically (if you set `HF_TOKEN`), and you can use it just like in the example above!

## How it works

Tiny Audio uses a parameter-efficient three-component architecture with **LoRA adapters on both the encoder and decoder**:

```text
Audio Waveform (16kHz)
    ↓
HuBERT-XLarge Encoder (1.3B params, frozen)
    + LoRA Adapters (r=8, ~2M params)
    ↓
Audio Embeddings [~149 frames × 1280 dim]
    ↓
Audio Projector (SwiGLU MLP, ~122M params)
    - Frame stacking (5x downsample)
    - Pre-norm (RMSNorm)
    - gate_proj & up_proj (6400 → 8192)
    - SwiGLU activation
    - down_proj (8192 → 2048)
    - Post-norm (RMSNorm)
    ↓
Language Embeddings [~30 frames × 2048 dim]
    ↓
SmolLM3-3B Decoder (3B params, frozen)
    + LoRA Adapters (r=64, ~15M params)
    + Flash Attention 2
    ↓
Text Transcription
```

**Three trainable components:**

1. **Audio Encoder (LoRA Fine-tuned)**: HuBERT-XLarge (1.3B parameters frozen) with LoRA adapters on attention layers (q_proj, k_proj) - adds ~2M trainable parameters (r=8, alpha=8)
2. **Audio Projector (Fully Trainable)**: A SwiGLU MLP (~122M parameters) that:
   - Stacks 5 consecutive frames (5x time reduction: ~149 → ~30 frames)
   - Maps 6400-dim stacked features → 8192-dim hidden → 2048-dim output
   - Uses pre/post RMSNorm for stable training
3. **Language Decoder (LoRA Fine-tuned)**: SmolLM3-3B (3B parameters frozen) with LoRA adapters on attention layers (q_proj, v_proj) - adds ~15M trainable parameters (r=64, alpha=32). Generates text transcription with Flash Attention 2

**Total trainable: ~139M params (3.2% of 4.3B total)**

### Key Concepts Explained

**What is LoRA?** Low-Rank Adaptation adds small trainable "adapter" matrices to frozen model weights. Instead of updating a 2048×2048 weight matrix (4.2M params), LoRA uses two small matrices: 2048×8 and 8×2048 (32K params). This is 0.76% the size but captures most of the adaptation needed!

**What is SwiGLU?** Swish Gated Linear Unit - a modern activation function that gates information flow. Used in Llama and other state-of-the-art models. Formula: `SwiGLU(x) = Swish(Wx) ⊗ (Vx)` where `Swish(x) = x × sigmoid(x)`.

**What is RMSNorm?** Root Mean Square Normalization - a simpler, faster alternative to LayerNorm that just normalizes by the RMS value: `x / sqrt(mean(x²))`. Used in modern transformers like Llama.

**Why 16kHz audio?** This sample rate captures human speech frequency range (85-255 Hz fundamental + harmonics up to ~8kHz) while being computationally efficient. Industry standard for ASR.

**Why 5x downsampling?** Reduces computational cost in the decoder while maintaining ~100ms temporal resolution per frame - detailed enough for accurate transcription.

### LoRA Configuration

**Encoder LoRA:**

- Rank: 8 (default)
- Alpha: 8 (scaling factor)
- Target modules: q_proj, k_proj in HuBERT attention layers
- Adds ~2M trainable parameters

*Decoder LoRA:*

- Rank: 64 (default)
- Alpha: 32 (scaling factor = 0.5)
- Target modules: q_proj, v_proj in SmolLM3 attention layers
- Adds ~15M trainable parameters

Why use parameter-efficient training with LoRA on both encoder and decoder? Because:

- **You train ~139M parameters** instead of 4.3+ billion (projector: ~122M + encoder LoRA: ~2M + decoder LoRA: ~15M)
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
- **Parameter-efficient training**: Train only ~139M parameters (projector + encoder LoRA + decoder LoRA) instead of 4.3B+
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
