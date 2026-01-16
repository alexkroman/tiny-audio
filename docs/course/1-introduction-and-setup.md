# Class 1: Introduction and Setup

*1.5 hours (40 min lecture + 50 min hands-on)*

**Goal**: Understand the architecture and set up your environment.

---

## Part A: Lecture (40 min)

### What is ASR?

Automatic Speech Recognition converts speech to text. The challenge:

- **Acoustic variability**: accents, noise, recording quality
- **Linguistic variability**: homophones ("to" vs "two"), context, punctuation

### The Architecture

```
Audio → Pre-processing → Encoder → Projector → Decoder → Text
         (spectrogram)   (frozen)  (trained)   (frozen)
```

**Encoder (GLM-ASR)**: Converts audio to embeddings. Processes spectrogram, outputs features for each ~20ms segment. Frozen during training.

**Projector (MLP)**: Translates audio embeddings to text space. Maps encoder dimensions → decoder dimensions. **This is the only component we train.**

**Decoder (Qwen3)**: Generates text from embeddings. Handles spelling, punctuation, grammar. Frozen during training.

### Why This Works

| Component | Params | Status |
|-----------|--------|--------|
| GLM-ASR Encoder | ~600M | Frozen |
| MLP Projector | ~12M | **Trained** |
| Qwen3-0.6B | ~600M | Frozen |

- Train ~12M params instead of billions
- ~24 hours on A40, ~$12
- Leverage pretrained knowledge from both models

### How the Projector Works

The projector does two things:

1. **Dimension mapping**: GLM-ASR outputs 768-dim vectors, Qwen3 expects 1024-dim. The projector bridges this gap.

2. **Frame stacking (downsampling)**: Audio has many frames (~50/second). The projector stacks adjacent frames to reduce the sequence length:

```
output_length = (input_length - k) // k + 1
```

Where `k` is the pooling stride (default: 5). This reduces compute while preserving information.

---

## Part B: Hands-On (50 min)

### Exercise 1: Environment Setup (15 min)

**Create accounts:**

| Account | URL | Purpose |
|---------|-----|---------|
| GitHub | [github.com](https://github.com) | Code |
| Hugging Face | [huggingface.co](https://huggingface.co) | Models |
| Weights & Biases | [wandb.ai](https://wandb.ai) | Training (Class 2) |

**Install:**

```bash
# Check prerequisites
python --version  # 3.10+
git --version

# Clone repo
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio

# Install Poetry (macOS)
brew install poetry

# Or with pip (any platform)
pip install poetry

# Install dependencies
poetry install

# Verify
poetry run python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Exercise 2: Run Inference (15 min)

**Launch demo:**

```bash
poetry run ta demo --model mazesmazes/tiny-audio
```

Open [http://localhost:7860](http://localhost:7860). Try recording or uploading audio.

**Run from Python:**

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="mazesmazes/tiny-audio", trust_remote_code=True)
result = pipe("path/to/audio.wav")
print(result["text"])
```

**Run evaluation:**

```bash
poetry run ta eval -m mazesmazes/tiny-audio -n 100
```

Output shows Word Error Rate (WER) for each sample. Lower is better.

### Exercise 3: Explore the CLI (20 min)

The `ta` command (short for `tiny-audio`) provides many useful tools:

```bash
# See all commands
poetry run ta --help

# Evaluation commands
poetry run ta eval --help

# Analysis commands (find errors, compare models)
poetry run ta analysis --help

# Development commands
poetry run ta dev --help
```

**Try some commands:**

```bash
# Evaluate on 50 samples
poetry run ta eval -m mazesmazes/tiny-audio -n 50

# Run code quality checks
poetry run ta dev lint
poetry run ta dev test
```

---

## Understanding the Code

### Key Files

| File | Purpose |
|------|---------|
| `tiny_audio/asr_modeling.py` | Core model: encoder + projector + decoder |
| `tiny_audio/asr_config.py` | Configuration (model IDs, projector type, etc.) |
| `tiny_audio/projectors.py` | Projector architectures (MLP, MOSA, MoE, QFormer) |
| `tiny_audio/asr_pipeline.py` | HuggingFace pipeline for inference |
| `scripts/train.py` | Training script with Hydra configs |

### Configuration System

Tiny Audio uses Hydra for configuration:

```bash
# Default training
poetry run python scripts/train.py +experiments=mlp

# Override any value
poetry run python scripts/train.py +experiments=mlp training.learning_rate=1e-4

# Use key=value syntax (not --key value)
```

Config files live in `configs/`:
- `config.yaml` - Main defaults
- `experiments/` - Projector presets (mlp, mosa, moe)
- `training/` - Training hyperparameters
- `data/` - Dataset settings

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `poetry install` hangs | `poetry install -vvv` for verbose output |
| Python version mismatch | Install 3.10+ with `pyenv` or `brew install python@3.11` |
| Model download fails | Check HF_TOKEN; try `huggingface-cli login` |
| Port 7860 in use | Use `--port 7861` |
| Import errors | Run `poetry install` again |
| CUDA not found | Check `nvidia-smi`; install CUDA toolkit |

---

## Key Takeaways

1. **Architecture**: Encoder (frozen) → Projector (trained) → Decoder (frozen)
2. **Efficiency**: Only train ~12M parameters
3. **Cost**: ~24 hours, ~$12
4. **Tools**: `ta` CLI for evaluation, analysis, development

---

[← Course Overview](./0-course-overview.md) | [Class 2: Training →](./2-training.md)
