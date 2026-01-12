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

**Run evaluation:**

```bash
poetry run ta eval mazesmazes/tiny-audio --max-samples 100
```

Output shows Word Error Rate (WER) for each sample. Lower is better.

### Exercise 3: Visualize Data Flow (20 min)

```bash
poetry run python docs/course/examples/trace_data.py
open docs/course/examples/data_trace.html
```

You'll see:
1. **Waveform** — raw audio amplitude
2. **Spectrogram** — frequency over time
3. **Encoder output** — what the model "hears"
4. **Projector output** — translation to text space

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `poetry install` hangs | `poetry install -vvv` for verbose output |
| Python version mismatch | Install 3.10+ with `pyenv` or `brew install python@3.11` |
| Model download fails | Check HF_TOKEN; try `huggingface-cli login` |
| Port 7860 in use | Use `--port 7861` |

---

## Key Takeaways

1. **Architecture**: Encoder (frozen) → Projector (trained) → Decoder (frozen)
2. **Efficiency**: Only train ~12M parameters
3. **Cost**: ~24 hours, ~$12

---

[← Course Overview](./0-course-overview.md) | [Class 2: Training →](./2-training.md)
