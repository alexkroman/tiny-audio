# Build Your Own Speech Recognition Model

A hands-on course: build, train, evaluate, and deploy your own ASR model with Tiny Audio.

| | |
|---|---|
| **Duration** | 3.5 hours of class time, plus an unattended training run |
| **Cost** | Classes 1 and 3 and the local smoke test are free. Class 2's cloud run costs GPU time (see [Budget](#budget)) |
| **Prerequisites** | Python, the command line, git |

## What You'll Build

```
Audio → GLM-ASR encoder (frozen) → MLP projector (trained) → Qwen3-0.6B decoder (fine-tuned) → Text
```

| Component | Model | Params | During training |
|-----------|-------|--------|-----------------|
| Audio encoder | GLM-ASR-Nano-2512 (encoder only) | ~635M | Frozen |
| Projector | 2-layer MLP with frame stacking | ~6.3M | Trained from scratch |
| Decoder | Qwen3-0.6B | ~600M | Fine-tuned at a low learning rate |

You reuse two pretrained models and teach them to talk to each other. The projector is the
only part built from nothing; the decoder gets a gentle fine-tune so it learns to read audio
tokens. Then you publish the result to the Hugging Face Hub and put a live demo on the web.

---

## Course Structure

### [Class 1: Introduction and Setup](./1-introduction-and-setup.md)
*1.5 hours*

- How an encoder, a projector, and a decoder turn audio into text
- Exactly what the projector computes, with real shapes and sizes
- Environment setup
- Running inference with the published model and exploring the `ta` CLI

### [Class 2: Training](./2-training.md)
*1 hour, then training runs on its own*

- What `scripts/train.py` does, step by step
- A free local smoke test of the whole training loop
- Sizing a cloud GPU with `ta runpod plan`
- Writing your own experiment config and launching on RunPod
- Reading the loss and gradient curves

### [Class 3: Evaluation and Deployment](./3-evaluation-and-deployment.md)
*1 hour*

- Word Error Rate (WER) and text normalization
- Evaluating across 13 datasets and against commercial APIs
- Finding your model's worst samples and entity errors
- Publishing the model and deploying a Gradio demo to Hugging Face Spaces

### How the Classes Fit Together

Class 1 gives you a working install and the mental model. Class 2 uses both to write a config
and launch a run. The run keeps going after class, so leave a few hours (or a day) before
Class 3, which evaluates and deploys the checkpoints the run pushed to the Hub. If your run
isn't ready, Class 3 works unchanged with the published model `mazesmazes/tiny-audio`; nothing
in it is blocked on your own training.

### [Quick Reference](./4-quick-reference.md)

Commands, hyperparameters, troubleshooting.

### [Glossary](./5-glossary.md)

Key terms defined.

---

## Budget

The cost of Class 2 is entirely yours to choose. Three tiers:

| Tier | Data | Hardware | Cost |
|------|------|----------|------|
| **Smoke test** | 73 LibriSpeech clips | Your laptop (Apple Silicon or any CUDA GPU) | Free |
| **Course run** (what Class 2 walks through) | LoquaciousSet `small` (~250 hours) | One rented GPU, a few hours | A few GPU-hours at RunPod's hourly rate |
| **Production recipe** (`stage_1`) | The `multiasr` mix, ~3M clips, over a terabyte on disk | An 80 GB GPU for a day or more | Hundreds of dollars |

`ta runpod plan` measures the GPU memory and disk any config needs before you rent anything.
Class 2 shows you how to use it.

---

## Requirements

**Development (Classes 1 and 3):** any modern laptop with 8 GB RAM and 20 GB of free disk.
Loading the published model downloads about 6 GB (its own weights plus the encoder's repo).

**Local smoke test (Class 2, optional):** a machine with 16 GB of RAM or more. Apple Silicon
with 32 GB is comfortable. It runs the real training loop, including the decoder fine-tune, on
a tiny dataset.

**Cloud training (Class 2):** a RunPod account and a GPU. The class sizes it for you.

---

## Accounts Needed

1. [GitHub](https://github.com)
2. [Hugging Face](https://huggingface.co) (create a write token for pushing checkpoints)
3. [Weights & Biases](https://wandb.ai) (training curves)
4. [RunPod](https://runpod.io) (Class 2 only)

---

[Start Class 1 →](./1-introduction-and-setup.md)
