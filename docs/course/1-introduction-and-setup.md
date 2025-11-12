# Class 1: Introduction and Setup

**Duration**: 1 hour (20 min lecture + 40 min hands-on)

**Goal**: Understand ASR systems and run your first model inference

## Learning Objectives

By the end of this class, you will:

- Understand what automatic speech recognition (ASR) is and why it matters

- Know the three main components of the Tiny Audio architecture

- Have a working development environment

- Successfully run inference on an audio file

______________________________________________________________________

# PART A: LECTURE (20 minutes)

## 1. What is Automatic Speech Recognition? (5 min)

### The Problem

Humans communicate primarily through speech, but computers understand text. ASR is the magic that bridges this gap, turning the rich, messy, and beautiful complexity of human speech into structured, machine-readable text.

**Real-world applications:**

- Voice assistants (Siri, Alexa, Google Assistant)

- Transcription services (meetings, podcasts, interviews)

- Accessibility tools (live captioning for deaf/hard-of-hearing)

- Voice search and commands

- Medical dictation systems

### The Challenge

Speech recognition is hard because:

1. **Audio variability**: Different accents, speaking speeds, background noise
1. **Ambiguity**: "I scream" vs "ice cream" sound identical
1. **Context dependency**: Understanding requires linguistic knowledge
1. **Real-time constraints**: Users expect instant responses

### How Modern ASR Works

Modern ASR has evolved dramatically:

**Classic Era (1950s-2010s)**: Rule-based systems → Hidden Markov Models

- Limited by manual feature engineering
- Required careful tuning for each language/domain

**Deep Learning Era (2010s-2020)**: RNNs → Attention → Transformers

- Neural networks learn features automatically
- Transformers (2017) enabled parallel processing and better context
- Much better accuracy, but still data-hungry

**Modern Era (2020s-Present)**: Self-Supervised + Multimodal

- **Self-supervised pre-training**: Models learn from unlabeled audio
  - wav2vec 2.0, HuBERT, Whisper
  - Millions of hours of audio without transcriptions
- **Transfer learning**: Leverage pre-trained audio encoders + language models
- **Multimodal architectures**: Connect audio directly to text generation

**This is what Tiny Audio uses!** We combine:

- Pre-trained audio encoder (HuBERT or Whisper)
- Pre-trained language model (Qwen3-8B or SmolLM3-3B)
- Small trainable bridge (projector) to connect them

______________________________________________________________________

## 2. The Tiny Audio Architecture (10 min)

### High-Level Overview

Tiny Audio uses a three-component architecture:

```
Audio File → Audio Encoder → Audio Projector → Language Model → Text
            (HuBERT/Whisper)   (Linear MLP)     (Qwen3-8B/SmolLM3-3B)


```

**Experiment Preview**: Throughout the course, you'll be able to experiment with switching these components to customize your model.

Let's understand each component:

### Component 1: Audio Encoder (HuBERT or Whisper)

**Purpose**: Convert raw audio waveforms into meaningful feature representations

**What it does**

- Takes audio waveform (numbers representing sound pressure over time)

- Outputs a sequence of embedding vectors (one per ~20ms of audio)

- Each vector captures phonetic and acoustic information

**Default Encoder: HuBERT-XLarge**

- Pre-trained on 60K hours of LibriLight (self-supervised learning)
- 1.3 billion parameters (frozen during our training with optional LoRA adapters)
- Excellent for English speech recognition

**Alternative: Whisper**

- Pre-trained on 680K hours of multilingual data
- 1.5 billion parameters
- Better for multilingual tasks

**Analogy**: An expert musician who can listen to any piece of music and instantly transcribe the notes, rhythm, and instrumentation, without ever seeing the sheet music.

### Component 2: Audio Projector (~13-138M parameters)

**Purpose**: Bridge the gap between the audio and language worlds.

**What it does**

- Takes encoder's audio embeddings (1280-dim from HuBERT or Whisper)

- Downsamples by 5x (reduces sequence length for efficiency)

- Transforms to match the language model's input format (2048-dim for Qwen3-8B, 1536-dim for SmolLM3-3B)

**Architecture**: Simple yet effective linear projection (we'll dive deeper in Class 3)

- Pre-normalization layer (RMSNorm)

- Single linear projection with dropout for regularization

- Post-normalization layer (RMSNorm)

**Key insight**: This is the **largest trainable component** - the projector is fully trained from scratch (no pre-training).

**Analogy**: A skilled diplomat who can fluently translate between two very different cultures, ensuring the meaning and nuance are preserved.

### Component 3: Language Model Decoder (Qwen3-8B or SmolLM3-3B)

**Purpose**: Generate a coherent and grammatically correct text transcription.

**What it does**

- Receives audio embeddings (via projector)

- Uses its linguistic knowledge to predict the text

- Handles grammar, spelling, and punctuation

**Default Decoder: Qwen3-8B**

- 8 billion parameters
- Strong multilingual capabilities
- We use LoRA (rank 8) to efficiently adapt (~4M trainable params)

**Alternative: SmolLM3-3B**

- 3 billion parameters
- Smaller, faster, good for resource-constrained environments
- Also uses LoRA for efficient adaptation

**Analogy**: A master storyteller who can take a sequence of events (the audio features) and weave them into a compelling narrative (the final transcription).

### Why This Architecture?

**Efficiency**: We train ~150M parameters instead of 9.3+ billion (~1.6% of total model)

- Projector: ~138M (fully trained, size varies based on encoder/decoder choice)

- Encoder LoRA: ~4M (adapter weights, r=16) - OPTIONAL, can be disabled

- Decoder LoRA: ~4M (adapter weights, r=8) - OPTIONAL, can be disabled

**Flexibility**: You can train in different modes:

- **Full PEFT**: Projector + Encoder LoRA + Decoder LoRA (~146M params)
- **Projector + Decoder LoRA**: Frozen encoder, trainable projector and decoder adapters (~142M params)
- **Projector Only**: Frozen encoder and decoder, only projector learns (~138M params)

**Speed**: Training completes in ~24 hours on a single GPU

**Cost**: ~$12 for a full training run

**Quality**: Leverages pre-trained knowledge from both audio and language domains

**A Note on Architectural Choices:**

The Tiny Audio architecture is a **dense transformer-based** model. It's crucial to start with a **proven, stable baseline**. While other exciting architectures like Mixture-of-Experts (MoE) and Hybrids (combining transformers with other architectures like SSMs) exist, they introduce complexity that isn't necessary for our goal.

- **Dense models** are well-understood, stable to train, and perform exceptionally well, especially for a focused task like ours.

- **MoE and Hybrid models** are powerful but can be more complex to train and tune. They are often used for massive, general-purpose models, but for our specific use case, a dense model is the most direct path to a high-quality, custom ASR model.

By starting with a solid, well-understood architecture, we can focus on the nuances of audio processing, data, and training, which is where we'll see the biggest improvements.

______________________________________________________________________

## 3. Course Goals (5 min)

By the end of this 6-class course, each student will:

1. **Train** their own customized ASR model
1. **Evaluate** it on standard benchmarks
1. **Push** it to their own HuggingFace account
1. **Add** their results to the community leaderboard

This isn't just learning - you'll have a real, working, deployed model with your name on it!

______________________________________________________________________

# PART B: HANDS-ON WORKSHOP (40 minutes)

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Set up your environment (install dependencies, download samples)

- **Exercise 2**: Run inference with the pretrained model

- **Exercise 3**: Inspect the model configuration

By the end, you'll have a working setup and understand the complete model architecture!

______________________________________________________________________

## Workshop Exercise 1: Environment Setup (15 min)

### Goal

Get a working development environment with all dependencies installed.

### Your Task

Set up the Tiny Audio development environment.

### Instructions

**Step 1: Create Required Accounts**

Before we begin, you'll need accounts on three platforms (if you don't have them already):

**GitHub** (for version control and accessing the code):
- Visit https://github.com/signup
- Create a free account
- Verify your email address

**Hugging Face** (for downloading and sharing models):
- Visit https://huggingface.co/join
- Create a free account
- Go to Settings → Access Tokens and create a new token with "read" permissions
- Save this token somewhere safe - you'll need it later

**Weights & Biases** (for tracking training experiments):
- Visit https://wandb.ai/signup
- Create a free account
- After logging in, go to https://wandb.ai/authorize to get your API key
- Save this API key - you'll need it when you start training

**Note**: You can skip W&B for now if you want to get started quickly - it's only required for training (Class 4), not for running inference.

**Step 2: Check Prerequisites**

Open your terminal and verify:

```bash
python --version  # Should be 3.10 or newer
git --version     # Should show git version


```

**Step 3: Clone the Repository**

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio


```

**Step 4: Install Poetry (if needed)**

Poetry manages dependencies and virtual environments.

**Option A: Using Homebrew (macOS)**

If you don't have Homebrew, install it first:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then, install Poetry with Homebrew:

```bash
brew install poetry
```

**Option B: Using the official installer (Linux, Windows)**

```bash
# Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Add Poetry to your PATH if prompted, then verify:

```bash
poetry --version
```

**Step 5: Install Dependencies**

```bash
poetry install


```

This will:

- Create a virtual environment

- Install PyTorch, Transformers, and other dependencies

- Take ~5-10 minutes depending on your internet speed

**Troubleshooting:**

- If you get SSL errors, try: `poetry config certificates.default.cert false`

- If PyTorch installation fails, visit [pytorch.org](https://pytorch.org) for platform-specific instructions

**Step 6: Download Samples and Verify Installation**

```bash

poetry run python scripts/download_samples.py

poetry run python scripts/verify_setup.py

```

This will check:

- Python version (3.10+)

- All required packages

- Sample audio files

- Model configuration

You should see: `✅ All checks passed! You're ready to start the course.`

______________________________________________________________________

## Workshop Exercise 2: Launch the Demo and Run Evaluation (15 min)

### Goal

Experience the pre-trained Tiny Audio model through an interactive web interface and evaluate its performance on a benchmark dataset.

### Your Task

Launch the Gradio demo interface and run a quantitative evaluation on the LoquaciousSet benchmark.

### Instructions

**Step 1: Launch the Gradio Demo**

Start the interactive web interface:

```bash
poetry run python demo/gradio/app.py --model mazesmazes/tiny-audio --port 7860
```

**What's happening:**

1. Model downloads from HuggingFace Hub (~4GB, first time only)
1. Gradio interface starts on http://localhost:7860
1. You can now interact with the model through your web browser

**Step 2: Try the Demo**

Open your browser to http://localhost:7860 and experiment with:

- **Recording audio**: Click the microphone button to record yourself speaking
- **Uploading files**: Upload WAV, MP3, or other audio files
- **Different tasks**: Try "transcribe", "describe", "emotion", or create custom prompts
- **Text mode**: Test the language model capabilities without audio

Try recording yourself saying: "Hello, this is a test of the Tiny Audio speech recognition system."

**Note**: You might notice some mistakes - that's normal! Our model achieves ~12% Word Error Rate, meaning it gets about 88% of words correct.

**Step 3: Run Quantitative Evaluation**

Stop the demo (press Ctrl+C in the terminal where it's running), then evaluate the model on 100 samples from LoquaciousSet:

```bash
poetry run python scripts/eval.py mazesmazes/tiny-audio \
    --dataset loquacious \
    --max-samples 100 \
    --split test \
    --config medium
```

**What's happening:**

1. Downloads LoquaciousSet test split (streaming, so only downloads what's needed)
1. Runs inference on 100 randomly shuffled samples
1. Computes Word Error Rate (WER) using Whisper's text normalization
1. Shows per-sample results and cumulative statistics
1. Saves detailed results to `outputs/eval_loquacious_mazesmazes_tiny-audio/`

**Expected output:**

```
Sample 1: WER = 8.33%, Time = 1.23s
  Ref:  The quick brown fox jumps over the lazy dog
  Pred: The quick brown fox jumps over the lazy dog

Sample 2: WER = 15.00%, Time = 1.45s
  Ref:  She sells seashells by the seashore
  Pred: She sells sea shells by the sea shore

...

================================================================================
CHECKPOINT @ 100 samples:
  Corpus WER: 12.45%
  Avg Time/Sample: 1.35s
================================================================================
```

**Step 4: Inspect the Results**

Check the detailed results file:

```bash
cat outputs/eval_loquacious_mazesmazes_tiny-audio/results.txt
```

This file contains:

- Overall WER and average response time
- Per-sample predictions vs ground truth
- Individual WER scores for each sample

______________________________________________________________________

## Workshop Exercise 3: Exploring the Code (10 min)

### Goal

Familiarize yourself with the project structure and key files.

### Your Task

Explore the codebase and locate the main components of the Tiny Audio architecture.

### Instructions

1. **Open the project in your code editor** (e.g., VS Code).
1. **Inspect the `src/` directory**:
   - `asr_modeling.py`: This is where the core model architecture is defined. Can you find the `ASRModel` class?
   - `asr_pipeline.py`: This file defines the custom ASR pipeline used by transformers.
   - `train.py`: The main script for training the model. We'll use this in a later class.
1. **Look at the `configs/` directory**:
   - `experiments/`: This directory contains configuration files for different training experiments.
   - `model/`: Here you can find the configuration for different model architectures.
1. **Check the `demo/gradio/` directory**:
   - `app.py`: The Gradio demo interface you just launched.
1. **Explore the `scripts/` directory**:
   - `verify_setup.py`: The script you ran earlier to check your environment.
   - `eval.py`: The evaluation script you used to benchmark the model.
   - `transcribe.py`: A command-line tool for quick audio transcription.

## Key Takeaways

- ASR is a challenging but solvable problem with modern deep learning.
- The Tiny Audio architecture is a powerful and efficient way to build a custom ASR model.
- You have a working development environment and can run inference on your own audio files.

______________________________________________________________________

## Further Reading (Optional)

### Foundational Papers

- **HuBERT**: [Self-Supervised Speech Representation Learning](https://arxiv.org/abs/2106.07447)

- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

- **Attention**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (foundational)

### Documentation & Tutorials

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

- [PyTorch Tutorials](https://pytorch.org/tutorials/)

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar

- [The Illustrated Word2vec](http://jalammar.github.io/illustrated-word2vec/) by Jay Alammar

### Videos

- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) by 3Blue1Brown

- [Transformers, explained](https://www.youtube.com/watch?v=TQQlZhbC5ps) by AI Coffee Break with Letitia

[Previous: Course Overview](./0-course-overview.md) | [Next: Class 2: Audio Processing and Encoders](./2-audio-processing-and-encoders.md)

**Questions or stuck?** Open an issue on GitHub or check the discussions!
