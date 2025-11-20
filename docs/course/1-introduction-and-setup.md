# Class 1: Introduction and Setup

**Duration**: 1 hour (20 min lecture + 40 min hands-on)

**Goal**: Understand ASR systems and run your first model inference.

## 🎯 Learning Objectives

By the end of this class, you will be able to:

- Explain what Automatic Speech Recognition (ASR) is and why it matters.

- Identify the three main components of the Tiny Audio architecture.

- Set up a working development environment.

- Successfully run inference (get a transcription) on an audio file.

______________________________________________________________________

# PART A: LECTURE (20 minutes)

## 1. Course Goals (5 min)

By the end of this 6-class course, each student will:

1. **Train** their own customized ASR model.

1. **Evaluate** it on standard benchmarks.

1. **Push** it to their own Hugging Face account.

1. **Add** their results to the community leaderboard.

This isn't just theory—you'll build a real, working model and deploy it with your name on it!

______________________________________________________________________

## 2. What is Automatic Speech Recognition? (5 min)

### The Problem

Humans communicate primarily through speech, but computers understand text. ASR is the technology that bridges this gap, turning the rich, messy, and complex sound of human speech into structured, machine-readable text.

**Real-world applications:**

- Voice assistants (Siri, Alexa, Google Assistant)

- Transcription services (meetings, podcasts, interviews)

- Accessibility tools (live captioning for deaf/hard-of-hearing)

- Voice search and commands

- Medical dictation systems

### The Challenge

Speech recognition is difficult because it must handle two distinct types of variability:

**Acoustic Variability (Handled by the Audio Encoder)**

1. **Speaker differences**: Accents, pitch, speaking rate, gender, age.

1. **Environmental noise**: Background sounds, echo, interference.

1. **Recording quality**: Microphone type, compression, sample rate.

1. **Pronunciation variations**: Casual vs. formal speech, mumbling, emphasis.

**Linguistic Variability (Handled by the Language Model Decoder)**

1. **Homophone ambiguity**: "I scream" vs. "ice cream" sound identical.

1. **Context dependency**: "read" (present) vs. "read" (past) requires sentence context.

1. **Domain knowledge**: Technical terms, proper nouns, specialized vocabulary.

1. **Grammar & punctuation**: Determining sentence boundaries and structure.

This two-part challenge is why modern ASR systems use **specialized components** for each task. You'll see this exact design pattern in the Tiny Audio architecture.

### How Modern ASR Works

ASR has evolved dramatically:

**Classic Era (1950s-2010s)**: Rule-based systems → Hidden Markov Models (HMMs).

- Limited by manual feature engineering.

- Required careful tuning for each language and domain.

**Deep Learning Era (2010s-2020)**: RNNs → Attention → Transformers.

- Neural networks learn features automatically from data.

- Transformers (2017) enabled parallel processing and better handling of long-range context.

- Much better accuracy, but required massive labeled (audio + text) datasets.

**Modern Era (2020s-Present)**: Self-Supervised + Multimodal.

- **Self-supervised pre-training**: Models (like HuBERT, wav2vec 2.0) learn rich representations from massive amounts of unlabeled audio.

- **Transfer learning**: We can leverage these powerful, pre-trained audio encoders and language models.

- **Multimodal architectures**: Connect audio models directly to text-generation models.

**This is the exact approach Tiny Audio uses!** We combine:

- A pre-trained **Audio Encoder** (HuBERT or Whisper).

- A pre-trained **Language Model** (Qwen3-8B or SmolLM3-3B).

- A small, trainable **Projector** (a bridge) to connect them.

______________________________________________________________________

## 3. The Tiny Audio Architecture (10 min)

### High-Level Overview

Tiny Audio uses a simple three-component architecture:

```
Audio File → Audio Encoder → Audio Projector → Language Model → Text
            (HuBERT/Whisper)   (Simple MLP)     (Qwen3-8B/SmolLM3-3B)
```

**Experiment Preview**: Throughout the course, you'll experiment with switching these components to customize your model.

### Component 1: Audio Encoder (e.g., HuBERT)

**Purpose**: Convert the raw audio waveform into meaningful feature representations (embeddings).

**What it does:**

- Takes the audio waveform (a list of numbers representing sound pressure).

- Outputs a sequence of embedding vectors (one per ~20ms of audio).

- Each vector captures phonetic and acoustic information.

**Default (HuBERT-XLarge):**

- Pre-trained on 60,000 hours of unlabeled audio.

- 1.3 billion parameters.

- These are frozen (not trained) in our course for efficiency.

**Analogy**: An expert musician who can listen to any piece of music and instantly transcribe the notes, rhythm, and instrumentation, without needing the sheet music.

### Component 2: Audio Projector (~13M parameters)

**Purpose**: Act as a "translator" between the audio encoder and the language model.

**What it does:**

- Takes the audio embeddings from the encoder (e.g., 1280-dimension vectors from HuBERT).

- Downsamples by 5x (reduces the sequence length for better efficiency).

- Transforms (projects) these features to match the dimensions the language model expects (e.g., 2048-dimension for Qwen3-8B).

**Architecture**: A simple Multi-Layer Perceptron (MLP). We'll dive deeper in Class 3.

**Key Insight**: This is the main component we train from scratch.

**Analogy**: A skilled diplomat who fluently translates between two very different cultures (the "audio world" and the "language world"), ensuring meaning is preserved.

### Component 3: Language Model Decoder (e.g., Qwen3-8B)

**Purpose**: Take the translated audio features and generate a coherent, grammatically correct text transcription.

**What it does:**

- Receives the audio embeddings from the projector.

- Uses its vast linguistic knowledge (grammar, facts, context) to predict the text.

- Handles spelling, punctuation, and sentence structure.

**Default (Qwen3-8B):**

- 8 billion parameters.

- We keep this frozen during training and only train the projector that feeds into it.

**Analogy**: A master storyteller who takes a sequence of key events (the audio features) and weaves them into a compelling, complete narrative (the final transcription).

### Why This Architecture?

**Efficiency**: We only train ~13M parameters instead of the full 9.3+ billion (~0.14% of the total).

- Projector: ~13M (fully trained)

- Encoder: Frozen (no training)

- Decoder: Frozen (no training)

**Simplicity**: By focusing on just the projector, training is:

- **Straightforward**: No complex adapter configurations

- **Stable**: Fewer hyperparameters to tune

- **Fast**: Less computational overhead

**Fast Training**: A full run completes in ~24 hours on a single GPU.

**Low Cost**: ~$12 for a full training run on a cloud GPU.

**High Quality**: Leverages the "wisdom" of two massive pre-trained models.

______________________________________________________________________

# PART B: HANDS-ON WORKSHOP (40 minutes)

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Set up your environment.

- **Exercise 2**: Run inference with the pre-trained model.

- **Exercise 3**: Inspect the code and configuration.

By the end, you'll have a working setup and a clear map of the project's code.

______________________________________________________________________

## Workshop Exercise 1: Environment Setup (15 min)

### Goal

Get a working development environment with all dependencies installed.

### Your Task

Follow the steps to set up the Tiny Audio development environment.

### Instructions

**Step 1: Create Required Accounts**

You'll need accounts on three free platforms:

**GitHub** (for code):

- Visit <https://github.com/signup>

- Create an account and verify your email.

**Hugging Face** (for models):

- Visit <https://huggingface.co/join>

- Create an account.

- Go to Settings → Access Tokens and create a new token with "read" permissions.

- Save this token—you'll need it to log in from your terminal.

**Weights & Biases** (for experiment tracking):

- Visit <https://wandb.ai/signup>

- Create an account.

- Go to <https://wandb.ai/authorize> to get your API key.

- Save this key for when we start training.

**Note**: You can skip W&B for now if you just want to run inference. It's only required for training (Class 4).

**Step 2: Check Prerequisites**

Open your terminal and verify you have `python` (3.10+) and `git` installed:

```bash
python --version  # Should be 3.10 or newer
git --version     # Should show a git version
```

**Step 3: Clone the Repository**

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
```

**Step 4: Install Poetry (Dependency Manager)**

If you don't have Poetry, install it.

**Option A: Using Homebrew (macOS)**

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

After installation (you may need to restart your terminal), verify it works:

```bash
poetry --version
```

**Step 5: Install Dependencies**

This command creates a virtual environment and installs all required packages (like PyTorch and Transformers).

```bash
poetry install
```

This will take ~5-10 minutes.

**Troubleshooting:**

- If you get SSL errors: `poetry config certificates.default.cert false`

- If PyTorch installation fails, visit [pytorch.org](https://pytorch.org) for platform-specific instructions.

**Step 6: Download Samples and Verify Installation**

Run these two scripts from the `tiny-audio` directory:

```bash
poetry run python scripts/download_samples.py  # Optional, takes a few minutes
poetry run python scripts/verify_setup.py
```

**Note**: The `download_samples.py` script is optional if you already have your own audio files handy (e.g., .wav or .mp3 files). It downloads sample audio files from LibriSpeech for testing and will take a few minutes to complete.

The verify script will check your Python version, packages, and sample files. You should see: `✅ All checks passed! You're ready to start the course.`

______________________________________________________________________

## Workshop Exercise 2: Run Inference & Evaluation (15 min)

### Goal

Run the pre-trained model in a web demo and quantitatively evaluate its performance.

### Your Task

Launch the Gradio demo, test it, and then run the benchmark evaluation script.

### Instructions

**Step 1: Launch the Gradio Demo**

This command downloads the pre-trained model and starts an interactive web interface.

```bash
poetry run python demo/gradio/app.py --model mazesmazes/tiny-audio --port 7860
```

**What this command does:**

1. Downloads the `mazesmazes/tiny-audio` model from Hugging Face (~4GB, first time only).

1. Starts the Gradio web server on <http://localhost:7860>.

**Step 2: Try the Demo**

Open <http://localhost:7860> in your browser. Experiment with:

- **Recording audio**: Click the microphone to record yourself.

- **Uploading files**: Upload your own WAV or MP3 files.

- **Different tasks**: Try "transcribe", "describe", or custom prompts.

- **Text mode**: Test the language model's capabilities without audio.

Try recording: "Hello, this is a test of the Tiny Audio speech recognition system."

**Note**: You might see some mistakes. Our base model has a ~12% Word Error Rate (WER), meaning it gets about 88% of words correct.

**Step 3: Run Quantitative Evaluation**

Stop the demo (press Ctrl+C in your terminal). Now, let's evaluate the model on 100 samples from the LoquaciousSet benchmark:

```bash
poetry run python scripts/eval.py mazesmazes/tiny-audio \
    --dataset loquacious \
    --max-samples 100 \
    --split test \
    --config medium
```

**What this command does:**

1. Streams the `loquacious` dataset test split.

1. Runs inference on 100 random samples.

1. Computes the Word Error Rate (WER), a standard ASR metric.

1. Shows per-sample results and saves a detailed log.

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

Open and inspect the detailed results file. You can use a text editor or the `cat` command:

```bash
# On macOS/Linux
cat outputs/eval_loquacious_mazesmazes_tiny-audio/results.txt

# On Windows
type outputs\eval_loquacious_mazesmazes_tiny-audio\results.txt
```

This file contains the overall WER, average time, and per-sample predictions.

______________________________________________________________________

## Workshop Exercise 3: Exploring the Code (10 min)

### Goal

Familiarize yourself with the project structure and key files.

### Your Task

Open the project in your code editor and locate the main components.

### Instructions

1. Open the `tiny-audio` folder in your code editor (e.g., VS Code).

1. **Inspect the `src/` directory**: This is the main source code.

   - `asr_modeling.py`: Defines the core model. Look for the `ASRModel` class, which brings the 3-part architecture together.

   - `asr_pipeline.py`: Defines the custom Hugging Face pipeline for easy inference.

   - `train.py`: The main script for training, which we'll use in Class 4.

1. **Look at the `configs/` directory**: This holds all configuration files.

   - `experiments/`: YAML files defining different training experiments (datasets, hyperparameters, etc.).

   - `model/`: YAML files defining the components to use (which encoder, projector, and decoder).

1. **Check the `demo/gradio/` directory**:

   - `app.py`: The code for the Gradio demo you just launched.

1. **Explore the `scripts/` directory**:

   - `verify_setup.py`: The setup-check script you ran.

   - `eval.py`: The evaluation script you just used.

   - `transcribe.py`: A simple command-line tool for transcribing a single file.

______________________________________________________________________

## 💡 Key Takeaways

- ASR is a challenging but solvable problem with modern, modular deep learning.

- The Tiny Audio architecture is a powerful and efficient way to build a custom ASR model by combining pre-trained components.

- You now have a working development environment and can run inference on your own audio files.

______________________________________________________________________

## Further Reading (Optional)

### Foundational Papers

- **HuBERT**: [Self-Supervised Speech Representation Learning](https://arxiv.org/abs/2106.07447)

- **Attention**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (The paper that introduced Transformers)

### Documentation & Tutorials

- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/index)

- [PyTorch Tutorials](https://pytorch.org/tutorials/)

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar

### Videos

- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) by 3Blue1Brown

- [Transformers, explained](https://www.youtube.com/watch?v=TQQlZhbC5ps) by AI Coffee Break with Letitia

______________________________________________________________________

[Previous: Course Overview](./0-course-overview.md) | [Next: Class 2: The End-to-End ASR Architecture](./2-end-to-end-architecture.md)
