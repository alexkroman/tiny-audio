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

Modern ASR systems have evolved through several generations:

**1st Generation (1950s-1980s)**: Rule-based pattern matching

- Limited vocabulary (~100 words)

- Speaker-dependent

**2nd Generation (1980s-2010s)**: Hidden Markov Models (HMMs)

- Better accuracy

- Still struggled with noise and accents

**3rd Generation (2010s-2020s)**: Deep Learning Era

- **RNNs and LSTMs** (2012-2015): Sequential processing of audio

  - Recurrent Neural Networks and Long Short-Term Memory networks
  - Better at capturing temporal dependencies than HMMs

- **Attention Mechanisms** (2014-2017): Model learns what to focus on

  - Attention Is All You Need (2017) introduces the Transformer architecture
  - Enables parallelization and better long-range dependencies

- **RNN-Transducer (RNN-T)** (2012-2019): Streaming ASR

  - Combines RNNs with CTC-like alignment
  - Enables real-time, low-latency transcription
  - Used in production systems (Google Assistant, etc.)

- **Transformers** (2017-2020): Replaced RNNs as the dominant architecture

  - Self-attention for better context modeling
  - Parallelizable training (much faster than RNNs)
  - Foundation for modern ASR

**4th Generation (2020s-Present)**: Self-Supervised Multimodal Transformers

- **Self-supervised pre-training** on unlabeled audio (wav2vec 2.0, HuBERT)

- **Transfer learning** from massive language models

- **Multimodal architectures** connecting audio and text

- **This is what Tiny Audio uses!**

______________________________________________________________________

## 2. The Tiny Audio Architecture (10 min)

### High-Level Overview

Tiny Audio uses a three-component architecture:

```
Audio File → Audio Encoder → Audio Projector → Language Model → Text
            (HuBERT)         (SwiGLU MLP)      (Qwen-3 8B)


```

**Experiment Preview**: In this course, you'll experiment with:

- Swapping encoders (HuBERT vs Wav2Vec2)

- Adjusting projector dimensions

- Using different language models

- Testing on various audio types

Let's understand each component:

### Component 1: Audio Encoder (HuBERT-XLarge)

**Purpose**: Convert raw audio waveforms into meaningful feature representations

**What it does**

- Takes audio waveform (numbers representing sound pressure over time)

- Outputs a sequence of embedding vectors (one per ~20ms of audio)

- Each vector captures phonetic and acoustic information

**Key insight**: HuBERT is **pre-trained** on thousands of hours of unlabeled speech, so it already "understands" human speech patterns before we even start training!

**Analogy**: An expert musician who can listen to any piece of music and instantly transcribe the notes, rhythm, and instrumentation, without ever seeing the sheet music.

**Size**: 1.3 billion parameters (frozen during our training)

### Component 2: Audio Projector (~138M parameters)

**Purpose**: Bridge the gap between the audio and language worlds.

**What it does**

- Takes HuBERT's audio embeddings

- Downsamples by 5x (reduces sequence length for efficiency)

- Transforms to match the language model's input format

**Architecture**: SwiGLU MLP (we'll dive deeper in Class 3)

- Pre-normalization layer

- Gated projection (allows selective information flow)

- Post-normalization layer

**Key insight**: This is the **largest trainable component** - all ~138M parameters learn during training.

**Analogy**: A skilled diplomat who can fluently translate between two very different cultures, ensuring the meaning and nuance are preserved.

### Component 3: Language Model Decoder (Qwen-3 8B)

**Purpose**: Generate a coherent and grammatically correct text transcription.

**What it does**

- Receives audio embeddings (via projector)

- Uses its vast linguistic knowledge to predict the text

- Handles grammar, spelling, and punctuation

**Key insight**: Qwen-3 is also **pre-trained** on a massive amount of text, so it already has a deep understanding of language before it ever "hears" any audio.

**Size**: 8 billion parameters (we use LoRA to adapt efficiently)

**Analogy**: A master storyteller who can take a sequence of events (the audio features) and weave them into a compelling narrative (the final transcription).

### Why This Architecture?

**Efficiency**: We train ~146M parameters instead of 9.3+ billion (1.6% of total model)

- Projector: ~138M (fully trained)

- Encoder LoRA: ~4M (adapter weights, r=16)

- Decoder LoRA: ~4M (adapter weights, r=8)

**Speed**: Training completes in ~24 hours on a single GPU

**Cost**: ~$12 for a full training run

**Quality**: Leverages pre-trained knowledge from both audio and language domains

**A Note on Architectural Choices:**

The Tiny Audio architecture is a **dense transformer-based** model. It's crucial to start with a **proven, stable baseline**. While other exciting architectures like Mixture-of-Experts (MoE) and Hybrids (combining transformers with other architectures like SSMs) exist, they introduce complexity that isn't necessary for our goal.

Recent research has shown that **instruct-tuned models** perform significantly better for speech recognition tasks than their base counterparts. This is why we use Qwen-3 8B Instruct rather than the base model - the instruction-following capabilities help the model better understand the task of transcribing audio to text.

- **Dense models** are well-understood, stable to train, and perform exceptionally well, especially for a focused task like ours.

- **MoE and Hybrid models** are powerful but can be more complex to train and tune. They are often used for massive, general-purpose models, but for our specific use case, a dense model is the most direct path to a high-quality, custom ASR model.

By starting with a solid, well-understood architecture, we can focus on the nuances of audio processing, data, and training, which is where we'll see the biggest improvements.

### Rules of Engagement

A disciplined, empirical approach to model training is crucial. As we go through this course, we'll follow a few key "rules of engagement":

1. **Systematic Beats Intuitive**: Don't just guess what will work. We'll use systematic experiments (ablations) to validate our choices.
1. **Change One Thing at a Time**: When we run experiments, we'll only change one variable at a time. This is the only way to know for sure what's responsible for any improvements (or regressions!).
1. **Validate Every Change**: Every modification, no matter how small, should be tested. We'll rely on evaluation metrics, not just gut feelings, to guide our decisions.

Adopting this mindset will not only lead to a better final model but will also teach you the disciplined process that professionals use to build world-class models.

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

- **Exercise 4**: Count trainable parameters

By the end, you'll have a working setup and understand the complete model architecture!

______________________________________________________________________

## Workshop Exercise 1: Environment Setup (15 min)

### Goal

Get a working development environment with all dependencies installed.

### Your Task

Set up the Tiny Audio development environment.

### Instructions

**Step 1: Check Prerequisites**

Open your terminal and verify:

```bash
python --version  # Should be 3.10 or newer
git --version     # Should show git version


```

**Step 2: Clone the Repository**

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio


```

**Step 3: Install Poetry (if needed)**

Poetry manages dependencies and virtual environments:

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -


```

Add Poetry to your PATH if prompted, then verify:

```bash
poetry --version


```

**Step 4: Install Dependencies**

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

**Step 5: Verify Installation**

```bash
poetry run python scripts/verify_setup.py


```

This will check:

- Python version (3.10+)
- All required packages
- Sample audio files
- Model configuration

You should see: `✅ All checks passed! You're ready to start the course.`

### Success Checkpoint

- [ ] Python 3.10+ installed
- [ ] All required packages installed
- [ ] Verification script shows all checks passed

If you hit any issues, check the error messages from the verification script for guidance!

______________________________________________________________________

## Workshop Exercise 2: Run Your First Inference (15 min)

### Goal

Transcribe an audio file using the pre-trained Tiny Audio model and experiment with different inputs.

### Your Task

Run inference on an audio file, see the transcription output, and experiment with model behavior.

### Instructions

**Step 1: Get an Audio File**

You need an audio file to transcribe. Choose one option:

**Option A: Use your own**

- Any WAV, MP3, or FLAC file works

- Speech works best (podcasts, meetings, voice memos)

**Option B: Record your own**

- Use your phone or computer

- Record yourself saying: "Hello, this is a test of the Tiny Audio speech recognition system."

- Save as `test.wav`

**Option C: Download a sample**

```bash
# Download a LibriSpeech test sample
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
# Now you have many .flac files in test-clean/


```

**Step 2: Create inference script**

Create a file called `test_inference.py` in the `tiny-audio/` directory:

Create a file called `test_inference.py`:

```python
from transformers import pipeline

# Load the pre-trained Tiny Audio model
print("Loading model...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True
)

print("✓ Model loaded!")

# Transcribe an audio file
# Replace with your own audio file path
audio_path = "path/to/your/audio.wav"

print(f"Transcribing {audio_path}...")
result = pipe(audio_path)

print("\nTranscription:")
print(result["text"])


```

**Step 3: Update the audio path**

Edit `test_inference.py` and change this line:

```python
audio_path = "path/to/your/audio.wav"


```

To point to your actual audio file, for example:

```python
audio_path = "test.wav"
# or
audio_path = "test-clean/1089/134686/1089-134686-0000.flac"


```

**Step 4: Run inference**

```bash
poetry run python test_inference.py


```

**What's happening:**

1. Model downloads from HuggingFace Hub (~4GB, first time only)
1. Audio file is loaded and resampled to 16kHz
1. Audio passes through: Encoder → Projector → Decoder
1. Text transcription is returned

**Expected output:**

```
Loading model...
✓ Model loaded!
Transcribing audio.wav...

Transcription:
Hello, this is a test of the Tiny Audio speech recognition system.


```

**What's happening:**

1. Model downloads from HuggingFace Hub (~4GB, first time only - be patient!)
1. Audio is loaded and resampled to 16kHz
1. Audio passes through: Encoder → Projector → Decoder
1. Text transcription is returned

**Expected output:**

```
Loading model...
✓ Model loaded!
Transcribing test.wav...

Transcription:
Hello, this is a test of the Tiny Audio speech recognition system.


```

### Success Checkpoint

- [ ] I successfully ran the script

- [ ] I saw a transcription output

- [ ] The transcription is mostly accurate (doesn't have to be perfect!)

**Note**: You might notice some mistakes - that's normal! Our model achieves ~12% Word Error Rate, meaning it gets about 88% of words correct.

______________________________________________________________________

## Understanding Generation Parameters (5 min)

Before we experiment, let's understand how text generation works and what parameters we can control.

### How ASR Generation Works

The language model generates text **one token at a time** (auto-regressively):

1. Receives audio embeddings from the projector
1. Predicts the most likely next token
1. Adds that token to the sequence
1. Repeats until it generates an end-of-sequence token

**Key insight**: At each step, the model has a probability distribution over all possible next tokens. Generation parameters control how we sample from this distribution.

### Important Generation Parameters

**1. `max_new_tokens` (default: 128)**

- Maximum number of tokens to generate
- **Too low**: Transcription gets cut off mid-sentence
- **Too high**: Slower inference, may generate extra text
- **Rule of thumb**: ~1 token per 0.4 seconds of audio

**2. `num_beams` (default: 1)**

- Controls beam search width
- **1**: Greedy decoding (fastest, picks most likely token each step)
- **3-5**: Beam search (explores multiple paths, often more accurate)
- **Trade-off**: Higher beams = slower but potentially better quality

**3. `temperature` (default: 1.0, only used with sampling)**

- Controls randomness in token selection
- **< 1.0**: More deterministic (conservative, repetitive)
- **= 1.0**: Balanced (use model's actual probabilities)
- **> 1.0**: More random (creative but less accurate)
- **For ASR**: Usually keep at 1.0 or use greedy decoding

**4. `do_sample` (default: False)**

- **False**: Use deterministic decoding (greedy or beam search)
- **True**: Sample from probability distribution (adds randomness)
- **For ASR**: Usually False (we want accurate transcription, not creativity)

**5. `chunk_length_s` (default: 30)**

- How many seconds of audio to process at once
- **Shorter (15-20s)**: Lower memory, more chunks to process
- **Longer (30s)**: More context, fewer chunks
- **Trade-off**: Memory usage vs. processing overhead

**6. `stride_length_s` (default: (5, 5))**

- Overlap between chunks as tuple (left_overlap, right_overlap) in seconds
- Helps smooth transitions between chunks
- Default (5, 5) means 5 seconds overlap on each side

### Quick Reference Table

| Parameter | Default | When to Change | Effect |
|-----------|---------|----------------|---------|
| `num_beams` | 1 | Want better quality | Slower, more accurate |
| `chunk_length_s` | 30 | Long audio or low memory | Memory usage |
| `max_new_tokens` | 128 | Very long audio | Prevents cutoff |
| `do_sample` | False | Experimenting only | Adds randomness |
| `temperature` | 1.0 | With sampling only | Controls randomness |
| `stride_length_s` | (5, 5) | Custom chunking | Overlap between chunks |

______________________________________________________________________

### Experimentation Time

Now let's experiment with these parameters and see how they affect transcription!

Create a new file `experiment_generation.py`:

```python
from transformers import pipeline
import time

# Load model once
print("Loading model...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True
)
print("✓ Model loaded!\n")

# Your audio file
audio_path = "test.wav"  # Change to your file

print("=" * 70)
print("GENERATION PARAMETER EXPERIMENTS")
print("=" * 70)

# Experiment 1: Greedy vs Beam Search
print("\n📊 Experiment 1: Greedy vs Beam Search")
print("-" * 70)

configs = [
    {"name": "Greedy (num_beams=1)", "params": {"num_beams": 1}},
    {"name": "Beam Search (num_beams=3)", "params": {"num_beams": 3}},
    {"name": "Beam Search (num_beams=5)", "params": {"num_beams": 5}},
]

for config in configs:
    start = time.time()
    result = pipe(audio_path, **config["params"])
    duration = time.time() - start

    print(f"\n{config['name']}:")
    print(f"  Time: {duration:.2f}s")
    print(f"  Text: {result['text'][:100]}...")

# Experiment 2: Different Chunk Lengths
print("\n\n📊 Experiment 2: Chunk Length Impact")
print("-" * 70)

chunk_configs = [
    {"name": "Short chunks (15s)", "chunk_length_s": 15, "stride_length_s": (2, 2)},
    {"name": "Medium chunks (20s)", "chunk_length_s": 20, "stride_length_s": (3, 3)},
    {"name": "Long chunks (30s)", "chunk_length_s": 30, "stride_length_s": (5, 5)},
]

for config in chunk_configs:
    start = time.time()
    result = pipe(audio_path,
                  chunk_length_s=config["chunk_length_s"],
                  stride_length_s=config["stride_length_s"])
    duration = time.time() - start

    print(f"\n{config['name']}:")
    print(f"  Time: {duration:.2f}s")
    print(f"  Text: {result['text'][:100]}...")

# Experiment 3: Sampling vs Deterministic
print("\n\n📊 Experiment 3: Sampling vs Deterministic")
print("-" * 70)

# Run deterministic multiple times (should be identical)
print("\nDeterministic (do_sample=False) - running 3 times:")
for i in range(3):
    result = pipe(audio_path, do_sample=False, max_new_tokens=50)
    print(f"  Run {i+1}: {result['text'][:60]}...")

# Run with sampling (may vary slightly)
print("\nWith sampling (do_sample=True, temperature=0.8) - running 3 times:")
for i in range(3):
    result = pipe(audio_path, do_sample=True, temperature=0.8, max_new_tokens=50)
    print(f"  Run {i+1}: {result['text'][:60]}...")

# Experiment 4: Temperature Effects (with sampling)
print("\n\n📊 Experiment 4: Temperature Effects")
print("-" * 70)

temps = [0.5, 1.0, 1.5]
for temp in temps:
    result = pipe(audio_path, do_sample=True, temperature=temp, max_new_tokens=50)
    print(f"\nTemperature {temp}:")
    print(f"  {result['text'][:80]}...")

print("\n" + "=" * 70)
print("✅ Experiments complete!")
print("=" * 70)
```

**Run the experiments:**

```bash
poetry run python experiment_generation.py
```

**What to observe:**

1. **Beam Search vs Greedy**:

   - Does beam search produce different (better?) text?
   - How much slower is it?
   - Is the quality improvement worth the speed cost?

1. **Chunk Length**:

   - How does chunk length affect processing time?
   - Do you notice any quality differences at chunk boundaries?
   - Which is best for your use case?

1. **Sampling vs Deterministic**:

   - Are deterministic outputs truly identical?
   - How much does sampling vary?
   - Why might you want sampling (or not)?

1. **Temperature**:

   - How does temperature affect output?
   - Does higher temperature make transcription worse for ASR?
   - When might you want temperature > 1.0?

**Additional Quick Experiments**:

```python
# Test with different audio formats
formats = ["test.wav", "test.mp3", "test.flac"]
for audio_file in formats:
    result = pipe(audio_file)
    print(f"{audio_file}: {result['text'][:50]}...")

# Test with very short audio
short_result = pipe(audio_path, max_new_tokens=10)
print(f"Truncated (10 tokens): {short_result['text']}")

# Test edge cases
edge_cases = [
    ("silent.wav", "Silent audio"),
    ("very_short.wav", "< 1 second audio"),
    ("long_podcast.mp3", "Long audio (5+ minutes)"),
]

for audio, description in edge_cases:
    try:
        result = pipe(audio)
        print(f"{description}: ✓ Success - {len(result['text'])} chars")
    except Exception as e:
        print(f"{description}: ✗ Error - {e}")
```

______________________________________________________________________

# CLASS SUMMARY

## What We Covered Today

**Lecture (20 min):**

- What ASR is and why it matters

- The three-component Tiny Audio architecture

- Course goals: train, evaluate, publish

**Workshop (40 min):**

- Set up development environment

- Ran first inference

- Explored model behavior on different audio

______________________________________________________________________

## Further Reading (Optional)

### Foundational Papers

- **HuBERT**: [Self-Supervised Speech Representation Learning](https://arxiv.org/abs/2106.07447)

- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

- **Attention**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (foundational)

### Documentation

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

- [PyTorch Tutorials](https://pytorch.org/tutorials/)

- [Wav2Vec2 Guide](https://huggingface.co/docs/transformers/model_doc/wav2vec2)

### Videos

- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) by 3Blue1Brown

- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar

[Previous: Course Overview](./0-course-overview.md) | [Next: Class 2: Audio Processing and Encoders](./2-audio-processing-and-encoders.md)

**Questions or stuck?** Open an issue on GitHub or check the discussions!
