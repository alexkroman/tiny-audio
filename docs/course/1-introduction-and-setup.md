# Class 1: Introduction and Setup

**Duration**: 1 hour (20 min lecture + 40 min hands-on)
**Goal**: Understand ASR systems and run your first model inference

## Learning Objectives

By the end of this class, you will:

- Understand what automatic speech recognition (ASR) is and why it matters
- Know the three main components of the Tiny Audio architecture
- Have a working development environment
- Successfully run inference on an audio file

---

# PART A: LECTURE (20 minutes)

> **Instructor**: Present these concepts with opportunities for exploration and experimentation.

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
2. **Ambiguity**: "I scream" vs "ice cream" sound identical
3. **Context dependency**: Understanding requires linguistic knowledge
4. **Real-time constraints**: Users expect instant responses

### How Modern ASR Works

Modern ASR systems have evolved through several generations:

**1st Generation (1950s-1980s)**: Rule-based pattern matching

- Limited vocabulary (~100 words)
- Speaker-dependent

**2nd Generation (1980s-2010s)**: Hidden Markov Models (HMMs)

- Better accuracy
- Still struggled with noise and accents

**3rd Generation (2010s-2020s)**: Deep Learning

- Recurrent Neural Networks (RNNs)
- Attention mechanisms
- Dramatic accuracy improvements

**4th Generation (2020s-Present)**: Multimodal Transformers

- Self-supervised pre-training
- Transfer learning from language models
- **This is what Tiny Audio uses!**

---

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

### Component 2: Audio Projector (~122M parameters)

**Purpose**: Bridge the gap between the audio and language worlds.

**What it does**

- Takes HuBERT's audio embeddings
- Downsamples by 5x (reduces sequence length for efficiency)
- Transforms to match the language model's input format

**Architecture**: SwiGLU MLP (we'll dive deeper in Class 3)

- Pre-normalization layer
- Gated projection (allows selective information flow)
- Post-normalization layer

**Key insight**: This is the **largest trainable component** - all ~122M parameters learn during training.

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

**Efficiency**: We train ~139M parameters instead of 4.3+ billion (3.2% of total model)

- Projector: ~122M (fully trained)
- Encoder LoRA: ~2M (adapter weights, r=8)
- Decoder LoRA: ~15M (adapter weights, r=64)

**Speed**: Training completes in ~24 hours on a single GPU

**Cost**: ~$12 for a full training run

**Quality**: Leverages pre-trained knowledge from both audio and language domains

**A Note on Architectural Choices:**

The Tiny Audio architecture is a **dense transformer-based** model. It's crucial to start with a **proven, stable baseline**. While other exciting architectures like Mixture-of-Experts (MoE) and Hybrids (combining transformers with other architectures like SSMs) exist, they introduce complexity that isn't necessary for our goal.

- **Dense models** are well-understood, stable to train, and perform exceptionally well, especially for a focused task like ours.
- **MoE and Hybrid models** are powerful but can be more complex to train and tune. They are often used for massive, general-purpose models, but for our specific use case, a dense model is the most direct path to a high-quality, custom ASR model.

By starting with a solid, well-understood architecture, we can focus on the nuances of audio processing, data, and training, which is where we'll see the biggest improvements.

### Rules of Engagement

A disciplined, empirical approach to model training is crucial. As we go through this course, we'll follow a few key "rules of engagement":

1.  **Systematic Beats Intuitive**: Don't just guess what will work. We'll use systematic experiments (ablations) to validate our choices.
2.  **Change One Thing at a Time**: When we run experiments, we'll only change one variable at a time. This is the only way to know for sure what's responsible for any improvements (or regressions!).
3.  **Validate Every Change**: Every modification, no matter how small, should be tested. We'll rely on evaluation metrics, not just gut feelings, to guide our decisions.

Adopting this mindset will not only lead to a better final model but will also teach you the disciplined process that professionals use to build world-class models.

---

## 3. Course Goals (5 min)

By the end of this 6-class course, each student will:

1. **Train** their own customized ASR model
2. **Evaluate** it on standard benchmarks
3. **Push** it to their own HuggingFace account
4. **Add** their results to the community leaderboard

This isn't just learning - you'll have a real, working, deployed model with your name on it!

---

# PART B: HANDS-ON WORKSHOP (40 minutes)

> **Students**: Follow these instructions step-by-step on your own computer.
>
> **Instructor**: Circulate and help students who get stuck.

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Set up your environment (install dependencies, download samples)
- **Exercise 2**: Run inference with the pretrained model
- **Exercise 3**: Inspect the model configuration
- **Exercise 4**: Count trainable parameters

By the end, you'll have a working setup and understand the complete model architecture!

---

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
poetry run python -c "import torch; import transformers; print('✓ Installation successful!')"
```

You should see: `✓ Installation successful!`

### Success Checkpoint

You should see: `✓ Installation successful!`

If you hit any issues, raise your hand for help!

---

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
2. Audio file is loaded and resampled to 16kHz
3. Audio passes through: Encoder → Projector → Decoder
4. Text transcription is returned

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
2. Audio is loaded and resampled to 16kHz
3. Audio passes through: Encoder → Projector → Decoder
4. Text transcription is returned

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

### Experimentation Time!

Now let's experiment with different parameters:

**Experiment 1: Test confidence scores**
```python
# Modify your test_inference.py to show confidence
result = pipe(audio_path, return_timestamps=True)
print(f"Text: {result['text']}")
if 'chunks' in result:
    print("\nWord-level confidence:")
    for chunk in result['chunks'][:10]:  # Show first 10 words
        print(f"  {chunk['text']}: {chunk.get('confidence', 'N/A')}")
```

**Experiment 2: Try different audio formats**
- Test with WAV, MP3, FLAC files
- Try different sample rates (8kHz, 16kHz, 44.1kHz)
- Note any differences in accuracy or speed

**Experiment 3: Test edge cases**
- Very short audio (< 1 second)
- Very long audio (> 1 minute)
- Silent audio
- Multiple speakers
- Different languages

---

## Workshop Exercise 3: Explore Model Behavior & Components (10 min)

### Goal

Understand how the model performs on different types of audio and explore component interactions.

### Your Task

Test the model systematically and explore its components.

### Instructions

**Step 1: Systematic Testing**

Create a test script `systematic_test.py`:

```python
from transformers import pipeline
import time

pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True
)

# Test different scenarios
test_cases = [
    ("clear_speech.wav", "Clear speech"),
    ("noisy_audio.wav", "Noisy environment"),
    ("accented_speech.wav", "Different accent"),
    ("fast_speech.wav", "Fast speaking"),
    ("whisper.wav", "Quiet/whispered speech"),
]

results = []
for audio_path, description in test_cases:
    try:
        start = time.time()
        result = pipe(audio_path)
        duration = time.time() - start

        results.append({
            "description": description,
            "text": result["text"],
            "time": duration,
            "words": len(result["text"].split())
        })
        print(f"✓ {description}: {duration:.2f}s")
    except Exception as e:
        print(f"✗ {description}: {e}")

# Analyze results
print("\n=== Analysis ===")
for r in results:
    print(f"{r['description']}:")
    print(f"  Speed: {r['words']/r['time']:.1f} words/sec")
    print(f"  Text preview: {r['text'][:50]}...")
```

**Step 2: Component Exploration**

Explore how changes affect the pipeline:

```python
# Experiment with chunk length
result_short = pipe(audio_path, chunk_length_s=10)  # Shorter chunks
result_long = pipe(audio_path, chunk_length_s=30)   # Longer chunks

# Compare outputs
print(f"Short chunks: {result_short['text'][:100]}")
print(f"Long chunks: {result_long['text'][:100]}")
```

**Step 3: Record Your Experiments**

Create `experiment_log.md`:

```markdown
# Experiment Log

## Audio Quality Tests
- Clear speech: [accuracy/speed]
- Noisy audio: [accuracy/speed]
- Accents: [accuracy/speed]

## Parameter Experiments
- Chunk length impact:
- Batch size effect:

## Key Findings
1. Model performs best when...
2. Performance degrades with...
3. Surprising observation...
```

### Discussion Questions

- How does audio quality affect transcription accuracy?
- What's the relationship between chunk size and memory usage?
- Which component (encoder/projector/decoder) is the bottleneck?

---

## Bonus: Explore the Model (Optional)

If you finish early, try these exploration exercises:

### Model Files

When you load the model, HuggingFace downloads these files:

```
~/.cache/huggingface/hub/models--mazesmazes--tiny-audio/
├── config.json              # Model configuration
├── model.safetensors        # Trained weights (projector + LoRA)
├── preprocessor_config.json # Audio preprocessing settings
├── tokenizer_config.json    # Text tokenization settings
├── asr_modeling.py         # Model architecture code
├── asr_config.py           # Configuration class
├── asr_pipeline.py         # Pipeline integration
└── asr_processing.py       # Audio/text processing
```

### Inspecting the Config

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "mazesmazes/tiny-audio",
    trust_remote_code=True
)

print(f"Audio encoder: {config.audio_model_id}")
print(f"Language model: {config.text_model_id}")
print(f"Encoder dimension: {config.encoder_dim}")
print(f"LLM dimension: {config.llm_dim}")
print(f"Downsampling rate: {config.audio_downsample_rate}x")
```

**Output:**

```
Audio encoder: facebook/hubert-xlarge-ls960-ft
Language model: Qwen/Qwen3-8B
Encoder dimension: 1280
LLM dimension: 2048
Downsampling rate: 5x
```

---

## Bonus Exercise 1: Count Model Parameters

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "mazesmazes/tiny-audio",
    trust_remote_code=True
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# Explore structure
print("\nModel structure:")
print(model)
```

**Question**: Which components have trainable parameters?

## Bonus Exercise 2: Time Inference Speed

```python
import time
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True
)

audio_path = "your-audio.wav"

start = time.time()
result = pipe(audio_path)
end = time.time()

print(f"Transcription took {end - start:.2f} seconds")
print(f"Result: {result['text']}")
```

**Question**: How does inference time relate to audio length?

---

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

## Homework (Optional)

Before Class 2, experiment with:

1. **Audio Testing Challenge**: Test the model on 5+ different audio files
   - Try different languages
   - Test with music in the background
   - Record yourself at different distances from the microphone

2. **Component Exploration**:
   - Measure inference time for different audio lengths
   - Calculate the real-time factor (audio duration / processing time)
   - Try to make the model fail (what breaks it?)

3. **Code Exploration**:
   - Browse `src/asr_modeling.py` - don't worry if it's confusing yet!
   - Look for the three main components we discussed
   - Think: "How would I represent sound as numbers?"

4. **Experimentation Ideas**:
   - What happens with non-English audio?
   - How does the model handle singing vs. speaking?
   - Can it transcribe multiple speakers in conversation?

## Key Takeaways

✅ ASR converts audio waveforms to text using ML models
✅ Tiny Audio uses: Encoder (HuBERT) → Projector → Decoder (Qwen-3 8B)
✅ We train only ~139M params (1.5%) instead of the full 9.3B model
✅ You can run inference on any audio file and get transcriptions

## Check Your Understanding

Before moving to Class 2, make sure you can answer:

1. **What are the three main components of Tiny Audio?**
   - Audio Encoder (HuBERT)
   - Audio Projector (SwiGLU MLP)
   - Language Model Decoder (Qwen-3 8B)

2. **Why is parameter-efficient training important?**
   - Trains only ~139M parameters instead of 9.3B+
   - Faster, cheaper, accessible on consumer hardware
   - Enables $12 / 24-hour training runs

3. **What does the audio projector do?**
   - Bridges audio and text modalities
   - Downsamples by 5x for efficiency
   - Transforms encoder outputs to decoder inputs

4. **Can you successfully run inference on an audio file?**
   - Yes! You should have transcribed multiple audio files

---

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

---

## Next Class

In [Class 2: Audio Processing and Encoders](./2-audio-processing-and-encoders.md), we'll dive deep into:

- How audio becomes numbers
- Feature extraction and spectrograms
- Understanding the HuBERT encoder
- Exploring audio embeddings hands-on

**Prep for next class:**

- Ensure you can run inference successfully
- Browse `src/asr_modeling.py` (don't worry if it's confusing yet!)
- Think about: "How would you represent sound as numbers?"

---

[Previous: Course Overview](./0-course-overview.md) | [Next: Class 2: Audio Processing and Encoders](./2-audio-processing-and-encoders.md)

**Questions or stuck?** Open an issue on GitHub or check the discussions!
