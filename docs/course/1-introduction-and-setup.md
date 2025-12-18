# Class 1: Introduction, Architecture, and Setup

**Duration**: 1.5 hours (40 min lecture + 50 min hands-on)

**Goal**: Understand ASR systems, learn the encoder-projector-decoder architecture, and set up your development environment.

## Learning Objectives

By the end of this class, you will:

- Explain what Automatic Speech Recognition (ASR) is and why it matters
- Understand how audio is digitized and pre-processed
- Know the role of each component: encoder, projector, and decoder
- Visualize data flow from audio samples to text tokens
- Set up a working development environment
- Run inference on an audio file

______________________________________________________________________

## PART A: LECTURE (40 minutes)

### 1. Course Goals (5 min)

By the end of this 3-class course, you will:

1. **Train** your own customized ASR model
1. **Evaluate** it on standard benchmarks (Word Error Rate)
1. **Push** it to your own Hugging Face account
1. **Add** your results to the community leaderboard

This isn't just theory—you'll build a real, working model and deploy it with your name on it.

**Why Build AI Models?**

Most AI startups today are essentially wrappers on top of LLMs. Very few companies are actually building AI models. Understanding how to train models—not just use them—is a valuable and increasingly rare skill.

______________________________________________________________________

### 2. What is Automatic Speech Recognition? (5 min)

**The Problem**

Humans communicate through speech, but computers understand text. ASR bridges this gap, turning complex human speech into machine-readable text.

**Real-world applications:**

- Voice assistants (Siri, Alexa, Google Assistant)
- Transcription services (meetings, podcasts, interviews)
- Accessibility tools (live captioning)
- Voice search and commands
- Medical dictation systems

**The Challenge**

Speech recognition is difficult because it must handle two types of variability:

**Acoustic Variability** (handled by the encoder)

- Speaker differences: accents, pitch, speaking rate, gender, age
- Environmental noise: background sounds, echo, interference
- Recording quality: microphone type, compression, sample rate
- Pronunciation: casual vs. formal speech, mumbling

**Linguistic Variability** (handled by the decoder)

- Homophones: "I scream" vs. "ice cream" sound identical
- Context: "read" (present) vs. "read" (past) requires context
- Domain knowledge: technical terms, proper nouns
- Grammar & punctuation: sentence boundaries, structure

**Debugging tip**: When you see transcription errors, ask yourself: Is this an acoustic problem (the model misheard) or a linguistic problem (the model heard correctly but chose the wrong word)?

**How Modern ASR Works**

**Classic Era (1950s-2010s)**: Rule-based → Hidden Markov Models

- Rule-based: "If this audio segment looks like X, it's probably word Y"
- HMMs: Probabilistic (if you see T-H, next letter is probably E)
- Limited by manual feature engineering

**Deep Learning Era (2010s-2020)**: RNNs → Transformers

- Alexa and Google Voice pioneered large-scale deep learning ASR
- Transformers (2017) enabled parallel processing and better context
- Much better accuracy, but required massive labeled datasets

**Modern Era (2020s+)**: Self-Supervised + Multimodal

- Pre-trained encoders (Whisper, HuBERT) learn from unlabeled audio
- Transfer learning: leverage existing models instead of training from scratch
- Multimodal: connect audio models directly to language models

**This is what Tiny Audio uses!** We combine:

- A pre-trained **audio encoder** (Whisper)
- A pre-trained **language model** (SmolLM3)
- A small, trainable **projector** to connect them

______________________________________________________________________

### 3. The Architecture (15 min)

**High-Level Overview**

```
[Audio Wave] → [Pre-processing] → [Encoder] → [Projector] → [Decoder] → [Text]
     ↓              ↓                ↓            ↓             ↓
  Raw audio    Spectrogram      Embeddings   Translated    Transcription
                                            embeddings
```

- **Encoder**: "Listens" — creates numerical representations of audio
- **Projector**: "Translates" — bridges audio and text spaces
- **Decoder**: "Writes" — generates the final transcription

**Step 1: Audio Pre-processing**

Before a model can "hear" audio, we convert it to a standardized format.

**Digitization**: All audio formats (MP3, FLAC, etc.) get converted to WAV at 16kHz—16,000 samples per second, capturing all frequencies in human speech.

**Log-Mel Spectrogram**: The WAV is converted to a visual representation that's more similar to how humans hear:

- Bottom rows: Deep/bass sounds
- Middle rows: Vowel sounds (a, e, i, o, u)
- Top rows: High consonants (s, f)

This spectrogram is what the encoder actually processes.

**Step 2: The Encoder (Whisper)**

**Purpose**: Extract meaningful features from each time segment of audio.

**What it does:**

- Processes the spectrogram through a neural network
- Outputs 1280 numbers for each ~20ms of audio
- Each number corresponds to a "feature" (speech sounds, pitch, speaker characteristics)

**Why Whisper?** It trains fastest. Your model starts producing coherent transcriptions after ~45 minutes. HuBERT gives better accuracy but takes ~6 hours to see results.

**Other encoders to try:**

- **HuBERT** (Meta): Better accuracy, slower to train
- **wav2vec 2.0** (Meta): Popular alternative
- **WavLM** (Microsoft): Best architecture, less training data

**Key insight**: The encoder is audio-only. At this stage, "to," "too," and "two" all look identical—same sound, different spellings.

**Step 3: The MLP Projector**

**Purpose**: Translate between audio-space and text-space.

The encoder outputs 1280 audio dimensions. The LLM expects 2048 text dimensions. They speak different "languages." The projector translates.

**What it does:**

- Maps 1280 audio dimensions → 2048 text dimensions
- Uses convolutional downsampling (4x compression) for efficiency
- Passes through two linear layers to learn the translation
- Learns to solve problems sound alone can't answer

**What the projector learns:**

- "to" vs "too" vs "two" — same sound, different spelling
- "their" vs "there" vs "they're" — context determines spelling
- Question marks — rising tone → add "?"
- Capitalization — "apple" (fruit) vs "Apple" (company)

**Architecture**: Multi-Layer Perceptron (MLP) - convolutional downsampling followed by two linear layers and a GELU activation.

**Key insight**: This is the **only component we train**. The encoder and decoder stay frozen.

**Step 4: The Decoder (SmolLM3)**

**Purpose**: Generate coherent text from the translated embeddings.

**What it does:**

- Receives embeddings from the projector
- Uses linguistic knowledge to predict text, one token at a time
- Handles spelling, punctuation, sentence structure

**Why SmolLM3?** Completely open source—weights, training data, and recipes are all public. Not the best model (Qwen 4B would be better), but I like supporting openness.

**Why 3B parameters?**

- Trains faster (smaller model = more samples fit in GPU memory)
- Runs on smaller GPUs
- Actually performs surprisingly well

**Interesting fact**: LLMs can transcribe audio even though they weren't trained for it! The projector sends high-dimensional representations (not text), and the LLM figures out how to decode them.

______________________________________________________________________

### 4. Why This Architecture? (5 min)

**The Team Analogy**

Think of it as three specialists:

1. **The Listener (Encoder)**: Expert who listens to audio and writes detailed notes
1. **The Translator (Projector)**: Translates notes into the writer's language — *this is who we hire and train*
1. **The Writer (Decoder)**: Author who turns translated notes into fluent text

**Efficiency**

We only train the MLP projector instead of billions of parameters:

| Component | Parameters | Training |
|-----------|------------|----------|
| Encoder (Whisper) | ~809M | Frozen |
| MLP Projector | trainable | **Trained** |
| Decoder (SmolLM3) | ~3B | Frozen |

**Training is fast and cheap:**

- ~24 hours on a single A40 GPU
- ~$8-12 for a full training run

**Beyond Transcription**

Tiny Audio is actually a **multitask audio model**. Same architecture can do:

- **Transcribe**: Audio → text (what we'll train)
- **Describe**: Audio → "A person speaking in a quiet room"
- **Emotion**: Audio → emotional characteristics

______________________________________________________________________

## PART B: HANDS-ON WORKSHOP (50 minutes)

You will:

1. **Exercise 1**: Set up your environment (15 min)
1. **Exercise 2**: Run inference and evaluation (15 min)
1. **Exercise 3**: Visualize data flow (20 min)

**Note**: This is potentially the hardest part of the course—getting a Python project with many dependencies running locally. Once the demo works, everything else is straightforward.

______________________________________________________________________

## Exercise 1: Environment Setup (15 min)

### Goal

Get a working development environment with all dependencies installed.

### Instructions

**Step 1: Create required accounts**

| Account | URL | Purpose |
|---------|-----|---------|
| GitHub | [github.com/signup](https://github.com/signup) | Code |
| Hugging Face | [huggingface.co/join](https://huggingface.co/join) | Models |
| Weights & Biases | [wandb.ai/signup](https://wandb.ai/signup) | Training monitoring |

For Hugging Face, create an access token (Settings → Access Tokens) with "read" permissions.

**Note**: W&B is only needed for training (Class 2). Skip it for now if you just want to run inference.

**Step 2: Check prerequisites**

Open your terminal and verify Python and git are installed:

```bash
python --version  # Should be 3.10+
git --version     # Should show a version
```

**If git is not found (common on Mac):**

```bash
xcode-select --install
```

**Step 3: Install Homebrew (macOS only)**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 4: Clone the repository**

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
```

**Tip**: Fork the repo first if you want to submit changes.

**Step 5: Install Poetry**

```bash
# macOS
brew install poetry

# Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Verify: `poetry --version`

**Step 6: Install dependencies**

```bash
poetry install
```

This takes 5-10 minutes.

**Troubleshooting:**

- SSL errors: `poetry config certificates.default.cert false`
- PyTorch issues: See [pytorch.org](https://pytorch.org)

**Step 7: Verify installation**

```bash
poetry run python scripts/download_samples.py  # Optional, downloads test audio
poetry run python -c "import torch; print(f'PyTorch {torch.__version__}')"
poetry run python -c "from transformers import pipeline; print('Transformers OK')"
```

**Don't worry** about matplotlib or librosa warnings—torch-codec handles audio processing.

### Success Checkpoint

- [ ] Accounts created (GitHub, Hugging Face, optionally W&B)
- [ ] Repository cloned
- [ ] Poetry installed and dependencies installed
- [ ] Python imports work

______________________________________________________________________

## Exercise 2: Run Inference & Evaluation (15 min)

### Goal

Run the pre-trained model and get quantitative performance metrics.

### Instructions

**Step 1: Launch the demo**

Every Python script uses `poetry run` to access the correct environment:

```bash
poetry run python demo/gradio/app.py --model mazesmazes/tiny-audio --port 7860
```

This downloads the model (~4GB first time) and starts a web server.

**Step 2: Try the demo**

Open [http://localhost:7860](http://localhost:7860). The demo has two modes:

- **Transcribe**: Upload audio → get transcription
- **Text**: Pure LLM output (test the language model without audio)

Try:

- Recording yourself with the microphone
- Uploading WAV/MP3 files
- Long files automatically chunk into 30-second segments

**Note**: Expect ~6-12% Word Error Rate (88-94% accuracy).

**Step 3: Run evaluation**

Stop the demo (Ctrl+C), then run:

```bash
poetry run eval mazesmazes/tiny-audio --dataset loquacious --max-samples 100
```

**What this does:**

- Streams audio from Hugging Face (no huge downloads)
- Runs inference on 100 samples (seed 42 for reproducibility)
- Normalizes text (Whisper normalizer: handles capitalization, numbers, contractions)
- Computes Word Error Rate

**Output:**

```
Sample 1: WER = 8.33%, Time = 1.23s
  Ref:  The quick brown fox jumps over the lazy dog
  Pred: The quick brown fox jumps over the lazy dog
...
================================================================================
CHECKPOINT @ 100 samples:
  Corpus WER: 12.45%
  Avg Time/Sample: 1.35s
================================================================================
```

**Tip**: Always use `--max-samples`. Full datasets have 10,000+ samples and take hours.

### Success Checkpoint

- [ ] Demo launched and working
- [ ] Successfully transcribed audio
- [ ] Ran evaluation script
- [ ] Understand WER output

______________________________________________________________________

## Exercise 3: Visualize Data Flow (20 min)

### Goal

See exactly how audio transforms at each stage of the pipeline.

### Instructions

**Step 1: Update code (if needed)**

```bash
git pull origin main
```

**Step 2: Run visualization**

```bash
poetry run python docs/course/examples/trace_data.py
```

**Step 3: Open the output**

Open `data_trace.html` in your browser. You'll see:

**1. Waveform**

Raw audio—amplitude over time.

**2. Spectrogram**

Frequency heatmap:

- Bottom = bass
- Middle = vowels
- Top = consonants (s, f)
- Purple area = padding (encoders use fixed-length input)

**3. Encoder Output**

What Whisper "understands":

- X-axis = time steps (~20ms each)
- Y-axis = 64 of 1280 feature dimensions (sorted by activation)
- Brighter = higher activation
- Top ~10 features do most of the work

**4. Projector Output**

Translation from audio-space to text-space:

- All dimensions contributing = well-trained projector
- "Nearest tokens" section shows what text tokens each time step maps to
- Mostly gibberish, but related to the speech—the LLM decodes the rest

**Key insight**: The projector output is NOT the transcription. It's a high-dimensional representation that the LLM interprets.

### Optional: Try Different Encoders

Edit `src/asr_config.py` and change the encoder:

- `"openai/whisper-tiny"` — Smaller, faster
- `"facebook/hubert-xlarge-ft-ls960"` — Different architecture
- `"microsoft/wavlm-large"` — Best architecture

Re-run the visualization to see differences.

### Success Checkpoint

- [ ] Generated visualization HTML
- [ ] Understand each stage (waveform → spectrogram → encoder → projector)
- [ ] Can identify what the projector does

______________________________________________________________________

## Key Takeaways

1. **Architecture**: Encoder (listens) → MLP Projector (translates) → Decoder (writes)

1. **Efficiency**: Only train the MLP projector, freeze everything else

1. **Cost**: ~24 hours, ~$8-12 for a full training run

1. **Modality gap**: The projector bridges audio and text representations

1. **You're ready**: Environment set up, can run inference and evaluation

______________________________________________________________________

## Further Reading

### Papers

- [Whisper](https://arxiv.org/abs/2212.04356) — Robust Speech Recognition via Large-Scale Weak Supervision
- [HuBERT](https://arxiv.org/abs/2106.07447) — Self-Supervised Speech Representation Learning
- [WavLM](https://arxiv.org/abs/2110.13900) — Large-Scale Self-Supervised Pre-Training
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The Transformer paper

### Tutorials

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

______________________________________________________________________

[← Course Overview](./0-course-overview.md) | [Class 2: Training →](./2-training.md)
