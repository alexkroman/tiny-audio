# Class 2: The End-to-End ASR Architecture

**Duration**: 1 hour (20 min lecture + 40 min hands-on)

**Goal**: Understand how audio is transformed into text through the full encoder-projector-decoder pipeline.

## Learning Objectives

By the end of this class, you will:

- Understand how audio is digitized and pre-processed.
- Know the role of the Whisper audio encoder.
- Understand why a projector is needed to bridge the "modality gap".
- Know the role of the language model decoder (e.g., SmolLM3).
- Visualize the entire data flow from audio samples to text tokens.

______________________________________________________________________

# PART A: LECTURE (20 minutes)

## 1. The Full ASR Pipeline

Today we're looking at the entire journey from a sound wave to a final text transcription. The core of our ASR system is an **encoder-projector-decoder** architecture.

Here's a high-level overview:

```
[Audio Wave] -> [Pre-processing] -> [Whisper Encoder] -> [Projector] -> [SmolLM3 Decoder] -> [Text]
```

- **Encoder**: "Listens" to the audio and creates a rich numerical representation.
- **Decoder**: A powerful language model that writes the transcription.
- **Projector**: A small but crucial "translator" that connects the encoder and decoder.

Let's look at each step.

______________________________________________________________________

## 2. Step 1: Audio Pre-processing

Before a model can "hear" audio, we must convert it into a standardized numerical format.

### Digitization

Speech starts as vibrations in the air. A microphone captures these vibrations **16,000 times per second** (16 kHz sampling rate). This is the standard for speech recognition because it captures all the frequencies in human speech.

### Feature Extraction

We convert the audio into a **log-mel spectrogram** - think of it like a musical score that shows different frequencies over time. Like a heat map, it shows:
- **Bottom rows**: Deep voice tones
- **Middle rows**: Vowel sounds (a, e, i, o, u)
- **Top rows**: High consonants like 's' and 'f'

This spectrogram is what the encoder actually sees.

______________________________________________________________________

## 3. Step 2: The Whisper Audio Encoder

The **encoder's** job is to "listen" to the spectrogram and understand what was said.

- **What it is**: We use the **OpenAI Whisper encoder**, a massive, pre-trained model that learned from 680,000 hours of diverse audio.
- **What it detects**: Each moment in the audio gets analyzed for:
  - Individual speech sounds (like "t", "s", "ah")
  - Whether sounds are voiced or whispered
  - Pitch and rhythm patterns
  - Speaker characteristics (accent, gender, age)
- **Time Compression**: The encoder compresses time. For example, 3 seconds of audio (48,000 samples) becomes just ~150 vectors, making everything much faster.
- **Frozen**: We keep the encoder **frozen** (don't train it). This preserves its powerful knowledge and saves massive computational resources.

At this stage, we have vectors (1280 dimensions each) that capture what was heard. **Important**: "to," "too," and "two" all look the same because they sound the same! The encoder knows the sound but not yet the spelling.

______________________________________________________________________

## 4. Step 3: The Projector - Bridging the Modality Gap

Now we face the **modality gap**. The encoder speaks "audio language" (1280 dimensions), but the text decoder speaks "text language" (1536 dimensions for SmolLM3).

The **AudioProjector** is the bridge - it translates "what was heard" into "how to write it." It's a small neural network with two key jobs:

1. **Dimension Transformation**: Changes from 1280-dimensional audio vectors to 1536-dimensional text vectors.
2. **Temporal Downsampling**: Stacks frames together (typically by 5x), making everything faster.

The projector learns to solve problems that sound alone can't answer:
- **"to" vs "too" vs "two"** - Same sound, different spelling based on meaning
- **"their" vs "there" vs "they're"** - Context determines which one
- **Question marks** - Rising tone at the end → add "?"
- **Capitalization** - "apple" (fruit) vs "Apple" (company)

The projector is the **only major component we train from scratch**. It learns to be a perfect translator between the audio and text worlds.

______________________________________________________________________

## 5. Step 4: The Language Model Decoder

The final piece is the **decoder**, a large language model (LLM) like **SmolLM3**.

- **What it does**: Takes the translated embeddings from the projector and writes the final text transcription, one word (or sub-word piece) at a time.
- **How it works**: Given the audio and the words it has already written, it predicts the most likely next word. Like autocomplete, but for transcription.
- **Frozen**: Like the encoder, the decoder is also **frozen**. We use its powerful, pre-trained language capabilities without expensive fine-tuning.

### The Complete Picture: A Team of Specialists

Think of the architecture as a team:

1. **The Listener (Whisper Encoder)**: A world-class expert who listens to any audio and writes down detailed notes in a technical shorthand.
1. **The Translator (Projector)**: A specialist who translates the listener's technical notes into the native language of the writer. This is the team member we hire and train.
1. **The Writer (SmolLM3 Decoder)**: A master author who can take the translated notes and write a perfectly fluent and coherent sentence.

By using pre-trained, frozen specialists for listening and writing, we only need to train the translator. This makes building a powerful ASR system incredibly efficient.

______________________________________________________________________

# PART B: HANDS-ON WORKSHOP (40 minutes)

## Goal

Visually trace a single audio sample as it gets transformed by each part of the ASR pipeline, making the abstract concepts of "embeddings" and the "modality gap" tangible.

### Your Task

You will run a script that processes a single audio file step-by-step and generates a self-contained, interactive HTML report to visualize the output at each stage.

### Instructions

**Step 1: Locate the script**

The script for this workshop, `trace_data.py`, is located in the `docs/course/examples/` directory.

Take a moment to open `docs/course/examples/trace_data.py` and look through the code. You'll see it performs the following steps:
1.  Loads a sample audio file.
2.  Processes it through the feature extractor, encoder, and projector.
3.  Creates visualizations using matplotlib showing the data at each stage.
4.  Generates a self-contained HTML file (`data_trace.html`) with embedded images and clear explanations of what's happening at each step.

**Step 2: Run the script**

Execute the script from the root directory of the project:

```bash
poetry run python docs/course/examples/trace_data.py
```

### Analysis and Key Insights

After running the script, open the newly created `data_trace.html` file in your web browser.

This report shows you exactly how speech becomes text, step by step:

1.  **Sound Waves**: The starting point—ups and downs showing how loud the sound is at each moment.
2.  **Spectrogram**: A heat map showing frequencies over time. You'll see vowels in the middle rows and consonants in the top rows. The purple area on the right is just padding.
3.  **Encoder Output**: What the AI "understands" about the audio. Brighter colors mean the AI detected something important. At this stage, "to," "too," and "two" still look the same because they sound the same.
4.  **Projector Output**: Now the AI is thinking in text, not sound. You'll see:
    *   The pattern looks completely different - sound has been translated to text concepts
    *   The "nearest tokens" section shows a jumble of words, proving the projector is working
    *   It's now ready to figure out spelling, punctuation, and capitalization

**Key insight**: The projector is the bridge that makes this all work. It learns to answer questions like "which 'to/too/two' should I write?" based on context.

______________________________________________________________________

[Previous: Class 1: Introduction and Setup](./1-introduction-and-setup.md) | [Next: Class 3: Training](./3-training.md)