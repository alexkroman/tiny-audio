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

We first digitize the audio by **sampling** it 16,000 times per second (16 kHz). This rate is the standard for speech recognition as it captures the full range of human speech.

### Feature Extraction

Raw audio samples are too high-dimensional. We extract more meaningful features by converting the audio into a **log-mel spectrogram**. This is a visual representation of how the frequencies in the audio change over time, much like a musical score. This spectrogram is the actual input to our encoder.

______________________________________________________________________

## 3. Step 2: The Whisper Audio Encoder

The **encoder's** job is to take the spectrogram and create a rich, contextualized representation of the speech.

- **What it is**: We use the **OpenAI Whisper encoder**, a massive, pre-trained model that has learned the nuances of human speech from 680,000 hours of diverse audio.
- **What it outputs**: A sequence of embeddings (vectors), where each embedding represents a small chunk of audio (~20-30ms).
- **Time Compression**: The encoder significantly compresses the temporal dimension. For example, 3 seconds of audio (48,000 samples) might become just ~150 embedding vectors. This makes the downstream processing much more efficient.
- **Frozen**: We keep the encoder **frozen**. We don't train it. This preserves its powerful, pre-trained knowledge and saves massive amounts of computational resources.

At the end of this stage, we have a sequence of high-dimensional vectors (1280 dimensions) that represent the *meaning* of the audio.

______________________________________________________________________

## 4. Step 3: The Projector - Bridging the Modality Gap

Now we face the **modality gap**. The audio encoder outputs embeddings in one "language" (1280 dimensions), but the text decoder expects embeddings in a different "language" (e.g., 1536 dimensions for SmolLM3).

The **AudioProjector** is the bridge. It's a small, trainable neural network with two key jobs:

1. **Dimension Transformation**: It projects the 1280-dimensional audio embeddings into the 1536-dimensional space the language model expects.
1. **Temporal Downsampling**: It stacks frames together to further reduce the sequence length (typically by a factor of 5), making the decoder's job much easier and faster.

The projector is the **only major component we train from scratch**. It learns to be a perfect translator between the audio and text worlds.

______________________________________________________________________

## 5. Step 4: The Language Model Decoder

The final piece is the **decoder**, a large language model (LLM) like **SmolLM3**.

- **What it does**: It receives the sequence of translated embeddings from the projector and generates the final text transcription, one token (word or sub-word) at a time.
- **How it works**: It's an auto-regressive, decoder-only model. Given the audio representation and the words it has already generated, it predicts the most likely next word.
- **Frozen**: Like the encoder, the decoder is also **frozen**. We use its powerful, pre-trained language capabilities without the need for expensive fine-tuning.

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
3.  Serializes the data (waveform, spectrogram, embeddings) into a JSON format.
4.  Generates a self-contained HTML file (`data_trace.html`) that uses the Observable JS runtime to create interactive visualizations of the data.

**Step 2: Run the script**

Execute the script from the root directory of the project:

```bash
poetry run python docs/course/examples/trace_data.py
```

### Analysis and Key Insights

After running the script, open the newly created `data_trace.html` file in your web browser.

This interactive report visually documents the entire journey of your audio data:

1.  **Raw Audio Waveform**: The starting point—a simple 1D representation of sound pressure over time.
2.  **Log-Mel Spectrogram**: You'll see how the audio is converted into a 2D representation that the encoder can understand, showing frequency changes over time.
3.  **Encoder Output**: This visualization shows the rich, high-dimensional embeddings produced by the Whisper encoder. Notice the shape and statistics—this is the "language" of the audio world.
4.  **Projector Output**: Here you can clearly see the "modality gap" being bridged.
    *   **Shape Change**: The dimensions of the embeddings are transformed to match what the language model expects.
    *   **Visual Difference**: The pattern of activations is completely different, showing that the projector has successfully "translated" the audio representation into a text-like one.

By reviewing this report, you've witnessed the crucial role of each component in transforming sound into a format ready for text generation.

______________________________________________________________________

[Previous: Class 1: Introduction and Setup](./1-introduction-and-setup.md) | [Next: Class 3: Training](./3-training.md)