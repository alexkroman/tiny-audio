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

---

## 2. Step 1: Audio Pre-processing

Before a model can "hear" audio, we must convert it into a standardized numerical format.

### Digitization
We first digitize the audio by **sampling** it 16,000 times per second (16 kHz). This rate is the standard for speech recognition as it captures the full range of human speech.

### Feature Extraction
Raw audio samples are too high-dimensional. We extract more meaningful features by converting the audio into a **log-mel spectrogram**. This is a visual representation of how the frequencies in the audio change over time, much like a musical score. This spectrogram is the actual input to our encoder.

---

## 3. Step 2: The Whisper Audio Encoder

The **encoder's** job is to take the spectrogram and create a rich, contextualized representation of the speech.

- **What it is**: We use the **OpenAI Whisper encoder**, a massive, pre-trained model that has learned the nuances of human speech from 680,000 hours of diverse audio.
- **What it outputs**: A sequence of embeddings (vectors), where each embedding represents a small chunk of audio (~20-30ms).
- **Time Compression**: The encoder significantly compresses the temporal dimension. For example, 3 seconds of audio (48,000 samples) might become just ~150 embedding vectors. This makes the downstream processing much more efficient.
- **Frozen**: We keep the encoder **frozen**. We don't train it. This preserves its powerful, pre-trained knowledge and saves massive amounts of computational resources.

At the end of this stage, we have a sequence of high-dimensional vectors (1280 dimensions) that represent the *meaning* of the audio.

---

## 4. Step 3: The Projector - Bridging the Modality Gap

Now we face the **modality gap**. The audio encoder outputs embeddings in one "language" (1280 dimensions), but the text decoder expects embeddings in a different "language" (e.g., 1536 dimensions for SmolLM3).

The **AudioProjector** is the bridge. It's a small, trainable neural network with two key jobs:

1.  **Dimension Transformation**: It projects the 1280-dimensional audio embeddings into the 1536-dimensional space the language model expects.
2.  **Temporal Downsampling**: It stacks frames together to further reduce the sequence length (typically by a factor of 5), making the decoder's job much easier and faster.

The projector is the **only major component we train from scratch**. It learns to be a perfect translator between the audio and text worlds.

---

## 5. Step 4: The Language Model Decoder

The final piece is the **decoder**, a large language model (LLM) like **SmolLM3**.

- **What it does**: It receives the sequence of translated embeddings from the projector and generates the final text transcription, one token (word or sub-word) at a time.
- **How it works**: It's an auto-regressive, decoder-only model. Given the audio representation and the words it has already generated, it predicts the most likely next word.
- **Frozen**: Like the encoder, the decoder is also **frozen**. We use its powerful, pre-trained language capabilities without the need for expensive fine-tuning.

### The Complete Picture: A Team of Specialists

Think of the architecture as a team:

1.  **The Listener (Whisper Encoder)**: A world-class expert who listens to any audio and writes down detailed notes in a technical shorthand.
2.  **The Translator (Projector)**: A specialist who translates the listener's technical notes into the native language of the writer. This is the team member we hire and train.
3.  **The Writer (SmolLM3 Decoder)**: A master author who can take the translated notes and write a perfectly fluent and coherent sentence.

By using pre-trained, frozen specialists for listening and writing, we only need to train the translator. This makes building a powerful ASR system incredibly efficient.

______________________________________________________________________

# PART B: HANDS-ON WORKSHOP (40 minutes)

## Goal

Visually trace a single audio sample as it gets transformed by each part of the ASR pipeline, making the abstract concepts of "embeddings" and the "modality gap" tangible.

### Your Task

You will process a single audio file step-by-step through the encoder and projector, visualizing the output at each stage to see how the data's shape and meaning change.

### Instructions

**Step 1: Create `trace_data.py`**

This script will load a sample audio file, pass it through the encoder and projector, and generate plots to visualize the transformations.

```python
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel

# --- 1. Load a single audio sample ---
print("Loading audio sample...")
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = dataset[0]["audio"]
# Convert to a batch of one
audio_input = torch.tensor(audio_sample["array"]).unsqueeze(0)
sampling_rate = audio_sample["sampling_rate"]
print(f"✓ Audio loaded. Shape: {audio_input.shape}, Rate: {sampling_rate} Hz")

# --- 2. Load the full ASR model ---
print("\nLoading ASR model (this may take a moment)...")
config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
model = ASRModel(config)
model.eval() # Set model to evaluation mode
print("✓ Model loaded.")

# --- 3. Process through the Encoder ---
print("\nStep 1: Passing audio through the Whisper Encoder...")
with torch.no_grad():
    # The `encode` method runs both pre-processing and the encoder
    encoder_output = model.encode(audio_input, sampling_rate)

print(f"✓ Encoder output shape: {encoder_output.shape}")
print("   - Batch size: 1")
print(f"   - Time steps: {encoder_outpu t.shape[1]} (Each step is ~30ms of audio)")
print(f"   - Embedding dimension: {encoder_output.shape[2]} (A rich representation of the audio)")

# --- 4. Process through the Projector ---
print("\nStep 2: Passing encoder embeddings through the Projector...")
with torch.no_grad():
    projector_output = model.audio_projector(encoder_output)

print(f"✓ Projector output shape: {projector_output.shape}")
print(f"   - Time steps: {projector_output.shape[1]} (Downsampled by {config.audio_downsample_rate}x)")
print(f"   - Embedding dimension: {projector_output.shape[2]} (Projected from {config.encoder_dim} to {config.llm_dim})")
print("This output is now ready for the Language Model Decoder!")

# --- 5. Visualization ---
print("\nStep 3: Generating visualizations...")
# Function to plot embeddings
def plot_embeddings(tensor, title, filename):
    plt.figure(figsize=(15, 5))
    plt.imshow(tensor.squeeze(0).T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label="Activation")
    plt.xlabel("Time Steps")
    plt.ylabel("Embedding Dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✓ Saved '{filename}'")

# Plot encoder and projector outputs
plot_embeddings(encoder_output, "Whisper Encoder Output (Audio Embeddings)", "encoder_output.png")
plot_embeddings(projector_output, "Projector Output (Text-like Embeddings)", "projector_output.png")
```

**Step 2: Run the script**

```bash
poetry run python trace_data.py
```

### Analysis and Key Insights

After running the script, you will have two image files: `encoder_output.png` and `projector_output.png`.

1.  **Open `encoder_output.png`**: This is what the audio "looks like" to the Whisper encoder. You'll see a detailed, high-dimensional representation. Notice the number of time steps on the x-axis and the 1280 dimensions on the y-axis.

2.  **Open `projector_output.png`**: This is what the audio looks like after being "translated" for the language model.
    *   **Fewer Time Steps**: Compare the x-axis to the first plot. It's much shorter, showing the effect of temporal downsampling. This makes it much more efficient for the LLM to process.
    *   **Different Dimension**: The y-axis now represents the dimensionality of the *language model* (1536), not the audio encoder.
    *   **Different Pattern**: The visual pattern of activations will be completely different, clearly showing the transformation that occurred.

This exercise makes the abstract "modality gap" visible. You've witnessed the crucial role of the projector in bridging the gap between the world of sound and the world of language.

______________________________________________________________________

## Further Reading (Optional)

### Papers

- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Whisper Paper)

[Previous: Class 1: Introduction and Setup](./1-introduction-and-setup.md) | [Next: Class 3: Training](./3-training.md)