# Class 2: The End-to-End ASR Architecture

**Duration**: 1 hour (20 min lecture + 40 min hands-on)

**Goal**: Understand how audio is transformed into text through the full encoder-projector-decoder pipeline.

## Learning Objectives

By the end of this class, you will:

- Understand how audio is digitized and pre-processed.
- Know the role of the Whisper audio encoder.
- Understand why a projector is needed to bridge the "modality gap".
- Know the role of the language model decoder (e.g., Qwen).
- Visualize the entire data flow from audio samples to text tokens.

______________________________________________________________________

# PART A: LECTURE (20 minutes)

## 1. The Full ASR Pipeline

Today we're looking at the entire journey from a sound wave to a final text transcription. The core of our ASR system is an **encoder-projector-decoder** architecture.

Here's a high-level overview:

```
[Audio Wave] -> [Pre-processing] -> [Whisper Encoder] -> [Projector] -> [Qwen Decoder] -> [Text]
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

Now we face the **modality gap**. The audio encoder outputs embeddings in one "language" (1280 dimensions), but the text decoder expects embeddings in a different "language" (e.g., 2048 dimensions for Qwen).

The **AudioProjector** is the bridge. It's a small, trainable neural network with two key jobs:

1.  **Dimension Transformation**: It projects the 1280-dimensional audio embeddings into the 2048-dimensional space the language model expects.
2.  **Temporal Downsampling**: It stacks frames together to further reduce the sequence length (typically by a factor of 5), making the decoder's job much easier and faster.

The projector is the **only major component we train from scratch**. It learns to be a perfect translator between the audio and text worlds.

---

## 5. Step 4: The Language Model Decoder

The final piece is the **decoder**, a large language model (LLM) like **Qwen**.

- **What it does**: It receives the sequence of translated embeddings from the projector and generates the final text transcription, one token (word or sub-word) at a time.
- **How it works**: It's an auto-regressive, decoder-only model. Given the audio representation and the words it has already generated, it predicts the most likely next word.
- **Frozen**: Like the encoder, the decoder is also **frozen**. We use its powerful, pre-trained language capabilities without the need for expensive fine-tuning.

### The Complete Picture: A Team of Specialists

Think of the architecture as a team:

1.  **The Listener (Whisper Encoder)**: A world-class expert who listens to any audio and writes down detailed notes in a technical shorthand.
2.  **The Translator (Projector)**: A specialist who translates the listener's technical notes into the native language of the writer. This is the team member we hire and train.
3.  **The Writer (Qwen Decoder)**: A master author who can take the translated notes and write a perfectly fluent and coherent sentence.

By using pre-trained, frozen specialists for listening and writing, we only need to train the translator. This makes building a powerful ASR system incredibly efficient.

______________________________________________________________________

# PART B: HANDS-ON WORKSHOP (40 minutes)

## Goal

Explore the complete Tiny Audio architecture through its configuration and model structure.

### Your Task

Inspect the `ASRConfig` and `ASRModel` to see how the components we discussed are defined and connected.

### Instructions

**Step 1: Create `explore_architecture.py`**

This script will load the configuration and the model from the Hugging Face Hub and print out their key components.

```python
from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel

# --- Part 1: Inspecting the Configuration ---
print("="*60)
print("PART 1: The ASRConfig - Our Architectural Blueprint")
print("="*60)

# Load the configuration from the Hub
# This object stores all the high-level architectural choices.
config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)

print(f"1. Audio Encoder ID  -> {config.audio_model_id}")
print(f"   - This is our 'Listener'. It defines which pre-trained audio model to use.")
print(f"   - Encoder output dimension: {config.encoder_dim}\n")

print(f"2. Text Decoder ID   -> {config.text_model_id}")
print(f"   - This is our 'Writer'. It defines which pre-trained language model to use.")
print(f"   - LLM input dimension: {config.llm_dim}\n")

print(f"3. Projector Config")
print(f"   - Downsample Rate: {config.audio_downsample_rate}x")
print(f"   - This defines how many audio frames are stacked by the 'Translator'.\n")


# --- Part 2: Inspecting the Model ---
print("="*60)
print("PART 2: The ASRModel - The Full Pipeline")
print("="*60)

# Initialize the model from the configuration.
# This will download the weights for the encoder and decoder if not cached.
# Note: This may take a moment and requires significant memory.
model = ASRModel(config)

print("ASRModel class contains the three main components:\n")
print(f"1. model.audio_encoder:\n   {model.audio_encoder.__class__.__name__}\n")
print(f"2. model.audio_projector:\n   {model.audio_projector}\n")
print(f"3. model.text_decoder:\n   {model.text_decoder.__class__.__name__}\n")

print("This single `ASRModel` object orchestrates the entire pipeline from audio to text.")
```

**Step 2: Run the script**

```bash
poetry run python explore_architecture.py
```

### Expected Output

You will see a printout that clearly maps the concepts from the lecture to the actual configuration and model components.

```
============================================================
PART 1: The ASRConfig - Our Architectural Blueprint
============================================================
1. Audio Encoder ID  -> openai/whisper-large-v3
   - This is our 'Listener'. It defines which pre-trained audio model to use.
   - Encoder output dimension: 1280

2. Text Decoder ID   -> Qwen/Qwen2-7B-Instruct
   - This is our 'Writer'. It defines which pre-trained language model to use.
   - LLM input dimension: 3584

3. Projector Config
   - Downsample Rate: 5x
   - This defines how many audio frames are stacked by the 'Translator'.

============================================================
PART 2: The ASRModel - The Full Pipeline
============================================================
ASRModel class contains the three main components:

1. model.audio_encoder:
   WhisperModel

2. model.audio_projector:
   AudioProjector(
     (ln_pre): RMSNorm()
     (proj): Linear(in_features=6400, out_features=3584, bias=True)
     (ln_post): RMSNorm()
   )

3. model.text_decoder:
   Qwen2ForCausalLM

This single `ASRModel` object orchestrates the entire pipeline from audio to text.
```

### Key Insight

The beauty of this architecture is its **modularity**. You can easily swap out the encoder or decoder just by changing the model ID in the configuration. The projector is the flexible glue that allows these powerful, pre-existing components to work together on a new task.

______________________________________________________________________

## Further Reading (Optional)

### Papers

- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Whisper Paper)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)

[Previous: Class 1: Introduction and Setup](./1-introduction-and-setup.md) | [Next: Class 3: Training](./4-training.md)
