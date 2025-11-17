# Class 2: Audio Processing and Encoders

**Duration**: 1 hour (20 min lecture + 40 min hands-on)

**Goal**: Understand how audio becomes data and how HuBERT processes it

## Learning Objectives

By the end of this class, you will:

- Understand how audio is digitized (sampling, bit depth)

- Know what feature extraction does

- Understand HuBERT's self-supervised pre-training

- Visualize audio waveforms and embeddings

- Explore the encoder's outputs hands-on

______________________________________________________________________

# PART A: LECTURE (20 minutes)

## 1. The Audio Processing Pipeline (5 min)

Today we're answering a fundamental question: **How does a computer "hear" and understand audio?**

The journey from sound waves to meaningful embeddings involves three critical steps:

1. **Digitization**: Converting continuous sound waves into discrete numbers
1. **Normalization**: Cleaning and standardizing the audio data
1. **Encoding**: Transforming raw audio into rich, semantic representations

Think of it like preparing ingredients for cooking:

- **Raw ingredients** (sound waves) → **Cleaned and prepped** (normalized audio) → **Cooked dish** (embeddings)

Each step is crucial. Skip normalization, and your model learns from noisy, inconsistent data. Use a weak encoder, and you lose the semantic richness needed for accurate transcription.

**The goal**: Turn messy, real-world audio into clean, dense representations that a language model can transcribe.

______________________________________________________________________

## 2. From Sound Waves to Numbers (5 min)

### What is Sound?

Sound = vibrations traveling through air as pressure waves

**Key Properties:**

- **Amplitude**: How loud (wave height)

- **Frequency**: Pitch (oscillation speed)

- **Duration**: How long

### Digitizing Audio

Computers need numbers, not waves. We use **sampling**:

**Sampling Rate**: Measurements per second

- CD quality: 44,100 Hz

- Phone: 8,000 Hz

- **Tiny Audio: 16,000 Hz** ← Perfect for speech!

**Why 16 kHz?**

- Captures human speech range (85-255 Hz fundamental + harmonics to ~8 kHz)

- Balances quality vs efficiency

- Industry standard for ASR

**Example:**

- 1 second of audio at 16kHz = 16,000 numbers

- 3 second audio file = 48,000 numbers!

**Quick Experiment**: Later we'll test how different sample rates affect quality:

- Downsample to 8kHz (telephone quality)

- Upsample to 44.1kHz (CD quality)

- Compare transcription accuracy

______________________________________________________________________

## 3. Feature Extraction: Preparing Audio for Models (5 min)

### The Challenge

Raw waveforms create three problems for training:

1. **Inconsistent scale**: One file's amplitude ranges [-0.1, 0.1], another's [-1.0, 1.0]
1. **Varying sample rates**: 8kHz phone audio mixed with 44.1kHz CD quality
1. **Different lengths**: 2-second clips vs 10-minute recordings in the same batch

Without preprocessing, the model wastes capacity learning to handle these variations instead of learning to transcribe speech.

### The Solution: Wav2Vec2FeatureExtractor

This preprocessing pipeline standardizes all audio:

**Step 1: Resampling**

- Convert any sample rate → 16 kHz
- Ensures consistent time resolution

**Step 2: Z-Normalization**

```python
normalized = (audio - mean) / std
```

- Centers audio around 0 (zero mean)
- Scales to unit variance (std ≈ 1)
- Makes all files comparable in amplitude

**Step 3: Padding & Batching**

Training is much faster when we process multiple audio files simultaneously (batching). But GPUs require all items in a batch to be the same length.

*The problem*: Audio files have different durations

- File A: 2.5 seconds = 40,000 samples
- File B: 1.3 seconds = 20,800 samples
- File C: 3.2 seconds = 51,200 samples

*The solution*: Padding

- Find the longest file in the batch (File C: 51,200 samples)
- Pad shorter files with zeros to match
- File A: [audio data... + 11,200 zeros]
- File B: [audio data... + 30,400 zeros]
- File C: [audio data... (no padding needed)]

Now all three files are 51,200 samples long and can be processed in parallel as a tensor with shape `[3, 51200]`.

The model learns to ignore padded regions using an **attention mask** that marks which values are real audio vs padding.

**Why this matters**: Batching improves training speed by ~10-50x compared to processing files one at a time.

**Step 4: Tensor Conversion**

PyTorch (our deep learning framework) operates on **tensors**, not NumPy arrays.

*What's a tensor?*

- A multi-dimensional array optimized for GPU computation
- Similar to NumPy arrays but with automatic differentiation (gradients)
- Can be moved to GPU for massive parallelization

*The conversion*:

```python
# Before: NumPy array (CPU-only)
audio_numpy = np.array([0.1, 0.2, 0.3, ...])  # shape: (48000,)

# After: PyTorch tensor (CPU or GPU)
audio_tensor = torch.tensor([0.1, 0.2, 0.3, ...])  # shape: torch.Size([48000])

# Can move to GPU for fast processing
audio_tensor = audio_tensor.to('cuda')  # Now on GPU
```

*Why this matters*:

- **Speed**: GPU operations are 10-100x faster than CPU
- **Gradients**: Automatic computation of derivatives for training
- **Compatibility**: All PyTorch models expect tensor inputs

After this step, our audio is ready to be consumed by the HuBERT model.

**Why this matters**: Normalization reduces the model's learning burden. Instead of learning "loud audio = this, quiet audio = that," it focuses on the actual speech patterns.

**Result**: Clean, consistent, model-ready audio!

______________________________________________________________________

## 4. The HuBERT Encoder: From Audio to Meaning (10 min)

### What is HuBERT?

**HuBERT** = **H**idden **U**nit **BERT**

HuBERT solves a critical problem: **How do you learn rich audio representations without expensive manual transcriptions?**

**The Innovation**: Self-supervised learning on unlabeled audio

Tiny Audio uses HuBERT-XLarge by default - pre-trained on 60,000 hours of unlabeled speech from LibriLight. That's nearly 7 years of continuous audio! During pre-training, it learned to predict masked audio segments, similar to how BERT predicts masked words in text.

**What makes this powerful**:

- No transcriptions needed (unlabeled data is abundant)
- Learns universal speech patterns (phonemes, prosody, speaker characteristics)
- Transfers to any language or domain
- Can be used frozen without any fine-tuning

**Why we use HuBERT as default**:

- Excellent performance on English speech
- Proven track record in ASR tasks
- Works well when frozen (no training needed)
- 1.3 billion parameters

**Alternative: Whisper Encoder**:
Tiny Audio also supports OpenAI's Whisper encoder (openai/whisper-large-v3):

- Pre-trained on 680K hours of multilingual, weakly-supervised data
- 1.5 billion parameters
- Better for multilingual tasks
- Can be swapped by changing the encoder config

**Analogy**: Like a child learning language by listening for years before speaking. They internalize patterns, rhythms, and sounds without explicit grammar lessons.

### Architecture

```
Audio waveform (16 kHz)
    ↓
CNN Feature Encoder (7 conv layers)
    ↓ (~320x time compression)
Audio features (~50 Hz)
    ↓
24 Transformer Layers
    ↓
1280-dim embeddings per frame


```

### What HuBERT Learned

During 60,000 hours of self-supervised pre-training, HuBERT developed internal representations of:

**Phonetic Knowledge**:

- Phonemes (/t/, /d/, /k/, etc.) - the atomic units of speech
- Phoneme boundaries and transitions
- Contextual pronunciation variations

**Acoustic Understanding**:

- Speaker characteristics (gender, age, accent)
- Environmental acoustics (room reverb, background noise)
- Channel effects (microphone quality, compression)

**Prosodic Patterns**:

- Rhythm and timing
- Stress and emphasis
- Intonation and pitch patterns

**Why this matters**: Training from scratch would require labeled data for all these patterns. HuBERT learned them "for free" from unlabeled audio, saving us millions of dollars and months of annotation work.

**This is why pre-trained encoders are game-changers** - we inherit this knowledge and focus our training budget on the transcription task.

### Time Compression: From Samples to Semantics

**What is time compression?**

Time compression is the process of reducing the temporal resolution (number of time steps) while preserving or even enhancing the information content. Think of it like this:

- **Before compression**: Every 1/16000th of a second gets its own data point (raw samples)
- **After compression**: Every ~20 milliseconds gets a rich summary vector (embeddings)

Instead of storing raw air pressure measurements 16,000 times per second, we create 50 summary vectors per second that capture the *meaning* of what was said during those time windows.

**Why compress?**

1. **Computational efficiency**: Language models can't process 16,000 tokens per second of audio
1. **Semantic grouping**: ~20ms is roughly one phoneme - the right granularity for speech
1. **Information density**: Embeddings encode patterns, not just raw amplitudes

HuBERT performs dramatic temporal compression while increasing semantic density:

```
3 seconds audio at 16kHz = 48,000 samples (just amplitude values)
    ↓ (CNN Feature Encoder: 7 conv layers)
~149 frame features
    ↓ (24 Transformer Layers)
~149 embeddings × 1280 dimensions (rich semantic vectors)
```

**The transformation**:

- **Input**: 48,000 numbers representing air pressure over time
- **Output**: 149 vectors, each capturing ~20ms of speech meaning
- **Compression ratio**: ~320x in time dimension
- **Information density**: ↑↑↑ (much more meaningful)

**Why compression matters**:

1. **Efficiency**: Decoder processes 149 frames instead of 48,000 samples
1. **Context**: Each frame summarizes 20ms of audio context
1. **Semantics**: Embeddings encode meaning, not just waveform shape

**Think of it this way**: Instead of describing every brush stroke in a painting (raw samples), we describe what the painting depicts (embeddings). Fewer words, more meaning.

### Why We Keep the Encoder Frozen

**The Challenge**: HuBERT has 1.3 billion parameters. Full fine-tuning would:

- Require massive GPU memory (40GB+)
- Take weeks to train
- Cost hundreds of dollars
- Risk destroying the pre-trained knowledge (catastrophic forgetting)

**The Solution**: Frozen Encoder with Trainable Projector

Instead of fine-tuning the massive encoder, we keep it completely frozen and only train the projector that transforms its outputs. This approach:

1. **Preserves knowledge**: The original 1.3B parameters contain years of learned speech patterns
1. **Saves resources**: No gradient computation or optimizer states for 1.3B params
1. **Speeds training**: Dramatically faster forward and backward passes
1. **Prevents overfitting**: Can't accidentally destroy the pre-trained representations

**Why this works**:

The encoder already produces excellent speech representations from its extensive pre-training. The projector's job is simply to translate these representations into the language model's embedding space - a much simpler task that doesn't require modifying the encoder itself.

**Analogy**: Imagine HuBERT is a master translator who speaks hundreds of languages fluently. Instead of teaching them a new language from scratch, we just give them a simple translation guide (the projector) to convert their understanding into the specific format our language model expects.

______________________________________________________________________

# PART B: HANDS-ON WORKSHOP (40 minutes)

>

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1 (20 min)**: Explore and visualize audio embeddings from HuBERT
- **Exercise 2 (20 min)**: Swap the encoder to Whisper and compare results

By the end, you'll understand how different encoders affect model performance and learn hands-on how to experiment with modular architectures!

______________________________________________________________________

## Workshop Exercise 1: Explore Audio Embeddings (20 min)

### Goal

Understand what audio encoders actually output by visualizing and analyzing embeddings.

### Your Task

Extract and visualize embeddings from HuBERT to see how audio is represented.

### Instructions

**Step 1: Extract embeddings from an audio file**

Create `explore_embeddings.py`:

```python
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import numpy as np

# Load audio (using sample from Exercise 1)
from datasets import load_dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio = dataset[0]["audio"]

# Load HuBERT model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-xlarge-ls960-ft")

# Extract features
inputs = feature_extractor(audio["array"], sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print(f"Audio duration: {len(audio['array']) / 16000:.2f} seconds")
print(f"Embedding shape: {embeddings.shape}")
print(f"  Batch size: {embeddings.shape[0]}")
print(f"  Time steps: {embeddings.shape[1]}")
print(f"  Embedding dim: {embeddings.shape[2]}")
print(f"\nEmbedding statistics:")
print(f"  Mean: {embeddings.mean():.4f}")
print(f"  Std: {embeddings.std():.4f}")
print(f"  Min: {embeddings.min():.4f}")
print(f"  Max: {embeddings.max():.4f}")
```

Run it:

```bash
poetry run python explore_embeddings.py
```

**Expected output:**

```
Audio duration: 5.56 seconds
Embedding shape: torch.Size([1, 174, 1280])
  Batch size: 1
  Time steps: 174
  Embedding dim: 1280

Embedding statistics:
  Mean: 0.0234
  Std: 0.8765
  Min: -3.4567
  Max: 4.1234
```

**Key insight**: ~5 seconds of audio → 174 time steps → ~32ms per frame!

**Step 2: Visualize embeddings (optional)**

If you have matplotlib installed, add this to visualize:

```python
import matplotlib.pyplot as plt

# Visualize first 50 dimensions over time
plt.figure(figsize=(12, 6))
plt.imshow(embeddings[0, :, :50].T, aspect='auto', cmap='viridis')
plt.colorbar(label='Activation')
plt.xlabel('Time steps')
plt.ylabel('Embedding dimensions (first 50)')
plt.title('HuBERT Audio Embeddings')
plt.tight_layout()
plt.savefig('embeddings_heatmap.png')
print("\nSaved visualization to embeddings_heatmap.png")
```

### Success Checkpoint

- [ ] Successfully extracted embeddings
- [ ] Understand shape: [batch, time_steps, 1280]
- [ ] See the temporal structure (174 frames from 5.6s audio)
- [ ] (Optional) Visualized embedding patterns

______________________________________________________________________

## Workshop Exercise 2: Swap the Encoder (20 min)

### Goal

Experiment with swapping encoders by comparing HuBERT vs Whisper performance.

### Why This Matters

Different encoders have different strengths:

- **HuBERT**: Self-supervised on 60K hours, excellent for English
- **Whisper**: Trained on 680K hours multilingual data, better for multiple languages

### Your Task

Compare inference with Whisper encoder to see the difference.

### Instructions

**Step 1: Understand the current encoder configuration**

Look at the current model config:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)

print("Current encoder:", config.audio_model_id)
print("Encoder output dim:", config.encoder_dim)
print("Downsampling rate:", config.audio_downsample_rate)
```

**Step 2: Create a new config with Whisper encoder**

Create `swap_encoder.py`:

```python
from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel
import torch

# Load base config
base_config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)

print("="*60)
print("ORIGINAL CONFIGURATION")
print("="*60)
print(f"Encoder: {base_config.audio_model_id}")
print(f"Encoder dim: {base_config.encoder_dim}")
print(f"Decoder: {base_config.text_model_id}")
print(f"LLM dim: {base_config.llm_dim}")
print(f"Downsampling: {base_config.audio_downsample_rate}x")

# Create new config with Whisper encoder
new_config = ASRConfig(
    audio_model_id="openai/whisper-large-v3",  # Swap to Whisper
    encoder_dim=1280,  # Whisper-large outputs 1280-dim embeddings
    text_model_id=base_config.text_model_id,  # Keep same decoder
    llm_dim=base_config.llm_dim,
    audio_downsample_rate=5,  # Keep same downsampling
    system_prompt=base_config.system_prompt,
    max_new_tokens=base_config.max_new_tokens,
)

print("\n" + "="*60)
print("NEW CONFIGURATION (with Whisper)")
print("="*60)
print(f"Encoder: {new_config.audio_model_id}")
print(f"Encoder dim: {new_config.encoder_dim}")
print(f"Decoder: {new_config.text_model_id}")
print(f"LLM dim: {new_config.llm_dim}")
print(f"Downsampling: {new_config.audio_downsample_rate}x")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"HuBERT: Pre-trained on 60K hours (LibriLight)")
print(f"Whisper: Pre-trained on 680K hours (multilingual, weakly-supervised)")
print(f"\nBoth output 1280-dimensional embeddings ✓")
print(f"Drop-in replacement possible!")
```

**Step 3: Understanding the implications**

When you swap encoders, consider:

**What stays the same:**

- Projector architecture (it just transforms 1280-dim → 2048-dim)
- Decoder (Qwen3-8B or SmolLM3-3B + LoRA)
- Training procedure

**What changes:**

- Audio representations (different "listening" capabilities)
- Multilingual support (Whisper handles 100+ languages)
- Pre-training domain (Whisper saw more diverse data)

**Trade-offs:**
| Aspect | HuBERT | Whisper |
|--------|---------|----------|
| Training data | 60K hours (English) | 680K hours (multilingual) |
| Languages | Primarily English | 100+ languages |
| Model size | 1.3B params | 1.5B params (large-v3) |
| Speed | Fast | Slightly slower |
| Domain | General speech | Diverse (YouTube, podcasts, etc.) |

**Step 4: Test the swap (conceptual)**

To actually train with Whisper, you would:

```bash
# Create a new experiment config: configs/hydra/experiments/whisper_encoder.yaml
model:
  audio_model_id: "openai/whisper-large-v3"
  encoder_dim: 1280

# Train with the new encoder
poetry run python src/train.py +experiments=whisper_encoder
```

**What to expect:**

- **Initialization**: Projector reinitialized (encoder dim matches)
- **Training**: Encoder LoRA adapts Whisper instead of HuBERT
- **Performance**: May be better on multilingual data, similar on English

### Discussion Questions

1. **When would you choose Whisper over HuBERT?**

   - Multilingual ASR required
   - Training data matches Whisper's domain (YouTube, podcasts)
   - Want stronger baseline (more pre-training data)

1. **What if encoder dimensions don't match?**

   - Projector input dimension must match encoder output
   - Would need to adjust `encoder_dim` in config
   - Example: Wav2Vec2-base outputs 768-dim (not 1280-dim)

1. **Can you mix and match any encoder/decoder?**

   - Yes! As long as dimensions are compatible
   - Projector bridges the gap
   - This is the power of modular architecture

### Key Insight

The beauty of the encoder-projector-decoder architecture is **modularity**. You can:

- Swap encoders (HuBERT → Whisper → Wav2Vec2)
- Swap decoders (Qwen → Llama → Mistral)
- Adjust projector (SwiGLU → simple MLP)

Each component is independent. Experiment freely!

______________________________________________________________________

# CLASS SUMMARY

## What We Covered Today

**Lecture (20 min):**

- The audio processing pipeline (digitization, normalization, encoding)

- How sound becomes numbers (sampling at 16kHz)

- Feature extraction and preprocessing (z-normalization, padding, tensors)

- HuBERT architecture and self-supervised pre-training

- LoRA adaptation for efficient fine-tuning

**Workshop (40 min):**

- Hands-on encoder swapping (HuBERT → Whisper)

- Understanding modular architecture and component trade-offs

______________________________________________________________________

## Further Reading (Optional)

### Papers

- [HuBERT: Self-Supervised Speech Representation Learning](https://arxiv.org/abs/2106.07447)

- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477)

### Tutorials

- [Librosa documentation](https://librosa.org/doc/main/index.html)

- [Audio processing basics](https://pytorch.org/audio/stable/tutorials/audio_preprocessing_tutorial.html)

[Previous: Class 1: Introduction and Setup](./1-introduction-and-setup.md) | [Next: Class 3: Language Models and Projectors](./3-language-models-and-projectors.md)
