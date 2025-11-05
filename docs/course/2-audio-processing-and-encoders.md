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

---

# PART A: LECTURE (20 minutes)

## 1. The Audio Processing Pipeline (5 min)

Today we're answering a fundamental question: **How does a computer "hear" and understand audio?**

The journey from sound waves to meaningful embeddings involves three critical steps:

1. **Digitization**: Converting continuous sound waves into discrete numbers
2. **Normalization**: Cleaning and standardizing the audio data
3. **Encoding**: Transforming raw audio into rich, semantic representations

Think of it like preparing ingredients for cooking:
- **Raw ingredients** (sound waves) → **Cleaned and prepped** (normalized audio) → **Cooked dish** (embeddings)

Each step is crucial. Skip normalization, and your model learns from noisy, inconsistent data. Use a weak encoder, and you lose the semantic richness needed for accurate transcription.

**The goal**: Turn messy, real-world audio into clean, dense representations that a language model can transcribe.

---

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

---

## 3. Feature Extraction: Preparing Audio for Models (5 min)

### The Challenge

Raw waveforms create three problems for training:

1. **Inconsistent scale**: One file's amplitude ranges [-0.1, 0.1], another's [-1.0, 1.0]
2. **Varying sample rates**: 8kHz phone audio mixed with 44.1kHz CD quality
3. **Different lengths**: 2-second clips vs 10-minute recordings in the same batch

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

---

## 4. The HuBERT Encoder: From Audio to Meaning (10 min)


### What is HuBERT?

**HuBERT** = **H**idden **U**nit **BERT**

HuBERT solves a critical problem: **How do you learn rich audio representations without expensive manual transcriptions?**

**The Innovation**: Self-supervised learning on unlabeled audio

HuBERT was pre-trained on 60,000 hours of unlabeled speech from LibriLight - that's nearly 7 years of continuous audio! During pre-training, it learned to predict masked audio segments, similar to how BERT predicts masked words in text.

**What makes this powerful**:
- No transcriptions needed (unlabeled data is abundant)
- Learns universal speech patterns (phonemes, prosody, speaker characteristics)
- Transfers to any language or domain

**Analogy**: Like a child learning language by listening for years before speaking. They internalize patterns, rhythms, and sounds without explicit grammar lessons.

**Why we use pre-trained HuBERT**: It already "understands" speech. We just need to fine-tune it for our specific transcription task.


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
2. **Semantic grouping**: ~20ms is roughly one phoneme - the right granularity for speech
3. **Information density**: Embeddings encode patterns, not just raw amplitudes

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
2. **Context**: Each frame summarizes 20ms of audio context
3. **Semantics**: Embeddings encode meaning, not just waveform shape

**Think of it this way**: Instead of describing every brush stroke in a painting (raw samples), we describe what the painting depicts (embeddings). Fewer words, more meaning.


### LoRA Adaptation: Efficient Fine-Tuning

**The Challenge**: HuBERT has 1.3 billion parameters. Full fine-tuning would:
- Require massive GPU memory (40GB+)
- Take weeks to train
- Cost hundreds of dollars
- Risk destroying the pre-trained knowledge (catastrophic forgetting)

**The Solution**: LoRA (Low-Rank Adaptation)

**What is LoRA?**

LoRA is a technique that lets us adapt a large pre-trained model without modifying its weights. Instead of updating the original 1.3B parameters, we add small "adapter" matrices that learn the adjustments.

**How LoRA Works**:

In a transformer, attention layers compute queries (Q) and keys (K) using weight matrices:
```
Q = input × W_q    (where W_q is a 1280×1280 matrix = 1.6M parameters)
K = input × W_k    (where W_k is a 1280×1280 matrix = 1.6M parameters)
```

Instead of updating W_q and W_k directly, LoRA adds small adapter matrices:
```
Q = input × (W_q + ΔW_q)    where ΔW_q = A × B
```

The magic: ΔW is factored into two small matrices:
- **A**: 1280 × 16 (rank r=16)
- **B**: 16 × 1280

Total parameters in ΔW: (1280 × 16) + (16 × 1280) = 40,960 parameters

**The Savings**:
- **Original**: 1,638,400 parameters per projection (1280²)
- **LoRA**: 40,960 parameters per projection
- **Reduction**: 40x fewer parameters!

**In Tiny Audio**:

- **Base model**: Frozen (1.3B params) - never updated during training
- **LoRA adapters**: Trainable (~4M params, r=16)
  - Applied to `q_proj` and `k_proj` (query and key projections) in all 24 attention layers
  - Each layer gets two small adapter matrices
- **Result**: We train only 0.3% of the encoder's parameters!

**What is "rank" (r=16)?**

The rank controls the adapter size:
- **Low rank (r=4-8)**: Fewer parameters, faster training, less adaptation capacity
- **Medium rank (r=16-32)**: Good balance for most tasks
- **High rank (r=64-128)**: More parameters, stronger adaptation, slower training

We use **r=16** for the encoder because HuBERT's pre-trained representations are already excellent - we only need light adaptation for our specific dataset.

**What are q_proj and k_proj?**

In transformer attention:
- **q_proj** (query projection): Transforms input into "what am I looking for?"
- **k_proj** (key projection): Transforms input into "what information do I have?"
- Together they compute attention scores: which parts of the audio to focus on

We adapt these because they control *what* the model pays attention to - crucial for ASR where we need to focus on speech-relevant features.

**Why this works**:
1. **Preserves knowledge**: The original 1.3B weights stay frozen, keeping learned speech patterns
2. **Task-specific adaptation**: LoRA learns adjustments for our transcription task
3. **Efficient**: 0.3% of parameters means 40x less memory and much faster training
4. **Modular**: Can save/load different LoRA adapters for different tasks

**Analogy**: Imagine HuBERT is a master chef with 1.3 billion skills. Instead of retraining the chef entirely (expensive, risky), we give them a small recipe card (LoRA adapter) with adjustments: "add a pinch more salt here, cook 2 minutes longer there." The chef's core skills remain intact, but the output is customized for your taste.

---

# PART B: HANDS-ON WORKSHOP (40 minutes)

>

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1** (15 min): Visualize audio processing (raw vs normalized waveforms)
- **Exercise 2** (25 min): Swap the encoder (HuBERT → Whisper)

By the end, you'll understand audio preprocessing and experience the power of modular architecture by swapping components!

---

## Workshop Exercise 1: Visualize Audio Processing (15 min)


### Goal

See how raw audio becomes normalized features.


### Your Task

Create visualizations showing audio before and after processing.


### Instructions

**Step 1: Create `explore_audio.py`**

(Note: librosa and matplotlib are already installed as part of the course dependencies)


```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from transformers import Wav2Vec2FeatureExtractor

# Load an audio file
audio_path = "test.wav"  # Update to your audio file
waveform, sr = librosa.load(audio_path, sr=16000)

print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(waveform) / sr:.2f} seconds")
print(f"Shape: {waveform.shape}")
print(f"Value range: [{waveform.min():.4f}, {waveform.max():.4f}]")

# Apply feature extraction
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-xlarge-ls960-ft"
)
inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

print(f"\nAfter feature extraction:")
print(f"Shape: {inputs.input_values.shape}")
print(f"Mean: {inputs.input_values.mean():.4f}")
print(f"Std: {inputs.input_values.std():.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Original waveform
ax1.plot(np.arange(len(waveform)) / sr, waveform)
ax1.set_title("Raw Waveform")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)

# Normalized waveform
normalized = inputs.input_values.squeeze().numpy()
ax2.plot(np.arange(len(normalized)) / sr, normalized)
ax2.set_title("After Z-Normalization")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Normalized Amplitude")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("audio_processing.png", dpi=150)
print("\n✓ Saved visualization to audio_processing.png")


```

**Step 2: Run the script**


```bash
poetry run python explore_audio.py


```

**Step 3: Open and examine `audio_processing.png`**


### Success Checkpoint

- [ ] Script ran without errors

- [ ] Generated `audio_processing.png`

- [ ] Can see difference between raw and normalized waveforms

- [ ] Normalized waveform is centered around 0

**Observations**: Notice how normalization centers the audio and makes it more uniform!

---

## Workshop Exercise 2: Swap the Encoder (25 min)

### Goal

Learn how to experiment with different audio encoders by swapping HuBERT for Whisper's encoder.

### Why This Matters

One of the most powerful aspects of modular architectures is the ability to swap components. The encoder is the "ear" of your ASR system - different encoders have different strengths:
- **HuBERT**: Self-supervised on 60K hours, excellent general-purpose representations
- **Whisper**: Trained on 680K hours of weakly-supervised multilingual data, strong multilingual capabilities

### Your Task

Modify the Tiny Audio configuration to use OpenAI's Whisper encoder instead of HuBERT.

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
- Decoder (Qwen-3 8B + LoRA)
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

2. **What if encoder dimensions don't match?**
   - Projector input dimension must match encoder output
   - Would need to adjust `encoder_dim` in config
   - Example: Wav2Vec2-base outputs 768-dim (not 1280-dim)

3. **Can you mix and match any encoder/decoder?**
   - Yes! As long as dimensions are compatible
   - Projector bridges the gap
   - This is the power of modular architecture

### Key Insight

The beauty of the encoder-projector-decoder architecture is **modularity**. You can:
- Swap encoders (HuBERT → Whisper → Wav2Vec2)
- Swap decoders (Qwen → Llama → Mistral)
- Adjust projector (SwiGLU → simple MLP)

Each component is independent. Experiment freely!

---

# CLASS SUMMARY

## What We Covered Today

**Lecture (20 min):**

- The audio processing pipeline (digitization, normalization, encoding)

- How sound becomes numbers (sampling at 16kHz)

- Feature extraction and preprocessing (z-normalization, padding, tensors)

- HuBERT architecture and self-supervised pre-training

- LoRA adaptation for efficient fine-tuning

**Workshop (40 min):**

- Visualized raw vs normalized audio waveforms

- Experimented with swapping encoders (HuBERT vs Whisper)

## Key Takeaways


✅ Audio is digitized through sampling (16kHz for speech)

✅ Wav2Vec2FeatureExtractor normalizes audio for training

✅ HuBERT compresses audio ~320x (48k samples → 149 embeddings)

✅ Each embedding represents ~20ms of audio in 1280 dimensions

✅ Only 1.6% of model parameters are trainable (146M / 9.3B)

## Homework (Optional)

Before Class 3, experiment with:

1. **Audio Preprocessing Challenge**:
   - Test 5 different audio files with varying quality
   - Measure embedding dimensions for each
   - Plot the relationship between audio length and number of embeddings

2. **Encoder Experiments**:
   - Compare HuBERT vs Wav2Vec2 on the same audio
   - Test inference speed on different hardware (CPU vs GPU if available)
   - Try different model sizes (base, large, xlarge)

3. **LoRA Analysis**:
   - Calculate memory savings for different rank values
   - Read about LoRA in the original paper
   - Think: Why do we use different ranks for encoder (r=16) vs decoder (r=8)?

4. **Code Exploration**:
   - Read the `AudioProjector` class in `src/asr_modeling.py:29-77`
   - Find where LoRA is applied in the code
   - Think: "How would you bridge 1280-dim audio to 2048-dim text?"

5. **Advanced Experiments**:
   - What happens if you skip normalization?
   - Can you visualize attention patterns in HuBERT?
   - How do embeddings change for different speakers?

## Check Your Understanding

1. **Why do we use 16kHz sampling rate?**
   - Captures human speech frequency range
   - Balances quality with computational efficiency
   - Industry standard for ASR

2. **What does z-normalization do?**
   - Centers audio around 0 (zero mean)
   - Scales to unit variance (std=1)
   - Stabilizes training

3. **How much does HuBERT compress the time dimension?**
   - ~320x reduction
   - 48,000 samples → ~149 frames
   - Each frame = ~20ms of audio

4. **Why use LoRA instead of full fine-tuning?**
   - Only trains 4M params vs 1.3B (0.3%)
   - Faster, less memory, cheaper
   - Preserves pre-trained knowledge

---

## Further Reading (Optional)


### Papers

- [HuBERT: Self-Supervised Speech Representation Learning](https://arxiv.org/abs/2106.07447)

- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477)


### Tutorials

- [Librosa documentation](https://librosa.org/doc/main/index.html)

- [Audio processing basics](https://pytorch.org/audio/stable/tutorials/audio_preprocessing_tutorial.html)

[Previous: Class 1: Introduction and Setup](./1-introduction-and-setup.md) | [Next: Class 3: Language Models and Projectors](./3-language-models-and-projectors.md)
