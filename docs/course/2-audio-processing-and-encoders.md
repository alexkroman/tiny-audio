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

> **Instructor**: Present these concepts. Students should just listen.

## 1. From Sound Waves to Numbers (5 min)

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

---

## 2. Feature Extraction with Wav2Vec2 (5 min)

### The Problem

Raw waveforms are:

- High-dimensional (16,000 numbers/second!)
- Noisy
- Hard for models to learn from

### The Solution: Wav2Vec2FeatureExtractor

Transforms audio through:

1. **Resampling**: Convert any sample rate → 16 kHz
2. **Z-Normalization**: `(x - mean) / std`
   - Centers audio around 0
   - Scales to unit variance
   - Stabilizes training
3. **Padding**: Makes all samples same length (enables batching)
4. **Tensor Conversion**: NumPy arrays → PyTorch tensors

**Result**: Clean, normalized, ready-to-train audio!

---

## 3. The HuBERT Encoder (10 min)

### What is HuBERT?

**HuBERT** = **H**idden **U**nit **BERT**

Key innovation: **Self-supervised learning on unlabeled audio**

- No transcriptions needed!
- Trained on 60,000 hours of speech
- Learns by predicting masked audio segments

**Analogy**: Like learning English by filling in blanks:

- "The quick brown ___ jumped over the lazy dog"
- No explicit grammar lessons needed!

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

**HuBERT-XLarge Stats:**

- 24 transformer layers
- 1280 hidden dimensions
- 16 attention heads/layer
- ~1.3 billion parameters
- Pre-trained on LibriLight (60K hours)

### What HuBERT Learned

During pre-training, HuBERT learned:

- Phonemes (speech sounds)
- Speaker characteristics
- Acoustic environments
- Prosody (rhythm, stress, intonation)

**This is why we don't train from scratch!**

### Time Compression

```
3 seconds audio at 16kHz = 48,000 samples
    ↓ (HuBERT encoder)
~149 embeddings × 1280 dimensions
```

- **Compression**: ~320x reduction
- **Each embedding**: Represents ~20ms of audio
- **Dense representation**: More meaningful than raw waveform

### LoRA Adaptation

In Tiny Audio, we add small LoRA adapters:

- **Base model**: Frozen (1.3B params)
- **LoRA adapters**: Trainable (~2M params, r=8)
- **Target**: q_proj, k_proj in attention layers
- **Result**: 0.15% of encoder params are trainable!

**Analogy**: Putting adjustable glasses on a camera - camera unchanged, but output is tuned.

---

# PART B: HANDS-ON WORKSHOP (40 minutes)

> **Students**: Follow these step-by-step instructions.
>
> **Instructor**: Circulate and help students.

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

## Workshop Exercise 2: Explore HuBERT Outputs (15 min)

### Goal

See how HuBERT converts audio waveforms into embeddings.

### Your Task

Pass audio through HuBERT and examine the output dimensions.

### Instructions

**Step 1: Create `explore_hubert.py`**

```python
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import librosa

print("Loading HuBERT model...")
encoder = AutoModel.from_pretrained("facebook/hubert-xlarge-ls960-ft")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-xlarge-ls960-ft"
)
print("✓ Model loaded!\n")

# Load audio
audio_path = "test.wav"  # Update to your audio file
waveform, sr = librosa.load(audio_path, sr=16000)

# Extract features
inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

# Pass through encoder
print("Processing audio through HuBERT...")
with torch.no_grad():
    outputs = encoder(**inputs)

embeddings = outputs.last_hidden_state

# Print dimensions
print("\n" + "="*50)
print("INPUT (Raw Audio)")
print("="*50)
print(f"Shape: {inputs.input_values.shape}")
print(f"Interpretation: [batch_size=1, samples={inputs.input_values.shape[-1]}]")
print(f"Duration: {inputs.input_values.shape[-1] / sr:.2f} seconds")

print("\n" + "="*50)
print("OUTPUT (Audio Embeddings)")
print("="*50)
print(f"Shape: {embeddings.shape}")
print(f"Interpretation: [batch_size=1, time_steps={embeddings.shape[1]}, embedding_dim={embeddings.shape[2]}]")

# Calculate compression
time_reduction = inputs.input_values.shape[-1] / embeddings.shape[1]
time_per_frame = (inputs.input_values.shape[-1] / embeddings.shape[1]) / sr

print(f"\nTime dimension reduction: {time_reduction:.1f}x")
print(f"Each embedding represents: ~{time_per_frame * 1000:.1f}ms of audio")
print(f"Embedding dimensionality: {embeddings.shape[-1]}D")

# Statistics
print("\n" + "="*50)
print("EMBEDDING STATISTICS")
print("="*50)
print(f"Mean: {embeddings.mean():.4f}")
print(f"Std: {embeddings.std():.4f}")
print(f"Min: {embeddings.min():.4f}")
print(f"Max: {embeddings.max():.4f}")

print("\n✓ Processing complete!")
```

**Step 2: Run the script**

```bash
poetry run python explore_hubert.py
```

**Expected output:**

```
Loading HuBERT model...
✓ Model loaded!

Processing audio through HuBERT...

==================================================
INPUT (Raw Audio)
==================================================
Shape: torch.Size([1, 48000])
Interpretation: [batch_size=1, samples=48000]
Duration: 3.00 seconds

==================================================
OUTPUT (Audio Embeddings)
==================================================
Shape: torch.Size([1, 149, 1280])
Interpretation: [batch_size=1, time_steps=149, embedding_dim=1280]

Time dimension reduction: 322.1x
Each embedding represents: ~20.1ms of audio
Embedding dimensionality: 1280D

==================================================
EMBEDDING STATISTICS
==================================================
Mean: 0.0123
Std: 0.4567
Min: -2.3456
Max: 3.1234

✓ Processing complete!
```

### Success Checkpoint

- [ ] Script ran successfully
- [ ] Saw input dimensions (e.g., `[1, 48000]`)
- [ ] Saw output dimensions (e.g., `[1, 149, 1280]`)
- [ ] Understand the ~320x compression ratio

**Key Insight**: 48,000 audio samples → 149 embeddings of 1280 dimensions each!

---

## Workshop Exercise 3: Count Trainable Parameters (10 min)

### Goal

Understand which parts of the model are trainable vs frozen.

### Your Task

Count parameters in the encoder, projector, and decoder.

### Instructions

**Step 1: Create `count_params.py`**

```python
from src.asr_modeling import ASRModel
from src.asr_config import ASRConfig

print("Loading Tiny Audio model...")
config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
model = ASRModel.from_pretrained("mazesmazes/tiny-audio", config=config)
print("✓ Model loaded!\n")

def count_params(module, name):
    """Count total and trainable parameters in a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = total - trainable
    percent = 100 * trainable / total if total > 0 else 0

    print(f"{name}")
    print(f"{'='*50}")
    print(f"  Total params:      {total:>15,}")
    print(f"  Trainable params:  {trainable:>15,}")
    print(f"  Frozen params:     {frozen:>15,}")
    print(f"  Trainable:         {percent:>14.2f}%")
    print()

# Count by component
count_params(model.encoder, "ENCODER (HuBERT + LoRA)")
count_params(model.projector, "PROJECTOR (SwiGLU MLP)")
count_params(model.decoder, "DECODER (SmolLM3 + LoRA)")

# Overall
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("="*50)
print("OVERALL MODEL")
print("="*50)
print(f"  Total params:      {total:>15,}")
print(f"  Trainable params:  {trainable:>15,}")
print(f"  Frozen params:     {total - trainable:>15,}")
print(f"  Efficiency:        {100 * trainable / total:>14.2f}%")
print("\n✓ We train only 3.2% of the total parameters!")
```

**Step 2: Run the script**

```bash
poetry run python count_params.py
```

**Expected output:**

```
ENCODER (HuBERT + LoRA)
==================================================
  Total params:       1,267,200,000
  Trainable params:       1,966,080
  Frozen params:      1,265,233,920
  Trainable:                   0.16%

PROJECTOR (SwiGLU MLP)
==================================================
  Total params:         121,643,264
  Trainable params:     121,643,264
  Frozen params:                  0
  Trainable:                 100.00%

DECODER (SmolLM3 + LoRA)
==================================================
  Total params:       2,953,383,936
  Trainable params:      15,335,424
  Frozen params:      2,938,048,512
  Trainable:                   0.52%

==================================================
OVERALL MODEL
==================================================
  Total params:       4,342,227,200
  Trainable params:     138,944,768
  Frozen params:      4,203,282,432
  Efficiency:                  3.20%

✓ We train only 3.2% of the total parameters!
```

### Success Checkpoint

- [ ] Script ran successfully
- [ ] Saw parameter counts for all three components
- [ ] Understand that projector is 100% trainable
- [ ] Understand that encoder/decoder use small LoRA adapters

**Key Insight**: We're only training 139M out of 4.3B parameters - that's the magic of parameter-efficient fine-tuning!

---

# CLASS SUMMARY

## What We Covered Today

**Lecture (20 min):**

- How audio becomes numbers (sampling, bit depth)
- Feature extraction with Wav2Vec2
- HuBERT architecture and self-supervised pre-training
- LoRA adaptation strategy

**Workshop (40 min):**

- Visualized raw vs normalized audio
- Explored HuBERT's embedding outputs
- Counted trainable vs frozen parameters

## Key Takeaways

✅ Audio is digitized through sampling (16kHz for speech)
✅ Wav2Vec2FeatureExtractor normalizes audio for training
✅ HuBERT compresses audio ~320x (48k samples → 149 embeddings)
✅ Each embedding represents ~20ms of audio in 1280 dimensions
✅ Only 3.2% of model parameters are trainable (139M / 4.3B)

## Homework (Optional)

Before Class 3:

1. Try different audio files and observe embedding dimensions
2. Read the `AudioProjector` class in `src/asr_modeling.py:29-77`
3. Think: "How would you bridge 1280-dim audio to 2048-dim text?"

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
   - Only trains 2M params vs 1.3B (0.16%)
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

---

## Next Class

In [Class 3: Language Models and Projectors](./3-language-models-and-projectors.md), we'll explore:

- How language models generate text
- The SmolLM3 decoder architecture
- Deep dive into the AudioProjector and SwiGLU
- Why downsampling matters for efficiency

[Previous: Class 1: Introduction and Setup](./1-introduction-and-setup.md) | [Next: Class 3: Language Models and Projectors](./3-language-models-and-projectors.md)

**See you next time!**
