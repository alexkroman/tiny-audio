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

> **Instructor**: Present these concepts with interactive demonstrations and experiments.

## 1. The Importance of Data Quality (5 min)

Before we dive into the technical details of audio processing, let's talk about the single most important factor in training a great model: **data quality**.

No amount of architectural cleverness or hyperparameter tuning can make up for a poor-quality dataset. The goal of all the processing steps we're about to discuss is to create a **clean, consistent, and high-quality** dataset that our model can learn from effectively.

Think of it this way:

- **Good Data**: A clear, consistent signal that the model can learn from.
- **Bad Data**: Noise that confuses the model and hurts performance.

Our job in this chapter is to turn raw, messy audio into good data.

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

Key innovation: **Self-supervised learning as data curation at scale**.

HuBERT was pre-trained on 60,000 hours of unlabeled speech. This is a powerful example of the principle from "The Smol Training Playbook": leveraging massive, diverse datasets to build foundational knowledge. Instead of needing transcriptions, HuBERT learns the structure of speech by predicting masked audio segments.

**Analogy**: Like learning a language by listening to thousands of hours of conversation and learning to predict missing words, without ever opening a dictionary.

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

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Visualize audio processing and experiment with preprocessing
- **Exercise 2**: Explore HuBERT outputs and compare with Wav2Vec2
- **Exercise 3**: Count trainable parameters and experiment with LoRA ranks

By the end, you'll see exactly how audio becomes embeddings and how different choices affect the model!

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

### Experimentation Time!

**Experiment 1: Test different sample rates**

Add this to your `explore_audio.py`:

```python
# Experiment with different sample rates
sample_rates = [8000, 16000, 32000, 44100]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, target_sr in enumerate(sample_rates):
    # Resample audio
    resampled = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    # Plot spectrogram
    D = librosa.stft(resampled, n_fft=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    img = librosa.display.specshow(S_db, sr=target_sr, x_axis='time',
                                    y_axis='hz', ax=axes[idx])
    axes[idx].set_title(f'Sample Rate: {target_sr} Hz')
    axes[idx].set_ylim(0, 8000)  # Focus on speech range

plt.tight_layout()
plt.savefig("sample_rate_comparison.png", dpi=150)
print("✓ Saved sample rate comparison to sample_rate_comparison.png")
```

**Experiment 2: Test noise addition**

```python
# Add different levels of noise
noise_levels = [0, 0.01, 0.05, 0.1]
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes = axes.flatten()

for idx, noise_level in enumerate(noise_levels):
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, len(waveform))
    noisy_waveform = waveform + noise

    # Apply feature extraction
    inputs = feature_extractor(noisy_waveform, sampling_rate=sr, return_tensors="pt")
    normalized = inputs.input_values.squeeze().numpy()

    # Plot
    axes[idx].plot(normalized[:1000])  # Show first 1000 samples
    axes[idx].set_title(f'Noise Level: {noise_level}')
    axes[idx].set_ylabel('Normalized Amplitude')

plt.tight_layout()
plt.savefig("noise_robustness.png", dpi=150)
print("✓ Saved noise analysis to noise_robustness.png")
```

**Questions to explore:**
- How does sample rate affect frequency resolution?
- Does normalization help with noise robustness?
- What's the minimum sample rate for intelligible speech?

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

### Encoder Comparison Experiment

Now let's compare HuBERT with Wav2Vec2:

**Step 1: Create `compare_encoders.py`**

```python
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import librosa
import time

# Load audio
audio_path = "test.wav"
waveform, sr = librosa.load(audio_path, sr=16000)

# Test different encoders
encoders_to_test = [
    ("facebook/hubert-xlarge-ls960-ft", "HuBERT-XLarge"),
    ("facebook/wav2vec2-large-960h", "Wav2Vec2-Large"),
]

results = []

for model_id, name in encoders_to_test:
    print(f"\nTesting {name}...")

    # Load model
    encoder = AutoModel.from_pretrained(model_id)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    # Prepare input
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

    # Time the forward pass
    start = time.time()
    with torch.no_grad():
        outputs = encoder(**inputs)
    inference_time = time.time() - start

    embeddings = outputs.last_hidden_state

    # Collect results
    results.append({
        "name": name,
        "params": sum(p.numel() for p in encoder.parameters()) / 1e6,
        "embedding_dim": embeddings.shape[-1],
        "num_frames": embeddings.shape[1],
        "inference_time": inference_time,
        "mean": embeddings.mean().item(),
        "std": embeddings.std().item()
    })

# Compare results
print("\n" + "="*60)
print("ENCODER COMPARISON")
print("="*60)
for r in results:
    print(f"\n{r['name']}:")
    print(f"  Parameters: {r['params']:.1f}M")
    print(f"  Embedding dim: {r['embedding_dim']}")
    print(f"  Frames produced: {r['num_frames']}")
    print(f"  Inference time: {r['inference_time']:.3f}s")
    print(f"  Output stats: mean={r['mean']:.4f}, std={r['std']:.4f}")

# Performance comparison
if len(results) == 2:
    speedup = results[1]['inference_time'] / results[0]['inference_time']
    size_ratio = results[0]['params'] / results[1]['params']
    print(f"\nHuBERT is {size_ratio:.1f}x larger but only {1/speedup:.1f}x slower")
```

**Step 2: Visualize embedding differences**

```python
# Add to compare_encoders.py
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, r in enumerate(results[:2]):
    # For visualization, we need to recompute embeddings
    # (In practice, save them during the loop above)
    model_id = encoders_to_test[idx][0]
    encoder = AutoModel.from_pretrained(model_id)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        outputs = encoder(**inputs)
    embeddings = outputs.last_hidden_state.squeeze().numpy()

    # Plot first 50 dimensions of first 100 frames
    im = axes[idx].imshow(embeddings[:100, :50].T, aspect='auto', cmap='coolwarm')
    axes[idx].set_title(r['name'])
    axes[idx].set_xlabel('Time (frames)')
    axes[idx].set_ylabel('Embedding dimensions')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig("encoder_embeddings.png", dpi=150)
print("\n✓ Saved embedding comparison to encoder_embeddings.png")
```

**Questions to explore:**
- Which encoder is faster? Why?
- How do embedding patterns differ between encoders?
- Would Wav2Vec2 work as a drop-in replacement?

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
count_params(model.decoder, "DECODER (Qwen-3 8B + LoRA)")

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
print("\n✓ We train only 1.5% of the total parameters!")
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

DECODER (Qwen-3 8B + LoRA)
==================================================
  Total params:       8,000,000,000
  Trainable params:      15,335,424
  Frozen params:      7,984,664,576
  Trainable:                   0.19%

==================================================
OVERALL MODEL
==================================================
  Total params:       9,388,843,264
  Trainable params:     138,944,768
  Frozen params:      9,249,898,496
  Efficiency:                  1.48%

✓ We train only 1.5% of the total parameters!
```

### Success Checkpoint

- [ ] Script ran successfully
- [ ] Saw parameter counts for all three components
- [ ] Understand that projector is 100% trainable
- [ ] Understand that encoder/decoder use small LoRA adapters

**Key Insight**: We're only training 139M out of 9.3B parameters - that's the magic of parameter-efficient fine-tuning!

### LoRA Rank Experiment

Let's explore how LoRA rank affects trainable parameters:

**Create `lora_experiment.py`:**

```python
def calculate_lora_params(in_dim, out_dim, rank):
    """Calculate LoRA parameters for a given rank."""
    # LoRA: W = W0 + BA where B: in_dim x rank, A: rank x out_dim
    down_proj = in_dim * rank  # B matrix
    up_proj = rank * out_dim    # A matrix
    return down_proj + up_proj

# HuBERT attention dimensions
hubert_dims = {
    "hidden_size": 1280,
    "num_attention_heads": 16,
    "head_dim": 80,  # 1280 / 16
}

# Calculate for different ranks
ranks = [1, 2, 4, 8, 16, 32, 64, 128]
print("="*60)
print("LoRA RANK ANALYSIS FOR HUBERT")
print("="*60)

for rank in ranks:
    # Q and K projections (typical LoRA targets)
    params_per_layer = 2 * calculate_lora_params(
        hubert_dims["hidden_size"],
        hubert_dims["hidden_size"],
        rank
    )

    # 24 transformer layers
    total_params = 24 * params_per_layer

    # Percentage of original model
    original_params = 1.3e9  # 1.3B for HuBERT
    percent = 100 * total_params / original_params

    print(f"Rank {rank:3d}: {total_params/1e6:6.2f}M params ({percent:.2f}% of model)")

print("\n" + "="*60)
print("MEMORY & SPEED TRADEOFFS")
print("="*60)

# Estimate memory and speed impact
baseline_rank = 8
for rank in [4, 8, 16, 32]:
    memory_factor = rank / baseline_rank
    # Speed impact is roughly linear with rank for small ranks
    speed_factor = 1 + 0.1 * (rank - baseline_rank) / baseline_rank

    print(f"Rank {rank:2d}:")
    print(f"  Memory: {memory_factor:.1f}x vs rank-8")
    print(f"  Speed impact: ~{speed_factor:.1f}x training time")
    print(f"  Recommendation: {recommend_usage(rank)}")

def recommend_usage(rank):
    if rank <= 4:
        return "Quick experiments, limited adaptation"
    elif rank <= 16:
        return "Good balance for most tasks"
    elif rank <= 32:
        return "Complex adaptation, domain shift"
    else:
        return "Maximum flexibility, longer training"
```

**Experiment: Test different LoRA configurations**

Add this analysis:

```python
# Compare LoRA targets
lora_targets = [
    (["q_proj", "v_proj"], "Attention QV"),
    (["q_proj", "k_proj"], "Attention QK"),
    (["q_proj", "k_proj", "v_proj", "o_proj"], "All Attention"),
    (["mlp.fc1", "mlp.fc2"], "FFN layers"),
]

print("\n" + "="*60)
print("LoRA TARGET COMPARISON")
print("="*60)

rank = 8  # Fixed rank for comparison
for targets, name in lora_targets:
    # Rough parameter calculation
    if "mlp" in targets[0]:
        params = len(targets) * calculate_lora_params(1280, 5120, rank)
    else:
        params = len(targets) * calculate_lora_params(1280, 1280, rank)

    params_total = 24 * params  # 24 layers
    print(f"{name:20s}: {params_total/1e6:6.2f}M params")
```

**Discussion Questions:**
- How does rank affect model capacity?
- What's the sweet spot for rank vs performance?
- Which LoRA targets are most important?

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
   - Think: Why do we use different ranks for encoder (r=8) vs decoder (r=64)?

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
- The Qwen-3 8B decoder architecture
- Deep dive into the AudioProjector and SwiGLU
- Why downsampling matters for efficiency

[Previous: Class 1: Introduction and Setup](./1-introduction-and-setup.md) | [Next: Class 3: Language Models and Projectors](./3-language-models-and-projectors.md)

**See you next time!**
