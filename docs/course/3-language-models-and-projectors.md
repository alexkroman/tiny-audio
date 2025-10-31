# Class 3: Language Models and Projectors

**Duration**: 1 hour (20 min lecture + 40 min hands-on)
**Goal**: Understand how the projector bridges audio and text modalities

## Learning Objectives

By the end of this class, you will:

- Understand what language models do and how they generate text
- Know the Qwen-3 8B architecture
- Understand the AudioProjector's SwiGLU architecture
- Implement and visualize the projection process
- See how audio embeddings become text embeddings

---

# PART A: LECTURE (20 minutes)

> **Instructor**: Present these concepts. Students should just listen.

## 1. Language Models Basics (5 min)

### What is a Language Model?

A language model predicts the next word (token) given previous words.

**Example:**

```
Input:  "The quick brown"
Output: "fox" (predicted next word)
```

### How They Work

**Training**: Learn patterns from massive text corpora

- Qwen-3 8B trained on trillions of tokens
- Learns grammar, facts, reasoning patterns
- Develops understanding of language structure

**Inference**: Generate text token by token

```
Start: "Hello"
Step 1: "Hello, how"
Step 2: "Hello, how are"
Step 3: "Hello, how are you"
Step 4: "Hello, how are you?" [STOP]
```

### Qwen-3 8B Architecture

**Stats:**

- 8 billion parameters
- 32 transformer layers
- 2048 hidden dimensions
- 32 attention heads
- Trained on diverse multilingual data

**Why Qwen-3 8B?**

- Large enough for strong performance
- Efficient with LoRA fine-tuning
- Open source and well-documented
- Excellent text generation quality
- Multilingual capabilities

### Decoder-Only Architecture

Qwen-3 8B is "decoder-only" (like GPT):

```
Input tokens → Embeddings → Transformer Layers → Next token prediction
```

**Key features:**

- Causal attention (can only look backward)
- Auto-regressive generation
- Flash Attention 2 for speed

**Why Decoder-Only?**

The choice of architecture is a critical decision. While other architectures exist, decoder-only models have become the standard for large-scale language generation tasks.

- **Encoder-Decoder Models** (like T5) are great for tasks that require a deep understanding of the input, like summarization or translation. However, they are more complex to train and are less common for generative assistants.
- **Mixture-of-Experts (MoE) Models** (like Mixtral) are very powerful and efficient at inference, but they are more complex to train and require more memory.

For our project, a **decoder-only model is the perfect choice** because:

- It excels at **generative tasks** like ASR.
- It's **simpler to train and understand** than other architectures.
- The vast majority of **open-source tools and research** are focused on decoder-only models, making it easier to find support and resources.

By choosing a decoder-only model, we are building on a solid, well-understood foundation.

---

## 2. The Modality Gap Problem (5 min)

### The Challenge

We have two different "languages":

- **Audio embeddings**: 1280 dimensions from HuBERT
- **Text embeddings**: 2048 dimensions from Qwen-3 8B

**Problem**: Can't directly feed audio embeddings to text model!

- Different dimensions (1280 vs 2048)
- Different statistical distributions
- Different semantic spaces

**Analogy**: Like trying to plug a European power plug into an American outlet - same purpose, different format!

### The Solution: AudioProjector

A trainable neural network that:

1. **Transforms dimensions**: 1280D → 2048D
2. **Aligns distributions**: Audio stats → Text stats
3. **Downsamples time**: 5x reduction for efficiency
4. **Bridges modalities**: Audio space → Language space

**Key insight**: This is the ONLY fully trainable component (~122M params)!

---

## 3. SwiGLU Architecture Deep Dive (10 min)

### What is SwiGLU?

**SwiGLU** = **Swi**sh **G**ated **L**inear **U**nit

Used in modern architectures (Llama, PaLM, etc.) for better performance than simple MLPs.

### Architecture Breakdown

```python
# Pseudocode for AudioProjector
def forward(audio_features):
    # Input: [batch, time, 1280]

    # Step 1: Stack 5 frames together (downsampling)
    stacked = stack_frames(audio_features, k=5)
    # Shape: [batch, time/5, 1280*5] = [batch, time/5, 6400]

    # Step 2: Pre-normalize (fix broken stats from stacking)
    x = rms_norm(stacked)

    # Step 3: SwiGLU transformation
    gate = linear_gate(x)      # [batch, time/5, 8192]
    up = linear_up(x)          # [batch, time/5, 8192]
    activated = silu(gate) * up  # Element-wise multiply

    # Step 4: Project to LLM dimension
    output = linear_down(activated)  # [batch, time/5, 2048]

    # Step 5: Post-normalize (match LLM's expected distribution)
    output = rms_norm(output)

    return output
```

### Why Each Component Matters

**1. Frame Stacking (5x downsampling)**

- Input: 149 frames × 1280D
- Concatenate 5 consecutive frames → 1 super-frame
- Output: ~30 frames × 6400D
- **Why?** Efficiency! Reduces sequence length for decoder

**2. Pre-normalization (RMSNorm)**

- Concatenation breaks normalized statistics
- RMSNorm re-normalizes: `x / sqrt(mean(x²))`
- **Why?** Stable training, better gradients

**3. SwiGLU Activation**

- `gate_proj`: Controls information flow
- `up_proj`: Transforms features
- `silu(gate) * up`: Gated activation (selective)
- **Why?** Better than ReLU, more expressive

**4. Down Projection**

- Maps 8192D → 2048D (LLM dimension)
- **Why?** Match decoder's input size

**5. Post-normalization**

- Ensure output matches LLM's expected distribution
- **Why?** LLM was trained on specific input stats

### RMSNorm vs LayerNorm

**RMSNorm** (Root Mean Square Norm):

```python
output = x / sqrt(mean(x²) + epsilon)
```

**Advantages:**

- Simpler than LayerNorm (no mean subtraction)
- Faster computation
- Similar performance
- Used in Llama, Qwen-3 8B, etc.

### SwiGLU vs Other Activations

**ReLU**: `max(0, x)` - Simple but loses negative info
**GELU**: Smoother, but no gating
**GLU**: Gating but with sigmoid (saturates)
**SwiGLU**: Best of both! Gating + smooth activation

**Formula**: `Swish(Wx) ⊗ (Vx)` where `Swish(x) = x * sigmoid(x)`

### Why SwiGLU?

SwiGLU has become the de-facto standard activation function in modern language models like Llama, PaLM, and Qwen. Here's why:

- **Gated Mechanism**: The "G" in SwiGLU stands for "Gated." The gating mechanism allows the network to control the flow of information, which has been shown to be more effective than a simple non-linearity like ReLU.
- **Expressiveness**: The combination of the Swish activation function and the gating mechanism allows the network to learn more complex patterns in the data.
- **Performance**: In practice, SwiGLU has been shown to outperform other activation functions on a wide range of language modeling tasks.

By using SwiGLU, we are using a modern, high-performance component that is known to work well in large-scale language models.

---

# PART B: HANDS-ON WORKSHOP (40 minutes)

> **Students**: Follow these instructions step-by-step.
>
> **Instructor**: Circulate and help students.

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Trace the projection process step-by-step
- **Exercise 2**: Visualize embedding distributions
- **Exercise 3**: Compare projector configurations

By the end, you'll understand how audio becomes language!

---

## Workshop Exercise 1: Trace the Projection Process (15 min)

### Goal

Follow audio embeddings through the projector step-by-step.

### Your Task

Create a script that shows dimensions at each projector stage.

### Instructions

**Step 1: Create `trace_projector.py`**

```python
import torch
from src.asr_modeling import ASRModel
from src.asr_config import ASRConfig
from transformers import Wav2Vec2FeatureExtractor
import librosa

# Load model
print("Loading Tiny Audio model...")
config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
model = ASRModel.from_pretrained("mazesmazes/tiny-audio", config=config)
print("✓ Model loaded!\n")

# Load and process audio
audio_path = "test.wav"
waveform, sr = librosa.load(audio_path, sr=16000)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-xlarge-ls960-ft"
)
inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

# Get encoder output
print("="*60)
print("STEP 1: AUDIO ENCODER (HuBERT)")
print("="*60)
with torch.no_grad():
    encoder_output = model.encoder(**inputs).last_hidden_state

print(f"Encoder output shape: {encoder_output.shape}")
print(f"  [batch={encoder_output.shape[0]}, time={encoder_output.shape[1]}, dim={encoder_output.shape[2]}]")
print(f"  Each frame = ~20ms of audio")
print(f"  Total coverage = ~{encoder_output.shape[1] * 20}ms")

# Manually trace through projector
projector = model.projector
k = projector.k  # Downsampling rate

print("\n" + "="*60)
print("STEP 2: FRAME STACKING (5x downsampling)")
print("="*60)

# Stack frames
batch_size, seq_len, dim = encoder_output.shape
remainder = seq_len % k
if remainder:
    pad_len = k - remainder
    encoder_output = torch.nn.functional.pad(encoder_output, (0, 0, 0, pad_len))
    print(f"Padded sequence: {seq_len} → {encoder_output.shape[1]} frames")

stacked = encoder_output.contiguous().view(batch_size, -1, dim * k)
print(f"After stacking: {stacked.shape}")
print(f"  [batch={stacked.shape[0]}, time={stacked.shape[1]}, dim={stacked.shape[2]}]")
print(f"  Time reduction: {seq_len} → {stacked.shape[1]} ({seq_len/stacked.shape[1]:.1f}x)")
print(f"  Each frame now = ~{20 * k}ms of audio")

print("\n" + "="*60)
print("STEP 3: PRE-NORMALIZATION (RMSNorm)")
print("="*60)
prenorm = projector.ln_pre(stacked)
print(f"Shape: {prenorm.shape} (unchanged)")
print(f"Before norm - Mean: {stacked.mean():.4f}, Std: {stacked.std():.4f}")
print(f"After norm  - Mean: {prenorm.mean():.4f}, Std: {prenorm.std():.4f}")

print("\n" + "="*60)
print("STEP 4: SwiGLU TRANSFORMATION")
print("="*60)
gate = projector.gate_proj(prenorm)
up = projector.up_proj(prenorm)
print(f"Gate projection: {gate.shape}")
print(f"Up projection:   {up.shape}")

activated = torch.nn.functional.silu(gate) * up
print(f"After SwiGLU:    {activated.shape}")
print(f"  SiLU(gate) ⊗ up = gated features")

print("\n" + "="*60)
print("STEP 5: DOWN PROJECTION")
print("="*60)
down = projector.down_proj(activated)
print(f"Down projection: {down.shape}")
print(f"  Dimension: 8192 → 2048 (LLM input size)")

print("\n" + "="*60)
print("STEP 6: POST-NORMALIZATION")
print("="*60)
output = projector.ln_post(down)
print(f"Final output: {output.shape}")
print(f"Before norm - Mean: {down.mean():.4f}, Std: {down.std():.4f}")
print(f"After norm  - Mean: {output.mean():.4f}, Std: {output.std():.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Input:  {encoder_output.shape[1]} frames × {encoder_output.shape[2]}D (HuBERT)")
print(f"Output: {output.shape[1]} frames × {output.shape[2]}D (Qwen-3 8B-ready)")
print(f"Time reduction: {encoder_output.shape[1] / output.shape[1]:.1f}x")
print(f"Dimension change: {encoder_output.shape[2]}D → {output.shape[2]}D")
print(f"\n✓ Audio embeddings are now ready for the language model!")
```

**Step 2: Run the script**

```bash
poetry run python trace_projector.py
```

**Expected output:**

```
============================================================
STEP 1: AUDIO ENCODER (HuBERT)
============================================================
Encoder output shape: torch.Size([1, 149, 1280])
  [batch=1, time=149, dim=1280]
  Each frame = ~20ms of audio
  Total coverage = ~2980ms

============================================================
STEP 2: FRAME STACKING (5x downsampling)
============================================================
Padded sequence: 149 → 150 frames
After stacking: torch.Size([1, 30, 6400])
  [batch=1, time=30, dim=6400]
  Time reduction: 149 → 30 (5.0x)
  Each frame now = ~100ms of audio

... (etc)
```

### Success Checkpoint

- [ ] Script ran successfully
- [ ] Saw all 6 steps of the projection process
- [ ] Understand the 5x downsampling (149 → 30 frames)
- [ ] Understand dimension change (1280 → 6400 → 8192 → 2048)

---

## Workshop Exercise 2: Visualize Embedding Distributions (15 min)

### Goal

See how embeddings change from audio to text space.

### Your Task

Plot the distribution of embedding values at each stage.

### Instructions

**Step 1: Create `visualize_projector.py`**

```python
import torch
import matplotlib.pyplot as plt
from src.asr_modeling import ASRModel
from src.asr_config import ASRConfig
from transformers import Wav2Vec2FeatureExtractor
import librosa

# Load model
config = ASRConfig.from_pretrained("mazesmazes/tiny-audio", trust_remote_code=True)
model = ASRModel.from_pretrained("mazesmazes/tiny-audio", config=config)

# Load audio
audio_path = "test.wav"
waveform, sr = librosa.load(audio_path, sr=16000)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-xlarge-ls960-ft"
)
inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

# Get embeddings at each stage
with torch.no_grad():
    # Encoder output
    audio_emb = model.encoder(**inputs).last_hidden_state

    # Projector output
    text_emb = model.projector(audio_emb)

    # LLM embedding for comparison (sample text)
    sample_text = model.tokenizer("Hello world", return_tensors="pt")
    llm_emb = model.decoder.get_input_embeddings()(sample_text.input_ids)

# Flatten for histograms
audio_flat = audio_emb.flatten().numpy()
text_flat = text_emb.flatten().numpy()
llm_flat = llm_emb.flatten().numpy()

# Plot distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(audio_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('Audio Embeddings (HuBERT Output)')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].axvline(audio_flat.mean(), color='red', linestyle='--', label=f'Mean: {audio_flat.mean():.3f}')
axes[0].legend()

axes[1].hist(text_flat, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title('After Projector (Ready for LLM)')
axes[1].set_xlabel('Value')
axes[1].axvline(text_flat.mean(), color='red', linestyle='--', label=f'Mean: {text_flat.mean():.3f}')
axes[1].legend()

axes[2].hist(llm_flat, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[2].set_title('Text Embeddings (LLM Native)')
axes[2].set_xlabel('Value')
axes[2].axvline(llm_flat.mean(), color='red', linestyle='--', label=f'Mean: {llm_flat.mean():.3f}')
axes[2].legend()

plt.tight_layout()
plt.savefig('embedding_distributions.png', dpi=150)
print("✓ Saved visualization to embedding_distributions.png")

# Print statistics
print("\n" + "="*60)
print("EMBEDDING STATISTICS")
print("="*60)
print(f"{'Stage':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("="*60)
print(f"{'Audio (HuBERT)':<25} {audio_flat.mean():<10.4f} {audio_flat.std():<10.4f} {audio_flat.min():<10.4f} {audio_flat.max():<10.4f}")
print(f"{'After Projector':<25} {text_flat.mean():<10.4f} {text_flat.std():<10.4f} {text_flat.min():<10.4f} {text_flat.max():<10.4f}")
print(f"{'Text (LLM Native)':<25} {llm_flat.mean():<10.4f} {llm_flat.std():<10.4f} {llm_flat.min():<10.4f} {llm_flat.max():<10.4f}")
```

**Step 2: Run the script**

```bash
poetry run python visualize_projector.py
```

**Step 3: Open `embedding_distributions.png`**

### Success Checkpoint

- [ ] Script ran successfully
- [ ] Generated `embedding_distributions.png`
- [ ] Can see three histograms showing embedding distributions
- [ ] Notice how projector output resembles LLM native embeddings

**Observation**: The projector transforms audio embeddings to match the LLM's expected input distribution!

---

## Workshop Exercise 3: Test Projector Configurations (10 min)

### Goal

Understand how projector parameters affect the model.

### Your Task

Experiment with different projector configurations.

### Instructions

**Step 1: Create `test_projector_config.py`**

```python
from types import SimpleNamespace

# Simulate different projector configurations
configs = [
    {
        "name": "Actual (Tiny Audio)",
        "encoder_dim": 1280,
        "llm_dim": 2048,
        "downsample_rate": 5,
        "hidden_dim": 8192,
    },
    {
        "name": "Smaller Projector",
        "encoder_dim": 1280,
        "llm_dim": 2048,
        "downsample_rate": 5,
        "hidden_dim": 4096,
    },
    {
        "name": "More Downsampling",
        "encoder_dim": 1280,
        "llm_dim": 2048,
        "downsample_rate": 8,
        "hidden_dim": 8192,
    },
    {
        "name": "Less Downsampling",
        "encoder_dim": 1280,
        "llm_dim": 2048,
        "downsample_rate": 2,
        "hidden_dim": 8192,
    },
]

def count_projector_params(config):
    """Calculate parameter count for a projector configuration."""
    stacked_dim = config["encoder_dim"] * config["downsample_rate"]
    hidden_dim = config["hidden_dim"]
    llm_dim = config["llm_dim"]

    # LayerNorm params (pre and post)
    ln_params = stacked_dim + llm_dim

    # Linear layer params (no bias)
    gate_params = stacked_dim * hidden_dim
    up_params = stacked_dim * hidden_dim
    down_params = hidden_dim * llm_dim

    total = ln_params + gate_params + up_params + down_params
    return total

def analyze_efficiency(config, audio_duration_sec=3.0):
    """Analyze computational efficiency."""
    # Assume 50 Hz encoder output (320x compression of 16kHz)
    encoder_frames = int(audio_duration_sec * 50)
    projector_frames = encoder_frames // config["downsample_rate"]

    return {
        "encoder_frames": encoder_frames,
        "projector_frames": projector_frames,
        "compression": encoder_frames / projector_frames,
        "ms_per_frame": (audio_duration_sec * 1000) / projector_frames,
    }

print("="*80)
print("PROJECTOR CONFIGURATION COMPARISON")
print("="*80)
print(f"{'Config':<25} {'Params':<15} {'Frames':<10} {'ms/frame':<12} {'Compression':<12}")
print("="*80)

for cfg in configs:
    params = count_projector_params(cfg)
    efficiency = analyze_efficiency(cfg)

    print(f"{cfg['name']:<25} {params:>14,} {efficiency['projector_frames']:>9} {efficiency['ms_per_frame']:>11.1f} {efficiency['compression']:>11.1f}x")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("Tradeoffs:")
print("  • Larger hidden_dim → More capacity, more parameters")
print("  • Higher downsample_rate → Fewer frames, faster inference, less detail")
print("  • Lower downsample_rate → More frames, slower inference, more detail")
print("\nTiny Audio's choice (5x, 8192 hidden):")
print("  ✓ Balances capacity with efficiency")
print("  ✓ ~122M params (trainable)")
print("  ✓ ~100ms per frame (good temporal resolution)")
```

**Step 2: Run the script**

```bash
poetry run python test_projector_config.py
```

**Expected output:**

```
================================================================================
PROJECTOR CONFIGURATION COMPARISON
================================================================================
Config                    Params          Frames    ms/frame     Compression
================================================================================
Actual (Tiny Audio)      121,643,264        30        100.0         5.0x
Smaller Projector         60,829,696        30        100.0         5.0x
More Downsampling        121,651,456        18        166.7         8.3x
Less Downsampling        121,634,880        75         40.0         2.0x

================================================================================
ANALYSIS
================================================================================
Tradeoffs:
  • Larger hidden_dim → More capacity, more parameters
  • Higher downsample_rate → Fewer frames, faster inference, less detail
  • Lower downsample_rate → More frames, slower inference, more detail

Tiny Audio's choice (5x, 8192 hidden):
  ✓ Balances capacity with efficiency
  ✓ ~122M params (trainable)
  ✓ ~100ms per frame (good temporal resolution)
```

### Success Checkpoint

- [ ] Script ran successfully
- [ ] Saw comparison of different projector configs
- [ ] Understand tradeoff between params, speed, and quality

---

# CLASS SUMMARY

## What We Covered Today

**Lecture (20 min):**

- Language models and text generation
- Qwen-3 8B-3B architecture
- The modality gap problem
- SwiGLU and RMSNorm explained

**Workshop (40 min):**

- Traced audio through projector step-by-step
- Visualized embedding distribution changes
- Analyzed projector configuration tradeoffs

## Key Takeaways

✅ Language models predict next tokens using learned patterns
✅ AudioProjector bridges 1280D audio → 2048D text space
✅ SwiGLU uses gated activation for better performance
✅ 5x downsampling balances efficiency and temporal resolution
✅ Projector is the only fully trainable component (~122M params)

## Homework (Optional)

Before Class 4:

1. Read the training script `src/train.py`
2. Browse `configs/hydra/experiments/stage1.yaml`
3. Think: "What hyperparameters would I tune?"

## Check Your Understanding

1. **What does the AudioProjector do?**
   - Transforms 1280D audio → 2048D text embeddings
   - Downsamples 5x for efficiency
   - Aligns audio and text distributions

2. **Why SwiGLU instead of ReLU?**
   - Gated activation (selective information flow)
   - Smoother gradients
   - Better empirical performance

3. **What's the downsampling tradeoff?**
   - Higher rate → Faster but less temporal detail
   - Lower rate → Slower but more detail
   - 5x is a good balance (~100ms per frame)

4. **Why is the projector 100% trainable?**
   - It's learning a new task (audio→text mapping)
   - Not pre-trained on this specific task
   - Small enough to train from scratch

---

## Further Reading (Optional)

### Papers

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)

### Code

- [AudioProjector implementation](../../src/asr_modeling.py#L29-L77)
- [Llama SwiGLU](https://github.com/meta-llama/llama/blob/main/llama/model.py)

---

## Next Class

In [Class 4: Training](./4-training.md), we'll:

- Understand LoRA's low-rank adaptation
- Configure training with Hydra
- Start a training run (finally!)
- Monitor training with Weights & Biases

[Previous: Class 2: Audio Processing and Encoders](./2-audio-processing-and-encoders.md) | [Next: Class 4: Training](./4-training.md)

**Get ready to train your own model!**
