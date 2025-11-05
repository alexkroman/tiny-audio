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

**Experiment Preview**: Later we'll test:

- Different decoder models (Qwen vs Llama vs Mistral)

- Various model sizes (1B, 3B, 8B parameters)

- Impact of decoder choice on accuracy


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

>

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Trace the projection process and experiment with dimensions

- **Exercise 2**: Visualize embeddings and test different activations

- **Exercise 3**: Compare projector configurations and decoder models

By the end, you'll understand how audio becomes language and how to optimize this bridge!

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


### Experimentation Time

**Experiment 1: Test different downsampling rates**

Add this to your script:


```python
# Test different downsampling rates
rates = [2, 3, 5, 8, 10]
for rate in rates:
    # Simulate downsampling
    frames_in = 149  # typical for 3-second audio
    frames_out = (frames_in + rate - 1) // rate  # ceiling division
    ms_per_frame = (3000 / frames_out)  # 3000ms total

    print(f"\nDownsample {rate}x:")
    print(f"  Frames: {frames_in} → {frames_out}")
    print(f"  Time resolution: {ms_per_frame:.1f}ms per frame")
    print(f"  LLM tokens to process: {frames_out}")

    # Memory calculation
    dim = 2048
    memory_mb = (frames_out * dim * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"  Memory for embeddings: {memory_mb:.2f} MB")


```

**Experiment 2: Compare activation functions**


```python
import torch.nn.functional as F

# Sample input
x = torch.randn(1, 30, 8192)

# Test different activations
activations = {
    "ReLU": F.relu,
    "GELU": F.gelu,
    "SiLU/Swish": F.silu,
    "Tanh": torch.tanh,
}

for name, act_fn in activations.items():
    output = act_fn(x)

    # Analyze sparsity (zeros)
    sparsity = (output == 0).float().mean().item()

    # Analyze dynamic range
    range_val = (output.max() - output.min()).item()

    print(f"\n{name}:")
    print(f"  Sparsity: {sparsity:.2%} zeros")
    print(f"  Range: {range_val:.2f}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")


```

**Questions to explore:**

- What's the optimal downsampling rate?

- How does activation choice affect gradient flow?

- Why is SwiGLU better than simple activations?

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


### Advanced Visualization Experiments

**Experiment 1: Compare before/after normalization**

Add this analysis:


```python
# Trace through projector with intermediate saves
intermediates = {}

with torch.no_grad():
    audio_emb = model.encoder(**inputs).last_hidden_state

    # Manual projector forward pass
    k = model.projector.k
    batch_size, seq_len, dim = audio_emb.shape

    # Stack frames
    remainder = seq_len % k
    if remainder:
        pad_len = k - remainder
        audio_emb = F.pad(audio_emb, (0, 0, 0, pad_len))
    stacked = audio_emb.contiguous().view(batch_size, -1, dim * k)
    intermediates['stacked'] = stacked.clone()

    # Pre-norm
    prenorm = model.projector.ln_pre(stacked)
    intermediates['prenorm'] = prenorm.clone()

    # SwiGLU
    gate = model.projector.gate_proj(prenorm)
    up = model.projector.up_proj(prenorm)
    activated = F.silu(gate) * up
    intermediates['swiglu'] = activated.clone()

    # Down projection
    down = model.projector.down_proj(activated)
    intermediates['down'] = down.clone()

    # Post-norm
    output = model.projector.ln_post(down)
    intermediates['final'] = output.clone()

# Plot all stages
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

stages = ['stacked', 'prenorm', 'swiglu', 'down', 'final']
for idx, stage in enumerate(stages):
    data = intermediates[stage].flatten().numpy()
    axes[idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{stage.capitalize()} (μ={data.mean():.3f}, σ={data.std():.3f})')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')

# Compare with target LLM distribution
llm_data = llm_emb.flatten().numpy()
axes[5].hist(llm_data, bins=50, alpha=0.7, color='gold', edgecolor='black')
axes[5].set_title(f'Target LLM (μ={llm_data.mean():.3f}, σ={llm_data.std():.3f})')

plt.tight_layout()
plt.savefig('projection_stages.png', dpi=150)
print("✓ Saved stage-by-stage visualization")


```

**Experiment 2: Analyze dimension importance**


```python
# Compute variance per dimension
with torch.no_grad():
    text_emb = model.projector(audio_emb)

    # Variance per dimension
    dim_variance = text_emb.var(dim=(0, 1))  # variance across batch and time

    # Sort dimensions by importance
    sorted_dims, indices = torch.sort(dim_variance, descending=True)

# Plot dimension importance
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(sorted_dims.numpy())
plt.xlabel('Dimension (sorted)')
plt.ylabel('Variance')
plt.title('Dimension Importance Distribution')
plt.yscale('log')

plt.subplot(1, 2, 2)
top_k = 100
plt.bar(range(top_k), sorted_dims[:top_k].numpy())
plt.xlabel('Top Dimensions')
plt.ylabel('Variance')
plt.title(f'Top {top_k} Most Important Dimensions')

plt.tight_layout()
plt.savefig('dimension_importance.png', dpi=150)

# Analysis
total_var = sorted_dims.sum().item()
top_100_var = sorted_dims[:100].sum().item()
print(f"\nTop 100 dims capture {100*top_100_var/total_var:.1f}% of variance")


```

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


### Decoder Model Comparison Experiment

**Create `compare_decoders.py`:**


```python
# Compare different decoder models
decoders = [
    {
        "name": "Qwen-3 8B (Actual)",
        "model_id": "Qwen/Qwen3-8B",
        "params": 8e9,
        "hidden_dim": 2048,
        "vocab_size": 151936,
    },
    {
        "name": "Llama-3 8B",
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "params": 8e9,
        "hidden_dim": 4096,
        "vocab_size": 128256,
    },
    {
        "name": "Mistral 7B",
        "model_id": "mistralai/Mistral-7B-v0.1",
        "params": 7e9,
        "hidden_dim": 4096,
        "vocab_size": 32000,
    },
    {
        "name": "Qwen-3 1.5B",
        "model_id": "Qwen/Qwen3-1.5B",
        "params": 1.5e9,
        "hidden_dim": 1536,
        "vocab_size": 151936,
    },
]

print("="*80)
print("DECODER MODEL COMPARISON FOR ASR")
print("="*80)
print(f"{'Model':<20} {'Params':<10} {'Hidden':<10} {'Vocab':<10} {'Projector Params':<20}")
print("="*80)

for decoder in decoders:
    # Calculate projector params for this decoder
    encoder_dim = 1280
    downsample = 5
    stacked_dim = encoder_dim * downsample
    hidden_proj = 8192
    llm_dim = decoder["hidden_dim"]

    # Projector params
    ln_params = stacked_dim + llm_dim
    gate_params = stacked_dim * hidden_proj
    up_params = stacked_dim * hidden_proj
    down_params = hidden_proj * llm_dim
    proj_params = ln_params + gate_params + up_params + down_params

    print(f"{decoder['name']:<20} {decoder['params']/1e9:.1f}B {decoder['hidden_dim']:<10} {decoder['vocab_size']:<10} {proj_params/1e6:.1f}M")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("Considerations for decoder choice:")
print("  • Larger models → Better language understanding")
print("  • Smaller hidden_dim → Smaller projector")
print("  • Vocab size → Affects tokenization efficiency")
print("  • Qwen-3 8B balances size, performance, and efficiency")

# Memory estimation
print("\n" + "="*80)
print("MEMORY REQUIREMENTS (LoRA fine-tuning)")
print("="*80)

for decoder in decoders[:3]:  # Top 3 models
    # LoRA memory estimate
    lora_rank = 64
    num_layers = 32
    hidden = decoder["hidden_dim"]

    # LoRA params (Q,K,V,O projections)
    lora_params = 4 * 2 * hidden * lora_rank * num_layers

    # Total trainable
    proj_params = 122e6  # Fixed projector
    encoder_lora = 2e6  # Fixed encoder LoRA
    total_trainable = proj_params + encoder_lora + lora_params

    # Memory estimate (params + gradients + optimizer states)
    memory_gb = (total_trainable * 4 * 3) / 1e9  # 4 bytes, 3x for Adam

    print(f"{decoder['name']:<20} ~{memory_gb:.1f} GB training memory")


```

**Questions to explore:**

- How does decoder size affect ASR accuracy?

- Is a larger decoder always better?

- What's the speed/accuracy tradeoff?

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

Before Class 4, experiment with:

1. **Projector Architecture Experiments**:
   - Calculate params for different hidden dimensions (4096, 8192, 16384)
   - Test different downsampling rates (2x, 5x, 10x)
   - Think: How would you modify the projector for streaming ASR?

2. **Decoder Exploration**:
   - Research different decoder models (Phi, Gemma, StableLM)
   - Compare tokenizer vocabularies
   - Calculate LoRA memory requirements for each

3. **Activation Function Study**:
   - Implement and test GLU, ReGLU, GeGLU variants
   - Measure gradient flow through different activations
   - Compare convergence speed in toy experiments

4. **Code Reading**:
   - Read the training script `src/train.py`
   - Browse `configs/hydra/experiments/stage1.yaml`
   - Find where the projector is initialized
   - Identify hyperparameters you'd want to tune

5. **Advanced Experiments**:
   - What if we used multiple projectors (ensemble)?
   - Can we make the downsampling rate learnable?
   - How would cross-attention compare to our concatenation approach?

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

[Previous: Class 2: Audio Processing and Encoders](./2-audio-processing-and-encoders.md) | [Next: Class 4: Training](./4-training.md)

**Get ready to train your own model!**
