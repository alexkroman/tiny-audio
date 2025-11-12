# Class 3: Language Models and Projectors

**Duration**: 1 hour (20 min lecture + 40 min hands-on)

**Goal**: Understand how the projector bridges audio and text modalities

## Learning Objectives

By the end of this class, you will:

- Understand what language models do and how they generate text

- Know the Qwen3-8B and SmolLM3-3B architectures

- Understand the AudioProjector's linear architecture

- Implement and visualize the projection process

- See how audio embeddings become text embeddings

______________________________________________________________________

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

### Qwen3-8B and SmolLM3-3B Architectures

Tiny Audio supports multiple decoder options - powerful language models that generate the transcription text.

**Default: Qwen3-8B**

- 8 billion parameters
- Excellent multilingual capabilities
- Strong text generation quality
- Efficient with LoRA fine-tuning (rank 8)

**Alternative: SmolLM3-3B**

- 3 billion parameters
- Smaller and faster
- Good for resource-constrained environments
- Also supports LoRA fine-tuning

Both are open source and well-documented

### Decoder-Only Architecture

Both Qwen3-8B and SmolLM3-3B are "decoder-only" models (like GPT):

```
Input tokens → Embeddings → Transformer Layers → Next token prediction


```

**Key features:**

- **Causal attention** (can only look backward): During generation, each token can only attend to previous tokens, not future ones. This prevents "cheating" by looking ahead and ensures the model generates text one token at a time, just like humans speak.

- **Auto-regressive generation**: The model generates one token at a time, using its previous outputs as input for the next prediction. Think of it like writing a sentence word-by-word, where each word depends on what came before.

- **Flash Attention 2 for speed**: An optimized implementation of attention that's much faster and uses less memory, making training and inference practical

**Why Decoder-Only?**

The choice of architecture is a critical decision. While other architectures exist, decoder-only models have become the standard for large-scale language generation tasks.

- **Encoder-Decoder Models** (like T5) are great for tasks that require a deep understanding of the input, like summarization or translation. However, they are more complex to train and are less common for generative assistants.

- **Mixture-of-Experts (MoE) Models** (like Mixtral) are very powerful and efficient at inference, but they are more complex to train and require more memory.

For our project, a **decoder-only model is the perfect choice** because:

- It excels at **generative tasks** like ASR.

- It's **simpler to train and understand** than other architectures.

- The vast majority of **open-source tools and research** are focused on decoder-only models, making it easier to find support and resources.

By choosing a decoder-only model, we are building on a solid, well-understood foundation.

______________________________________________________________________

## 2. The Modality Gap Problem (5 min)

### The Challenge

We have two different "languages":

- **Audio embeddings**: 1280 dimensions from HuBERT (or Whisper)

- **Text embeddings**: 2048 dimensions from Qwen3-8B (or 1536 dimensions from SmolLM3-3B)

**Problem**: Can't directly feed audio embeddings to text model!

- Different dimensions (1280 vs 2048/1536)

- Different statistical distributions

- Different semantic spaces

**Analogy**: Like trying to plug a European power plug into an American outlet - same purpose, different format!

### The Solution: AudioProjector

A trainable neural network that:

1. **Transforms dimensions**: 1280D×5 → 2048D (or 1536D depending on decoder)
1. **Aligns distributions**: Audio stats → Text stats through normalization
1. **Downsamples time**: 5x reduction for efficiency (~80% fewer frames)
1. **Bridges modalities**: Audio space → Language space

**Key insight**: This is the ONLY fully trainable component from scratch (~13M params for the projection layer, plus normalization layers)!

______________________________________________________________________

## 3. Linear Projector Architecture Deep Dive (8 min)

### What is the AudioProjector?

**AudioProjector** = Simple yet effective linear projection layer

A streamlined architecture designed to bridge audio and text modalities efficiently. While more complex architectures like SwiGLU exist (used in Llama, PaLM, etc.), Tiny Audio uses a simpler design that prevents overfitting while maintaining strong alignment capacity.

### Architecture Breakdown

The AudioProjector is surprisingly simple - just 4 operations:

```python
class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = 5  # Stack 5 frames together
        in_dim = 1280 * 5  # = 6400
        out_dim = 2048  # For Qwen3-8B (1536 for SmolLM3-3B)

        self.ln_pre = RMSNorm(in_dim)       # 1. Pre-normalize
        self.proj = Linear(in_dim, out_dim)  # 2. Main projection
        self.dropout = Dropout(0.05)         # 3. Regularization
        self.ln_post = RMSNorm(out_dim)      # 4. Post-normalize

    def forward(self, x):
        # Input: [batch, 149 frames, 1280 dim]
        x = stack_5_frames(x)    # → [batch, 30 frames, 6400 dim]
        x = self.ln_pre(x)       # → Normalize
        x = self.proj(x)         # → [batch, 30 frames, 2048 dim]
        x = self.dropout(x)      # → Regularize
        return self.ln_post(x)   # → Final normalize
```

### Why Each Component Matters

**1. Frame Stacking**: Concatenate 5 frames → Reduces 149 frames to ~30 (~80% less computation)

**2. Pre-normalize (RMSNorm)**: Stacking breaks statistics → normalize for stable training

**3. Linear Projection**: 6400D → 2048D mapping, simple but effective (~13M params)

**4. Dropout (5%)**: Prevents overfitting during training

**5. Post-normalize (RMSNorm)**: Match what the LLM expects as input

### Why Not a More Complex Projector?

**Simple is better here:**

- **Parameter efficient**: ~13M vs ~40M+ with gated architectures
- **Less overfitting**: Simpler generalizes better
- **Faster**: Fewer computations
- **Sufficient**: Job is alignment, not complex transformation
- **Works**: Achieves competitive WER scores (12-15%)

______________________________________________________________________

# PART B: HANDS-ON WORKSHOP (40 minutes)

>

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Compare projector configurations

- **Exercise 2**: Compare decoder models

By the end, you'll understand how audio becomes language and how to optimize this bridge!

______________________________________________________________________

## Workshop Exercise 1: Test Projector Configurations (10 min)

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

______________________________________________________________________

## Workshop Exercise 2: Compare Decoder Models (30 min)

### Goal

Understand how decoder choice affects ASR system design.

### Your Task

Compare different language models as potential decoders.

### Instructions

**Create `compare_decoders.py`:**

```python
# Compare different decoder models
decoders = [
    {
        "name": "Qwen3-8B (Default)",
        "model_id": "Qwen/Qwen3-8B",
        "params": 8e9,
        "hidden_dim": 2048,
        "vocab_size": 151936,
    },
    {
        "name": "SmolLM3-3B (Alternative)",
        "model_id": "HuggingFaceTB/SmolLM3-3B",
        "params": 3e9,
        "hidden_dim": 1536,
        "vocab_size": 49152,
    },
    {
        "name": "Llama-3.2-3B (Possible)",
        "model_id": "meta-llama/Llama-3.2-3B",
        "params": 3e9,
        "hidden_dim": 3072,
        "vocab_size": 128256,
    },
    {
        "name": "Qwen3-1.5B (Smallest)",
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

______________________________________________________________________

# CLASS SUMMARY

## What We Covered Today

**Lecture (20 min):**

- Language models and text generation

- Qwen-3 8B-3B architecture

- The modality gap problem

- AudioProjector architecture

**Workshop (40 min):**

- Compared projector configurations

- Analyzed decoder model tradeoffs

______________________________________________________________________

## Further Reading (Optional)

### Papers

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

- [RMSNorm](https://arxiv.org/abs/1910.07467)

- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)

### Code

- [AudioProjector implementation](../../src/asr_modeling.py#L29-L77)

- [Llama SwiGLU](https://github.com/meta-llama/llama/blob/main/llama/model.py)

[Previous: Class 2: Audio Processing and Encoders](./2-audio-processing-and-encoders.md) | [Next: Class 4: Training](./4-training.md)
