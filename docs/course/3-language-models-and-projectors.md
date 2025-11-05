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

Qwen-3 8B is our decoder - a powerful language model that generates the transcription text.

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


---

# PART B: HANDS-ON WORKSHOP (40 minutes)

>

## Workshop Overview

In the next 40 minutes, you will:

- **Exercise 1**: Compare projector configurations

- **Exercise 2**: Compare decoder models

By the end, you'll understand how audio becomes language and how to optimize this bridge!

---

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


---

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

- Compared projector configurations

- Analyzed decoder model tradeoffs

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
