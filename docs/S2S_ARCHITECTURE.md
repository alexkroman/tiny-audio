# S2S (Speech-to-Speech) Architecture

This document describes the speech-to-speech pipeline for generating audio responses from audio input.

## Architecture Overview

```
Audio In → Whisper Encoder (frozen) → Projector (trained) → LLM 
                                                            ↓
                                                    [LLM hidden states]
                                                            ↓
                        Pre-NN (3 layers, bidirectional) ← input_proj
                                                            ↓
                            AR Decoder (6 layers, causal) → codebook 0
                                                            ↓
                                Depformer (4 layers) → codebooks 1-7
                                                            ↓
                                    Mimi Decoder (frozen) → Audio Out
```

## Training Stages

S2S training follows a multi-stage approach building on ASR:

### Stage 1: ASR Pre-training

Train the projector for speech recognition (transcription only):

- **Trained**: Projector
- **Frozen**: Encoder, LLM
- **Task**: Audio → Text transcription

### Stage 2: Omni Training (S2S)

Joint training of projector and audio head for speech-to-speech:

- **Trained**: Projector + Audio Head (Pre-NN, AR Decoder, Depformer)
- **Frozen**: Encoder, LLM, Mimi decoder
- **Task**: Audio → Text + Audio response

This allows the projector to adapt its representations for both transcription quality and downstream audio generation.

## Pipeline Steps

1. **Audio In** — Capture speech from microphone or file (16kHz sample rate)

1. **Understand Speech** — Whisper encoder (frozen) converts audio to mel spectrograms, then to encoder hidden states

1. **Bridge to Language** — Projector (trained) downsamples and projects encoder states to LLM embedding space via frame stacking

1. **Think** — LLM (frozen) processes the audio embeddings and generates a text response. The LLM's last-layer hidden states (from assistant token positions) are passed to the audio head

1. **Process Hidden States** — Pre-NN (3 bidirectional transformer layers) processes LLM hidden states with full attention to build contextualized representations

1. **Generate Semantic Codes** — AR Decoder (6 causal transformer layers) autoregressively generates Mimi codebook 0 (semantic tokens) with top-k sampling

1. **Generate Acoustic Codes** — Depformer (4 transformer layers) predicts codebooks 1-7 conditioned on AR hidden states and codebook 0, using acoustic delays for quality

1. **Audio Out** — Mimi decoder (frozen) converts all 8 codebooks to 24kHz audio waveform

## Audio Head Architecture

The Audio Head (`tiny_audio/audio_head.py`) is the trainable component for S2S:

| Component | Layers | Type | Purpose |
|-----------|--------|------|---------|
| input_proj | 1 | Linear | Project LLM dim → 1024 hidden dim |
| Pre-NN | 3 | Bidirectional Transformer | Context processing of LLM states |
| AR Decoder | 6 | Causal Transformer | Generate semantic codebook 0 |
| Depformer | 4 | Transformer | Predict acoustic codebooks 1-7 |

### Dimensions

- Hidden dim: 1024
- Intermediate dim: 4096 (FFN)
- Attention heads: 16
- Codec vocab size: 2048 (Mimi)
- Depformer hidden: 512
- Depformer heads: 8

### AR Decoder

Generates semantic tokens (codebook 0) autoregressively:

- Uses causal attention with KV caching
- Top-k sampling with temperature
- Repetition penalty over sliding window
- Special tokens: BOS, SOS, EOS, PAD (offsets from vocab_size)

### Depformer

Predicts acoustic codebooks 1-7 following Moshi's architecture:

- Processes all timesteps in parallel during inference
- Sequential codebook generation (cb1 → cb2 → ... → cb7)
- Uses **acoustic delays** for improved audio quality:
  - Codebook k at AR position t is for audio time t - k
  - Higher codebooks benefit from "future" semantic context
  - Delays are undone in post-processing to align with audio time

### Mimi Codec

The Kyutai Mimi codec (`kyutai/mimi`) handles audio encoding/decoding:

- Sample rate: 24kHz
- Frame rate: 12.5 Hz (each frame = 1920 samples)
- 8 codebooks (RVQ): codebook 0 = semantic, codebooks 1-7 = acoustic
- Vocab size: 2048 per codebook

## Training

**What's trained**: Projector + Audio Head (Pre-NN, AR Decoder, Depformer)

**What's frozen**: Whisper encoder, LLM, Mimi decoder

During omni training, both the projector and audio head are trained jointly. This allows the projector to learn representations that work well for both ASR and audio generation.

### Loss Computation

- Semantic loss: Cross-entropy on AR decoder predicting codebook 0
- Acoustic loss: Cross-entropy on Depformer predicting codebooks 1-7
- Total loss: `semantic_loss + acoustic_loss` (equal weighting)

### Training Data

Training requires paired data:

- Input: Audio waveform (16kHz) or mel spectrograms
- Targets: Mimi codec tokens `[batch, 8, seq_len]`

Use `audio_head.encode_audio()` to extract Mimi codes from audio for creating training data.

## Inference Methods

### Speech-to-Speech (`generate_with_audio`)

```python
result = model.generate_with_audio(audio, sampling_rate=16000)
text = result["text"]      # Transcription
audio = result["audio"]    # Response audio [batch, samples] @ 24kHz
```

### Text-to-Speech (`generate_speech`)

```python
result = model.generate_speech("Hello, how are you?")
text = result["text"]      # LLM response
audio = result["audio"]    # Spoken response @ 24kHz
```

## Configuration

Enable S2S in `ASRConfig`:

```python
config = ASRConfig(
    use_audio_head=True,           # Enable audio head
    max_audio_tokens=500,          # Max codec frames to generate
    audio_top_k=50,                # Top-k sampling
    audio_temperature=1.0,         # Sampling temperature
    audio_repetition_penalty=1.1,  # Penalty for repeated tokens
)
```

## File Structure

```
tiny_audio/
├── audio_head.py           # AudioHead class (main S2S component)
└── modules/
    ├── ar_decoder.py       # PreNN + CodecARDecoder
    └── depformer.py        # Depformer for acoustic codebooks
```

## References

- **Moshi** (Kyutai): Depformer architecture with acoustic delays
- **Freeze-Omni**: Pre-NN + AR decoder design pattern
- **Mimi** (Kyutai): Neural audio codec with RVQ
