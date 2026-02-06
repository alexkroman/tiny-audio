# Speech-to-Speech Architecture

Technical architecture for the speech-to-speech (S2S) model. Only the Projector and Audio Head are trained; all other components are frozen.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AUDIO INPUT                                    │
│                         Raw waveform @ 16kHz                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WHISPER FEATURE EXTRACTOR                            │
│                                                                             │
│  Input:  Raw audio [batch, samples]                                         │
│  Output: Mel spectrogram [batch, 128, mel_frames]                           │
│                                                                             │
│  - 128 mel bins, 80ms window, 10ms hop                                      │
│  - Returns attention_mask for variable-length audio                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WHISPER ENCODER (frozen)                           │
│                                                                             │
│  Input:  Mel spectrogram [batch, 128, mel_frames]                           │
│  Output: Hidden states [batch, encoder_frames, 1280]                        │
│                                                                             │
│  Architecture: whisper-large-v3-turbo                                       │
│  - 2 Conv1d layers: (k=3,s=1,p=1) → (k=3,s=2,p=1)                           │
│  - 32 transformer encoder layers                                            │
│  - Output: encoder_frames ≈ mel_frames // 2                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROJECTOR (trained)                               │
│                                                                             │
│  Input:  Encoder hidden states [batch, encoder_frames, 1280]                │
│  Output: LLM embeddings [batch, audio_tokens, 3072]                         │
│                                                                             │
│  Variants:                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ MLP (default)                                                          │ │
│  │ - Frame stacking: concat k=5 adjacent frames → [batch, T/5, 1280*5]    │ │
│  │ - Linear(6400, 3072) → RMSNorm → GELU → Linear(3072, 3072)             │ │
│  │ - Output length: (encoder_frames - k) // k + 1                         │ │
│  │ - ~12M parameters                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ MOSA (Mixture of Simple Adapters)                                      │ │
│  │ - Conv1d downsampling: 2 layers, stride 2 each → 4x reduction          │ │
│  │ - Router: Linear(3072, 512) → ReLU → Linear(512, 4)                    │ │
│  │ - 4 experts: Linear(3072, 4096) → GELU → Linear(4096, 3072)            │ │
│  │ - Dense mixture: softmax routing, all experts contribute               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ MoE (Sparse with Shared Expert)                                        │ │
│  │ - Frame stacking + RMSNorm (same as MLP)                               │ │
│  │ - Router: Linear(6400, 4) with jitter noise                            │ │
│  │ - Shared expert: always active                                         │ │
│  │ - 4 sparse experts: top-2 routing with load balancing loss             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ QFormer (BLIP-2 style)                                                 │ │
│  │ - Window-based processing: window=15, downsample=5                     │ │
│  │ - Learnable queries: 3 queries per window                              │ │
│  │ - Cross-attention to encoder features                                  │ │
│  │ - Linear projection to LLM dim                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SmolLM3-3B (frozen)                              │
│                                                                             │
│  Input:  Token embeddings [batch, seq_len, 3072]                            │
│          - Audio embeddings injected at <audio> token positions             │
│          - Text tokens embedded normally                                    │
│  Output: Hidden states [batch, seq_len, 3072]                               │
│                                                                             │
│  Architecture:                                                              │
│  - 36 transformer decoder layers                                            │
│  - 3072 hidden dim, 24 attention heads                                      │
│  - RoPE positional encoding                                                 │
│  - Greedy decoding (do_sample=False)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AUDIO HEAD (trained)                               │
│                                                                             │
│  Input:  LLM hidden states [batch, response_len, 3072]                      │
│  Output: Mimi codec tokens [batch, 8, codec_frames]                         │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input Projection                                                       │ │
│  │ Linear(3072, 1024, bias=False)                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Pre-NN (bidirectional context processing)                              │ │
│  │ - 3 LlamaDecoderLayer blocks (SDPA attention)                          │ │
│  │ - 1024 hidden dim, 16 heads, 4096 FFN                                  │ │
│  │ - Bidirectional attention (no causal mask)                             │ │
│  │ - RoPE positional encoding                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ AR Decoder (semantic codebook 0)                                       │ │
│  │ - 6 LlamaDecoderLayer blocks (SDPA attention)                          │ │
│  │ - 1024 hidden dim, 16 heads, 4096 FFN                                  │ │
│  │ - Causal attention with cross-attention to Pre-NN output               │ │
│  │ - Vocabulary: 2048 codec + 4 special tokens (BOS, SOS, EOS, PAD)       │ │
│  │ - Embedding + output projection (tied weights)                         │ │
│  │ - Outputs: semantic tokens + hidden states for Depformer               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Depformer (acoustic codebooks 1-7)                                     │ │
│  │ - 4 LlamaDecoderLayer blocks (eager attention)                         │ │
│  │ - 512 hidden dim, 8 heads, 2048 FFN                                    │ │
│  │ - Input: AR hidden states + previous codebook embeddings               │ │
│  │ - 7 codebook embeddings: Embedding(2048, 512) each                     │ │
│  │ - 7 output projections: Linear(512, 2048) each                         │ │
│  │ - Acoustic delays: CB_k delayed by k timesteps (Moshi-style)           │ │
│  │ - Sequential generation: CB1 → CB2 → ... → CB7                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MIMI DECODER (frozen)                              │
│                                                                             │
│  Input:  Codec tokens [batch, 8, codec_frames]                              │
│  Output: Audio waveform [batch, samples] @ 24kHz                            │
│                                                                             │
│  Architecture: kyutai/mimi                                                  │
│  - 8 RVQ codebooks, 2048 codes each                                         │
│  - 12.5 Hz frame rate (1 frame = 1920 samples at 24kHz)                     │
│  - Neural audio codec with semantic + acoustic quantization                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AUDIO OUTPUT                                   │
│                         Raw waveform @ 24kHz                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Dimensions

| Component | Input Shape | Output Shape | Parameters |
|-----------|-------------|--------------|------------|
| Whisper Encoder | [B, 128, T_mel] | [B, T_mel//2, 1280] | 809M (frozen) |
| MLP Projector | [B, T_enc, 1280] | [B, T_enc//5, 3072] | ~12M |
| SmolLM3-3B | [B, T_seq, 3072] | [B, T_seq, 3072] | 3B (frozen) |
| Audio Head | [B, T_resp, 3072] | [B, 8, T_codec] | ~50M |
| Mimi Decoder | [B, 8, T_codec] | [B, T_codec\*1920] | 100M (frozen) |
