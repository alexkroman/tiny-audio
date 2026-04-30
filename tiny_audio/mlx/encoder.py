"""MLX port of the GLM-ASR-Nano-2512 audio encoder.

Mirrors `transformers.models.glmasr.modeling_glmasr.GlmAsrEncoder` parameter-for-parameter
so PT state_dict keys map onto our modules without remapping (only Conv1d weight axes
need a swap due to MLX's NLC channel layout — handled by the test's weight-copy helper).

Architecture (verified against zai-org/GLM-ASR-Nano-2512):
- 2x Conv1d subsampler (n_mels=128 -> 1280, stride 1 then 2) + GELU
- 32 transformer layers, each:
    pre-LN (input_layernorm) -> self-attn (q/v/o have bias, k_proj does NOT) -> residual
    pre-LN (post_attention_layernorm) -> MLP (fc1+gelu+fc2) -> residual
- Llama-style partial RoPE on q/k inside attention. Only the first 32 of head_dim=64
  are rotated; remainder passes through. Uses `rotate_half` (the `[-x2, x1]` half-split
  convention), NOT the interleaved-pair convention from the older 4-bit modeling_audio.py.
- Final LayerNorm `norm`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class EncoderConfig:
    n_mels: int
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5


def encoder_config_from_hf(hf_audio_cfg) -> EncoderConfig:
    """Build EncoderConfig from a transformers GlmAsrEncoderConfig."""
    head_dim = getattr(hf_audio_cfg, "head_dim", None) or (
        hf_audio_cfg.hidden_size // hf_audio_cfg.num_attention_heads
    )
    rope_params = getattr(hf_audio_cfg, "rope_parameters", {}) or {}
    return EncoderConfig(
        n_mels=hf_audio_cfg.num_mel_bins,
        hidden_size=hf_audio_cfg.hidden_size,
        num_layers=hf_audio_cfg.num_hidden_layers,
        num_heads=hf_audio_cfg.num_attention_heads,
        head_dim=head_dim,
        intermediate_size=hf_audio_cfg.intermediate_size,
        partial_rotary_factor=rope_params.get("partial_rotary_factor", 0.5),
        rope_theta=rope_params.get("rope_theta", 10000.0),
        layer_norm_eps=1e-5,
    )


def _rope_cos_sin(seq_len: int, rotary_dim: int, theta: float) -> tuple[mx.array, mx.array]:
    """Compute Llama-style cos/sin for partial RoPE.

    Returns cos, sin each with shape [1, seq_len, rotary_dim], ready to broadcast over
    [B, n_heads, T, head_dim] after an unsqueeze on dim=1.

    rotary_dim is the number of head dims that get rotated (= head_dim * partial_rotary_factor).
    Following PT's `cat((freqs, freqs), dim=-1)`, the cos/sin are tiled across the rotary_dim.
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (mx.arange(0, half, dtype=mx.float32) * 2 / rotary_dim))
    # PT: arange(0, dim, 2) / dim -> here arange(0, half) * 2 / rotary_dim. Equivalent.
    pos = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(pos, inv_freq)  # [T, half]
    emb = mx.concatenate([freqs, freqs], axis=-1)  # [T, rotary_dim]
    cos = mx.cos(emb)[None, :, :]  # [1, T, rotary_dim]
    sin = mx.sin(emb)[None, :, :]
    return cos, sin


def _rotate_half(x: mx.array) -> mx.array:
    """Llama-style: split last dim in half, rotate to (-second, first)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_partial_rope(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    """Apply Llama-style partial RoPE to first `rotary_dim` of q/k.

    q, k: [B, n_heads, T, head_dim]
    cos, sin: [1, T, rotary_dim] -> broadcasts after unsqueezing dim=1 to [1, 1, T, rotary_dim]
    """
    cos_b = cos[:, None, :, :]  # [1, 1, T, rotary_dim]
    sin_b = sin[:, None, :, :]
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos_b) + (_rotate_half(q_rot) * sin_b)
    k_embed = (k_rot * cos_b) + (_rotate_half(k_rot) * sin_b)

    q_out = mx.concatenate([q_embed, q_pass], axis=-1)
    k_out = mx.concatenate([k_embed, k_pass], axis=-1)
    return q_out, k_out


class GlmAsrAttention(nn.Module):
    """Multi-head self-attention with partial RoPE on q/k.

    Note bias asymmetry mirroring the PT model:
    - q_proj, v_proj, o_proj: bias=True
    - k_proj:                 bias=False
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, inner, bias=True)
        self.k_proj = nn.Linear(hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(hidden_size, inner, bias=True)
        self.o_proj = nn.Linear(inner, hidden_size, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        b, t, _ = x.shape
        # Project & reshape to [B, n_heads, T, head_dim]
        q = self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k = _apply_partial_rope(q, k, cos, sin)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        # PT eager_attention_forward casts softmax to float32 then back.
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out)


class GlmAsrMLP(nn.Module):
    """fc1 + GELU + fc2, both Linear with bias."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class GlmAsrEncoderLayer(nn.Module):
    """One Llama-style encoder block with pre-norms."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        h = cfg.hidden_size
        self.self_attn = GlmAsrAttention(h, cfg.num_heads, cfg.head_dim)
        self.mlp = GlmAsrMLP(h, cfg.intermediate_size)
        self.input_layernorm = nn.LayerNorm(h, eps=cfg.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(h, eps=cfg.layer_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class GLMASREncoder(nn.Module):
    """GLM-ASR-Nano-2512 audio encoder, MLX port.

    Input: mel features [B, n_mels, T_mel] (PT NCL convention — we permute internally
    to MLX's NLC).
    Output: [B, T_enc, hidden_size] where T_enc = ceil(T_mel / 2) due to conv2 stride=2.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        # MLX Conv1d uses NLC: weight shape [out, kernel, in]. PT uses [out, in, kernel].
        # The weight-copy helper in tests handles the swap.
        self.conv1 = nn.Conv1d(
            cfg.n_mels, cfg.hidden_size, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv1d(
            cfg.hidden_size, cfg.hidden_size, kernel_size=3, stride=2, padding=1, bias=True
        )
        self.layers = [GlmAsrEncoderLayer(cfg) for _ in range(cfg.num_layers)]
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, mel: mx.array) -> mx.array:
        # mel arrives as [B, n_mels, T_mel] (PT convention). Permute to [B, T_mel, n_mels].
        x = mel.transpose(0, 2, 1)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        # x: [B, T_enc, hidden_size]
        rotary_dim = int(self.cfg.head_dim * self.cfg.partial_rotary_factor)
        cos, sin = _rope_cos_sin(x.shape[1], rotary_dim, self.cfg.rope_theta)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


def compute_mel_unpadded(
    audio: np.ndarray,
    *,
    audio_model_id: str = "zai-org/GLM-ASR-Nano-2512",
    sampling_rate: int = 16000,
) -> tuple[mx.array, int]:
    """Compute log-mel spectrogram for a single audio array, no 30s padding.

    Uses transformers.WhisperFeatureExtractor (loaded from the audio model's
    HF repo) so the mel is bit-exact to the PT inference pipeline. The encoder
    in this module expects shape [B, n_mels, T_mel].

    Args:
        audio: 1D float32 numpy array, 16kHz mono.
        audio_model_id: HF repo ID for the feature extractor. Default
            "zai-org/GLM-ASR-Nano-2512" (128-mel).
        sampling_rate: must be 16000.

    Returns:
        (mel, mel_length): mel is an mx.array of shape [1, n_mels, T_mel];
        mel_length is the unpadded T_mel from the extractor's attention_mask.
    """
    from transformers import AutoFeatureExtractor

    if audio.ndim != 1:
        raise ValueError(f"audio must be 1D, got shape {audio.shape}")
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    fe = AutoFeatureExtractor.from_pretrained(audio_model_id)
    fe.padding = False
    out = fe(
        audio,
        sampling_rate=sampling_rate,
        return_attention_mask=True,
        return_tensors="np",
    )
    mel_np = out["input_features"]  # [1, n_mels, T_mel]
    mel_length = int(out["attention_mask"].sum())
    return mx.array(mel_np), mel_length
