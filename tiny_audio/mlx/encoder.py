"""GLM-ASR-Nano-2512 audio encoder for MLX.

Wraps `mlx_audio.stt.models.glmasr.WhisperEncoder` plus the encoder's final
LayerNorm. The PT model `transformers.models.glmasr.GlmAsrEncoder` exposes
the final norm as `audio_tower.norm`; mlx-audio's `WhisperEncoder` does not
include it (it lives one level up in `AudioEncoder.layer_norm`), so we add
it here. Net effect: a single MLX module that mirrors `audio_tower`
parameter-for-parameter, ready for weight loading.

The internal encoder uses MLX-native ops:
- `mx.fast.scaled_dot_product_attention` for fused attention
- `nn.RoPE(head_dim // 2, traditional=False)` for Llama half-split partial RoPE
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.stt.models.glmasr.glmasr import WhisperConfig, WhisperEncoder


@dataclass
class EncoderConfig:
    """Slim config that survives JSON round-trip and feeds WhisperConfig."""

    n_mels: int
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    rope_theta: float = 10000.0


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
        rope_theta=float(rope_params.get("rope_theta", 10000.0)),
    )


class GLMASREncoder(nn.Module):
    """GLM-ASR-Nano-2512 audio encoder = mlx-audio WhisperEncoder + final LayerNorm.

    Input: mel features [B, n_mels, T_mel] (PT NCL convention; we permute
    internally to MLX's NLC). Output: [B, T_enc, hidden_size] where
    T_enc = ceil(T_mel / 2) due to conv2 stride=2.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = WhisperEncoder(
            WhisperConfig(
                d_model=cfg.hidden_size,
                encoder_attention_heads=cfg.num_heads,
                encoder_ffn_dim=cfg.intermediate_size,
                encoder_layers=cfg.num_layers,
                num_mel_bins=cfg.n_mels,
                rope_traditional=False,  # Llama half-split convention
            ),
            use_rope=True,
        )
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def __call__(self, mel: mx.array) -> mx.array:
        # mel arrives as [B, n_mels, T_mel] (PT convention). Permute to [B, T_mel, n_mels].
        x = mel.transpose(0, 2, 1)
        return self.norm(self.encoder(x))


# ----- PT -> MLX weight remapping --------------------------------------------


# PT GlmAsrEncoder uses Llama-style names (input_layernorm, mlp.fc1/fc2, o_proj).
# mlx-audio WhisperEncoder uses Whisper-style names (self_attn_layer_norm,
# fc1/fc2 flat, out_proj). The PT root has `norm` for the final LayerNorm.
def _remap_pt_key(k: str) -> str:
    if k.startswith("norm."):
        return "norm." + k[len("norm.") :]
    new = k.replace("input_layernorm", "self_attn_layer_norm")
    new = new.replace("post_attention_layernorm", "final_layer_norm")
    new = new.replace("mlp.fc1", "fc1")
    new = new.replace("mlp.fc2", "fc2")
    new = new.replace("self_attn.o_proj", "self_attn.out_proj")
    return "encoder." + new


def pt_encoder_state_to_mlx(pt_state_dict) -> list[tuple[str, mx.array]]:
    """Convert a PT GlmAsrEncoder state_dict to a list of (mlx_name, mx.array)
    suitable for `tree_unflatten` + `Module.update`. Handles Conv1d weight
    axis swap (PT [out, in, kernel] -> MLX [out, kernel, in])."""
    flat: list[tuple[str, mx.array]] = []
    for k, v in pt_state_dict.items():
        arr = v.detach().cpu().numpy()
        new_k = _remap_pt_key(k)
        if new_k.endswith("encoder.conv1.weight") or new_k.endswith("encoder.conv2.weight"):
            arr = np.swapaxes(arr, 1, 2)
        flat.append((new_k, mx.array(arr)))
    return flat


# ----- Mel preprocessing ------------------------------------------------------


def compute_mel_unpadded(
    audio: np.ndarray,
    *,
    feature_extractor=None,
    audio_model_id: str = "zai-org/GLM-ASR-Nano-2512",
    sampling_rate: int = 16000,
) -> tuple[mx.array, int]:
    """Compute log-mel spectrogram for a single audio array, no 30s padding.

    Uses transformers.WhisperFeatureExtractor so the mel is bit-exact to the
    PT inference pipeline. The encoder expects shape [B, n_mels, T_mel].

    Args:
        audio: 1D float32 numpy array, 16kHz mono.
        feature_extractor: pre-loaded extractor with `padding=False` already set.
            Pass this in hot paths to avoid the ~50-100ms `from_pretrained` cost.
        audio_model_id: HF repo id used when `feature_extractor` is None.
        sampling_rate: must be 16000 for Whisper-style 128-mel.

    Returns:
        (mel, mel_length): mel is mx.array [1, n_mels, T_mel_real];
        mel_length is the unpadded T_mel (== mel.shape[2]).
    """
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1D, got shape {audio.shape}")
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if feature_extractor is None:
        from transformers import AutoFeatureExtractor

        feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_id)

    out = feature_extractor(
        audio,
        sampling_rate=sampling_rate,
        return_attention_mask=True,
        return_tensors="np",
    )
    mel_np = out["input_features"]  # [1, n_mels, T_mel_padded]
    mel_length = int(out["attention_mask"].sum())
    mel_np = mel_np[:, :, :mel_length]  # truncate to real length
    return mx.array(mel_np), mel_length
