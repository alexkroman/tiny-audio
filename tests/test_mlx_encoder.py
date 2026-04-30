"""Equivalence test: MLX GLM-ASR encoder vs PT GLM-ASR encoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")


@pytest.fixture(scope="session")
def glm_asr_pt():
    """Session-scoped PT GLM-ASR encoder (fp32, eval mode)."""
    from transformers import AutoModelForSeq2SeqLM

    m = AutoModelForSeq2SeqLM.from_pretrained(
        "zai-org/GLM-ASR-Nano-2512", trust_remote_code=True, dtype=torch.float32
    )
    m.train(False)
    return m


def _copy_pt_encoder_to_mlx(pt_encoder, mlx_encoder):
    """Copy weights from PT GlmAsrEncoder to our MLX encoder.

    Walks pt_encoder.state_dict() and updates mlx_encoder via tree_unflatten + update.
    Both sides use identical parameter names.

    The only fix-up: PT Conv1d weights are [out, in, kernel]; MLX Conv1d weights are
    [out, kernel, in] (NLC channel layout). We swap axes 1 and 2 for conv1/conv2 weights.
    """
    from mlx.utils import tree_unflatten

    pt_sd = pt_encoder.state_dict()
    flat = []
    for k, v in pt_sd.items():
        arr = v.detach().cpu().numpy()
        if k in ("conv1.weight", "conv2.weight"):
            # PT (O, I, K) -> MLX (O, K, I)
            arr = np.swapaxes(arr, 1, 2)
        flat.append((k, mx.array(arr)))
    mlx_encoder.update(tree_unflatten(flat))


def test_glm_asr_encoder_matches_pt(glm_asr_pt):
    from tiny_audio.mlx.encoder import GLMASREncoder, encoder_config_from_hf

    pt_full = glm_asr_pt
    pt_encoder = pt_full.audio_tower
    cfg = encoder_config_from_hf(pt_full.config.audio_config)

    mlx_encoder = GLMASREncoder(cfg)
    _copy_pt_encoder_to_mlx(pt_encoder, mlx_encoder)

    # Synthetic mel input (B=1, n_mels=128, T_mel=300 = 3s of audio)
    rng = np.random.default_rng(0)
    mel_np = rng.standard_normal((1, cfg.n_mels, 300)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_encoder(torch.from_numpy(mel_np)).last_hidden_state.detach().numpy()
    mlx_out = np.array(mlx_encoder(mx.array(mel_np)))

    assert mlx_out.shape == pt_out.shape, f"shape mismatch: {mlx_out.shape} vs {pt_out.shape}"
    np.testing.assert_allclose(mlx_out, pt_out, atol=2e-3, rtol=2e-3)
