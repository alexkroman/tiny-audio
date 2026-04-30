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


def test_glm_asr_encoder_matches_pt(glm_asr_pt):
    """End-to-end parity: load PT weights into our mlx-audio-based encoder
    and confirm forward outputs match within fp32 tolerance."""
    from mlx.utils import tree_unflatten

    from tiny_audio.mlx.encoder import (
        GLMASREncoder,
        encoder_config_from_hf,
        pt_encoder_state_to_mlx,
    )

    pt_full = glm_asr_pt
    pt_encoder = pt_full.audio_tower
    cfg = encoder_config_from_hf(pt_full.config.audio_config)

    mlx_encoder = GLMASREncoder(cfg)
    mlx_encoder.update(tree_unflatten(pt_encoder_state_to_mlx(pt_encoder.state_dict())))

    # Synthetic mel input (B=1, n_mels=128, T_mel=300 = 3s of audio)
    rng = np.random.default_rng(0)
    mel_np = rng.standard_normal((1, cfg.n_mels, 300)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_encoder(torch.from_numpy(mel_np)).last_hidden_state.detach().numpy()
    mlx_out = np.array(mlx_encoder(mx.array(mel_np)))

    assert mlx_out.shape == pt_out.shape, f"shape mismatch: {mlx_out.shape} vs {pt_out.shape}"
    np.testing.assert_allclose(mlx_out, pt_out, atol=2e-3, rtol=2e-3)
