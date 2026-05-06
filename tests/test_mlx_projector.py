"""Equivalence tests: MLX projector vs reference PyTorch projector."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from tests.conftest import MockProjectorConfig  # noqa: E402
from tiny_audio.mlx.projector import MLXMLPProjector  # noqa: E402
from tiny_audio.projectors import MLPAudioProjector  # noqa: E402


def _copy_pt_to_mlx(pt: MLPAudioProjector, mlx_proj: MLXMLPProjector) -> None:
    mlx_proj.linear_1.weight = mx.array(pt.linear_1.weight.detach().numpy())
    mlx_proj.norm.weight = mx.array(pt.norm.weight.detach().numpy())
    mlx_proj.linear_2.weight = mx.array(pt.linear_2.weight.detach().numpy())
    mlx_proj.norm_2.weight = mx.array(pt.norm_2.weight.detach().numpy())


def test_mlx_projector_matches_pt():
    cfg = MockProjectorConfig(
        encoder_dim=512,
        llm_dim=960,
        projector_pool_stride=4,
        projector_hidden_dim=512,
    )
    pt = MLPAudioProjector(cfg).train(False).to(torch.float32)
    mlx_proj = MLXMLPProjector(encoder_dim=512, llm_dim=960, hidden_dim=512, pool_stride=4)
    _copy_pt_to_mlx(pt, mlx_proj)

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((2, 100, 512)).astype(np.float32)

    pt_out = pt(torch.from_numpy(x_np)).detach().numpy()
    mlx_out = np.array(mlx_proj(mx.array(x_np)))

    np.testing.assert_allclose(mlx_out, pt_out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("L", [4, 5, 8, 100, 101, 999])
def test_mlx_projector_output_length_matches_pt(L: int):  # noqa: N803
    cfg = MockProjectorConfig(
        encoder_dim=512, llm_dim=960, projector_pool_stride=4, projector_hidden_dim=512
    )
    pt = MLPAudioProjector(cfg)
    mlx_proj = MLXMLPProjector(encoder_dim=512, llm_dim=960, hidden_dim=512, pool_stride=4)
    assert mlx_proj.get_output_length(L) == pt.get_output_length(L)
