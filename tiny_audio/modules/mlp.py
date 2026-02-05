"""Flow matching MLP with adaptive layer normalization.

Adapted from pocket-tts, originally from:
https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/models/diffloss.py

Reference: https://arxiv.org/abs/2406.11838
"""

import math

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive normalization modulation."""
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        var = self.eps + x.var(dim=-1, keepdim=True)
        return (x * (self.alpha.to(var) * torch.rsqrt(var))).to(x_dtype)


class LayerNorm(nn.Module):
    """LayerNorm that supports JVP (for flow matching gradients)."""

    def __init__(self, channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if hasattr(self, "weight"):
            x = x * self.weight + self.bias
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            RMSNorm(hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t * self.freqs.to(t.dtype)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class ResBlock(nn.Module):
    """Residual block with adaptive layer normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.in_ln = LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """Final layer with adaptive normalization (DiT-style)."""

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    """MLP for flow matching with adaptive layer normalization.

    Takes conditioning from an AR transformer and predicts flow velocity.

    Args:
        in_channels: Input/output latent dimension (e.g., 256 for Mimi)
        model_channels: Hidden dimension of the MLP
        out_channels: Output dimension (same as in_channels for flow matching)
        cond_channels: Conditioning dimension from LLM
        num_res_blocks: Number of residual blocks
        num_time_conds: Number of time conditions (2 for start/end time in LSD)
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        num_time_conds: int = 2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds

        assert num_time_conds == 2, "LSD requires exactly 2 time conditions (start, end)"

        self.time_embed = nn.ModuleList(
            [TimestepEmbedder(model_channels) for _ in range(num_time_conds)]
        )
        self.cond_embed = nn.Linear(cond_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)

    def forward(
        self,
        c: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Predict flow velocity.

        Args:
            c: Conditioning from LLM, shape [N, cond_channels]
            s: Start time, shape [N, 1]
            t: Target time, shape [N, 1]
            x: Noisy latent, shape [N, in_channels]

        Returns:
            Predicted velocity, shape [N, out_channels]
        """
        x = self.input_proj(x)

        # Combine time embeddings (average of start and end time embeddings)
        ts = [s, t]
        t_combined = sum(self.time_embed[i](ts[i]) for i in range(self.num_time_conds))
        t_combined = t_combined / self.num_time_conds

        # Add conditioning
        c = self.cond_embed(c)
        y = t_combined + c

        # Residual blocks
        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)
