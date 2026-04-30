"""MLX port of MLPAudioProjector (frame-stack + 2-layer MLP)."""

import mlx.core as mx
import mlx.nn as nn


class MLXMLPProjector(nn.Module):
    """Frame-stack (k=pool_stride) -> Linear -> RMSNorm -> GELU -> Linear."""

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        hidden_dim: int,
        pool_stride: int = 4,
    ):
        super().__init__()
        self.k = pool_stride
        in_dim = encoder_dim * pool_stride
        self.linear_1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, llm_dim, bias=False)

    def get_output_length(self, input_length: int) -> int:
        return (input_length - self.k) // self.k + 1

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq, dim = x.shape
        out_len = (seq - self.k) // self.k + 1
        x = x[:, : out_len * self.k, :]
        x = x.reshape(batch, out_len, dim * self.k)
        x = self.linear_1(x)
        x = self.norm(x)
        x = self.act(x)
        return self.linear_2(x)
