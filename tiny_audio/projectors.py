"""Audio projector module for bridging encoder and decoder embeddings.

MLPAudioProjector: Simple 2-layer MLP with frame stacking downsampling.
"""

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class MLPAudioProjector(nn.Module):
    """2-layer MLP projector with frame-stacking downsampling (matches GLM-ASR)."""

    def __init__(self, config):
        """Initialize MLP projector.

        Args:
            config: ASRConfig with encoder_dim, llm_dim, projector_pool_stride
        """
        super().__init__()

        encoder_dim = getattr(config, "encoder_dim", 768)
        llm_dim = getattr(config, "llm_dim", 2048)
        self.k = getattr(config, "projector_pool_stride", 4)

        # Frame stacking: concat k adjacent frames then project
        in_dim = encoder_dim * self.k
        # Hidden dim defaults to llm_dim, can be overridden via config
        hidden_dim = getattr(config, "projector_hidden_dim", None) or llm_dim
        self.linear_1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm = LlamaRMSNorm(hidden_dim, eps=1e-6)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, llm_dim, bias=False)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length (matches GLM-ASR)."""
        # GLM-ASR formula: (L - merge_factor) // merge_factor + 1
        return (input_length - self.k) // self.k + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features to LLM embedding space.

        Args:
            x: Audio encoder output of shape [batch, seq_len, encoder_dim]

        Returns:
            Projected features of shape [batch, (seq_len - k) // k + 1, llm_dim]
        """
        batch, seq, dim = x.shape
        # Truncate to match GLM-ASR: use (seq - k) // k + 1 frames
        # This drops trailing frames that don't fill a complete k-frame window
        out_len = (seq - self.k) // self.k + 1
        x = x[:, : out_len * self.k, :]  # Truncate to exact multiple
        x = x.reshape(batch, out_len, dim * self.k)

        x = self.linear_1(x)
        x = self.norm(x)
        x = self.act(x)
        return self.linear_2(x)


PROJECTOR_CLASSES = {
    "mlp": MLPAudioProjector,
}
