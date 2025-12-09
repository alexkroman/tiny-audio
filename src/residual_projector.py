"""Residual MLP projector for Whisper → LLM feature space translation.

Philosophy: Whisper features are already information-complete. The projector
learns a nonlinear correction/refinement to align them with the LLM's expected
input distribution, rather than replacing them entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class ResidualMLP(nn.Module):
    """MLP block with residual connection.

    Output = x + MLP(x)

    At initialization (weights near zero), output ≈ input, providing a stable
    starting point. The network learns to add nonlinear corrections as needed.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class ResidualAudioProjector(nn.Module):
    """Residual MLP projector for audio-to-LLM feature translation.

    Architecture:
        1. Temporal pooling (concatenate k consecutive frames)
        2. Linear projection to LLM dimension
        3. N residual MLP blocks for nonlinear refinement
        4. Final layer norm

    The linear projection handles dimension matching, while residual MLPs
    learn the nonlinear corrections needed to align acoustic features
    with semantic embedding space.
    """

    def __init__(self, config):
        super().__init__()

        # Temporal downsampling factor
        self.k = getattr(config, "projector_pool_stride", 4)

        # Dimensions
        in_dim = config.encoder_dim * self.k  # After concatenating k frames
        out_dim = config.llm_dim
        hidden_dim = getattr(config, "projector_hidden_dim", None) or out_dim * 4

        # Number of residual blocks
        self.num_layers = getattr(config, "projector_num_layers", 2)

        dropout_rate = getattr(config, "projector_dropout", 0.0)

        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        # Initial projection: encoder_dim * k → llm_dim
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.ln_input = LlamaRMSNorm(out_dim, eps=1e-6)

        # Residual MLP blocks for nonlinear refinement
        self.layers = nn.ModuleList(
            [ResidualMLP(out_dim, hidden_dim, dropout=dropout_rate) for _ in range(self.num_layers)]
        )

        # Per-layer norms (applied after each residual block)
        self.layer_norms = nn.ModuleList(
            [LlamaRMSNorm(out_dim, eps=1e-6) for _ in range(self.num_layers)]
        )

        self.output_dropout = nn.Dropout(dropout_rate)

        # Initialize for stable training
        self._init_weights(config)

    def _init_weights(self, config):
        """Initialize weights for stable residual learning.

        Key insight: Initialize fc2 of each residual block to near-zero
        so that initially output ≈ input (identity function).
        """
        std = getattr(config, "projector_init_std", 0.02)

        with torch.no_grad():
            # Input projection: standard init
            nn.init.normal_(self.input_proj.weight, mean=0.0, std=std)
            if self.input_proj.bias is not None:
                nn.init.zeros_(self.input_proj.bias)

            # Layer norms
            self.ln_input.weight.data.fill_(1.0)
            for ln in self.layer_norms:
                ln.weight.data.fill_(1.0)  # type: ignore[operator]

            # Residual blocks: small init on output projection
            for layer in self.layers:
                nn.init.normal_(layer.fc1.weight, mean=0.0, std=std)
                # Initialize fc2 smaller so residual starts near identity
                nn.init.normal_(layer.fc2.weight, mean=0.0, std=std * 0.1)
                if layer.fc1.bias is not None:
                    nn.init.zeros_(layer.fc1.bias)
                if layer.fc2.bias is not None:
                    nn.init.zeros_(layer.fc2.bias)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, encoder_dim] from Whisper encoder

        Returns:
            [batch_size, seq_len // k, llm_dim] projected features
        """
        batch_size, seq_len, dim = x.size()

        # Ensure correct dtype
        target_dtype = self.input_proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Pad sequence to be divisible by k
        remainder = seq_len % self.k
        if remainder:
            pad_len = self.k - remainder
            x = F.pad(x, (0, 0, 0, pad_len))

        # Temporal pooling: concatenate k consecutive frames
        # [B, T, D] → [B, T//k, D*k]
        x = x.contiguous().view(batch_size, -1, dim * self.k)

        # Project to LLM dimension
        x = self.input_proj(x)
        x = self.ln_input(x)

        # Apply residual MLP blocks
        for layer, ln in zip(self.layers, self.layer_norms):
            x = layer(x)
            x = ln(x)

        return self.output_dropout(x)
