"""Simple SwiGLU-based audio projector."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=False, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_gate = self.act(self.w1(x))
        x_val = self.w2(x)
        x = x_gate * x_val
        x = self.dropout(x)
        return self.w3(x)


class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = getattr(config, "projector_pool_stride", 4)
        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim
        hidden_dim = config.projector_hidden_dim
        if hidden_dim is None:
            hidden_dim = config.encoder_dim * 2

        dropout_rate = getattr(config, "projector_dropout", 0.0)
        self.use_post_norm = getattr(config, "projector_post_norm", False)

        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        self.ln_pre = LlamaRMSNorm(in_dim, eps=1e-6)
        self.proj1 = SwiGLU(in_dim, hidden_dim, hidden_dim, dropout=dropout_rate)
        self.proj2 = SwiGLU(hidden_dim, hidden_dim, out_dim, dropout=dropout_rate)
        self.ln_post = LlamaRMSNorm(out_dim, eps=1e-6) if self.use_post_norm else None
        self.layer_scale = nn.Parameter(torch.ones(out_dim) * 1e-4)
        self.output_dropout = nn.Dropout(dropout_rate)

        with torch.no_grad():
            std = getattr(config, "projector_init_std", 0.02)
            self.ln_pre.weight.data.fill_(1.0)
            if self.ln_post is not None:
                self.ln_post.weight.data.fill_(1.0)
            # Initialize first layer
            nn.init.normal_(self.proj1.w1.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj1.w2.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj1.w3.weight, mean=0.0, std=std)
            # Initialize second layer
            nn.init.normal_(self.proj2.w1.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj2.w2.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj2.w3.weight, mean=0.0, std=std)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        target_dtype = self.proj1.w1.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        remainder = seq_len % self.k
        if remainder:
            pad_len = self.k - remainder
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.contiguous().view(batch_size, -1, dim * self.k)
        x = self.ln_pre(x)
        x = self.proj1(x)
        x = self.proj2(x)
        if self.ln_post is not None:
            x = self.ln_post(x)
        # Apply layer scale for training stability
        x = x * self.layer_scale

        return self.output_dropout(x)
