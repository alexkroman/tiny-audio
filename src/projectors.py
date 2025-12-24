"""Audio projector modules for bridging encoder and decoder embeddings.

This module contains all projector architectures:
- MLPAudioProjector: Simple 2-layer MLP with conv downsampling
- MoEAudioProjector: MOSA-style dense mixture of experts
- SwiGLUAudioProjector: SwiGLU-based projector with temporal pooling
- ResidualAudioProjector: Residual MLP blocks with linear projection
- SharedMoEAudioProjector: Shared expert + sparse routed experts
- QFormerAudioProjector: BLIP-2 QFormer with learnable queries (Granite-style)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModel, Blip2QFormerConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# =============================================================================
# MLP Projector
# =============================================================================


class MLPAudioProjector(nn.Module):
    """2-layer MLP projector with conv-based 2x temporal downsampling."""

    def __init__(self, config):
        super().__init__()

        encoder_dim = getattr(config, "encoder_dim", 768)
        llm_dim = getattr(config, "llm_dim", 2048)

        self.downsample = nn.Conv1d(
            encoder_dim, encoder_dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.linear_1 = nn.Linear(encoder_dim, llm_dim, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_dim, llm_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        # Conv stride=2 halves the length (with padding=1, kernel=3)
        return (input_length + 1) // 2

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        Returns: [Batch, Seq_Len // 2, llm_dim]
        """
        # Conv1d expects [Batch, Channels, Seq_Len]
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)

        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)


# =============================================================================
# MoE Projector (MOSA-style)
# =============================================================================


class SimpleAdapter(nn.Module):
    """Simple adapter: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class MoEAudioProjector(nn.Module):
    """
    MOSA-style projector: Mixture of Simple Adapters.

    From paper (arXiv:2508.18998):
    - Dense mixture (softmax over ALL experts) instead of sparse Top-K
    - Simple Linear->ReLU->Linear adapters
    - No auxiliary losses - just cross-entropy on transcripts
    - Conv downsampling: stride 4 total (two conv layers, stride 2 each)
    """

    def __init__(self, config):
        super().__init__()

        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.num_experts = getattr(config, "num_experts", 4)
        adapter_hidden = getattr(config, "projector_hidden_dim", None) or 4096
        self.dropout_rate = getattr(config, "projector_dropout", 0.1)

        # Convolutional Subsampling (stride 4 total)
        self.conv = nn.Sequential(
            nn.Conv1d(self.encoder_dim, self.llm_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.llm_dim, self.llm_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Router
        router_hidden = 512
        self.router = nn.Sequential(
            nn.Linear(self.encoder_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, self.num_experts),
        )

        # Experts
        self.experts = nn.ModuleList(
            [
                SimpleAdapter(self.llm_dim, adapter_hidden, self.llm_dim, dropout=self.dropout_rate)
                for _ in range(self.num_experts)
            ]
        )

        self.ln_post = LlamaRMSNorm(self.llm_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        with torch.no_grad():
            for module in self.conv:
                if isinstance(module, nn.Conv1d):
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            for module in self.router:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            for expert in self.experts:
                nn.init.normal_(expert.fc1.weight, mean=0.0, std=std)
                nn.init.normal_(expert.fc2.weight, mean=0.0, std=std)
                if expert.fc1.bias is not None:
                    nn.init.zeros_(expert.fc1.bias)
                if expert.fc2.bias is not None:
                    nn.init.zeros_(expert.fc2.bias)

            self.ln_post.weight.data.fill_(1.0)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        # Two conv layers with stride=2 each = stride 4 total
        # Pad to multiple of 4, then divide by 4
        padded = input_length + (4 - input_length % 4) % 4
        return padded // 4

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Pad to be divisible by stride (4)
        pad_amt = (4 - (seq_len % 4)) % 4
        if pad_amt > 0:
            x = F.pad(x, (0, 0, 0, pad_amt))
            seq_len = x.shape[1]

        # Convolutional Downsampling
        h_conv = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Router on high-res input, then downsample weights
        router_logits = self.router(x)
        router_logits = router_logits.view(batch_size, seq_len // 4, 4, self.num_experts).mean(
            dim=2
        )
        routing_weights = F.softmax(router_logits, dim=-1)

        # Weighted sum of expert outputs
        final_out = torch.zeros_like(h_conv)
        for i, expert in enumerate(self.experts):
            expert_out = expert(h_conv)
            expert_weight = routing_weights[:, :, i : i + 1]
            final_out.add_(expert_out * expert_weight)

        return self.ln_post(final_out)

    def get_aux_loss(self) -> torch.Tensor:
        """Return auxiliary loss (none for dense MoE)."""
        return torch.tensor(0.0)


# =============================================================================
# SwiGLU Projector
# =============================================================================


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


class SwiGLUAudioProjector(nn.Module):
    """SwiGLU-based projector with temporal pooling."""

    def __init__(self, config):
        super().__init__()
        self.k = getattr(config, "projector_pool_stride", 4)
        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim
        hidden_dim = config.projector_hidden_dim
        if hidden_dim is None:
            hidden_dim = config.encoder_dim * 2

        dropout_rate = getattr(config, "projector_dropout", 0.0)

        self.proj1 = SwiGLU(in_dim, hidden_dim, hidden_dim, dropout=dropout_rate)
        self.proj2 = SwiGLU(hidden_dim, hidden_dim, out_dim, dropout=dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

        with torch.no_grad():
            std = getattr(config, "projector_init_std", 0.02)
            nn.init.normal_(self.proj1.w1.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj1.w2.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj1.w3.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj2.w1.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj2.w2.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj2.w3.weight, mean=0.0, std=std)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        # Temporal pooling with stride k
        remainder = input_length % self.k
        if remainder:
            input_length += self.k - remainder
        return input_length // self.k

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
        x = self.proj1(x)
        x = self.proj2(x)

        return self.output_dropout(x)


# Alias for backwards compatibility
AudioProjector = SwiGLUAudioProjector


# =============================================================================
# Residual Projector
# =============================================================================


class ResidualMLP(nn.Module):
    """MLP block with residual connection: Output = x + MLP(x)."""

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
    """Residual MLP projector for audio-to-LLM feature translation."""

    def __init__(self, config):
        super().__init__()

        self.k = getattr(config, "projector_pool_stride", 4)
        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim
        hidden_dim = getattr(config, "projector_hidden_dim", None) or out_dim * 4
        self.num_layers = getattr(config, "projector_num_layers", 2)
        dropout_rate = getattr(config, "projector_dropout", 0.0)

        self.input_proj = nn.Linear(in_dim, out_dim)
        self.ln_input = LlamaRMSNorm(out_dim, eps=1e-6)

        self.layers = nn.ModuleList(
            [ResidualMLP(out_dim, hidden_dim, dropout=dropout_rate) for _ in range(self.num_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [LlamaRMSNorm(out_dim, eps=1e-6) for _ in range(self.num_layers)]
        )

        self.output_dropout = nn.Dropout(dropout_rate)
        self._init_weights(config)

    def _init_weights(self, config):
        std = getattr(config, "projector_init_std", 0.02)

        with torch.no_grad():
            nn.init.normal_(self.input_proj.weight, mean=0.0, std=std)
            if self.input_proj.bias is not None:
                nn.init.zeros_(self.input_proj.bias)

            self.ln_input.weight.data.fill_(1.0)
            for ln in self.layer_norms:
                ln.weight.data.fill_(1.0)

            for layer in self.layers:
                nn.init.normal_(layer.fc1.weight, mean=0.0, std=std)
                nn.init.normal_(layer.fc2.weight, mean=0.0, std=std * 0.1)
                if layer.fc1.bias is not None:
                    nn.init.zeros_(layer.fc1.bias)
                if layer.fc2.bias is not None:
                    nn.init.zeros_(layer.fc2.bias)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        # Temporal pooling with stride k
        remainder = input_length % self.k
        if remainder:
            input_length += self.k - remainder
        return input_length // self.k

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        target_dtype = self.input_proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        remainder = seq_len % self.k
        if remainder:
            pad_len = self.k - remainder
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.contiguous().view(batch_size, -1, dim * self.k)
        x = self.input_proj(x)
        x = self.ln_input(x)

        for layer, ln in zip(self.layers, self.layer_norms):
            x = layer(x)
            x = ln(x)

        return self.output_dropout(x)


# =============================================================================
# Shared MoE Projector
# =============================================================================


class RMSNorm(nn.Module):
    """RMS Normalization (SOTA normalization for transformers)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed


class SwiGLUExpert(nn.Module):
    """SwiGLU expert MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Bias=False is strictly preferred for MoE experts to reduce memory/compute
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class SharedMoEBlock(nn.Module):
    """MoE block with Shared + Sigmoid-Routed Experts."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim

        # RMSNorm before routing
        self.norm = RMSNorm(input_dim)

        self.router = nn.Linear(input_dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        self.shared_expert = SwiGLUExpert(input_dim, hidden_dim, output_dim)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )

        self.last_router_logits = None
        self.last_router_probs = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape

        # 1. Apply Shared Expert
        normed_states = self.norm(hidden_states)
        shared_out = self.shared_expert(normed_states)

        # 2. Router Logic (Sigmoid Style)
        flat_hidden = normed_states.view(-1, dim)
        router_logits = self.router(flat_hidden)

        # Sigmoid routing
        router_probs = torch.sigmoid(router_logits)

        self.last_router_logits = router_logits
        self.last_router_probs = router_probs

        # 3. Top-K Selection
        top_k_scores, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize weights
        top_k_weights = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        # 4. Dispatch
        routed_out = self._dispatch_experts(flat_hidden, top_k_indices, top_k_weights)
        routed_out = routed_out.view(batch_size, seq_len, -1)

        return shared_out + routed_out

    def _dispatch_experts(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        output = torch.zeros(
            num_tokens, self.output_dim, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = top_k_indices == expert_idx
            if not expert_mask.any():
                continue

            token_indices, slot_indices = torch.where(expert_mask)
            expert_input = hidden_states[token_indices]
            expert_output = expert(expert_input).to(output.dtype)
            weights = top_k_weights[token_indices, slot_indices].unsqueeze(-1)
            output.index_add_(0, token_indices, expert_output * weights)

        return output


def load_balancing_loss(router_probs: torch.Tensor, num_experts: int, top_k: int) -> torch.Tensor:
    """Auxiliary loss to encourage balanced expert usage."""
    prob_per_expert = router_probs.mean(dim=0)
    target_mean = prob_per_expert.mean()
    return (prob_per_expert - target_mean).square().sum() * num_experts


def z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Z-loss to prevent router logits from growing too large."""
    return torch.logsumexp(router_logits.float(), dim=-1).square().mean()


class SharedMoEAudioProjector(nn.Module):
    """Shared expert + sparse routed experts projector."""

    def __init__(self, config):
        super().__init__()

        # Default stride is now 2 (was 4)
        self.k = getattr(config, "projector_pool_stride", 4)
        encoder_dim = config.encoder_dim

        # Depthwise Conv for temporal mixing
        self.temporal_conv = nn.Conv1d(
            encoder_dim, encoder_dim, kernel_size=3, padding=1, groups=encoder_dim
        )

        in_dim = encoder_dim * self.k
        out_dim = config.llm_dim
        hidden_dim = getattr(config, "projector_hidden_dim", None) or in_dim

        self.num_experts = getattr(config, "num_experts", 4)
        self.top_k = getattr(config, "num_experts_per_tok", 2)
        self.aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.02)
        self.z_loss_coef = getattr(config, "router_z_loss_coef", 0.001)

        self.moe = SharedMoEBlock(in_dim, hidden_dim, out_dim, self.num_experts, self.top_k)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.orthogonal_(self.moe.shared_expert.gate_proj.weight)
            nn.init.orthogonal_(self.moe.shared_expert.up_proj.weight)
            nn.init.orthogonal_(self.moe.shared_expert.down_proj.weight, gain=0.5)

            for expert in self.moe.experts:
                nn.init.orthogonal_(expert.gate_proj.weight)
                nn.init.orthogonal_(expert.up_proj.weight)
                nn.init.orthogonal_(expert.down_proj.weight, gain=0.01)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        # Temporal pooling with stride k
        if input_length % self.k:
            input_length += self.k - input_length % self.k
        return input_length // self.k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()

        target_dtype = self.moe.shared_expert.gate_proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Temporal Context Injection
        x_ctx = x.transpose(1, 2)
        x_ctx = self.temporal_conv(x_ctx)
        x = x + x_ctx.transpose(1, 2)

        if seq_len % self.k:
            x = F.pad(x, (0, 0, 0, self.k - seq_len % self.k))

        x = x.view(batch_size, -1, dim * self.k)

        return self.moe(x)

    def get_aux_loss(self) -> torch.Tensor:
        if self.moe.last_router_logits is None:
            return torch.tensor(0.0, device=self.moe.router.weight.device)

        balance = load_balancing_loss(self.moe.last_router_probs, self.num_experts, self.top_k)
        z = z_loss(self.moe.last_router_logits)

        return self.aux_loss_coef * balance + self.z_loss_coef * z


# =============================================================================
# QFormer Projector (Granite-style)
# =============================================================================


class QFormerAudioProjector(nn.Module):
    """
    BLIP-2 QFormer projector with learnable queries.

    Based on GraniteSpeechEncoderProjector - uses a QFormer model with learnable
    query embeddings to compress and project audio encoder outputs. The audio
    sequence is processed in windows and downsampled via cross-attention.
    """

    def __init__(self, config):
        super().__init__()

        encoder_dim = config.encoder_dim
        llm_dim = config.llm_dim

        # Window and downsampling parameters (Granite defaults: window=15, downsample=5)
        self.window_size = getattr(config, "qformer_window_size", 15)
        self.downsample_rate = getattr(config, "downsample_rate", 5)
        self.num_queries = self.window_size // self.downsample_rate

        # QFormer hidden size (matches encoder for cross-attention)
        qformer_hidden = getattr(config, "qformer_hidden_size", None) or encoder_dim
        qformer_num_layers = getattr(config, "qformer_num_layers", 2)
        qformer_num_heads = getattr(config, "qformer_num_heads", 16)
        qformer_intermediate = getattr(config, "qformer_intermediate_size", None) or (qformer_hidden * 4)

        # Learnable query embeddings (Granite uses std=1.0)
        self.query = nn.Parameter(torch.zeros(1, self.num_queries, qformer_hidden))
        self.query.data.normal_(mean=0.0, std=1.0)

        # Optional projection if encoder dim != qformer hidden
        if encoder_dim != qformer_hidden:
            self.encoder_proj = nn.Linear(encoder_dim, qformer_hidden, bias=False)
        else:
            self.encoder_proj = None

        # Configure QFormer to match Granite's exact config
        qformer_config = Blip2QFormerConfig(
            hidden_size=qformer_hidden,
            num_hidden_layers=qformer_num_layers,
            num_attention_heads=qformer_num_heads,
            intermediate_size=qformer_intermediate,
            encoder_hidden_size=qformer_hidden,
            cross_attention_frequency=1,
            # Granite-specific settings
            hidden_act="gelu",
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            initializer_range=0.02,
        )
        self.qformer = AutoModel.from_config(qformer_config)

        # Final projection to LLM dimension (Granite uses bias=True)
        self.linear = nn.Linear(qformer_hidden, llm_dim)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        # QFormer uses window-based processing with num_queries per window
        nblocks = math.ceil(input_length / self.window_size)
        return nblocks * self.num_queries

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, encoder_dim]

        Returns:
            projected: [batch_size, num_output_tokens, llm_dim]
        """
        batch_size, seq_len, dim = hidden_states.size()

        # Ensure float dtype for QFormer
        target_dtype = self.query.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)

        # Optional encoder projection
        if self.encoder_proj is not None:
            hidden_states = self.encoder_proj(hidden_states)

        # Compute number of windows and pad to fit
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        if pad > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad), "constant", 0)

        # Reshape to process each window: [batch*nblocks, window_size, dim]
        effective_batch = batch_size * nblocks
        hidden_states = hidden_states.view(effective_batch, self.window_size, -1)

        # Expand queries to match batch size
        query_embeds = self.query.expand(effective_batch, -1, -1)

        # QFormer cross-attention
        query_output = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=hidden_states,
            return_dict=True,
        )

        # Reshape back: [batch, nblocks * num_queries, hidden]
        output_tokens = nblocks * self.num_queries
        query_proj = query_output.last_hidden_state.view(batch_size, output_tokens, -1)

        # Project to LLM dimension
        return self.linear(query_proj)


# =============================================================================
# Projector Registry
# =============================================================================

PROJECTOR_CLASSES = {
    "mlp": MLPAudioProjector,
    "moe": MoEAudioProjector,
    "swiglu": SwiGLUAudioProjector,
    "residual": ResidualAudioProjector,
    "shared_moe": SharedMoEAudioProjector,
    "qformer": QFormerAudioProjector,
}
