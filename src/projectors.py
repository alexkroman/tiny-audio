"""Audio projector modules for bridging encoder and decoder embeddings.

This module contains all projector architectures:
- MLPAudioProjector: Simple 2-layer MLP with frame stacking downsampling
- MOSAProjector: MOSA-style dense mixture of experts
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
    """2-layer MLP projector with frame-stacking downsampling (matches GLM-ASR)."""

    def __init__(self, config):
        super().__init__()

        encoder_dim = getattr(config, "encoder_dim", 768)
        llm_dim = getattr(config, "llm_dim", 2048)
        self.k = getattr(config, "projector_pool_stride", 4)

        # Frame stacking: concat k adjacent frames then project
        # Matches GLM-ASR: in_dim -> 2*llm_dim -> llm_dim
        in_dim = encoder_dim * self.k
        hidden_dim = llm_dim * 2
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, llm_dim)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        return input_length // self.k

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        Returns: [Batch, Seq_Len // k, llm_dim]
        """
        batch, seq, dim = x.shape
        # Reshape to combine k frames: [B, S, D] -> [B, -1, D*k]
        # -1 infers sequence length, implicitly downsampling by factor k
        x = x.reshape(batch, -1, dim * self.k)

        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)


# =============================================================================
# MoE Projector (MOSA-style)
# =============================================================================


class SimpleAdapter(nn.Module):
    """Simple 2-layer ReLU adapter (from MOSA paper)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUExpert(nn.Module):
    """SwiGLU expert (gated MLP with SiLU activation)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MOSAProjector(nn.Module):
    """MOSA-Base projector: simple 2-layer router with 4 simple adapters.

    Based on "MOSA: Mixtures of Simple Adapters" (arXiv:2508.18998).
    Uses softmax gating over all experts (dense MoE) with only cross-entropy loss.
    Uses frame-stacking for downsampling (like MLP projector).
    """

    def __init__(self, config):
        super().__init__()
        self.encoder_dim = getattr(config, "encoder_dim", None) or 1280
        self.llm_dim = getattr(config, "llm_dim", None) or 2048
        self.k = getattr(config, "projector_pool_stride", 4)
        self.num_experts = getattr(config, "num_experts", None) or 4  # MOSA-Base uses 4
        adapter_hidden = getattr(config, "adapter_hidden_dim", None) or 4096

        # Frame stacking: concat k adjacent frames then project
        in_dim = self.encoder_dim * self.k

        # --- 1. Simple Router (MOSA-Base: 2 layers with ReLU) ---
        # Maps encoder_dim -> 512 -> num_experts
        router_hidden = getattr(config, "router_hidden_dim", None) or 512
        self.router = nn.Sequential(
            nn.Linear(self.encoder_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, self.num_experts),
        )

        # --- 2. Experts (Simple 2-layer ReLU adapters per MOSA paper) ---
        # Each expert: in_dim (stacked frames) -> hidden -> llm_dim
        self.experts = nn.ModuleList(
            [
                SimpleAdapter(in_dim, adapter_hidden, self.llm_dim)
                for _ in range(self.num_experts)
            ]
        )

        # Using PyTorch default initialization (like MOSA paper)

    def forward(self, x):
        # x: (B, S, encoder_dim)
        batch_size, seq_len, dim = x.shape

        # --- 1. Router Branch ---
        # Mean pool encoder outputs for routing decisions
        x_pooled = x.reshape(batch_size, -1, self.k, self.encoder_dim).mean(dim=2)  # (B, S//k, D)

        # Router logits and softmax gating (dense MoE)
        routing_weights = F.softmax(self.router(x_pooled), dim=-1)  # (B, S//k, num_experts)

        # --- 2. Frame stacking for experts ---
        # Reshape to combine k frames: [B, S, D] -> [B, S//k, D*k]
        x_stacked = x.reshape(batch_size, -1, dim * self.k)

        # --- 3. Expert Mixture (Dense Execution) ---
        # Run all experts and compute weighted sum
        expert_outputs = torch.stack([expert(x_stacked) for expert in self.experts])  # (E, B, S//k, D)
        return torch.einsum("ebsd, bse -> bsd", expert_outputs, routing_weights)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        return input_length // self.k


# =============================================================================
# MoE Projector (Shared Expert + Sparse Routed Experts)
# =============================================================================


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
        self.norm = LlamaRMSNorm(input_dim, eps=1e-8)

        self.router = nn.Linear(input_dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        self.shared_expert = SimpleAdapter(input_dim, hidden_dim, output_dim)
        self.experts = nn.ModuleList(
            [SimpleAdapter(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
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


class MoEAudioProjector(nn.Module):
    """MoE projector with shared expert + sparse routed experts."""

    def __init__(self, config):
        super().__init__()

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
            nn.init.orthogonal_(self.moe.shared_expert.fc1.weight)
            nn.init.orthogonal_(self.moe.shared_expert.fc2.weight, gain=0.5)

            for expert in self.moe.experts:
                nn.init.orthogonal_(expert.fc1.weight)
                nn.init.orthogonal_(expert.fc2.weight, gain=0.01)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        # Temporal pooling with stride k
        if input_length % self.k:
            input_length += self.k - input_length % self.k
        return input_length // self.k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()

        target_dtype = self.moe.shared_expert.fc1.weight.dtype
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
        qformer_intermediate = getattr(config, "qformer_intermediate_size", None) or (
            qformer_hidden * 4
        )

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
    "mosa": MOSAProjector,
    "moe": MoEAudioProjector,
    "qformer": QFormerAudioProjector,
}
