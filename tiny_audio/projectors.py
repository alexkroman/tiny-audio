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


# =============================================================================
# MoE Projector (MOSA-style)
# =============================================================================


class SimpleAdapter(nn.Module):
    """Simple 2-layer GELU adapter (from MOSA paper)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear units (used in LLaMA, Mistral, etc.)."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)  # Value
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)  # Output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class AsymmetricSwiGLU(nn.Module):
    """SwiGLU that handles different input and output dimensions."""

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, bias: bool = False
    ):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)  # Gate
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)  # Value
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)  # Output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MOSAProjector(nn.Module):
    """MOSA-Base projector: simple 2-layer ReLU router with 4 simple adapters.

    Based on "MOSA: Mixtures of Simple Adapters" (arXiv:2508.18998).
    Uses softmax gating over all experts (dense MoE) with only cross-entropy loss.
    Uses Conv1d for downsampling (2 layers, stride 2 each = 4x total).
    """

    def __init__(self, config):
        """Initialize MOSA projector.

        Args:
            config: ASRConfig with encoder_dim, llm_dim, num_experts
        """
        super().__init__()
        self.encoder_dim = getattr(config, "encoder_dim", None) or 1280
        self.llm_dim = getattr(config, "llm_dim", None) or 2048
        self.num_experts = getattr(config, "num_experts", None) or 4  # MOSA-Base uses 4
        adapter_hidden = getattr(config, "adapter_hidden_dim", None) or 4096
        router_hidden = getattr(config, "router_hidden_dim", None) or 512

        # --- 1. Conv1d Downsampler (4x reduction) ---
        # 2 layers of stride-2 convolution
        self.downsampler = nn.Sequential(
            nn.Conv1d(self.encoder_dim, self.encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(self.encoder_dim, self.llm_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # --- 2. Simple Router (MOSA-Base: 2 layers with ReLU) ---
        # Takes downsampled features (llm_dim) -> 512 -> num_experts
        self.router = nn.Sequential(
            nn.Linear(self.llm_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, self.num_experts),
        )

        # --- 3. Experts (Simple 2-layer GELU adapters) ---
        # Each expert: llm_dim -> hidden -> llm_dim (much smaller than frame-stacking)
        self.experts = nn.ModuleList(
            [
                SimpleAdapter(self.llm_dim, adapter_hidden, self.llm_dim)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features using mixture of experts.

        Args:
            x: Audio encoder output of shape [batch, seq_len, encoder_dim]

        Returns:
            Projected features of shape [batch, out_len, llm_dim]
        """
        # --- 1. Conv1d Downsampling ---
        # Permute for Conv1d: [B, S, D] -> [B, D, S]
        x = x.transpose(1, 2)
        x = self.downsampler(x)
        # Permute back: [B, D, S] -> [B, S, D]
        x = x.transpose(1, 2)

        # --- 2. Routing ---
        routing_weights = F.softmax(self.router(x), dim=-1)  # (B, out_len, num_experts)

        # --- 3. Expert Mixture (Dense Execution) ---
        expert_outputs = torch.stack([expert(x) for expert in self.experts])  # (E, B, out_len, D)
        return torch.einsum("ebsd, bse -> bsd", expert_outputs, routing_weights)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length after Conv1d downsampling (4x reduction)."""
        # Conv1d with stride 2, kernel 3, padding 1: out = (in + 2*1 - 3) // 2 + 1 = (in - 1) // 2 + 1
        # Applied twice for 4x total reduction
        after_conv1 = (input_length + 2 * 1 - 3) // 2 + 1
        return (after_conv1 + 2 * 1 - 3) // 2 + 1


# =============================================================================
# MoE Projector (Pure PyTorch with Shared Expert)
# =============================================================================


class MoEAudioProjector(nn.Module):
    """MoE projector with shared expert (DeepSeek-style), pure PyTorch implementation.

    Uses 4 sparse experts with top-2 routing plus a shared expert that processes all tokens.
    No external dependencies (megablocks removed).

    Architecture matches main branch: norm → experts(in_dim → hidden → out_dim)
    """

    def __init__(self, config):
        """Initialize MoE projector.

        Args:
            config: ASRConfig with encoder_dim, llm_dim, num_experts, num_experts_per_tok
        """
        super().__init__()

        self.k = getattr(config, "projector_pool_stride", 4)
        self.aux_coef = getattr(config, "router_aux_loss_coef", 0.01)

        # Stability coefficients
        self.router_z_loss_coef = getattr(
            config, "router_z_loss_coef", 1e-4
        )  # Prevents logit explosion
        self.router_jitter_noise = getattr(
            config, "router_jitter_noise", 0.01
        )  # Prevents expert collapse

        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim

        # Expert hidden dim (default = output dim)
        hidden_dim = getattr(config, "projector_hidden_dim", None) or out_dim

        # Number of experts and top-k selection
        self.num_experts = getattr(config, "num_experts", 4)
        self.top_k = getattr(config, "num_experts_per_tok", 2)

        # A. Normalize stacked input (like main branch SharedMoEBlock)
        self.norm = LlamaRMSNorm(in_dim, eps=1e-6)

        # B. Router (operates on stacked input)
        self.router = nn.Linear(in_dim, self.num_experts, bias=False)

        # C. Experts: simple 2-layer MLP (same as MLPAudioProjector)
        self.experts = nn.ModuleList(
            [SimpleAdapter(in_dim, hidden_dim, out_dim) for _ in range(self.num_experts)]
        )

        # D. Shared Expert (same architecture)
        self.shared_expert = SimpleAdapter(in_dim, hidden_dim, out_dim)

        # E. Initialize weights for stable training
        self._init_weights()

        self.last_aux_loss = torch.tensor(0.0)

    def _init_weights(self):
        """Initialize weights for stable training start."""
        with torch.no_grad():
            # Router: small weights -> uniform probability
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

            # Experts: xavier for fc1, small for fc2 (output)
            for expert in [self.shared_expert, *self.experts]:
                nn.init.xavier_uniform_(expert.fc1.weight)
                nn.init.normal_(expert.fc2.weight, mean=0.0, std=0.01)  # Small init

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length (matches MLP projector)."""
        return (input_length - self.k) // self.k + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features using shared + sparse MoE.

        Args:
            x: Audio encoder output of shape [batch, seq_len, encoder_dim]

        Returns:
            Projected features of shape [batch, out_len, llm_dim]
        """
        # 1. Frame Stacking
        batch, seq, dim = x.shape
        out_len = (seq - self.k) // self.k + 1
        x = x[:, : out_len * self.k, :]
        x = x.reshape(batch, out_len, dim * self.k)

        # 2. Normalize stacked input (like main branch SharedMoEBlock)
        x = self.norm(x)
        flat_x = x.view(-1, x.size(-1))  # [tokens, in_dim]

        # 3. Shared Expert (compute first, creates output tensor)
        output = self.shared_expert(flat_x)

        # 4. Sparse Experts (in-place add to shared output)
        self.last_aux_loss = self._forward_sparse(flat_x, output)

        return output.view(batch, out_len, -1)

    def _forward_sparse(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Stability-hardened sparse expert dispatch (in-place add to output).

        Args:
            x: Flattened input of shape [tokens, dim]
            output: Output tensor to add sparse expert results into (in-place)

        Returns:
            Auxiliary loss tensor
        """
        # A. Router Logic with Jitter
        logits = self.router(x)

        if self.training and self.router_jitter_noise > 0:
            # Jitter: multiply by uniform noise (1-eps, 1+eps) to shake decision boundary
            # Prevents router from getting stuck on one expert early in training
            noise = torch.empty_like(logits).uniform_(
                1.0 - self.router_jitter_noise, 1.0 + self.router_jitter_noise
            )
            logits = logits * noise

        # Force float32 for softmax (bf16/fp16 exponentials can overflow)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(x)

        # B. Top-K Selection
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Normalize weights so they sum to 1.0
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # C. Aux Loss + Z-Loss
        aux_loss = torch.tensor(0.0, device=x.device)

        if self.training:
            # Load balancing loss (batch-size invariant)
            prob_per_expert = probs.mean(0)  # [num_experts]
            target = 1.0 / self.num_experts
            balance_loss = (
                self.aux_coef * ((prob_per_expert - target) ** 2).mean() * self.num_experts
            )

            # Z-loss: penalty on large logits to prevent softmax saturation
            z_loss = self.router_z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()

            aux_loss = balance_loss + z_loss

        # D. Dispatch Loop (in-place add to output)
        for i, expert in enumerate(self.experts):
            # Create boolean mask for tokens that selected Expert 'i'
            mask = top_k_indices == i

            if mask.any():
                # token_idx = which tokens, k_idx = 1st or 2nd choice
                token_idx, k_idx = torch.where(mask)

                # Gather inputs and compute
                expert_input = x[token_idx]
                expert_output = expert(expert_input)

                # Apply routing weight
                weight = top_k_weights[token_idx, k_idx].unsqueeze(-1)
                weighted_output = (expert_output * weight).type_as(output)

                # Scatter back in-place (index_add_ is atomic and deterministic)
                output.index_add_(0, token_idx, weighted_output)

        return aux_loss

    def get_aux_loss(self) -> torch.Tensor:
        """Return auxiliary load balancing loss."""
        return self.last_aux_loss


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
        """Initialize QFormer projector.

        Args:
            config: ASRConfig with encoder_dim, llm_dim, qformer_* settings
        """
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
