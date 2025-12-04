"""Shared MoE Audio Projector.

A simplified MoE projector combining the best ideas:
- Shared expert: Always-on baseline processing (from GLM4)
- Zero-initialized router: Learns specialization naturally (from Qwen3)
- Simple top-k softmax: No grouping complexity (from Mixtral)
- Renormalized weights: Top-k weights sum to 1

Architecture:
    Output = SharedExpert(x) + TopKRoutedExperts(x)

The shared expert ensures every audio token gets consistent baseline
processing, while routed experts can specialize for different patterns
(e.g., vowels vs consonants, silence vs speech).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class RMSNorm(nn.Module):
    """High-precision RMSNorm to ensure stability in mixed-precision training."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr__(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SharedExpert(nn.Module):
    """Shared expert MLP that processes all tokens."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class RoutedExperts(nn.Module):
    """
    Sparse routed experts using a vectorized, torch.compile-friendly implementation.
    """

    def __init__(
        self, num_experts: int, top_k: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Fused gate+up projection: [num_experts, 2*hidden, input]
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, input_dim))
        # Down projection: [num_experts, output, hidden]
        self.down_proj = nn.Parameter(torch.empty(num_experts, output_dim, hidden_dim))
        self.act = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        A vectorized, torch.compile-friendly forward pass for routed experts.

        Args:
            hidden_states: [num_tokens, input_dim]
            top_k_indices: [num_tokens, top_k]
            top_k_weights: [num_tokens, top_k]
        """
        output = torch.zeros(
            hidden_states.shape[0],
            self.output_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # self.top_k is a small, static number, so this loop can be unrolled by torch.compile.
        for k in range(self.top_k):
            # Get the k-th expert indices and weights for each token
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_weights[:, k]

            # Gather the weights for all tokens at once for the k-th expert choice
            gate_up_w = self.gate_up_proj[expert_indices]
            down_w = self.down_proj[expert_indices]

            # Apply the expert MLPs in a batched manner using torch.bmm
            # hidden_states: [N, D_in] -> [N, D_in, 1]
            # gate_up_w:   [N, 2*D_hidden, D_in]
            # Result:      [N, 2*D_hidden, 1]
            gate_up_out = torch.bmm(gate_up_w, hidden_states.unsqueeze(-1))

            # SwiGLU activation
            gate, up = gate_up_out.squeeze(-1).chunk(2, dim=-1)
            h = self.act(gate) * up

            # Down projection
            # h:      [N, D_hidden] -> [N, D_hidden, 1]
            # down_w: [N, D_out, D_hidden]
            # Result: [N, D_out, 1]
            down_out = torch.bmm(down_w, h.unsqueeze(-1)).squeeze(-1)

            # Weight by router score and accumulate
            down_out = down_out * expert_weights.unsqueeze(-1)
            output += down_out

        return output


class SharedMoEBlock(nn.Module):
    """MoE block with shared expert + sparse routed experts."""

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

        # Router: zero-initialized for natural learning
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        nn.init.zeros_(self.router.weight)

        # Shared expert (always active)
        self.shared_expert = SharedExpert(input_dim, hidden_dim, output_dim)

        # Routed experts (sparse)
        self.routed_experts = RoutedExperts(
            num_experts, self.top_k, input_dim, hidden_dim, output_dim
        )

        # LayerScale: for training stability in deep networks
        self.layer_scale = nn.Parameter(torch.ones(output_dim) * 1e-4)

        # For auxiliary loss
        self.last_router_logits = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape

        # Shared expert output (all tokens)
        shared_out = self.shared_expert(hidden_states)

        # Routing
        flat_hidden = hidden_states.view(-1, dim)
        router_logits = self.router(flat_hidden)
        self.last_router_logits = router_logits

        # Softmax -> top-k -> renormalize
        router_probs = F.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        # Routed expert output
        routed_out = self.routed_experts(flat_hidden, top_k_indices, top_k_weights)
        routed_out = routed_out.view(batch_size, seq_len, -1)

        # Combine: shared + routed, then apply LayerScale
        return (shared_out + routed_out) * self.layer_scale


def load_balancing_loss(router_logits: torch.Tensor, num_experts: int, top_k: int) -> torch.Tensor:
    """Auxiliary loss to encourage balanced expert usage."""
    if router_logits is None:
        return torch.tensor(0.0)

    probs = F.softmax(router_logits.float(), dim=-1)
    _, selected = torch.topk(probs, top_k, dim=-1)

    # Fraction of tokens per expert
    expert_mask = F.one_hot(selected, num_experts).float()
    tokens_per_expert = expert_mask.mean(dim=(0, 1))

    # Average probability per expert
    prob_per_expert = probs.mean(dim=0)

    # Balance loss
    return (tokens_per_expert * prob_per_expert).sum() * num_experts


def z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Z-loss to prevent router logits from growing too large.

    From DeepSeek/Switch Transformer: penalizes large logits to keep
    softmax in its "soft" regime where gradients flow properly.
    """
    if router_logits is None:
        return torch.tensor(0.0)

    # logsumexp ≈ max(logits), squaring penalizes large values
    return torch.logsumexp(router_logits.float(), dim=-1).square().mean()


class SharedMoEAudioProjector(nn.Module):
    """Shared MoE Audio Projector.

    Combines a shared expert (always-on) with sparse routed experts.
    Uses zero-initialized router for natural specialization learning.

    Config options:
        - num_experts: Number of routed experts (default: 4)
        - num_experts_per_tok: Top-k routing (default: 2)
        - router_aux_loss_coef: Load balancing loss weight (default: 0.01)
        - router_z_loss_coef: Z-loss weight to prevent large logits (default: 0.001)
    """

    def __init__(self, config):
        super().__init__()

        # Temporal downsampling
        self.k = getattr(config, "projector_pool_stride", 4)

        # Dimensions
        self.encoder_dim = config.encoder_dim
        in_dim = self.encoder_dim * self.k
        out_dim = config.llm_dim
        # No expansion - keep hidden dim same as input dim
        hidden_dim = getattr(config, "projector_hidden_dim", None) or in_dim

        # MoE config
        self.num_experts = getattr(config, "num_experts", 4)
        self.top_k = getattr(config, "num_experts_per_tok", 2)
        self.aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.01)
        self.z_loss_coef = getattr(config, "router_z_loss_coef", 0.001)

        # Layers
        self.ln_pre = RMSNorm(in_dim, eps=1e-6)
        self.moe = SharedMoEBlock(in_dim, hidden_dim, out_dim, self.num_experts, self.top_k)
        self.ln_post = RMSNorm(out_dim, eps=1e-6)

        # Init
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.ln_pre.weight.fill_(1.0)
            self.ln_post.weight.fill_(1.0)

            # Xavier init: std = 1/sqrt(fan_in)
            in_dim = self.encoder_dim * self.k
            std = 1.0 / (in_dim ** 0.5)

            # Use a smaller std for the final projection in the shared expert's residual path
            down_proj_std = std / 2.0

            # Shared expert
            nn.init.normal_(self.moe.shared_expert.gate_proj.weight, std=std)
            nn.init.normal_(self.moe.shared_expert.up_proj.weight, std=std)
            nn.init.normal_(self.moe.shared_expert.down_proj.weight, std=down_proj_std)

            # Routed experts - zero init down_proj so they "grow in" from zero
            nn.init.normal_(self.moe.routed_experts.gate_up_proj, std=std)
            nn.init.zeros_(self.moe.routed_experts.down_proj)

            # Router stays zero-initialized

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()

        # Dtype
        target_dtype = self.moe.shared_expert.gate_proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Pad for pooling
        if seq_len % self.k:
            x = F.pad(x, (0, 0, 0, self.k - seq_len % self.k))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, self.k - seq_len % self.k), value=0)

        # Store pooled attention mask for aux loss
        if attention_mask is not None:
            # Max-pool the attention mask
            pooled_mask = F.max_pool1d(attention_mask.float().unsqueeze(1), self.k, self.k)
            self.last_attention_mask = pooled_mask.squeeze(1).bool()
        else:
            self.last_attention_mask = None

        # Temporal pooling
        x = x.view(batch_size, -1, dim * self.k)

        # Forward
        x = self.ln_pre(x)
        x = self.moe(x)
        x = self.ln_post(x)

        return x

    def get_aux_loss(self) -> torch.Tensor:
        """Get auxiliary losses (call after forward).

        Combines:
        - Load balancing loss: encourages balanced expert usage
        - Z-loss: prevents router logits from growing too large
        """
        router_logits = self.moe.last_router_logits
        if router_logits is None:
            return torch.tensor(0.0, device=self.moe.router.weight.device)

        # Retrieve the attention mask stored during the forward pass
        attention_mask = getattr(self, "last_attention_mask", None)

        # If a mask exists, filter the logits to only include un-padded tokens
        if attention_mask is not None:
            flat_mask = attention_mask.view(-1)
            # Ensure the mask is not all False, which would create an empty tensor
            if flat_mask.any():
                active_logits = router_logits[flat_mask]
            else:
                # If the mask is all False, there are no tokens to compute loss on
                return torch.tensor(0.0, device=router_logits.device)
        else:
            active_logits = router_logits

        balance_loss = load_balancing_loss(active_logits, self.num_experts, self.top_k)
        z = z_loss(active_logits)

        return self.aux_loss_coef * balance_loss + self.z_loss_coef * z
