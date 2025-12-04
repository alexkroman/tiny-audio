import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class SwiGLUExpert(nn.Module):
    """SwiGLU expert MLP (used for both shared and routed experts)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


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
        self.output_dim = output_dim

        # Router: zero-initialized for natural learning
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        nn.init.zeros_(self.router.weight)

        # Shared expert (always active)
        self.shared_expert = SwiGLUExpert(input_dim, hidden_dim, output_dim)

        # Routed experts (sparse)
        self.experts = nn.ModuleList([
            SwiGLUExpert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

        # For auxiliary loss (cached to avoid recomputation)
        self.last_router_logits = None
        self.last_router_probs = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape

        # Shared expert output (all tokens)
        shared_out = self.shared_expert(hidden_states)

        # Routing
        flat_hidden = hidden_states.view(-1, dim)
        router_logits = self.router(flat_hidden)
        router_probs = F.softmax(router_logits.float(), dim=-1)

        # Cache for aux loss
        self.last_router_logits = router_logits
        self.last_router_probs = router_probs

        # Top-k selection and renormalization
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        # Routed expert output via token dispatch
        routed_out = self._dispatch_experts(flat_hidden, top_k_indices, top_k_weights)
        routed_out = routed_out.view(batch_size, seq_len, -1)

        # Combine: shared expert baseline + routed experts (grow in via zero-init down_proj)
        return shared_out + routed_out

    def _dispatch_experts(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Token dispatch - gather tokens per expert, process, scatter back."""
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
            expert_output = expert(expert_input)
            weights = top_k_weights[token_indices, slot_indices].unsqueeze(-1)
            output.index_add_(0, token_indices, expert_output * weights)

        return output


def load_balancing_loss(router_probs: torch.Tensor, num_experts: int, top_k: int) -> torch.Tensor:
    """Auxiliary loss to encourage balanced expert usage."""
    _, selected = torch.topk(router_probs, top_k, dim=-1)
    expert_mask = F.one_hot(selected, num_experts).float()
    tokens_per_expert = expert_mask.mean(dim=(0, 1))
    prob_per_expert = router_probs.mean(dim=0)
    return (tokens_per_expert * prob_per_expert).sum() * num_experts


def z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Z-loss to prevent router logits from growing too large."""
    return torch.logsumexp(router_logits.float(), dim=-1).square().mean()


class SharedMoEAudioProjector(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Temporal downsampling
        self.k = getattr(config, "projector_pool_stride", 4)

        # Dimensions
        encoder_dim = config.encoder_dim
        in_dim = encoder_dim * self.k
        out_dim = config.llm_dim
        hidden_dim = getattr(config, "projector_hidden_dim", None) or in_dim

        # MoE config
        self.num_experts = getattr(config, "num_experts", 4)
        self.top_k = getattr(config, "num_experts_per_tok", 2)
        self.aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.01)
        self.z_loss_coef = getattr(config, "router_z_loss_coef", 0.001)

        # Layers
        self.moe = SharedMoEBlock(in_dim, hidden_dim, out_dim, self.num_experts, self.top_k)

        # Init
        self._init_weights(in_dim)

    def _init_weights(self, in_dim: int):
        with torch.no_grad():
            std = 1.0 / (in_dim ** 0.5)

            # Shared expert (smaller std for down_proj since it's the "residual" path)
            nn.init.normal_(self.moe.shared_expert.gate_proj.weight, std=std)
            nn.init.normal_(self.moe.shared_expert.up_proj.weight, std=std)
            nn.init.normal_(self.moe.shared_expert.down_proj.weight, std=std / 2.0)

            # Routed experts - zero init down_proj so they "grow in" from zero
            for expert in self.moe.experts:
                nn.init.normal_(expert.gate_proj.weight, std=std)
                nn.init.normal_(expert.up_proj.weight, std=std)
                nn.init.zeros_(expert.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()

        target_dtype = self.moe.shared_expert.gate_proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # Pad for pooling (at most k-1 frames -> 1 extra token, negligible impact)
        if seq_len % self.k:
            x = F.pad(x, (0, 0, 0, self.k - seq_len % self.k))

        # Temporal pooling
        x = x.view(batch_size, -1, dim * self.k)

        return self.moe(x)

    def get_aux_loss(self) -> torch.Tensor:
        """Get auxiliary losses (call after forward)."""
        if self.moe.last_router_logits is None:
            return torch.tensor(0.0, device=self.moe.router.weight.device)

        balance = load_balancing_loss(self.moe.last_router_probs, self.num_experts, self.top_k)
        z = z_loss(self.moe.last_router_logits)

        return self.aux_loss_coef * balance + self.z_loss_coef * z
