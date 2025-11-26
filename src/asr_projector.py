import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=False, dropout_rate=0.05):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        input_dtype = x.dtype
        w12_out = self.w12(x)
        x_gate, x_val = w12_out.chunk(2, dim=-1)
        x = F.silu(x_gate) * x_val
        x = self.w3(x)
        x = self.dropout(x)
        return x.to(input_dtype)


class MoEAudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = getattr(config, "projector_pool_stride", 2)
        self.num_experts = getattr(config, "num_experts", 8)
        self.top_k = getattr(config, "moe_top_k", 2)

        self.router_scale = getattr(config, "router_scale", 16.0)
        self.bias_scale = getattr(config, "router_bias_scale", 0.5)
        self.load_update_rate = getattr(config, "router_load_update_rate", 0.1)
        self.z_loss_coef = getattr(config, "router_z_loss_coef", 1e-4)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.expert_load: torch.Tensor
        self.register_buffer("expert_load", torch.ones(self.num_experts) / self.num_experts)

        in_dim = config.encoder_dim * self.k
        self.out_dim = config.llm_dim

        routed_expert_hidden = getattr(config, "projector_hidden_dim", None) or 2048
        shared_expert_hidden = getattr(config, "shared_projector_hidden_dim", None) or 2048

        self.ln_pre = RMSNorm(in_dim, eps=1e-6)
        self.ln_post = RMSNorm(self.out_dim, eps=1e-6)

        self.router_weights = nn.Parameter(torch.randn(self.num_experts, in_dim) * 0.02)

        self.shared_expert = SwiGLU(in_dim, shared_expert_hidden, self.out_dim)

        self.experts = nn.ModuleList(
            [SwiGLU(in_dim, routed_expert_hidden, self.out_dim) for _ in range(self.num_experts)]
        )

        with torch.no_grad():
            nn.init.normal_(self.shared_expert.w12.weight, std=0.02)
            nn.init.normal_(self.shared_expert.w3.weight, std=0.02)
            for expert in self.experts:
                nn.init.normal_(expert.w12.weight, std=0.02)
                nn.init.normal_(expert.w3.weight, std=0.02)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        remainder = seq_len % self.k
        if remainder:
            x = F.pad(x, (0, 0, 0, self.k - remainder))

        x = x.contiguous().view(batch_size, -1, dim * self.k)
        x_flat = x.view(-1, dim * self.k)

        norm_x = self.ln_pre(x_flat)
        shared_out = self.shared_expert(norm_x)

        input_normed = F.normalize(norm_x, dim=-1)
        router_normed = F.normalize(self.router_weights, dim=-1)
        router_logits = F.linear(input_normed, router_normed) * self.router_scale
        routing_probs = torch.sigmoid(router_logits)

        choice_probs = routing_probs - (self.bias_scale * self.expert_load)
        _, top_k_indices = torch.topk(choice_probs, self.top_k, dim=-1)
        top_k_weights = torch.gather(routing_probs, -1, top_k_indices)

        denominator = top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
        top_k_weights = top_k_weights / denominator
        top_k_weights = top_k_weights * self.routed_scaling_factor
        top_k_weights = top_k_weights.to(x.dtype)

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.training:
            with torch.no_grad():
                flat_indices = top_k_indices.flatten()

                current_usage = torch.histc(
                    flat_indices.float(), bins=self.num_experts, min=0, max=self.num_experts - 1
                )

                total_tokens = flat_indices.numel()

                current_load = current_usage / total_tokens

                self.expert_load = (
                    1 - self.load_update_rate
                ) * self.expert_load + self.load_update_rate * current_load

            if self.z_loss_coef > 0:
                z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
                aux_loss = z_loss * self.z_loss_coef

        routed_out = torch.zeros_like(shared_out)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask_2d = top_k_indices == expert_idx

            if not expert_mask_2d.any():
                continue

            batch_indices, k_indices = torch.where(expert_mask_2d)
            expert_input = norm_x[batch_indices]
            expert_output = expert(expert_input)

            current_weights = top_k_weights[batch_indices, k_indices].unsqueeze(-1)
            weighted_output = expert_output * current_weights
            routed_out.index_add_(0, batch_indices, weighted_output)

        final_out = self.ln_post(shared_out + routed_out)
        return final_out.view(batch_size, -1, self.out_dim), aux_loss
