import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class SimpleAdapter(nn.Module):
    """
    MOSA Section III-B:
    "consists of two linear layers with a ReLU activation in between,
    projecting the hidden dimension from 3072 to 4096 and back to 3072."
    """

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MoEAudioProjector(nn.Module):
    """
    MOSA-style projector: Mixture of Simple Adapters.

    From paper (arXiv:2508.18998):
    - Dense mixture (softmax over ALL experts) instead of sparse Top-K
    - Simple Linear->ReLU->Linear adapters (3072->4096->3072)
    - No auxiliary losses - just cross-entropy on transcripts
    - Conv downsampling: stride 4 total (two conv layers, stride 2 each)
    """

    def __init__(self, config):
        super().__init__()

        # Dimensions from paper (same as ours):
        # Whisper-large-v3 encoder_dim = 1280
        # Phi-3-mini / SmolLM3-3B hidden_size = 3072
        self.encoder_dim = config.encoder_dim  # 1280
        self.llm_dim = config.llm_dim  # 3072

        # Number of experts: Base=4, Large=8
        self.num_experts = getattr(config, "num_experts", 4)

        # Adapter hidden dim: paper uses 4096
        adapter_hidden = getattr(config, "projector_hidden_dim", None) or 4096

        # --- Convolutional Subsampling (Section III-B) ---
        # "two convolutional layers, each with a kernel size of 3 and a stride of 2"
        # Maps encoder_dim (1280) -> llm_dim (3072), total stride=4
        self.conv = nn.Sequential(
            nn.Conv1d(self.encoder_dim, self.llm_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.llm_dim, self.llm_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # --- Router (Section III-B) ---
        # Base: "two linear layers... mapping from 1280 to 512 and finally to 4"
        router_hidden = 512
        self.router = nn.Sequential(
            nn.Linear(self.encoder_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, self.num_experts),
        )

        # --- Experts / Adapters (Section III-B) ---
        # "projecting the hidden dimension from 3072 to 4096 and back to 3072"
        self.experts = nn.ModuleList(
            [SimpleAdapter(self.llm_dim, adapter_hidden, self.llm_dim) for _ in range(self.num_experts)]
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, encoder_dim] from Whisper encoder (1280)

        Returns:
            output: [batch_size, seq_len // 4, llm_dim] (3072)
            aux_loss: 0.0 (MOSA uses no auxiliary losses)
        """
        batch_size, seq_len, _ = x.shape

        # Pad to be divisible by stride (4)
        pad_amt = (4 - (seq_len % 4)) % 4
        if pad_amt > 0:
            x = F.pad(x, (0, 0, 0, pad_amt))
            seq_len = x.shape[1]

        # 1. Convolutional Downsampling
        # (B, T, C) -> (B, C, T) -> conv -> (B, C, T//4) -> (B, T//4, C)
        h_conv = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. Router on high-res input, then downsample weights
        router_logits = self.router(x)  # [B, T, num_experts]
        # Average over stride window to match conv output
        router_logits = router_logits.view(batch_size, seq_len // 4, 4, self.num_experts).mean(dim=2)
        # Dense softmax
        routing_weights = F.softmax(router_logits, dim=-1)  # [B, T//4, num_experts]

        # 3. Weighted sum of expert outputs (Eq. 2: y = sum(w_i * E_i(x)))
        final_out = torch.zeros_like(h_conv)
        for i, expert in enumerate(self.experts):
            expert_out = expert(h_conv)
            expert_weight = routing_weights[:, :, i : i + 1]
            final_out = final_out + expert_out * expert_weight

        # MOSA: "we compute only the cross-entropy loss on transcriptions"
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return final_out, aux_loss
