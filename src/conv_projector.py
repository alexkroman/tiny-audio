import torch
import torch.nn as nn

class ConvAudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        encoder_dim = getattr(config, "encoder_dim", 768)
        llm_dim = getattr(config, "llm_dim", 4096)
        k = getattr(config, "projector_pool_stride", 5)

        # 1. Downsampling Layer (Audio Patch Embedding)
        # Compresses time dimension by factor 'k'
        self.patch_embed = nn.Conv1d(
            in_channels=encoder_dim,
            out_channels=llm_dim,
            kernel_size=k,
            stride=k,
            bias=True
        )

        # 2. MLP Connector (LlavaNext Style)
        # Simple Linear -> GELU -> Linear topology
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        
        # Permute for Conv1d: [Batch, Dim, Seq_Len]
        x = x.transpose(1, 2)
        
        # Downsample
        x = self.patch_embed(x)
        
        # Permute back: [Batch, New_Seq_Len, Dim]
        x = x.transpose(1, 2)
        
        # Project
        x = self.mlp(x)
        
        return x