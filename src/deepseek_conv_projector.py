import torch
import torch.nn as nn
import math

class DeepSeekConvAudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        encoder_dim = getattr(config, "encoder_dim", 768)
        llm_dim = getattr(config, "llm_dim", 4096)
        self.target_downsample = getattr(config, "downsample_rate", 16)
        
        layers = []
        current_dim = encoder_dim
        current_stride = 1
        
        # 1. Progressive Convolutional Compressor
        while current_stride < self.target_downsample:
            next_dim = min(current_dim * 2, llm_dim)
            
            layers.append(
                nn.Conv1d(
                    in_channels=current_dim,
                    out_channels=next_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False 
                )
            )
            # OPTIMIZATION: Use GroupNorm (usually 32 groups) to avoid Transpose overhead
            # This works directly on [B, C, T]
            layers.append(nn.GroupNorm(num_groups=32, num_channels=next_dim))
            layers.append(nn.GELU(approximate='tanh'))  # Faster approximate GELU
            
            current_dim = next_dim
            current_stride *= 2
            
        self.compressor = nn.Sequential(*layers)

        # 2. MLP Connector with fused GELU for better performance
        self.mlp = nn.Sequential(
            nn.Linear(current_dim, llm_dim),
            nn.GELU(approximate='tanh'),  # Faster approximate GELU
            nn.Linear(llm_dim, llm_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask=None):
        """
        x: [Batch, Seq_Len, Dim] (Standard Transformer output)
        attention_mask: [Batch, Seq_Len] (0 for padding, 1 for valid)
        """
        B, T, D = x.shape
        
        # 1. Prepare inputs for Conv1d [B, Dim, Seq_Len]
        x = x.transpose(1, 2).contiguous()  # Ensure contiguous for Conv1d
        
        # 2. Calculate padding
        pad_amt = (self.target_downsample - (T % self.target_downsample)) % self.target_downsample
        
        if pad_amt > 0:
            x = torch.nn.functional.pad(x, (0, pad_amt)) # Pad end of time
            if attention_mask is not None:
                # Pad mask with 0 (ignored)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_amt), value=0)

        # 3. Progressive Compression
        x = self.compressor(x)
        
        # 4. Handle Mask Downsampling
        if attention_mask is not None:
            # We simply pool the mask. If ANY value in the window was valid, 
            # we consider the compressed token valid? 
            # Usually for stride convolution, we just slice or nearest-neighbor.
            # Easiest correct way for stride=N is slicing:
            # But since we have padding=1, the spatial dim is preserved then halved.
            # MaxPool is robust for masks (if at least one valid token exists in the window -> keep it)
            attention_mask = attention_mask.unsqueeze(1).float() # [B, 1, T]
            
            # We need to apply the exact same reduction geometry
            # Or analytically: Mask length / target_downsample
            final_T = x.shape[2]
            attention_mask = torch.nn.functional.interpolate(
                attention_mask, size=final_T, mode='nearest'
            ).squeeze(1).long()

        # 5. Final Projection [B, New_T, LLM_Dim]
        x = x.transpose(1, 2)
        x = self.mlp(x)
        
        return x, attention_mask