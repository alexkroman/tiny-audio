"""Prefix Bridge for KV-cache fine-tuning.

Implements Freeze-Omni's approach to bridge text-space hidden states
into audio decoder space via learnable prefix transformer layers.

The prefix bridge:
1. Takes LLM hidden states (text-space representations)
2. Processes them through separate learnable transformer layers
3. Produces KV cache entries that condition audio generation

This allows efficient transfer learning where:
- Main AR decoder layers can be frozen
- Only prefix layers are trained
- KV cache bridges the modality gap
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)


class PrefixBridge(nn.Module):
    """Learnable prefix layers for KV-cache fine-tuning.

    Transforms LLM text-space hidden states into audio-compatible KV caches
    that condition the speech token generation. This acts as a learned
    "bridge" between text and audio modalities.

    Following Freeze-Omni's architecture:
    - Separate transformer layers process text hidden states
    - Output KV caches are used to condition AR decoder
    - Only prefix layers are trained, enabling efficient fine-tuning

    Args:
        hidden_size: Model hidden dimension (must match AR decoder)
        num_layers: Number of prefix transformer layers (should match AR decoder)
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        input_dim: Input dimension from LLM (if different from hidden_size)
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        input_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim or hidden_size

        # Input projection if dimensions differ
        if self.input_dim != hidden_size:
            self.input_proj = nn.Linear(self.input_dim, hidden_size, bias=False)
        else:
            self.input_proj = None

        config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=4096,
            attention_dropout=dropout,
            _attn_implementation="sdpa",
        )

        # Separate transformer layers for prefix processing
        # These are the ONLY trainable components when using prefix tuning
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(num_layers)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Process text hidden states and populate KV cache.

        Args:
            hidden_states: LLM hidden states [batch, seq_len, input_dim]
            attention_mask: Optional mask [batch, seq_len] (True = valid)
            past_key_values: Optional existing KV cache to extend

        Returns:
            output_hidden: Processed hidden states [batch, seq_len, hidden_size]
            past_key_values: KV cache populated with prefix representations
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Project input if needed
        if self.input_proj is not None:
            hidden_states = self.input_proj(hidden_states)

        # Initialize KV cache if not provided
        if past_key_values is None:
            past_key_values = DynamicCache()

        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)

        # Compute rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Create bidirectional attention mask for prefix
        # Prefix tokens can attend to all other prefix tokens
        if attention_mask is not None:
            # Expand mask: [batch, 1, seq, seq]
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, -1, seq_len, -1).contiguous()
            attn_mask = attn_mask & attention_mask.unsqueeze(1).unsqueeze(-1)
            # Convert to float mask
            attn_mask = torch.where(
                attn_mask,
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype),
            )
        else:
            attn_mask = None

        # Forward through prefix layers, accumulating KV cache
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        return hidden_states, past_key_values

    def get_prefix_kv_cache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DynamicCache:
        """Convenience method to get just the KV cache.

        Args:
            hidden_states: LLM hidden states [batch, seq_len, input_dim]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            KV cache populated with prefix representations
        """
        _, past_key_values = self.forward(hidden_states, attention_mask)
        return past_key_values
