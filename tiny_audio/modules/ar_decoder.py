"""Autoregressive decoder for codec token generation.

Based on Freeze-Omni's LLM2TTSCodecAR architecture:
- Pre-NN layers process LLM hidden states (bidirectional) - half of AR decoder layers
- AR decoder generates codec tokens autoregressively (causal)
- Uses LlamaDecoderLayer from transformers for efficiency
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class CodecARDecoder(nn.Module):
    """Autoregressive decoder for generating codec tokens.

    Includes Pre-NN layers internally (Freeze-Omni style):
    - Pre-NN: num_layers // 2 bidirectional transformer layers
    - AR decoder: num_layers causal transformer layers

    Args:
        hidden_size: Model hidden dimension
        num_layers: Number of AR transformer layers
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        vocab_size: Codec vocabulary size (default 2048 for Mimi)
        dropout: Dropout rate for attention
        embedding: Optional shared embedding layer
    """

    # Special token offsets from vocab_size
    BOS_OFFSET = 0  # Beginning of LLM context
    SOS_OFFSET = 1  # Start of codec sequence
    EOS_OFFSET = 2  # End of codec sequence
    PAD_OFFSET = 3  # Padding
    NUM_SPECIAL = 4

    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        vocab_size: int = 2048,
        dropout: float = 0.1,
        embedding: nn.Embedding = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.total_vocab = vocab_size + self.NUM_SPECIAL
        self.num_layers = num_layers

        # Special token IDs
        self.bos_token_id = vocab_size + self.BOS_OFFSET
        self.sos_token_id = vocab_size + self.SOS_OFFSET
        self.eos_token_id = vocab_size + self.EOS_OFFSET
        self.pad_token_id = vocab_size + self.PAD_OFFSET

        config = LlamaConfig(
            vocab_size=self.total_vocab,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,  # No GQA
            max_position_embeddings=4096,
            attention_dropout=dropout,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            _attn_implementation="sdpa",
        )

        # Token embedding - use provided embedding or create new one
        # (Freeze-Omni style: single shared embedding for all tokens)
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(
                self.total_vocab,
                hidden_size,
                padding_idx=self.pad_token_id,
            )

        # Pre-NN layers (Freeze-Omni style): num_layers // 2 bidirectional layers
        # Processes LLM hidden states before AR decoding
        num_pre_nn_layers = num_layers // 2
        self.pre_nn_layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(num_pre_nn_layers)]
        )
        self.pre_nn_rotary_emb = LlamaRotaryEmbedding(config=config)

        # AR Transformer layers (causal)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(num_layers)]
        )
        self.norm = LlamaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, self.total_vocab, bias=False)

    def forward_pre_nn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process hidden states through Pre-NN with bidirectional attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional mask [batch, seq_len] (True = valid)

        Returns:
            Processed hidden states [batch, seq_len, hidden_size]
        """
        _, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Compute rotary embeddings
        position_embeddings = self.pre_nn_rotary_emb(hidden_states, position_ids)

        # Create bidirectional attention mask (all positions attend to all)
        if attention_mask is not None:
            # Expand mask for attention: [batch, 1, seq, seq]
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, -1, seq_len, -1).contiguous()
            attn_mask = attn_mask & attention_mask.unsqueeze(1).unsqueeze(-1)
            # IMPORTANT: contiguous() and correct dtype required to avoid CUBLAS errors
            attn_mask = torch.where(
                attn_mask,
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype),
            )
        else:
            attn_mask = None

        # Forward through Pre-NN layers
        for layer in self.pre_nn_layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        # Note: Freeze-Omni does NOT apply norm here - norm is applied
        # only in the AR decoder after the full forward pass
        return hidden_states

    def forward(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        target_ids: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        skip_pre_nn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Following Freeze-Omni's approach:
        - Input: [SOS, target_0, target_1, ..., target_{n-1}]
        - Labels: [target_0, target_1, ..., target_{n-1}, EOS]
        This teaches the model to predict EOS after the last token.

        Args:
            context: LLM hidden states or Pre-NN output [batch, context_len, hidden_size]
            context_mask: Mask for context [batch, context_len]
            target_ids: Target codec tokens [batch, target_len]
            target_mask: Mask for targets [batch, target_len]
            return_hidden: If True, also return hidden states for Depformer
            skip_pre_nn: If True, skip Pre-NN (context is already processed)

        Returns:
            logits: [batch, target_len + 1, vocab_size]
            loss: Cross-entropy loss
            hidden_states (if return_hidden): [batch, target_len, hidden_size]
        """
        # Process through Pre-NN if not already done
        if not skip_pre_nn:
            context = self.forward_pre_nn(context, context_mask)

        batch_size, context_len, _ = context.shape
        device = context.device
        dtype = context.dtype

        # Create special tokens
        sos = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)

        # Input: SOS + ALL target tokens (Freeze-Omni style)
        # Clamp target_ids to valid codec range (targets may have padding/special values)
        clamped_targets = target_ids.clamp(0, self.vocab_size - 1)
        input_ids = torch.cat([sos, clamped_targets], dim=1)  # [batch, max_target_len + 1]
        input_len = input_ids.shape[1]

        # Embed input tokens
        input_emb = self.embedding(input_ids)  # [batch, input_len, hidden]

        # Add BOS embedding to context
        bos_emb = self.embedding(torch.full((batch_size, 1), self.bos_token_id, device=device))
        context = torch.cat([bos_emb, context], dim=1)
        context_len = context.shape[1]

        # Concatenate context + input embeddings
        combined = torch.cat([context, input_emb], dim=1)
        total_len = combined.shape[1]

        # Create attention mask:
        # - Context: bidirectional within context
        # - Targets: attend to context + causal within targets
        attn_mask = torch.zeros(batch_size, total_len, total_len, dtype=torch.bool, device=device)

        # Context attends to context (bidirectional)
        attn_mask[:, :context_len, :context_len] = True

        # Targets attend to context
        attn_mask[:, context_len:, :context_len] = True

        # Targets attend causally to targets
        causal = torch.tril(torch.ones(input_len, input_len, dtype=torch.bool, device=device))
        attn_mask[:, context_len:, context_len:] = causal

        # Apply sequence masks if provided
        if context_mask is not None:
            # Zero out attention to masked context positions
            ctx_mask = context_mask.unsqueeze(1).expand(-1, total_len, -1)
            # Add column for BOS (always attend to it)
            bos_col = torch.ones(batch_size, total_len, 1, dtype=torch.bool, device=device)
            ctx_mask = torch.cat([bos_col, ctx_mask], dim=2)
            attn_mask[:, :, :context_len] &= ctx_mask

        if target_mask is not None:
            # Expand target_mask to include the +1 position for EOS prediction
            # The model needs to attend to valid targets + one more position for EOS
            extended_mask = torch.cat(
                [target_mask, torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1
            )
            tgt_mask = extended_mask.unsqueeze(1).expand(-1, total_len, -1)
            attn_mask[:, :, context_len:] &= tgt_mask

        # Convert to float mask
        # IMPORTANT: correct dtype required to avoid CUBLAS errors with BF16 on GPU
        attn_mask = torch.where(
            attn_mask.unsqueeze(1),
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype),
        )

        # Position IDs
        position_ids = torch.arange(total_len, device=device).unsqueeze(0)

        # Rotary embeddings
        position_embeddings = self.rotary_emb(combined, position_ids)

        # Forward through transformer
        hidden_states = combined
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            # Handle both tensor (transformers 5.0+) and tuple (older) outputs
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        hidden_states = self.norm(hidden_states)

        # Get target hidden states only
        target_hidden = hidden_states[:, context_len:]

        # Project to vocab
        logits = self.output_proj(target_hidden)

        # Create labels: [target_0, ..., target_{n-1}, EOS, PAD, ...]
        # Start with clamped targets + padding column (use clamped to match embedding inputs)
        labels = torch.cat(
            [
                clamped_targets,
                torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        # Insert EOS at the position after the last valid token
        if target_mask is not None:
            # Get actual lengths from mask
            lengths = target_mask.sum(dim=1)  # [batch]
            # Create position indices
            positions = torch.arange(input_len, device=device).expand(batch_size, -1)
            # EOS position is where position == length (right after last valid token)
            eos_positions = positions == lengths.unsqueeze(1)
            labels[eos_positions] = self.eos_token_id
            # Mask padding positions in labels
            pad_positions = positions > lengths.unsqueeze(1)
            labels[pad_positions] = self.pad_token_id
        else:
            # If no mask, assume all positions are valid, EOS at the end
            labels[:, -1] = self.eos_token_id

        loss = nn.functional.cross_entropy(
            logits.view(-1, self.total_vocab),
            labels.view(-1),
            ignore_index=self.pad_token_id,
        )

        if return_hidden:
            # Return hidden states aligned with target positions
            # target_hidden has shape [batch, target_len + 1, hidden] for positions:
            #   [SOS, target_0, target_1, ..., target_{n-1}]
            # Hidden at position i contains context AFTER processing input at position i
            # For Depformer conditioning at time t, we need hidden AFTER seeing semantic[t]
            # So we return positions 1..n (after target_0..target_{n-1}), not 0..n-1
            return logits, loss, target_hidden[:, 1:, :]

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        max_tokens: int = 500,
        top_k: int = 50,
        temperature: float = 1.0,
        repetition_penalty: float = 1.1,
        penalty_window: int = 20,
        return_hidden: bool = False,
        skip_pre_nn: bool = False,
    ):
        """Generate codec tokens autoregressively.

        Args:
            context: LLM hidden states or Pre-NN output [batch, context_len, hidden_size]
            context_mask: Mask for context [batch, context_len]
            max_tokens: Maximum tokens to generate
            top_k: Top-k sampling parameter
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeated tokens
            penalty_window: Window size for repetition penalty
            return_hidden: If True, yield (token, hidden_state) tuples for Depformer
            skip_pre_nn: If True, skip Pre-NN (context is already processed)

        Yields:
            If return_hidden=False: Generated token IDs one at a time
            If return_hidden=True: Tuples of (token_id, hidden_state [1, 1, hidden_size])
        """
        # Process through Pre-NN if not already done
        if not skip_pre_nn:
            context = self.forward_pre_nn(context, context_mask)

        batch_size = context.shape[0]
        device = context.device

        assert batch_size == 1, "Streaming generation only supports batch_size=1"

        # Add BOS to context
        bos_emb = self.embedding(torch.full((1, 1), self.bos_token_id, device=device))
        context = torch.cat([bos_emb, context], dim=1)
        context_len = context.shape[1]

        # Initialize KV cache
        past_key_values = DynamicCache()

        # First pass: process context
        position_ids = torch.arange(context_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(context, position_ids)

        # Build context attention mask if provided
        if context_mask is not None:
            # Add BOS position (always valid) to mask
            bos_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            full_context_mask = torch.cat([bos_mask, context_mask], dim=1)
            context_attn_mask = full_context_mask.unsqueeze(1).unsqueeze(2)
            context_attn_mask = (~context_attn_mask) * torch.finfo(context.dtype).min
        else:
            context_attn_mask = None

        hidden_states = context
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=context_attn_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        # Start with SOS token
        current_token = torch.full((1, 1), self.sos_token_id, device=device)
        generated: list[int] = []

        # Build base mask for generation
        if context_mask is not None:
            bos_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            base_valid_mask = torch.cat([bos_mask, context_mask], dim=1)
        else:
            base_valid_mask = None

        for _ in range(max_tokens):
            # Embed current token
            token_emb = self.embedding(current_token)

            # Position for this token
            pos = past_key_values.get_seq_length()
            position_ids = torch.tensor([[pos]], device=device)
            position_embeddings = self.rotary_emb(token_emb, position_ids)

            # Build attention mask for this step
            if base_valid_mask is not None:
                num_generated = pos - context_len
                if num_generated > 0:
                    gen_mask = torch.ones(1, num_generated, dtype=torch.bool, device=device)
                    step_mask = torch.cat([base_valid_mask, gen_mask], dim=1)
                else:
                    step_mask = base_valid_mask
                step_attn_mask = step_mask.unsqueeze(1).unsqueeze(2)
                step_attn_mask = (~step_attn_mask) * torch.finfo(token_emb.dtype).min
            else:
                step_attn_mask = None

            # Forward through layers
            hidden_states = token_emb
            for layer in self.layers:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=step_attn_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                hidden_states = (
                    layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
                )

            hidden_states = self.norm(hidden_states)

            # Project to vocab
            logits = self.output_proj(hidden_states[:, -1, :])  # [1, vocab]

            # Mask special tokens (BOS, SOS, PAD) to ensure valid Mimi codes
            logits[:, self.bos_token_id] = float("-inf")
            logits[:, self.sos_token_id] = float("-inf")
            logits[:, self.pad_token_id] = float("-inf")

            # Apply repetition penalty
            if penalty_window > 0 and len(generated) > 0:
                recent = generated[-penalty_window:]
                for token_id in set(recent):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k sampling
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[-1]))

            # Renormalize
            top_k_probs = top_k_probs / top_k_probs.sum()

            # Sample
            sampled_idx = torch.multinomial(top_k_probs, 1)
            next_token = top_k_indices[0, sampled_idx[0, 0]].item()

            # Check for EOS
            if next_token == self.eos_token_id:
                break

            generated.append(next_token)
            current_token = torch.tensor([[next_token]], device=device)

            if return_hidden:
                yield next_token, hidden_states.clone()
            else:
                yield next_token
