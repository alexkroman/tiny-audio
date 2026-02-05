"""Depformer module for predicting acoustic codebooks.

Based on Moshi's Depformer architecture:
- Small transformer that predicts codebooks 1-7 conditioned on codebook 0
- Processes codebooks sequentially, each conditioned on previous
- Uses shared transformer with per-codebook input/output projections
- Supports acoustic delays (Moshi-style) where each codebook is shifted in time
"""

import logging

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

logger = logging.getLogger(__name__)


class Depformer(nn.Module):
    """Depformer for predicting acoustic codebooks 1-7.

    Following Moshi's architecture:
    - Input: main AR decoder hidden states + previous codebook token
    - Output: logits for next codebook token
    - Processes codebooks sequentially during inference
    - Processes all codebooks in parallel during training
    - Supports acoustic delays for improved audio quality

    With acoustic delays enabled (default, Moshi-style flat delays):
    - CB0 (semantic) at AR position t is for audio time t
    - CB1-CB7 at AR position t are all for audio time t - 1

    This flat delay pattern (vs progressive delays) allows all acoustic
    codebooks to be decoded in parallel after CB0, reducing latency for
    streaming/real-time generation.

    Args:
        num_codebooks: Number of codebooks to predict (default: 7 for cb 1-7)
        vocab_size: Codec vocabulary size (default: 2048 for Mimi)
        main_dim: Dimension of main AR decoder (input projection source)
        hidden_size: Depformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        dropout: Dropout rate
        use_delays: Whether to use acoustic delays (default: True)
    """

    def __init__(
        self,
        num_codebooks: int = 7,
        vocab_size: int = 2048,
        main_dim: int = 1024,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        dropout: float = 0.0,
        use_delays: bool = True,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.main_dim = main_dim
        self.hidden_size = hidden_size
        self.use_delays = use_delays

        # Moshi-style flat delays: all acoustic codebooks at same delay
        # Input delays: [0, 0, 0, 0, 0, 0, 0] - all use same input timing
        # Target delays: [1, 1, 1, 1, 1, 1, 1] - all predict same target timing
        # This enables parallel decoding of all acoustic codebooks
        if use_delays:
            self.register_buffer("input_delays", torch.zeros(num_codebooks, dtype=torch.long))
            self.register_buffer("target_delays", torch.ones(num_codebooks, dtype=torch.long))
        else:
            self.register_buffer("input_delays", torch.zeros(num_codebooks, dtype=torch.long))
            self.register_buffer("target_delays", torch.zeros(num_codebooks, dtype=torch.long))

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=64,  # Small context for depformer
            attention_dropout=dropout,
            rope_theta=10000.0,
            _attn_implementation="eager",  # Use eager for small sequences
        )

        # Per-codebook input projections from main AR decoder (Moshi-style multi-linear)
        # Each codebook gets its own projection, allowing different information per codebook
        self.input_projs = nn.ModuleList(
            [nn.Linear(main_dim, hidden_size, bias=False) for _ in range(num_codebooks)]
        )

        # Embeddings for previous codebook tokens
        # cb_index 0: embed codebook 0 token (semantic)
        # cb_index 1-6: embed codebook 1-6 tokens (acoustic)
        self.codebook_emb = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden_size) for _ in range(num_codebooks)]
        )

        # Shared transformer layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(num_layers)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Per-codebook output normalization and projection
        self.output_norms = nn.ModuleList(
            [LlamaRMSNorm(hidden_size, eps=config.rms_norm_eps) for _ in range(num_codebooks)]
        )
        self.output_projs = nn.ModuleList(
            [nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(num_codebooks)]
        )

    @property
    def max_delay(self) -> int:
        """Maximum delay across all target codebooks."""
        return int(self.target_delays.max().item())

    def _apply_delay(
        self, tensor: torch.Tensor, delay: int, dim: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a single delay to a tensor along the specified dimension.

        At AR position t, we want the value for audio time t - delay.
        So shifted[..., t, ...] = original[..., t - delay, ...] (if t >= delay).
        Positions where t < delay are invalid (padded with zeros).

        Args:
            tensor: Input tensor
            delay: Number of positions to delay
            dim: Dimension to shift along

        Returns:
            shifted: Tensor with delay applied (zeros where invalid)
            valid_mask: Boolean mask of valid (non-padded) positions
        """
        if delay == 0:
            valid = torch.ones(tensor.shape[dim], dtype=torch.bool, device=tensor.device)
            return tensor, valid

        seq_len = tensor.shape[dim]
        shifted = torch.zeros_like(tensor)
        valid = torch.zeros(seq_len, dtype=torch.bool, device=tensor.device)

        if delay < seq_len:
            # shifted[t] = tensor[t - delay] for t >= delay
            if dim == -1 or dim == tensor.dim() - 1:
                shifted[..., delay:] = tensor[..., : seq_len - delay]
            elif dim == 1:
                shifted[:, delay:] = tensor[:, : seq_len - delay]
            elif dim == 2:
                shifted[:, :, delay:] = tensor[:, :, : seq_len - delay]
            valid[delay:] = True

        return shifted, valid

    def forward_training(
        self,
        main_hidden: torch.Tensor,
        codebook_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training - process all codebooks in parallel.

        With Moshi-style flat delays enabled:
        - All input codebooks at AR position t use audio time t (delay=0)
        - All target codebooks at AR position t are for audio time t-1 (delay=1)

        This enables parallel decoding during inference.
        Loss is computed only on valid positions (after delay padding).

        Args:
            main_hidden: Hidden states from main AR decoder [batch, seq_len, main_dim]
            codebook_targets: Target tokens for all codebooks [batch, num_codebooks+1, seq_len]
                              where [:, 0, :] is codebook 0 (semantic, from AR decoder)
                              and [:, 1:, :] are codebooks 1-7 (targets for depformer)

        Returns:
            logits: [batch, num_codebooks, seq_len, vocab_size]
            loss: Cross-entropy loss averaged over valid positions
        """
        batch_size, seq_len, _ = main_hidden.shape
        device = main_hidden.device
        dtype = main_hidden.dtype

        # Build inputs for each codebook with delays applied
        # Each codebook gets its own projection (multi-linear)
        depformer_inputs = []
        input_valid_masks = []

        for cb_idx in range(self.num_codebooks):
            # Project main hidden states with per-codebook projection
            projected = self.input_projs[cb_idx](main_hidden)  # [B, T, hidden]

            # Get previous codebook tokens
            prev_tokens = codebook_targets[:, cb_idx, :]  # [B, T]

            # Apply input delay
            delay = int(self.input_delays[cb_idx].item())
            shifted_tokens, valid = self._apply_delay(prev_tokens, delay, dim=-1)
            input_valid_masks.append(valid)

            # Clamp tokens to valid embedding range (codec targets may have special tokens)
            shifted_tokens = shifted_tokens.clamp(0, self.vocab_size - 1)

            # Embed previous tokens
            token_emb = self.codebook_emb[cb_idx](shifted_tokens)  # [B, T, hidden]

            # Combine with projected hidden states
            cb_input = projected + token_emb
            depformer_inputs.append(cb_input)

        # Stack all codebook inputs: [B, T, K, hidden]
        stacked = torch.stack(depformer_inputs, dim=2)

        # Reshape for transformer: [B*T, K, hidden]
        flat_batch = batch_size * seq_len
        stacked = stacked.view(flat_batch, self.num_codebooks, -1)

        # Create causal mask for codebook dimension
        # Must be same dtype as hidden states to avoid CUBLAS errors
        causal_mask = torch.tril(
            torch.ones(self.num_codebooks, self.num_codebooks, device=device, dtype=torch.bool)
        )
        attn_mask = torch.where(
            causal_mask.unsqueeze(0).unsqueeze(0),
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype),
        )

        # Position embeddings for codebook dimension
        position_ids = torch.arange(self.num_codebooks, device=device).unsqueeze(0)
        position_ids = position_ids.expand(flat_batch, -1)
        position_embeddings = self.rotary_emb(stacked, position_ids)

        # Forward through transformer
        hidden_states = stacked
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
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        # Ensure hidden_states maintains shape [B*T, K, hidden]
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.view(flat_batch, self.num_codebooks, -1)

        # Compute logits for each codebook
        logits_list = []
        for cb_idx in range(self.num_codebooks):
            cb_hidden = hidden_states[:, cb_idx, :]  # [B*T, hidden]
            cb_hidden = self.output_norms[cb_idx](cb_hidden)
            logits = self.output_projs[cb_idx](cb_hidden)  # [B*T, vocab]
            logits = logits.view(batch_size, seq_len, -1)  # [B, T, vocab]
            logits_list.append(logits)

        # Stack logits: [B, K, T, vocab]
        all_logits = torch.stack(logits_list, dim=1)

        # Apply delays to targets and compute loss with masking
        raw_targets = codebook_targets[:, 1:, :]  # [B, K, T]

        # Apply target delays and build validity mask
        shifted_targets = []
        target_valid_masks = []

        for cb_idx in range(self.num_codebooks):
            delay = int(self.target_delays[cb_idx].item())
            shifted, valid = self._apply_delay(raw_targets[:, cb_idx, :], delay, dim=-1)
            shifted_targets.append(shifted)
            target_valid_masks.append(valid)

        targets = torch.stack(shifted_targets, dim=1)  # [B, K, T]

        # Combine validity masks: position is valid if both input and target are valid
        combined_valid = torch.stack(
            [iv & tv for iv, tv in zip(input_valid_masks, target_valid_masks)], dim=0
        )  # [K, T]
        combined_valid = combined_valid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, T]

        # Mark out-of-range target values as invalid (targets may have special tokens)
        out_of_range = (targets < 0) | (targets >= self.vocab_size)
        combined_valid = combined_valid & ~out_of_range
        # Clamp targets to valid range for cross_entropy
        targets = targets.clamp(0, self.vocab_size - 1)

        # Compute loss only on valid positions
        ignore_index = -100
        masked_targets = targets.clone()
        masked_targets[~combined_valid] = ignore_index

        loss = nn.functional.cross_entropy(
            all_logits.permute(0, 3, 1, 2),  # [B, vocab, K, T]
            masked_targets,  # [B, K, T]
            ignore_index=ignore_index,
            reduction="mean",
        )

        return all_logits, loss

    @torch.no_grad()
    def generate_batch(
        self,
        main_hidden: torch.Tensor,
        semantic_tokens: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate acoustic codebooks for a batch of timesteps.

        Processes all timesteps in parallel, using sequential codebook generation
        with KV caching and position IDs matching training (0, 1, 2, ..., 6).

        When delays are enabled, generates tokens at AR positions, then aligns
        outputs to audio time by shifting left (undoing the delay). Invalid
        positions at the end (due to delay) are filled by repeating the last
        valid frame to avoid audio artifacts.

        Args:
            main_hidden: Hidden states from AR decoder [batch, seq_len, main_dim]
            semantic_tokens: Semantic codebook tokens [batch, seq_len]
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated tokens for codebooks 1-7 [batch, 7, seq_len]
        """
        batch_size, seq_len, _ = main_hidden.shape
        device = main_hidden.device

        # Validate sequence length vs delays - disable delays for short sequences
        use_delays = self.use_delays
        if use_delays and seq_len <= self.max_delay:
            logger.info(
                f"Sequence length ({seq_len}) <= max_delay ({self.max_delay}). "
                f"Disabling delays for this batch to avoid invalid output."
            )
            use_delays = False

        flat_batch = batch_size * seq_len

        # Initialize KV cache - each item in flat_batch gets independent cache entries
        # along the batch dimension, so this is correct for parallel processing
        past_key_values = DynamicCache()

        # Flatten semantic tokens as initial previous tokens
        # Validate token range
        out_of_range = (semantic_tokens < 0) | (semantic_tokens >= self.vocab_size)
        if out_of_range.any():
            num_invalid = out_of_range.sum().item()
            logger.warning(
                f"{num_invalid} semantic tokens outside valid range [0, {self.vocab_size}). "
                f"Clamping to valid range. This may indicate AR decoder generated special tokens."
            )
        prev_tokens = semantic_tokens.view(flat_batch).clamp(0, self.vocab_size - 1)
        all_generated = []

        for cb_idx in range(self.num_codebooks):
            # Project with per-codebook projection (multi-linear)
            projected = self.input_projs[cb_idx](main_hidden)  # [B, T, hidden]
            projected_flat = projected.view(flat_batch, 1, -1)  # [B*T, 1, hidden]

            # Embed previous tokens
            token_emb = self.codebook_emb[cb_idx](prev_tokens)  # [B*T, hidden]
            token_emb = token_emb.unsqueeze(1)  # [B*T, 1, hidden]

            # Combine with projected hidden
            cb_input = projected_flat + token_emb  # [B*T, 1, hidden]

            # Position ID matches training: cb_idx for this codebook
            position_ids = torch.full((flat_batch, 1), cb_idx, dtype=torch.long, device=device)
            position_embeddings = self.rotary_emb(cb_input, position_ids)

            # Forward through layers with KV cache
            hidden_states = cb_input
            for layer in self.layers:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                hidden_states = (
                    layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
                )

            # Get logits for current codebook
            cb_hidden = hidden_states[:, -1, :]  # [B*T, hidden]
            cb_hidden = self.output_norms[cb_idx](cb_hidden)
            logits = self.output_projs[cb_idx](cb_hidden)  # [B*T, vocab]

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k sampling
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[-1]))
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(top_k_probs, 1)
            next_tokens = top_k_indices.gather(1, sampled_idx).squeeze(-1)  # [B*T]

            # Store generated tokens and update prev_tokens for next codebook
            all_generated.append(next_tokens.view(batch_size, seq_len))
            prev_tokens = next_tokens

        # Stack: [B, K, T]
        generated = torch.stack(all_generated, dim=1)

        # Apply delay alignment for output if delays are enabled
        # Generated at AR position t is for audio time t - delay
        # To get audio-aligned output: aligned[audio_t] = generated[ar_t] where ar_t = audio_t + delay
        # So: aligned[t] = generated[t + delay] (shift left to undo delay)
        if use_delays:
            aligned = torch.zeros_like(generated)
            for cb_idx in range(self.num_codebooks):
                delay = int(self.target_delays[cb_idx].item())
                if delay < seq_len:
                    # Shift left: aligned[t] = generated[t + delay]
                    valid_len = seq_len - delay
                    aligned[:, cb_idx, :valid_len] = generated[:, cb_idx, delay:]
                    # Fill invalid positions at the end by repeating last valid frame
                    # This avoids audio artifacts from zero-padding
                    if valid_len > 0 and valid_len < seq_len:
                        last_valid = generated[:, cb_idx, -1:]  # [B, 1]
                        aligned[:, cb_idx, valid_len:] = last_valid.expand(-1, seq_len - valid_len)
                elif seq_len > 0:
                    # delay >= seq_len: no valid positions, fill with first generated token
                    aligned[:, cb_idx, :] = generated[:, cb_idx, 0:1].expand(-1, seq_len)
            return aligned

        return generated
