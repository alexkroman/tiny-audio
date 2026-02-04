"""Moshi-style Depformer for multi-codebook audio token prediction.

This module implements a Depth Transformer (Depformer) that generates multiple
codec codebooks, following Moshi's architecture closely.

Architecture:
    For each time step, all codebooks are processed together:
        - cb_0: depformer_in[0](llm_hidden) + text_emb(prev_text_token)
        - cb_1: depformer_in[1](llm_hidden) + audio_emb[0](prev_cb_0_token)
        - cb_2: depformer_in[2](llm_hidden) + audio_emb[1](prev_cb_1_token)
        - ...

    Inputs are stacked [B*T, K, D] and processed by depformer transformer.
    Per-codebook norms and output heads produce logits.

Training:
    Uses teacher forcing - previous codebook tokens come from ground truth.
    Acoustic delay τ: semantic tokens at time t, acoustic tokens from t-τ.
    Loss is weighted (first codebook 100x) and averaged.
"""

import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class Depformer(nn.Module):
    """Moshi-style Depth Transformer for multi-codebook prediction.

    Args:
        config: ASRConfig with:
            - llm_dim: LLM hidden dimension (default: 2048)
            - depformer_dim: Depformer hidden dimension (default: 1024)
            - depformer_num_layers: Number of transformer layers (default: 6)
            - num_codebooks: Number of codebooks to predict (default: 8)
            - codebook_size: Vocabulary size per codebook (default: 2048)
            - acoustic_delay: Delay τ for acoustic codebooks (default: 1)
            - first_codebook_weight: Loss weight for semantic tokens (default: 100.0)
    """

    def __init__(self, config):
        super().__init__()

        # Dimensions (Moshi code defaults, codebook_size matches Mimi codec)
        self.llm_dim = getattr(config, "llm_dim", 2048)
        self.hidden_dim = getattr(config, "depformer_dim", 512)  # 1/4 of SmolLM3-3B hidden
        self.num_layers = getattr(config, "depformer_num_layers", 6)
        self.num_codebooks = getattr(config, "num_codebooks", 8)
        self.vocab_size = getattr(config, "codebook_size", 2048)  # Mimi codec default

        # Acoustic delay τ (Moshi uses 1 or 2, "greatly improves quality")
        self.acoustic_delay = getattr(config, "acoustic_delay", 1)

        # Vocab includes special initial token
        self.initial_token_id = self.vocab_size  # Token for start of sequence
        self.total_vocab_size = self.vocab_size + 1

        # Number of attention heads (64 dims per head)
        num_heads = max(1, self.hidden_dim // 64)

        # Llama config for depformer layers with GELU activation (like Moshi)
        self.llama_config = LlamaConfig(
            vocab_size=self.total_vocab_size,
            hidden_size=self.hidden_dim,
            intermediate_size=self.hidden_dim * 4,
            num_hidden_layers=self.num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=self.num_codebooks,
            rms_norm_eps=1e-6,
            hidden_act="gelu",  # Moshi uses GELU (not SiLU)
            _attn_implementation="sdpa",
        )

        # Per-codebook input projections from LLM hidden states (like Moshi's depformer_in)
        self.depformer_in = nn.ModuleList(
            [
                nn.Linear(self.llm_dim, self.hidden_dim, bias=False)
                for _ in range(self.num_codebooks)
            ]
        )

        # Per-codebook embeddings for conditioning on previous codebook tokens
        # cb_0 doesn't need one (it's the first), so we have num_codebooks-1
        # Each embeds audio tokens from the previous codebook
        self.depformer_emb = nn.ModuleList(
            [
                nn.Embedding(self.total_vocab_size, self.hidden_dim)
                for _ in range(self.num_codebooks - 1)
            ]
        )

        # Depformer transformer layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(self.llama_config, layer_idx=i) for i in range(self.num_layers)]
        )

        # Per-codebook output norms (like Moshi's depformer_norms)
        self.depformer_norms = nn.ModuleList(
            [LlamaRMSNorm(self.hidden_dim, eps=1e-6) for _ in range(self.num_codebooks)]
        )

        # Rotary embeddings for position encoding across codebooks
        self.rotary_emb = LlamaRotaryEmbedding(config=self.llama_config)

        # Per-codebook output heads (like Moshi's linears)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
                for _ in range(self.num_codebooks)
            ]
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # First codebook (semantic tokens) gets higher weight (Moshi uses 100x)
        self.first_codebook_weight = getattr(config, "first_codebook_weight", 100.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        codec_targets: torch.Tensor | None = None,
        codec_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for training or inference.

        Args:
            hidden_states: LLM hidden states (batch, seq_len, llm_dim)
            codec_targets: Target tokens (batch, num_codebooks, seq_len) for training
            codec_lengths: Actual lengths per sample (batch,)

        Returns:
            Training: scalar loss (averaged across all codebooks)
            Inference: predicted tokens (batch, num_codebooks, seq_len)
        """
        if codec_targets is not None:
            return self._forward_train(hidden_states, codec_targets, codec_lengths)
        return self._forward_inference(hidden_states)

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Training with teacher forcing and acoustic delay.

        Following Moshi's approach:
        1. Semantic codebook (cb_0): predicts at time t
        2. Acoustic codebooks (cb_1-7): predict from time t-τ (acoustic delay)
        3. Build inputs for all codebooks (LLM projection + prev token embedding)
        4. Stack to [B*T, K, D] and process through depformer
        5. Apply per-codebook norms and output heads
        """
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        num_cbs = self.num_codebooks
        tau = self.acoustic_delay

        # targets shape: (batch, num_codebooks, seq_len) or (batch, seq_len)
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)

        num_target_codebooks = min(targets.shape[1], num_cbs)
        target_seq_len = targets.shape[2]

        # Effective sequence length accounting for acoustic delay
        # We need at least tau+1 steps to have valid acoustic targets
        eff_seq_len = min(seq_len, target_seq_len) - tau
        if eff_seq_len <= 0:
            # Sequence too short for acoustic delay, fall back to no delay
            return self._forward_train_no_delay(hidden_states, targets, lengths)

        # Semantic codebook: uses hidden[:, tau:] to predict targets[:, 0, tau:]
        # Acoustic codebooks: uses hidden[:, tau:] to predict targets[:, 1:, :eff_seq_len]
        # This way both are aligned at the same hidden state positions

        hidden_delayed = hidden_states[:, tau : tau + eff_seq_len]  # [bsz, eff_seq_len, dim]

        # Build depformer inputs for all codebooks
        depformer_inputs = []

        for cb_idx in range(num_target_codebooks):
            # Project LLM hidden states for this codebook
            transformer_in = self.depformer_in[cb_idx](hidden_delayed)

            # Add conditioning from previous codebook token (teacher forcing)
            if cb_idx == 0:
                # First codebook (semantic): just use transformer projection
                depformer_inputs.append(transformer_in)
            else:
                # Acoustic codebooks: embed previous codebook's tokens
                # Use tokens from the delayed position (t-tau for acoustic)
                prev_tokens = targets[:, cb_idx - 1, :eff_seq_len]
                token_in = self.depformer_emb[cb_idx - 1](prev_tokens)
                depformer_inputs.append(token_in + transformer_in)

        # Stack: [bsz, eff_seq_len, num_cbs, hidden_dim]
        depformer_input = torch.stack(depformer_inputs, dim=2)

        # Reshape to [bsz*eff_seq_len, num_cbs, hidden_dim] for processing
        depformer_input = depformer_input.view(bsz * eff_seq_len, num_target_codebooks, -1)

        # Position embeddings across codebook dimension
        position_ids = torch.arange(num_target_codebooks, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(depformer_input, position_ids)

        # Forward through depformer layers
        hidden = depformer_input
        for layer in self.layers:
            layer_out = layer(hidden, position_embeddings=position_embeddings)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Apply per-codebook norms and compute logits
        all_logits = []
        for cb_idx in range(num_target_codebooks):
            cb_hidden = self.depformer_norms[cb_idx](hidden[:, cb_idx])
            logits = self.linears[cb_idx](cb_hidden)
            logits = logits.view(bsz, eff_seq_len, -1)
            all_logits.append(logits)

        # Stack logits: [bsz, num_cbs, eff_seq_len, vocab]
        logits = torch.stack(all_logits, dim=1)

        # Create padding mask from lengths (adjusted for delay)
        if lengths is not None:
            adjusted_lengths = torch.clamp(lengths - tau, min=0)
            position_indices = torch.arange(eff_seq_len, device=device).unsqueeze(0)
            valid_mask = position_indices < adjusted_lengths.unsqueeze(1)
        else:
            valid_mask = torch.ones(bsz, eff_seq_len, dtype=torch.bool, device=device)

        # Compute loss with first codebook weighted higher (semantic tokens)
        total_loss = torch.tensor(0.0, device=device)
        total_weight = 0.0

        for cb_idx in range(num_target_codebooks):
            cb_logits = logits[:, cb_idx]

            if cb_idx == 0:
                # Semantic: predict targets at time t (with delay offset)
                cb_targets = targets[:, 0, tau : tau + eff_seq_len].clone()
            else:
                # Acoustic: predict targets from time 0 to eff_seq_len (no offset)
                cb_targets = targets[:, cb_idx, :eff_seq_len].clone()

            # Mask out padding
            cb_targets[~valid_mask] = -100

            loss = self.criterion(
                cb_logits.reshape(-1, cb_logits.shape[-1]),
                cb_targets.reshape(-1),
            )

            # First codebook (semantic) gets higher weight
            weight = self.first_codebook_weight if cb_idx == 0 else 1.0
            total_loss = total_loss + loss * weight
            total_weight += weight

        return total_loss / total_weight

    def _forward_train_no_delay(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fallback training without acoustic delay (for very short sequences)."""
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        num_cbs = self.num_codebooks

        num_target_codebooks = min(targets.shape[1], num_cbs)
        target_seq_len = targets.shape[2]
        tgt_len = min(seq_len, target_seq_len)

        # Build depformer inputs for all codebooks
        depformer_inputs = []

        for cb_idx in range(num_target_codebooks):
            transformer_in = self.depformer_in[cb_idx](hidden_states[:, :tgt_len])

            if cb_idx == 0:
                depformer_inputs.append(transformer_in)
            else:
                prev_tokens = targets[:, cb_idx - 1, :tgt_len]
                token_in = self.depformer_emb[cb_idx - 1](prev_tokens)
                depformer_inputs.append(token_in + transformer_in)

        depformer_input = torch.stack(depformer_inputs, dim=2)
        depformer_input = depformer_input.view(bsz * tgt_len, num_target_codebooks, -1)

        position_ids = torch.arange(num_target_codebooks, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(depformer_input, position_ids)

        hidden = depformer_input
        for layer in self.layers:
            layer_out = layer(hidden, position_embeddings=position_embeddings)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        all_logits = []
        for cb_idx in range(num_target_codebooks):
            cb_hidden = self.depformer_norms[cb_idx](hidden[:, cb_idx])
            logits = self.linears[cb_idx](cb_hidden)
            logits = logits.view(bsz, tgt_len, -1)
            all_logits.append(logits)

        logits = torch.stack(all_logits, dim=1)

        if lengths is not None:
            position_indices = torch.arange(tgt_len, device=device).unsqueeze(0)
            valid_mask = position_indices < lengths.unsqueeze(1)
        else:
            valid_mask = torch.ones(bsz, tgt_len, dtype=torch.bool, device=device)

        total_loss = torch.tensor(0.0, device=device)
        total_weight = 0.0

        for cb_idx in range(num_target_codebooks):
            cb_logits = logits[:, cb_idx]
            cb_targets = targets[:, cb_idx, :tgt_len].clone()
            cb_targets[~valid_mask] = -100

            loss = self.criterion(
                cb_logits.reshape(-1, cb_logits.shape[-1]),
                cb_targets.reshape(-1),
            )

            weight = self.first_codebook_weight if cb_idx == 0 else 1.0
            total_loss = total_loss + loss * weight
            total_weight += weight

        return total_loss / total_weight

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive inference generating all codebooks with acoustic delay.

        Args:
            hidden_states: LLM hidden states (batch, seq_len, llm_dim)
            temperature: Sampling temperature (1.0 = greedy)

        Returns:
            Generated tokens (batch, num_codebooks, seq_len)
        """
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        num_cbs = self.num_codebooks
        tau = self.acoustic_delay

        # Output tensor
        generated = torch.zeros(bsz, num_cbs, seq_len, dtype=torch.long, device=device)

        # Process each time step
        for t in range(seq_len):
            llm_hidden_t = hidden_states[:, t]  # [bsz, llm_dim]

            # Build inputs for all codebooks at this time step
            depformer_inputs = []

            for cb_idx in range(num_cbs):
                transformer_in = self.depformer_in[cb_idx](llm_hidden_t)

                if cb_idx == 0:
                    # Semantic: no previous token conditioning
                    depformer_inputs.append(transformer_in)
                else:
                    # Acoustic: use token from t-tau (or initial token if t < tau)
                    if t >= tau:
                        prev_token = generated[:, cb_idx - 1, t - tau]
                    else:
                        prev_token = torch.full(
                            (bsz,), self.initial_token_id, dtype=torch.long, device=device
                        )
                    token_in = self.depformer_emb[cb_idx - 1](prev_token)
                    depformer_inputs.append(token_in + transformer_in)

            # Stack: [bsz, num_cbs, hidden_dim]
            depformer_input = torch.stack(depformer_inputs, dim=1)

            # Position embeddings
            position_ids = torch.arange(num_cbs, device=device).unsqueeze(0)
            position_embeddings = self.rotary_emb(depformer_input, position_ids)

            # Forward through depformer
            hidden = depformer_input
            for layer in self.layers:
                layer_out = layer(hidden, position_embeddings=position_embeddings)
                hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            # Generate tokens for each codebook
            for cb_idx in range(num_cbs):
                cb_hidden = self.depformer_norms[cb_idx](hidden[:, cb_idx])
                logits = self.linears[cb_idx](cb_hidden)

                if temperature <= 0:
                    token = logits.argmax(dim=-1)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    token = torch.multinomial(probs, num_samples=1).squeeze(-1)

                generated[:, cb_idx, t] = token

                # Update input for next codebook (if not last)
                if cb_idx < num_cbs - 1:
                    next_transformer_in = self.depformer_in[cb_idx + 1](llm_hidden_t)
                    next_token_in = self.depformer_emb[cb_idx](token)
                    depformer_input[:, cb_idx + 1] = next_token_in + next_transformer_in

        return generated

    def get_output_length(self, input_length: int) -> int:
        """Estimate output codec token count."""
        return int(input_length * 2)
