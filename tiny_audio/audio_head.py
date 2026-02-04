"""Autoregressive Audio Head for Mimi codec token prediction.

This module implements a Freeze-Omni style AR decoder that predicts Mimi codec tokens
from LLM hidden states using cross-entropy loss.

Architecture:
    LLM Hidden States → Pre-NN (4 Llama layers) → AR Decoder (8 Llama layers) → Token logits

Training:
    Uses audio + pre-computed Mimi codes from mazesmazes/libritts-r-mimi-audio dataset.
    Learns to predict codec tokens from audio input (speech alignment).
    Simple cross-entropy loss on codec tokens, no teacher model needed.

Inference:
    Autoregressive generation of codec tokens that can be decoded to audio.
"""

import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from .projectors import SimpleAdapter


class AudioHead(nn.Module):
    """Freeze-Omni style AR decoder for Mimi codec tokens.

    Architecture:
        LLM hidden → Pre-NN (4 layers) → AR Decoder (8 layers) → Token logits

    The model uses teacher forcing during training and autoregressive generation
    during inference.
    """

    # Special token offsets from vocab_size
    BOS_OFFSET = 0  # Beginning of sequence (prepended to LLM hidden)
    SOS_OFFSET = 1  # Start of speech (first AR input token)
    EOS_OFFSET = 2  # End of speech (target for last position)
    PAD_OFFSET = 3  # Padding token

    def __init__(self, config):
        """Initialize AudioHead with AR decoder architecture.

        Args:
            config: ASRConfig with audio head parameters:
                - llm_dim: Main LLM hidden dimension (default: 1536)
                - audio_head_hidden_dim: AR decoder hidden dimension (default: 512)
                - codebook_size: Mimi codec vocabulary size (default: 2048)
                - num_codebooks: Number of codebooks to predict (default: 1)
        """
        super().__init__()

        # Dimensions
        self.llm_dim = getattr(config, "llm_dim", 1536)
        self.hidden_dim = getattr(config, "audio_head_hidden_dim", 512)
        self.vocab_size = getattr(config, "codebook_size", 2048)
        self.num_codebooks = getattr(config, "num_codebooks", 1)

        # Special tokens (after vocab)
        self.bos_id = self.vocab_size + self.BOS_OFFSET
        self.sos_id = self.vocab_size + self.SOS_OFFSET
        self.eos_id = self.vocab_size + self.EOS_OFFSET
        self.pad_id = self.vocab_size + self.PAD_OFFSET
        self.total_vocab_size = self.vocab_size + 4  # vocab + 4 special tokens

        # Calculate number of attention heads based on hidden dim
        # head_dim should be at least 8 for RoPE to work correctly
        num_heads = max(1, self.hidden_dim // 64)  # Each head is 64 dims
        if num_heads == 0:
            num_heads = 1  # Ensure at least 1 head

        # Llama config for decoder layers
        self.llama_config = LlamaConfig(
            vocab_size=self.total_vocab_size,
            hidden_size=self.hidden_dim,
            intermediate_size=self.hidden_dim * 4,
            num_hidden_layers=8,  # Main AR decoder depth
            num_attention_heads=num_heads,
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            _attn_implementation="sdpa",  # Use scaled dot product attention
        )

        # Input projection: LLM dim → hidden dim (2-layer MLP)
        self.input_proj = SimpleAdapter(
            input_dim=self.llm_dim,
            hidden_dim=self.llm_dim,
            output_dim=self.hidden_dim,
        )

        # Token embedding (vocab + 4 special tokens)
        self.embedding = nn.Embedding(
            self.total_vocab_size,
            self.hidden_dim,
            padding_idx=self.pad_id,
        )

        # Pre-NN: Process LLM hidden states (4 layers)
        self.pre_nn_layers = nn.ModuleList(
            [LlamaDecoderLayer(self.llama_config, layer_idx=i) for i in range(4)]
        )
        self.pre_nn_norm = LlamaRMSNorm(self.hidden_dim, eps=1e-6)

        # AR Decoder layers (8 layers)
        self.decoder_layers = nn.ModuleList(
            [LlamaDecoderLayer(self.llama_config, layer_idx=i) for i in range(8)]
        )
        self.decoder_norm = LlamaRMSNorm(self.hidden_dim, eps=1e-6)

        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config=self.llama_config)

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.total_vocab_size)

        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        codec_targets: torch.Tensor | None = None,
        codec_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for training or inference.

        Args:
            hidden_states: LLM hidden states (batch, llm_seq, llm_dim)
            codec_targets: Target codec tokens (batch, audio_len) for training
            codec_lengths: Actual lengths of targets (batch,)

        Returns:
            Training: scalar loss
            Inference: predicted token IDs (batch, audio_len)
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # Project LLM hidden states to decoder dimension
        hidden_states = self.input_proj(hidden_states)

        # Process through Pre-NN
        hidden_states = self._forward_pre_nn(hidden_states)

        # Add BOS embedding
        bos_emb = self.embedding(
            torch.full((batch_size, 1), self.bos_id, device=device, dtype=torch.long)
        )
        context = torch.cat([bos_emb, hidden_states], dim=1)  # (batch, 1+llm_seq, hidden)

        if codec_targets is not None:
            # Training: teacher forcing
            return self._forward_train(context, codec_targets, codec_lengths)
        # Inference: autoregressive generation
        return self._forward_inference(context)

    def _forward_pre_nn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process LLM hidden states through Pre-NN layers."""
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        # Get rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.pre_nn_layers:
            layer_out = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )
            # Transformers 5.0+ returns tensor directly, earlier returns tuple
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return self.pre_nn_norm(hidden_states)

    def _forward_train(
        self,
        context: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Training forward with teacher forcing."""
        batch_size = context.shape[0]
        context_len = context.shape[1]
        device = context.device
        max_target_len = targets.shape[1]

        # Prepare input: SOS + targets[:-1] (shifted right)
        sos_tokens = torch.full((batch_size, 1), self.sos_id, device=device, dtype=torch.long)
        input_tokens = torch.cat([sos_tokens, targets[:, :-1]], dim=1)  # (batch, max_target_len)
        input_embeds = self.embedding(input_tokens)

        # Prepare output targets: targets + EOS at length position
        output_targets = targets.clone()
        if lengths is not None:
            for i, length in enumerate(lengths):
                length = int(length.item())
                if length < max_target_len:
                    output_targets[i, length] = self.eos_id
                    output_targets[i, length + 1 :] = self.pad_id

        # Concatenate context + input embeddings
        full_embeds = torch.cat([context, input_embeds], dim=1)
        total_len = full_embeds.shape[1]

        # Create causal mask
        attn_mask = self._create_train_mask(batch_size, context_len, max_target_len, device)

        # Position IDs
        position_ids = torch.arange(total_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(full_embeds, position_ids)

        # Forward through decoder
        hidden = full_embeds
        for layer in self.decoder_layers:
            layer_out = layer(
                hidden,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
            )
            # Transformers 5.0+ returns tensor directly, earlier returns tuple
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden = self.decoder_norm(hidden)

        # Get logits for AR portion only
        ar_hidden = hidden[:, context_len:]  # (batch, max_target_len, hidden)
        logits = self.output_proj(ar_hidden)  # (batch, max_target_len, vocab+4)

        # Compute loss
        return self.criterion(
            logits.reshape(-1, logits.shape[-1]),
            output_targets.reshape(-1),
        )

    def _forward_inference(self, context: torch.Tensor, max_tokens: int = 500) -> torch.Tensor:
        """Autoregressive inference."""
        batch_size = context.shape[0]
        device = context.device

        # Start with SOS token
        generated = torch.full((batch_size, 1), self.sos_id, device=device, dtype=torch.long)

        for _ in range(max_tokens):
            # Embed generated tokens
            gen_embeds = self.embedding(generated)

            # Concatenate context + generated
            full_embeds = torch.cat([context, gen_embeds], dim=1)
            total_len = full_embeds.shape[1]

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(total_len, total_len, device=device), diagonal=1
            ).bool()
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask.expand(batch_size, 1, -1, -1)
            # Convert to float mask with -inf for masked positions
            attn_mask = torch.where(
                attn_mask,
                torch.tensor(float("-inf"), device=device),
                torch.tensor(0.0, device=device),
            )

            # Position embeddings
            position_ids = torch.arange(total_len, device=device).unsqueeze(0)
            position_embeddings = self.rotary_emb(full_embeds, position_ids)

            # Forward
            hidden = full_embeds
            for layer in self.decoder_layers:
                layer_out = layer(
                    hidden,
                    attention_mask=attn_mask,
                    position_embeddings=position_embeddings,
                )
                # Transformers 5.0+ returns tensor directly, earlier returns tuple
                hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            hidden = self.decoder_norm(hidden)
            logits = self.output_proj(hidden[:, -1:])  # Last position

            # Greedy decode
            next_token = logits.argmax(dim=-1)  # (batch, 1)

            # Check for EOS
            if (next_token == self.eos_id).all():
                break

            generated = torch.cat([generated, next_token], dim=1)

        # Remove SOS token from output
        return generated[:, 1:]

    def _create_train_mask(
        self,
        batch_size: int,
        context_len: int,
        target_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create attention mask for training.

        Context tokens can attend to all context.
        AR tokens can attend to context + previous AR tokens (causal).
        """
        total_len = context_len + target_len

        # Start with zeros (no masking)
        mask = torch.zeros(batch_size, 1, total_len, total_len, device=device)

        # AR tokens: causal attention to other AR tokens (upper triangular = -inf)
        ar_causal = torch.triu(
            torch.ones(target_len, target_len, device=device) * float("-inf"),
            diagonal=1,
        )
        mask[:, :, context_len:, context_len:] = ar_causal

        return mask

    def get_output_length(self, input_length: int) -> int:
        """Estimate output codec token count.

        Mimi uses 12.5 Hz frame rate. For simplicity, we estimate based on
        typical speech duration ratios.

        Args:
            input_length: Number of input LLM hidden states

        Returns:
            Estimated number of codec tokens
        """
        # Rough estimate: LLM tokens -> speech frames
        # This will be refined based on actual audio duration
        return int(input_length * 2)

    def state_dict(self, *args, **kwargs):
        """Return full state dict."""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict."""
        return super().load_state_dict(state_dict, strict=strict)
