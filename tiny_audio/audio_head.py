"""Audio head for speech-to-speech using a trainable AR decoder + Mimi codec.

Generates audio from LLM embeddings via a trainable LlamaModel decoder:
  LLM hidden (2048) -> Linear (2048->512) -> LlamaModel (6 layers) -> 8 codebook heads -> Mimi codes -> audio (24kHz)

All decoder parameters are trained (~30M params). Direct gradient path from loss to all params.

Training: S2SDataCollator prepares codec_input_ids/codec_labels (simple BOS + codes + EOS format).
AudioHead concatenates projected LLM hidden states with codec token embeddings, runs through
the decoder, and predicts per-codebook logits.

Inference: Autoregressive generation with KV cache.
"""

import logging
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812

logger = logging.getLogger(__name__)

# Mimi codec constants
MIMI_VOCAB_SIZE = 2048
NUM_MIMI_CODEBOOKS = 8
MIMI_SAMPLE_RATE = 24000

# Special tokens (above vocab range)
BOS_TOKEN = 2048
EOS_TOKEN = 2049
PAD_TOKEN = 2050
TOTAL_VOCAB = MIMI_VOCAB_SIZE + 3  # 2051


class AudioHead(nn.Module):
    """Trainable AR decoder for audio generation via Mimi codec.

    Training: projects LLM hidden states, concatenates with codec token embeddings,
    runs through LlamaModel, predicts per-codebook logits with cross-entropy loss.

    Inference: autoregressive generation with KV cache, then Mimi decode to audio.

    Args:
        config: ASRConfig with llm_dim, num_codebooks, decoder_dim, etc.
        llm_dim: Override for LLM dimension
    """

    def __init__(self, config, llm_dim: int = None):
        super().__init__()
        self.llm_dim = llm_dim or getattr(config, "llm_dim", None) or 2048
        self.num_codebooks = getattr(config, "num_codebooks", NUM_MIMI_CODEBOOKS)
        self.decoder_dim = getattr(config, "decoder_dim", 512)
        self.max_tokens = getattr(config, "max_audio_tokens", 500)
        self.vocab_size = MIMI_VOCAB_SIZE

        # Project LLM hidden states to decoder dim
        self.input_proj = nn.Linear(self.llm_dim, self.decoder_dim)

        # Codec token embedding (shared across codebooks, summed at each timestep)
        self.token_embedding = nn.Embedding(TOTAL_VOCAB, self.decoder_dim)

        # Small LlamaModel as decoder backbone (from config, NOT pretrained)
        from transformers import LlamaConfig, LlamaModel

        num_layers = getattr(config, "decoder_layers", 6)
        num_heads = getattr(config, "decoder_heads", 8)
        llama_config = LlamaConfig(
            hidden_size=self.decoder_dim,
            intermediate_size=self.decoder_dim * 4,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            vocab_size=TOTAL_VOCAB,
            max_position_embeddings=4096,
        )
        self.decoder = LlamaModel(llama_config)
        # We handle embeddings ourselves, remove the unused one to save memory
        self.decoder.embed_tokens = None

        # Per-codebook prediction heads (predict full vocab including special tokens)
        self.heads = nn.ModuleList(
            [nn.Linear(self.decoder_dim, TOTAL_VOCAB) for _ in range(self.num_codebooks)]
        )

        # Mimi model (loaded lazily, frozen, inference only)
        self.mimi_model = None

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        codec_labels: Optional[torch.Tensor] = None,
        codec_input_ids: Optional[torch.Tensor] = None,
        codec_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training or inference.

        Args:
            embeddings: LLM hidden states [batch, seq_len, hidden_dim]
            attention_mask: Encoder attention mask [batch, seq_len] (1=real, 0=padding)
            codec_labels: Target codes [batch, audio_len, num_codebooks] (-100 for ignore)
            codec_input_ids: Teacher-forced input [batch, audio_len, num_codebooks]
            codec_attention_mask: Codec attention mask [batch, audio_len]

        Returns:
            Training (codec_labels provided): scalar cross-entropy loss
            Inference (no codec_labels): tuple of (codes [batch, gen_len, num_codebooks], empty tensor)
        """
        # Project LLM hidden states
        prefix = self.input_proj(embeddings)  # [batch, text_len, decoder_dim]
        batch_size, text_len, _ = prefix.shape

        if codec_labels is not None:
            # Teacher forcing: embed codec input tokens, sum across codebooks
            # codec_input_ids: [batch, audio_len, num_codebooks]
            token_emb = self.token_embedding(codec_input_ids)  # [batch, audio_len, num_cb, dim]
            token_emb = token_emb.sum(dim=2)  # [batch, audio_len, dim]

            audio_len = token_emb.shape[1]

            # Concatenate prefix + codec tokens
            hidden = torch.cat([prefix, token_emb], dim=1)  # [batch, text+audio, dim]

            # Build combined attention mask
            # Prefix mask: from encoder attention_mask (or all ones)
            if attention_mask is not None:
                prefix_mask = attention_mask  # [batch, text_len]
            else:
                prefix_mask = torch.ones(
                    batch_size, text_len, device=hidden.device, dtype=torch.long
                )

            # Codec mask
            if codec_attention_mask is not None:
                audio_mask = codec_attention_mask  # [batch, audio_len]
            else:
                audio_mask = torch.ones(
                    batch_size, audio_len, device=hidden.device, dtype=torch.long
                )

            combined_mask = torch.cat([prefix_mask, audio_mask], dim=1)  # [batch, total_len]

            # Build causal mask for codec positions while prefix attends bidirectionally
            total_len = text_len + audio_len
            # Start with causal mask (lower triangular)
            causal_mask = torch.triu(
                torch.full((total_len, total_len), float("-inf"), device=hidden.device),
                diagonal=1,
            )
            # Allow prefix positions to attend bidirectionally to each other
            causal_mask[:text_len, :text_len] = 0.0
            # Expand for batch: [batch, 1, total_len, total_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

            # Apply padding mask: positions where combined_mask == 0 should not be attended to
            padding_mask = (1 - combined_mask).bool()  # True where padded
            # Expand to [batch, 1, 1, total_len] for broadcasting
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2).expand_as(causal_mask)
            causal_mask = causal_mask.masked_fill(padding_mask_expanded, float("-inf"))

            # Position IDs
            position_ids = (
                torch.arange(total_len, device=hidden.device).unsqueeze(0).expand(batch_size, -1)
            )

            # Run through LlamaModel
            outputs = self.decoder(
                inputs_embeds=hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )

            # Extract audio positions only
            audio_hidden = outputs.last_hidden_state[:, text_len:]  # [batch, audio_len, dim]

            # Predict per codebook and compute loss
            # codec_labels: [batch, audio_len, num_codebooks]
            # Clamp labels to valid range to prevent CUDA device-side asserts
            safe_labels = codec_labels.clone()
            valid_mask = safe_labels != -100
            safe_labels[valid_mask] = safe_labels[valid_mask].clamp(0, TOTAL_VOCAB - 1)

            loss = torch.tensor(0.0, device=hidden.device, dtype=audio_hidden.dtype)
            for cb, head in enumerate(self.heads):
                logits = head(audio_hidden)  # [batch, audio_len, total_vocab]
                cb_labels = safe_labels[:, :, cb].reshape(-1)  # [batch * audio_len]
                loss = loss + F.cross_entropy(
                    logits.reshape(-1, TOTAL_VOCAB),
                    cb_labels,
                    ignore_index=-100,
                )
            return loss / self.num_codebooks

        # Inference: autoregressive generation
        return self._generate(prefix, attention_mask)

    def _generate(
        self, prefix: torch.Tensor, prefix_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """AR generation: predict one timestep at a time with KV cache."""
        batch_size, text_len, _ = prefix.shape
        device = prefix.device

        # Start with BOS token for all codebooks
        current_tokens = torch.full(
            (batch_size, 1, self.num_codebooks), BOS_TOKEN, dtype=torch.long, device=device
        )
        all_codes = []

        # Build initial input: prefix + BOS embedding
        bos_emb = self.token_embedding(current_tokens).sum(dim=2)  # [batch, 1, dim]
        hidden = torch.cat([prefix, bos_emb], dim=1)  # [batch, text_len+1, dim]

        # Position IDs for initial forward
        position_ids = torch.arange(text_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)

        # Initial forward pass (no KV cache yet)
        outputs = self.decoder(
            inputs_embeds=hidden,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # Get prediction from last position
        last_hidden = outputs.last_hidden_state[:, -1:]  # [batch, 1, dim]

        for step in range(self.max_tokens):
            # Predict codes for current timestep from each head
            step_codes = []
            for head in self.heads:
                logits = head(last_hidden.squeeze(1))  # [batch, vocab]
                tokens = logits.argmax(dim=-1)  # [batch]
                step_codes.append(tokens)

            # Stack: [batch, num_codebooks]
            step_codes = torch.stack(step_codes, dim=1)

            # Check for EOS
            if (step_codes == EOS_TOKEN).any(dim=1).all():
                break

            all_codes.append(step_codes)

            # Embed predicted tokens for next step
            # [batch, 1, num_codebooks]
            next_input = step_codes.unsqueeze(1)
            next_emb = self.token_embedding(next_input).sum(dim=2)  # [batch, 1, dim]

            # Next position
            next_pos = torch.full(
                (batch_size, 1), text_len + 1 + step + 1, dtype=torch.long, device=device
            )

            # Forward with KV cache
            outputs = self.decoder(
                inputs_embeds=next_emb,
                position_ids=next_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_hidden = outputs.last_hidden_state  # [batch, 1, dim]

        if all_codes:
            codes = torch.stack(all_codes, dim=1)  # [batch, gen_len, num_codebooks]
        else:
            codes = torch.empty(batch_size, 0, self.num_codebooks, dtype=torch.long, device=device)

        return codes, torch.empty(0, device=device)

    def _load_mimi(self):
        """Load frozen Mimi model for audio decoding."""
        from transformers import MimiModel

        self.mimi_model = MimiModel.from_pretrained("kyutai/mimi")
        self.mimi_model.eval()
        self.mimi_model.requires_grad_(False)
        logger.info("Loaded frozen Mimi model for audio decoding")

    def decode_to_audio(self, codes: torch.Tensor) -> list[torch.Tensor]:
        """Decode Mimi codec tokens to audio waveforms.

        Args:
            codes: Codec tokens [batch, seq_len, num_codebooks]

        Returns:
            List of audio waveform tensors (one per batch item)
        """
        if self.mimi_model is None:
            self._load_mimi()
        assert self.mimi_model is not None

        # codes: [batch, seq_len, num_codebooks] -> [batch, num_codebooks, seq_len]
        codes_transposed = codes.transpose(1, 2).to(self.mimi_model.device)
        with torch.no_grad():
            decoder_output = self.mimi_model.decode(codes_transposed)
            audio_values = decoder_output.audio_values  # [batch, 1, samples]
            assert audio_values is not None

        return [audio_values[i, 0] for i in range(audio_values.shape[0])]

    def generate_streaming(
        self,
        embeddings: torch.Tensor,
        chunk_samples: int = 24000,
    ) -> Iterator[torch.Tensor]:
        """Generate audio and yield waveform chunks for streaming playback.

        Args:
            embeddings: LLM hidden states [batch, seq_len, llm_dim]
            chunk_samples: Audio samples per chunk (default 1s at 24kHz)

        Yields:
            Audio waveform chunks [samples]
        """
        codes, _ = self(embeddings)
        audios = self.decode_to_audio(codes)

        for audio in audios:
            for start in range(0, audio.shape[-1], chunk_samples):
                end = min(start + chunk_samples, audio.shape[-1])
                yield audio[..., start:end]
