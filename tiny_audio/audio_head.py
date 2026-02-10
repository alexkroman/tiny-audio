"""Audio head for speech-to-speech using a trainable AR decoder + NeuCodec.

Generates audio from text tokens via a trainable LlamaModel decoder:
  Text tokens -> Embedding -> LlamaModel -> head -> NeuCodec FSQ codes -> audio

NeuCodec uses a single FSQ codebook (levels=[4]*8, vocab=65536) at 50 tokens/sec,
outputting 24kHz audio. No multi-codebook handling needed.

Training: S2SDataCollator prepares codec_input_ids/codec_labels (both 2D: [batch, seq_len]).
AudioHead predicts FSQ codes via a single head with teacher forcing.

Inference: Autoregressive generation with KV cache, feeding back predicted codes.
"""

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)

# NeuCodec FSQ constants (levels=[4]*8, 1 quantizer -> 4^8 = 65536 codes)
NEUCODEC_VOCAB_SIZE = 65536
NEUCODEC_SAMPLE_RATE = 24000

# Special tokens (above vocab range)
BOS_TOKEN = NEUCODEC_VOCAB_SIZE
EOS_TOKEN = NEUCODEC_VOCAB_SIZE + 1
PAD_TOKEN = NEUCODEC_VOCAB_SIZE + 2
TOTAL_VOCAB = NEUCODEC_VOCAB_SIZE + 3  # 65539


class AudioHeadConfig(PretrainedConfig):
    """Configuration class for the AudioHead model."""

    model_type = "audio_head"

    def __init__(
        self,
        decoder_dim: int = 512,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        text_vocab_size: int = 32000,
        max_audio_tokens: int = 500,
        neucodec_model_id: str = "neuphonic/neucodec",
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs,
    ):
        self.decoder_dim = decoder_dim
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.text_vocab_size = text_vocab_size
        self.max_audio_tokens = max_audio_tokens
        self.neucodec_model_id = neucodec_model_id
        self.temperature = temperature
        self.top_k = top_k
        super().__init__(**kwargs)


@dataclass
class AudioHeadOutput(ModelOutput):
    """Output of AudioHead forward pass.

    Attributes:
        loss: Cross-entropy loss when codec_labels are provided.
        codes: Generated codec codes when in inference mode [batch, gen_len].
    """

    loss: Optional[torch.Tensor] = None
    codes: Optional[torch.Tensor] = None


class AudioHead(PreTrainedModel):
    """Trainable AR decoder that predicts NeuCodec FSQ codes.

    NeuCodec uses a single FSQ codebook (4^8 = 65536 codes) at 50 tokens/sec.
    No multi-codebook handling needed — just a flat sequence of codes.
    """

    config_class = AudioHeadConfig

    def __init__(self, config: AudioHeadConfig):
        super().__init__(config)
        self.text_vocab_size = config.text_vocab_size
        self.decoder_dim = config.decoder_dim
        self.max_tokens = config.max_audio_tokens
        self.vocab_size = NEUCODEC_VOCAB_SIZE

        # Embed text tokens to decoder dim
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.decoder_dim)

        # Codec token embedding (FSQ codes + special tokens)
        self.token_embedding = nn.Embedding(TOTAL_VOCAB, config.decoder_dim)

        # Small LlamaModel as decoder backbone (from config, NOT pretrained)
        from transformers import LlamaConfig, LlamaModel

        llama_config = LlamaConfig(
            hidden_size=config.decoder_dim,
            intermediate_size=config.decoder_dim * 4,
            num_hidden_layers=config.decoder_layers,
            num_attention_heads=config.decoder_heads,
            vocab_size=TOTAL_VOCAB,
            max_position_embeddings=4096,
        )
        self.decoder = LlamaModel(llama_config)
        # We handle embeddings ourselves, remove the unused one to save memory
        self.decoder.embed_tokens = None

        # Sampling parameters for inference
        self.temperature = config.temperature
        self.top_k = config.top_k

        # NeuCodec model (loaded lazily, frozen, inference only)
        self.neucodec_model = None

        # Initialize weights
        self.post_init()

    def forward(
        self,
        text_token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        codec_labels: Optional[torch.Tensor] = None,
        codec_input_ids: Optional[torch.Tensor] = None,
        codec_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AudioHeadOutput:
        """Forward pass for training or inference.

        Args:
            text_token_ids: Text token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len] (1=real, 0=padding)
            codec_labels: Target codes [batch, audio_len] (-100 for ignore)
            codec_input_ids: Teacher-forced input [batch, audio_len]
            codec_attention_mask: Codec attention mask [batch, audio_len]

        Returns:
            AudioHeadOutput with loss (training) or codes (inference).
        """
        # Embed text tokens (clamp to valid range)
        if (text_token_ids >= self.text_vocab_size).any() or (text_token_ids < 0).any():
            logger.warning(
                "text_token_ids out of range [0, %d): min=%d max=%d. Clamping.",
                self.text_vocab_size,
                text_token_ids.min().item(),
                text_token_ids.max().item(),
            )
            text_token_ids = text_token_ids.clamp(0, self.text_vocab_size - 1)
        prefix = self.text_embedding(text_token_ids)  # [batch, text_len, decoder_dim]
        batch_size, text_len, _ = prefix.shape

        if codec_labels is not None:
            # Teacher forcing: codec_input_ids is [batch, audio_len]
            assert codec_input_ids is not None, (
                "codec_input_ids required when codec_labels provided"
            )
            cb_input = codec_input_ids
            if (cb_input >= TOTAL_VOCAB).any() or (cb_input < 0).any():
                logger.warning(
                    "codec_input_ids out of range [0, %d): min=%d max=%d. Clamping.",
                    TOTAL_VOCAB,
                    cb_input.min().item(),
                    cb_input.max().item(),
                )
                cb_input = cb_input.clamp(0, TOTAL_VOCAB - 1)
            token_emb = self.token_embedding(cb_input)  # [batch, audio_len, dim]

            audio_len = token_emb.shape[1]

            # Concatenate prefix + codec tokens
            hidden = torch.cat([prefix, token_emb], dim=1)  # [batch, text+audio, dim]

            # Build combined attention mask
            if attention_mask is not None:
                prefix_mask = attention_mask
            else:
                prefix_mask = torch.ones(
                    batch_size, text_len, device=hidden.device, dtype=torch.long
                )

            if codec_attention_mask is not None:
                audio_mask = codec_attention_mask
            else:
                audio_mask = torch.ones(
                    batch_size, audio_len, device=hidden.device, dtype=torch.long
                )

            combined_mask = torch.cat([prefix_mask, audio_mask], dim=1)

            # Build causal mask for codec positions while prefix attends bidirectionally
            total_len = text_len + audio_len
            causal_mask = torch.triu(
                torch.full((total_len, total_len), float("-inf"), device=hidden.device),
                diagonal=1,
            )
            causal_mask[:text_len, :text_len] = 0.0
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

            padding_mask = (1 - combined_mask).bool()
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2).expand_as(causal_mask)
            causal_mask = causal_mask.masked_fill(padding_mask_expanded, float("-inf"))

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

            # Predict codes and compute loss
            labels = codec_labels.clone()  # [batch, audio_len]
            valid_mask = labels != -100
            labels[valid_mask] = labels[valid_mask].clamp(0, TOTAL_VOCAB - 1)

            logits = F.linear(
                audio_hidden, self.token_embedding.weight
            )  # [batch, audio_len, total_vocab]
            loss = F.cross_entropy(
                logits.reshape(-1, TOTAL_VOCAB),
                labels.reshape(-1),
                ignore_index=-100,
            )
            return AudioHeadOutput(loss=loss)

        # Inference: autoregressive generation
        codes = self._generate(prefix, attention_mask)
        return AudioHeadOutput(codes=codes)

    def _generate(self, prefix: torch.Tensor, prefix_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """AR generation: predict codes one timestep at a time with KV cache."""
        batch_size, text_len, _ = prefix.shape
        device = prefix.device

        all_codes = []

        # Build initial input: prefix + BOS embedding
        bos_token = torch.full((batch_size, 1), BOS_TOKEN, dtype=torch.long, device=device)
        bos_emb = self.token_embedding(bos_token)  # [batch, 1, dim]
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
        last_hidden = outputs.last_hidden_state[:, -1:]  # [batch, 1, dim]

        for step in range(self.max_tokens):
            # Predict code token
            logits = F.linear(last_hidden.squeeze(1), self.token_embedding.weight)  # [batch, vocab]

            # Apply temperature and top-k sampling
            if self.temperature > 0 and self.top_k > 0:
                logits = logits / self.temperature
                # Zero out logits below top-k threshold
                top_k_vals, _ = logits.topk(self.top_k, dim=-1)
                logits[logits < top_k_vals[:, -1:]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch]
            else:
                token = logits.argmax(dim=-1)  # [batch]

            # Check for EOS
            if (token == EOS_TOKEN).all():
                break

            all_codes.append(token)

            # Feed back prediction for next step
            next_emb = self.token_embedding(token.unsqueeze(1))  # [batch, 1, dim]

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
            # [batch, gen_len]
            codes = torch.stack(all_codes, dim=1)
        else:
            codes = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        return codes

    def _load_neucodec(self):
        """Load frozen NeuCodec model for audio decoding."""
        from neucodec import NeuCodec

        self.neucodec_model = NeuCodec.from_pretrained(self.config.neucodec_model_id)
        self.neucodec_model.eval()
        self.neucodec_model.requires_grad_(False)
        logger.info("Loaded frozen NeuCodec model for audio decoding")

    def decode_to_audio(self, codes: torch.Tensor) -> list[torch.Tensor]:
        """Decode NeuCodec FSQ tokens to audio waveforms.

        Args:
            codes: Codec tokens [batch, seq_len]

        Returns:
            List of audio waveform tensors (one per batch item)
        """
        if self.neucodec_model is None:
            self._load_neucodec()
        assert self.neucodec_model is not None

        # NeuCodec decode_code expects [batch, 1, seq_len]
        codes_3d = codes.unsqueeze(1).to(self.neucodec_model.device)

        with torch.no_grad():
            audio_values = self.neucodec_model.decode_code(codes_3d)  # [batch, 1, samples]

        return [audio_values[i, 0] for i in range(audio_values.shape[0])]

    def generate_streaming(
        self,
        text_token_ids: torch.Tensor,
        chunk_samples: int = 24000,
    ) -> Iterator[torch.Tensor]:
        """Generate audio and yield waveform chunks for streaming playback.

        Args:
            text_token_ids: Text token IDs [batch, seq_len]
            chunk_samples: Audio samples per chunk (default 1s at 24kHz)

        Yields:
            Audio waveform chunks [samples]
        """
        output = self(text_token_ids)
        codes = output.codes
        audios = self.decode_to_audio(codes)

        for audio in audios:
            for start in range(0, audio.shape[-1], chunk_samples):
                end = min(start + chunk_samples, audio.shape[-1])
                yield audio[..., start:end]
