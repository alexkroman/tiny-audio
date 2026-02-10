"""Audio head for speech-to-speech using a frozen pretrained TTS backbone.

Architecture (Freeze-Omni style):
  Text tokens → neutts-nano embed_tokens (frozen) → Projector MLP (trainable)
  → Concat with codec embeddings → neutts-nano LlamaForCausalLM (frozen)
  → lm_head → speech token logits → NeuCodec codes → audio

neutts-nano (neuphonic/neutts-nano) is a pretrained 24-layer LlamaForCausalLM
(dim=576, ~117M params) that generates NeuCodec codes as <|speech_N|> tokens.
Only the projector MLP (~1.2M params) is trained.

NeuCodec uses a single FSQ codebook (levels=[4]*8, vocab=65536) at 50 tokens/sec,
outputting 24kHz audio. Codes 0-65535 map to neutts-nano tokens <|speech_0|>..<|speech_65535|>.
"""

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)

# NeuCodec FSQ constants
NEUCODEC_VOCAB_SIZE = 65536
NEUCODEC_SAMPLE_RATE = 24000

# Special token IDs used by S2SDataCollator (above NeuCodec vocab range)
BOS_TOKEN = NEUCODEC_VOCAB_SIZE      # 65536
EOS_TOKEN = NEUCODEC_VOCAB_SIZE + 1  # 65537
PAD_TOKEN = NEUCODEC_VOCAB_SIZE + 2  # 65538
TOTAL_VOCAB = NEUCODEC_VOCAB_SIZE + 3  # 65539 (for backwards compat)


class AudioHeadConfig(PretrainedConfig):
    """Configuration for AudioHead with frozen TTS backbone + trainable projector."""

    model_type = "audio_head"

    def __init__(
        self,
        tts_model_id: str = "neuphonic/neutts-nano",
        projector_hidden: int = 1024,
        text_vocab_size: int = 32000,
        max_audio_tokens: int = 500,
        neucodec_model_id: str = "neuphonic/neucodec",
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs,
    ):
        self.tts_model_id = tts_model_id
        self.projector_hidden = projector_hidden
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
        codes: Generated NeuCodec codes in inference mode [batch, gen_len].
    """

    loss: Optional[torch.Tensor] = None
    codes: Optional[torch.Tensor] = None


class AudioHead(PreTrainedModel):
    """Frozen TTS backbone + trainable projector for speech generation.

    Loads neutts-nano (a pretrained LlamaForCausalLM that generates NeuCodec tokens)
    and freezes it entirely. Only a small MLP projector is trained to adapt
    text embeddings for the backbone's input space.

    Stage 1: text_token_ids → backbone embed_tokens → projector → backbone → speech codes
    Stage 2 (later): LLM hidden states → projector(3072→576) → backbone → speech codes
    """

    config_class = AudioHeadConfig
    # Prevent from_pretrained from using meta device init (which conflicts
    # with loading the backbone inside __init__ via its own from_pretrained)
    _supports_param_buffer_assignment = False

    def __init__(self, config: AudioHeadConfig):
        super().__init__(config)
        self.max_tokens = config.max_audio_tokens

        # Load frozen TTS backbone (skip if we're in meta device context,
        # which happens during from_pretrained — _load_backbone() is called after)
        self._backbone_loaded = False
        if not self._is_meta_init():
            self._load_backbone(config)

    def _is_meta_init(self) -> bool:
        """Check if we're inside a meta device context manager."""
        try:
            test = torch.empty(1)
            return test.device.type == "meta"
        except Exception:
            return False

    def _load_backbone(self, config: AudioHeadConfig) -> None:
        """Load the frozen TTS backbone and initialize the projector."""
        if self._backbone_loaded:
            return

        logger.info("Loading TTS backbone: %s", config.tts_model_id)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.tts_model_id,
            torch_dtype=torch.float32,
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Load tokenizer to resolve speech token IDs
        self.tts_tokenizer = AutoTokenizer.from_pretrained(config.tts_model_id)

        # Cache key token IDs
        self.speech_token_offset = self.tts_tokenizer.convert_tokens_to_ids("<|speech_0|>")
        self.speech_start_id = self.tts_tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_START|>"
        )
        self.speech_end_id = self.tts_tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_END|>"
        )

        # Backbone hidden size (auto-detected)
        backbone_dim = self.backbone.config.hidden_size  # 576 for neutts-nano

        # Trainable projector: 2-layer MLP (backbone_dim → hidden → backbone_dim)
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, config.projector_hidden),
            nn.SiLU(),
            nn.Linear(config.projector_hidden, backbone_dim),
        )

        # Sampling parameters for inference
        self.temperature = config.temperature
        self.top_k = config.top_k

        # NeuCodec model (loaded lazily, frozen, inference only)
        self.neucodec_model = None

        self._backbone_loaded = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load AudioHead: config + projector weights from disk, backbone from HF Hub."""
        from pathlib import Path

        from safetensors.torch import load_file

        path = Path(pretrained_model_name_or_path)

        # Load config
        config = AudioHeadConfig.from_pretrained(path)

        # Create model (loads backbone from HF Hub)
        model = cls(config)

        # Load projector weights from saved checkpoint
        safetensors_path = path / "model.safetensors"
        if safetensors_path.exists():
            projector_state = load_file(safetensors_path)
            model.load_state_dict(projector_state, strict=False)
            logger.info("Loaded projector weights from %s", safetensors_path)

        return model

    def train(self, mode: bool = True):
        """Override to keep backbone in eval mode (disables dropout, etc.)."""
        super().train(mode)
        # Always keep backbone in eval mode regardless of parent training state
        self.backbone.eval()
        return self

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens using the frozen backbone's embedding table."""
        return self.backbone.model.embed_tokens(token_ids)

    def _codec_to_speech_ids(self, codec_codes: torch.Tensor) -> torch.Tensor:
        """Map NeuCodec codes [0, 65535] to neutts-nano speech token IDs."""
        return codec_codes + self.speech_token_offset

    def _speech_ids_to_codec(self, speech_ids: torch.Tensor) -> torch.Tensor:
        """Map neutts-nano speech token IDs back to NeuCodec codes [0, 65535]."""
        return speech_ids - self.speech_token_offset

    def forward(
        self,
        text_token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        codec_labels: Optional[torch.Tensor] = None,
        codec_input_ids: Optional[torch.Tensor] = None,
        codec_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # noqa: ARG002 — absorbs extra keys from Trainer
    ) -> AudioHeadOutput:
        """Forward pass for training or inference.

        Args:
            text_token_ids: Text token IDs [batch, seq_len] (neutts-nano tokenizer vocab)
            attention_mask: Text attention mask [batch, seq_len] (1=real, 0=padding)
            codec_labels: Target NeuCodec codes [batch, audio_len] (-100 for ignore)
            codec_input_ids: Teacher-forced NeuCodec codes [batch, audio_len]
            codec_attention_mask: Codec attention mask [batch, audio_len]
            **kwargs: Absorbed silently (Trainer may pass extra keys).

        Returns:
            AudioHeadOutput with loss (training) or codes (inference).
        """
        batch_size, text_len = text_token_ids.shape
        device = text_token_ids.device

        # Embed text through frozen backbone embedding, then adapt via trainable projector.
        # We use no_grad for embed_tokens since it's a lookup (no grad needed for the
        # embedding table itself — it's frozen). Gradients flow through the projector.
        with torch.no_grad():
            text_emb = self._embed_tokens(text_token_ids)  # [batch, text_len, 576]
        prefix = self.projector(text_emb)  # [batch, text_len, 576] — trainable

        if codec_labels is None:
            # Inference: autoregressive generation
            codes = self._generate(prefix, attention_mask)
            return AudioHeadOutput(codes=codes)

        # Training: teacher forcing
        assert codec_input_ids is not None, (
            "codec_input_ids required when codec_labels provided"
        )

        # Map NeuCodec codes to neutts speech token IDs for embedding
        # codec_input_ids contains: BOS_TOKEN (65536), codec codes (0-65535), PAD (65538)
        # We need to map these to neutts-nano token space
        speech_input = self._map_collator_ids_to_speech(codec_input_ids)

        with torch.no_grad():
            token_emb = self._embed_tokens(speech_input)  # [batch, audio_len, 576]

        audio_len = token_emb.shape[1]

        # Concatenate: [projected_text, codec_token_embeddings]
        # prefix has grad (from projector), token_emb is detached (frozen embedding lookup)
        hidden = torch.cat([prefix, token_emb], dim=1)

        # Build 2D padding mask — backbone handles causal masking internally
        prefix_mask = attention_mask if attention_mask is not None else torch.ones(
            batch_size, text_len, device=device, dtype=torch.long
        )
        audio_mask = codec_attention_mask if codec_attention_mask is not None else torch.ones(
            batch_size, audio_len, device=device, dtype=torch.long
        )
        combined_mask = torch.cat([prefix_mask, audio_mask], dim=1)

        # Run through frozen backbone WITHOUT torch.no_grad().
        # The backbone weights have requires_grad=False so they won't accumulate grads,
        # but PyTorch still builds the computation graph through the matmuls, allowing
        # gradients to flow back from the loss through backbone → hidden → prefix → projector.
        outputs = self.backbone.model(
            inputs_embeds=hidden,
            attention_mask=combined_mask,
        )

        # Extract audio-position hidden states
        audio_hidden = outputs.last_hidden_state[:, text_len:]  # [batch, audio_len, 576]

        # Project through frozen lm_head to get logits over full vocab.
        # Same principle: lm_head weights are frozen but gradients flow through the
        # matmul back to audio_hidden (and ultimately to the projector).
        logits = self.backbone.lm_head(audio_hidden)  # [batch, audio_len, vocab_size]

        # Map codec_labels to speech token IDs for CE loss target
        speech_labels = self._map_collator_labels_to_speech(codec_labels)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            speech_labels.view(-1),
            ignore_index=-100,
        )

        return AudioHeadOutput(loss=loss)

    def _map_collator_ids_to_speech(self, codec_input_ids: torch.Tensor) -> torch.Tensor:
        """Map S2SDataCollator codec_input_ids to neutts-nano token IDs.

        S2SDataCollator produces:
          - BOS_TOKEN (65536) at position 0
          - NeuCodec codes (0-65535) for real audio
          - PAD_TOKEN (65538) for padding

        Maps to:
          - BOS_TOKEN → <|SPEECH_GENERATION_START|>
          - codes 0-65535 → <|speech_0|>..<|speech_65535|>
          - PAD_TOKEN → pad_token_id
        """
        result = codec_input_ids.clone()

        # Map BOS (65536)
        bos_mask = codec_input_ids == NEUCODEC_VOCAB_SIZE
        result[bos_mask] = self.speech_start_id

        # Map EOS (65537)
        eos_mask = codec_input_ids == (NEUCODEC_VOCAB_SIZE + 1)
        result[eos_mask] = self.speech_end_id

        # Map PAD (65538)
        pad_mask = codec_input_ids == (NEUCODEC_VOCAB_SIZE + 2)
        result[pad_mask] = self.tts_tokenizer.pad_token_id

        # Map codec codes (0-65535) → speech tokens
        codec_mask = codec_input_ids < NEUCODEC_VOCAB_SIZE
        result[codec_mask] = codec_input_ids[codec_mask] + self.speech_token_offset

        return result

    def _map_collator_labels_to_speech(self, codec_labels: torch.Tensor) -> torch.Tensor:
        """Map S2SDataCollator codec_labels to neutts-nano token IDs.

        codec_labels contains:
          - NeuCodec codes (0-65535) for real targets
          - EOS_TOKEN (65537) at the end
          - -100 for ignore positions
        """
        result = codec_labels.clone()

        valid = codec_labels != -100

        # Map EOS (65537)
        eos_mask = valid & (codec_labels == (NEUCODEC_VOCAB_SIZE + 1))
        result[eos_mask] = self.speech_end_id

        # Map codec codes (0-65535) → speech tokens
        codec_mask = valid & (codec_labels < NEUCODEC_VOCAB_SIZE)
        result[codec_mask] = codec_labels[codec_mask] + self.speech_token_offset

        return result

    def _generate(self, prefix: torch.Tensor, prefix_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """AR generation with KV cache on frozen backbone.

        Args:
            prefix: Projected text embeddings [batch, text_len, 576].
            prefix_mask: Attention mask for prefix tokens (unused for now,
                         reserved for batched generation with padding).
        """
        _ = prefix_mask  # Reserved for future batched generation
        batch_size, text_len, _ = prefix.shape
        device = prefix.device

        all_codes = []

        # Build initial input: prefix + SPEECH_GENERATION_START token
        start_token = torch.full(
            (batch_size, 1), self.speech_start_id, dtype=torch.long, device=device
        )
        start_emb = self._embed_tokens(start_token)  # [batch, 1, 576]
        hidden = torch.cat([prefix, start_emb], dim=1)  # [batch, text_len+1, 576]

        position_ids = torch.arange(
            text_len + 1, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        # Initial forward through frozen backbone
        with torch.no_grad():
            outputs = self.backbone.model(
                inputs_embeds=hidden,
                position_ids=position_ids,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_hidden = outputs.last_hidden_state[:, -1:]  # [batch, 1, 576]

            for step in range(self.max_tokens):
                # Get logits from lm_head
                logits = self.backbone.lm_head(last_hidden.squeeze(1))  # [batch, vocab]

                # Mask to speech tokens only
                speech_logits = logits[:, self.speech_token_offset:
                                       self.speech_token_offset + NEUCODEC_VOCAB_SIZE]

                # Also check speech_end token
                end_logit = logits[:, self.speech_end_id:self.speech_end_id + 1]

                # Combine speech + end logits for sampling
                combined = torch.cat([speech_logits, end_logit], dim=-1)  # [batch, 65537]

                # Apply temperature and top-k
                if self.temperature != 1.0:
                    combined = combined / self.temperature
                if self.top_k > 0:
                    topk_vals, _ = combined.topk(min(self.top_k, combined.size(-1)))
                    combined[combined < topk_vals[:, -1:]] = float("-inf")

                probs = F.softmax(combined, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)  # [batch]

                # Check for EOS (last position in combined = end token)
                is_eos = sampled == NEUCODEC_VOCAB_SIZE  # index 65536 = end token
                if is_eos.all():
                    break

                # Map sampled index to NeuCodec code (0-65535)
                codec_code = sampled.clamp(0, NEUCODEC_VOCAB_SIZE - 1)
                all_codes.append(codec_code)

                # Map to speech token ID for next step embedding
                next_token_id = codec_code + self.speech_token_offset
                # For EOS items, use speech_end_id (won't matter as we'll stop)
                next_token_id[is_eos] = self.speech_end_id

                next_emb = self._embed_tokens(
                    next_token_id.unsqueeze(1)
                )  # [batch, 1, 576]

                next_pos = torch.full(
                    (batch_size, 1), text_len + 1 + step + 1,
                    dtype=torch.long, device=device,
                )

                outputs = self.backbone.model(
                    inputs_embeds=next_emb,
                    position_ids=next_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                last_hidden = outputs.last_hidden_state  # [batch, 1, 576]

        if all_codes:
            codes = torch.stack(all_codes, dim=1)  # [batch, gen_len]
        else:
            codes = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        return codes

    def state_dict(self, *args, **kwargs):
        """Only save projector weights (backbone is frozen/pretrained)."""
        full = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full.items() if k.startswith("projector.")}

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
            codes: Codec tokens [batch, seq_len] (values 0-65535)

        Returns:
            List of audio waveform tensors (one per batch item)
        """
        if self.neucodec_model is None:
            self._load_neucodec()
        assert self.neucodec_model is not None

        codes_3d = codes.unsqueeze(1).to(self.neucodec_model.device)

        with torch.no_grad():
            audio_values = self.neucodec_model.decode_code(codes_3d)

        return [audio_values[i, 0] for i in range(audio_values.shape[0])]

    def generate_streaming(
        self,
        text_token_ids: torch.Tensor,
        chunk_samples: int = 24000,
    ) -> Iterator[torch.Tensor]:
        """Generate audio and yield waveform chunks for streaming playback."""
        output = self(text_token_ids)
        codes = output.codes
        audios = self.decode_to_audio(codes)

        for audio in audios:
            for start in range(0, audio.shape[-1], chunk_samples):
                end = min(start + chunk_samples, audio.shape[-1])
                yield audio[..., start:end]
