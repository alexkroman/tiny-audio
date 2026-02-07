"""Audio head for speech-to-speech using frozen Dia TTS decoder.

Generates audio from LLM embeddings via a trainable MLP projector and
a frozen pretrained Dia decoder:
  LLM hidden (2048) -> MLP (2048->1024) -> frozen Dia decoder -> DAC codes -> audio (44.1kHz)

Only the MLP projector is trained (~330K params). The frozen Dia decoder
handles temporal alignment via cross-attention (no interpolation needed).

Training: S2SDataCollator prepares Dia-ready labels (delay pattern, teacher forcing).
AudioHead just projects and delegates to frozen Dia.
"""

import logging
from typing import Iterator, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

logger = logging.getLogger(__name__)

# DAC codec constants (used by Dia)
DAC_VOCAB_SIZE = 1028
NUM_DAC_CODEBOOKS = 9
DAC_SAMPLE_RATE = 44100


class AudioHead(nn.Module):
    """MLP projector for audio generation via frozen Dia decoder.

    Training: projects LLM hidden states -> Dia encoder space, then Dia computes
    cross-entropy loss from pre-computed labels (prepared by S2SDataCollator).

    Inference: projects -> Dia generate() -> DAC codes -> audio waveform.

    Args:
        config: ASRConfig with llm_dim, dia_model_id
        llm_dim: Override for LLM dimension
    """

    DIA_DIM = 1024
    DEFAULT_MAX_TOKENS = 500

    def __init__(self, config, llm_dim: int = None):
        super().__init__()
        self.llm_dim = llm_dim or getattr(config, "llm_dim", None) or 2048
        self.dia_model_id = getattr(config, "dia_model_id", "nari-labs/Dia-1.6B-0626")
        self.vocab_size = DAC_VOCAB_SIZE
        self.num_codebooks = NUM_DAC_CODEBOOKS
        self.max_tokens = getattr(config, "max_audio_tokens", self.DEFAULT_MAX_TOKENS)

        # MLP projector: LLM dim -> Dia dim (only trainable component)
        self.projector = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.DIA_DIM),
        )

        # Dia model and processor (loaded lazily)
        self.dia_model = None
        self.dia_processor = None

    def train(self, mode=True):
        super().train(mode)
        if self.dia_model is not None:
            self.dia_model.eval()
        return self

    def state_dict(self, *args, **kwargs):
        return {f"projector.{k}": v for k, v in self.projector.state_dict().items()}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        projector_state = {
            k.removeprefix("projector."): v
            for k, v in state_dict.items()
            if k.startswith("projector.")
        }
        return self.projector.load_state_dict(projector_state, strict=strict)

    def load_dia_decoder(self, device: torch.device = None, dtype: torch.dtype = None):
        from transformers import DiaForConditionalGeneration, DiaProcessor

        self.dia_model = DiaForConditionalGeneration.from_pretrained(self.dia_model_id)
        self.dia_processor = DiaProcessor.from_pretrained(self.dia_model_id)
        self.dia_model.requires_grad_(False)
        self.dia_model.eval()

        if device is not None or dtype is not None:
            self.dia_model = self.dia_model.to(device=device, dtype=dtype)

        logger.info(f"Loaded frozen Dia model from {self.dia_model_id}")

    def forward(
        self,
        embeddings: torch.Tensor,
        dia_labels: Optional[torch.Tensor] = None,
        dia_decoder_input_ids: Optional[torch.Tensor] = None,
        dia_decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training or inference.

        Args:
            embeddings: LLM hidden states [batch, seq_len, hidden_dim]
            dia_labels: Pre-computed labels [batch*9, seq_len] from S2SDataCollator
            dia_decoder_input_ids: Teacher-forced decoder inputs [batch, seq_len, 9]
            dia_decoder_attention_mask: Decoder attention mask [batch, seq_len]

        Returns:
            Training (dia_labels provided): scalar cross-entropy loss
            Inference (no dia_labels): tuple of (codes [batch, gen_len, 9], empty tensor)
        """
        if self.dia_model is None:
            self.load_dia_decoder(device=embeddings.device, dtype=embeddings.dtype)

        projected = self.projector(embeddings)  # [batch, seq_len, DIA_DIM]
        encoder_outputs = BaseModelOutput(last_hidden_state=projected)

        if dia_labels is not None:
            output = self.dia_model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=dia_decoder_input_ids,
                decoder_attention_mask=dia_decoder_attention_mask,
                labels=dia_labels,
            )
            return output.loss

        # Inference: Dia handles delay pattern, sampling, multi-codebook internally
        with torch.no_grad():
            output = self.dia_model.generate(
                encoder_outputs=encoder_outputs,
                max_new_tokens=self.max_tokens,
            )
        codes = output if isinstance(output, torch.Tensor) else output.sequences
        return codes, torch.empty(0, device=projected.device)

    def decode_to_audio(self, codes: torch.Tensor) -> list[torch.Tensor]:
        """Decode DAC codec tokens to audio waveforms via DiaProcessor.

        Args:
            codes: Codec tokens [batch, seq_len, 9] (Dia's native format)

        Returns:
            List of audio waveform tensors (one per batch item)
        """
        if self.dia_processor is None:
            raise RuntimeError("Dia not loaded. Call load_dia_decoder() first.")
        return self.dia_processor.batch_decode(codes)

    def generate_streaming(
        self,
        embeddings: torch.Tensor,
        chunk_samples: int = 44100,
    ) -> Iterator[torch.Tensor]:
        """Generate audio and yield waveform chunks for streaming playback.

        Args:
            embeddings: LLM hidden states [batch, seq_len, llm_dim]
            chunk_samples: Audio samples per chunk (default 1s at 44.1kHz)

        Yields:
            Audio waveform chunks [samples]
        """
        codes, _ = self(embeddings)
        audios = self.decode_to_audio(codes)

        for audio in audios:
            for start in range(0, audio.shape[-1], chunk_samples):
                end = min(start + chunk_samples, audio.shape[-1])
                yield audio[..., start:end]
