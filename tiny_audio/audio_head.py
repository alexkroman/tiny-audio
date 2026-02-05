"""Autoregressive audio head for speech-to-speech.

Generates audio from LLM embeddings via discrete codec tokens:
  LLM embeddings -> Pre-NN -> AR Decoder -> Depformer -> Mimi codes -> audio

Architecture:
- Pre-NN transformer (3 layers, bidirectional) processes LLM hidden states
- AR decoder (6 layers, causal) generates semantic codebook 0 autoregressively
- Depformer (4 layers) predicts acoustic codebooks 1-7 conditioned on codebook 0
- Mimi decoder converts all 8 codebooks to audio waveform
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .modules.ar_decoder import CodecARDecoder, PreNN
from .modules.depformer import Depformer

logger = logging.getLogger(__name__)


class AudioHead(nn.Module):
    """AR codec head: LLM embeddings -> Mimi codes -> audio.

    Architecture:
        - input_proj: Projects LLM embeddings to hidden_dim
        - pre_nn: 3-layer bidirectional transformer for context processing
        - ar_decoder: 6-layer causal transformer for semantic codebook 0
        - depformer: 4-layer transformer for acoustic codebooks 1-7
        - Mimi decoder (frozen) for codes -> audio

    Args:
        config: ASRConfig with:
            - llm_dim: LLM embedding dimension (default: 3072 for SmolLM3)
        llm_dim: Override for LLM dimension (takes precedence over config)
    """

    # Architecture dimensions
    HIDDEN_DIM = 1024  # Transformer hidden dimension
    INTERMEDIATE_DIM = 4096  # FFN intermediate dimension
    NUM_HEADS = 16  # Attention heads

    # Pre-NN config
    PRE_NN_LAYERS = 3

    # AR Decoder config (for semantic codebook 0)
    AR_LAYERS = 6
    VOCAB_SIZE = 2048  # Mimi codebook size

    # Depformer config (for acoustic codebooks 1-7)
    DEPFORMER_DIM = 512
    DEPFORMER_LAYERS = 4
    DEPFORMER_HEADS = 8
    DEPFORMER_INTERMEDIATE = 2048
    NUM_ACOUSTIC_CODEBOOKS = 7  # Codebooks 1-7

    # Loss weighting (equal weights following Moshi's approach)
    SEMANTIC_LOSS_WEIGHT = 1.0  # Weight for codebook 0 loss
    ACOUSTIC_LOSS_WEIGHT = 1.0  # Weight for codebooks 1-7 loss

    # Generation defaults
    DEFAULT_MAX_TOKENS = 500
    DEFAULT_TOP_K = 50
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_REPETITION_PENALTY = 1.1

    # Mimi codec constants
    MIMI_SAMPLE_RATE = 24000  # Expected sample rate for Mimi

    def __init__(self, config, llm_dim: int = None):
        super().__init__()
        self.llm_dim = llm_dim or getattr(config, "llm_dim", None) or 3072
        self.hidden_dim = self.HIDDEN_DIM
        self.vocab_size = self.VOCAB_SIZE
        self.num_codebooks = 1 + self.NUM_ACOUSTIC_CODEBOOKS  # 8 total

        # Generation parameters from config
        self.max_tokens = getattr(config, "max_audio_tokens", self.DEFAULT_MAX_TOKENS)
        self.top_k = getattr(config, "audio_top_k", self.DEFAULT_TOP_K)
        self.temperature = getattr(config, "audio_temperature", self.DEFAULT_TEMPERATURE)
        self.repetition_penalty = getattr(
            config, "audio_repetition_penalty", self.DEFAULT_REPETITION_PENALTY
        )

        # Input projection: LLM dim -> hidden dim
        self.input_proj = nn.Linear(self.llm_dim, self.hidden_dim, bias=False)

        # Pre-NN: Process LLM hidden states with bidirectional attention
        self.pre_nn = PreNN(
            hidden_size=self.hidden_dim,
            num_layers=self.PRE_NN_LAYERS,
            num_heads=self.NUM_HEADS,
            intermediate_size=self.INTERMEDIATE_DIM,
        )

        # AR Decoder: Generate semantic codebook 0 autoregressively
        self.ar_decoder = CodecARDecoder(
            hidden_size=self.hidden_dim,
            num_layers=self.AR_LAYERS,
            num_heads=self.NUM_HEADS,
            intermediate_size=self.INTERMEDIATE_DIM,
            vocab_size=self.vocab_size,
        )

        # Depformer: Generate acoustic codebooks 1-7
        self.depformer = Depformer(
            num_codebooks=self.NUM_ACOUSTIC_CODEBOOKS,
            vocab_size=self.vocab_size,
            main_dim=self.hidden_dim,
            hidden_size=self.DEPFORMER_DIM,
            num_layers=self.DEPFORMER_LAYERS,
            num_heads=self.DEPFORMER_HEADS,
            intermediate_size=self.DEPFORMER_INTERMEDIATE,
        )

        # Mimi model (loaded separately via load_mimi_decoder)
        self.mimi = None

    @property
    def bos_token_id(self) -> int:
        return self.ar_decoder.bos_token_id

    @property
    def sos_token_id(self) -> int:
        return self.ar_decoder.sos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.ar_decoder.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.ar_decoder.pad_token_id

    def load_mimi_decoder(self, device: torch.device = None, dtype: torch.dtype = None):
        """Load Mimi model for encoding/decoding audio.

        Args:
            device: Device to load model on (e.g., 'cuda', 'cpu')
            dtype: Data type for model weights (e.g., torch.float16, torch.bfloat16)
        """
        from transformers import MimiModel

        self.mimi = MimiModel.from_pretrained("kyutai/mimi")
        self.mimi.requires_grad_(False)
        self.mimi.eval()

        # Single transfer for efficiency (instead of two separate .to() calls)
        if device is not None or dtype is not None:
            self.mimi = self.mimi.to(device=device, dtype=dtype)

        logger.info("Loaded Mimi model from kyutai/mimi")

    def encode_audio(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """Encode audio waveform to Mimi codec tokens.

        Args:
            audio: Audio waveform [batch, samples] or [batch, 1, samples]
            sample_rate: Sample rate of input audio. If provided and doesn't match
                MIMI_SAMPLE_RATE (24000), a warning is logged. Audio will still be
                encoded but results may be incorrect.

        Returns:
            Codec tokens [batch, 8, seq_len] for all codebooks
        """
        if self.mimi is None:
            raise RuntimeError("Mimi not loaded. Call load_mimi_decoder() first.")

        # Validate sample rate if provided
        if sample_rate is not None and sample_rate != self.MIMI_SAMPLE_RATE:
            logger.warning(
                f"Audio sample rate ({sample_rate} Hz) does not match Mimi's expected "
                f"rate ({self.MIMI_SAMPLE_RATE} Hz). This may produce incorrect codec "
                f"tokens. Consider resampling the audio first."
            )

        # Ensure [batch, channels, samples]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        with torch.no_grad():
            # Encode to codes: [batch, num_codebooks, seq_len]
            # Use num_quantizers=8 (Mimi supports up to 32, we use 8 for efficiency)
            encoder_outputs = self.mimi.encode(audio, num_quantizers=self.num_codebooks)
            codes = encoder_outputs.audio_codes
            assert codes is not None, "Mimi encode returned no codes"
            return codes  # [batch, 8, seq_len]

    def forward(
        self,
        embeddings: torch.Tensor,
        codec_targets: Optional[torch.Tensor] = None,
        codec_lengths: Optional[torch.Tensor] = None,
        embeddings_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training or inference.

        Args:
            embeddings: LLM token embeddings [batch, seq_len, llm_dim]
            codec_targets: Target Mimi codes for training.
                - [batch, 8, audio_seq_len] for full training (all codebooks)
                - [batch, audio_seq_len] for legacy (codebook 0 only)
            codec_lengths: Actual audio lengths per sample [batch]
            embeddings_mask: Mask for embeddings [batch, seq_len]

        Returns:
            Training: scalar cross-entropy loss (weighted combination of all codebooks)
            Inference: tuple of (generated codes [batch, 8, gen_len], empty tensor)
        """
        # Project to hidden dim
        hidden = self.input_proj(embeddings)

        # Process through Pre-NN
        context = self.pre_nn(hidden, attention_mask=embeddings_mask)

        if codec_targets is not None:
            return self._compute_loss(context, embeddings_mask, codec_targets, codec_lengths)
        return self._generate(context, embeddings_mask)

    def _compute_loss(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute cross-entropy loss for codec token prediction.

        Trains both AR decoder (codebook 0) and Depformer (codebooks 1-7).
        Uses equal weighting following Moshi's approach.

        Args:
            context: Pre-NN output [batch, context_len, hidden_dim]
            context_mask: Mask for context [batch, context_len]
            targets: Target codec tokens [batch, 8, target_len] for all codebooks
            lengths: Actual target lengths [batch]

        Returns:
            Weighted sum of semantic and acoustic losses
        """
        device = targets.device

        assert targets.dim() == 3, f"Expected [batch, 8, seq_len], got {targets.shape}"
        assert targets.shape[1] == self.num_codebooks, (
            f"Expected {self.num_codebooks} codebooks, got {targets.shape[1]}"
        )

        target_len = targets.shape[2]

        # Create target mask from lengths
        if lengths is not None:
            positions = torch.arange(target_len, device=device).unsqueeze(0)
            target_mask = positions < lengths.unsqueeze(1)
        else:
            target_mask = None

        # Extract semantic targets (codebook 0)
        semantic_targets = targets[:, 0, :]  # [batch, target_len]

        # Forward through AR decoder for semantic codebook
        # Get hidden states for Depformer conditioning
        _, semantic_loss, ar_hidden = self.ar_decoder(
            context=context,
            context_mask=context_mask,
            target_ids=semantic_targets,
            target_mask=target_mask,
            return_hidden=True,
        )

        # Train Depformer on acoustic codebooks 1-7
        # Use AR decoder hidden states as conditioning
        # Depformer expects [batch, num_codebooks, seq_len] where
        # [:, 0, :] is semantic (input condition)
        # [:, 1:, :] are targets for depformer
        _, acoustic_loss = self.depformer.forward_training(
            main_hidden=ar_hidden,
            codebook_targets=targets,
        )

        return self.SEMANTIC_LOSS_WEIGHT * semantic_loss + self.ACOUSTIC_LOSS_WEIGHT * acoustic_loss

    def _generate(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate codec tokens via AR decoding + Depformer.

        Uses AR decoder hidden states for Depformer conditioning, providing
        position-specific context at each generated timestep.

        Args:
            context: Pre-NN output [batch, context_len, hidden_dim]
            context_mask: Mask for context [batch, context_len]

        Returns:
            Tuple of:
                - Generated codec tokens [batch, 8, gen_len] (all codebooks)
                - Empty tensor (for API compatibility)
        """
        device = context.device

        # Collect generated semantic tokens AND AR decoder hidden states
        semantic_tokens = []
        ar_hidden_states = []

        for token, hidden in self.ar_decoder.generate(
            context=context,
            context_mask=context_mask,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            return_hidden=True,
        ):
            semantic_tokens.append(token)
            ar_hidden_states.append(hidden)

        if not semantic_tokens:
            return torch.empty(
                1, self.num_codebooks, 0, dtype=torch.long, device=device
            ), torch.empty(0, device=device)

        # Convert to tensors
        semantic = torch.tensor([semantic_tokens], device=device)  # [1, seq_len]
        # Stack AR hidden states: [1, seq_len, hidden_dim]
        ar_hidden = torch.cat(ar_hidden_states, dim=1)

        # Generate acoustic codebooks 1-7 using Depformer
        # Now using position-specific AR decoder hidden states
        acoustic = self.depformer.generate_batch(
            main_hidden=ar_hidden,
            semantic_tokens=semantic,
            temperature=self.temperature,
            top_k=self.top_k,
        )  # [1, 7, seq_len]

        # Combine semantic + acoustic: [1, 8, seq_len]
        all_codes = torch.cat([semantic.unsqueeze(1), acoustic], dim=1)

        return all_codes, torch.empty(0, device=device)

    def decode_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode Mimi codec tokens to audio waveform.

        Args:
            codes: Codec tokens [batch, 8, seq_len] (all codebooks)

        Returns:
            Audio waveform [batch, samples]
        """
        if self.mimi is None:
            raise RuntimeError("Mimi not loaded. Call load_mimi_decoder() first.")

        # Ensure we have all 8 codebooks
        if codes.dim() == 2:
            # Legacy: single codebook, pad with zeros for others
            codes = codes.unsqueeze(1)
            padding = torch.zeros(
                codes.shape[0],
                self.num_codebooks - 1,
                codes.shape[2],
                dtype=codes.dtype,
                device=codes.device,
            )
            codes = torch.cat([codes, padding], dim=1)

        assert codes.shape[1] == self.num_codebooks, (
            f"Expected {self.num_codebooks} codebooks, got {codes.shape[1]}"
        )

        with torch.no_grad():
            # Decode to audio - returns MimiDecoderOutput with audio_values
            output = self.mimi.decode(codes)
            audio = output.audio_values  # [batch, 1, samples]

        return audio.squeeze(1)  # [batch, samples]

    def get_output_length(self, input_length: int) -> int:
        """Estimate output audio samples from input hidden state length.

        Mimi operates at 12.5 Hz frame rate with 24kHz audio:
        Each codec frame = 24000 / 12.5 = 1920 audio samples

        Note: Actual length depends on AR generation, this is an estimate.
        """
        # Rough estimate: ~3 audio frames per text token
        estimated_frames = input_length * 3
        return estimated_frames * 1920
