"""Autoregressive audio head for speech-to-speech.

Generates audio from LLM embeddings via discrete codec tokens:
  LLM embeddings -> Pre-NN -> AR Decoder -> Depformer -> Mimi codes -> audio

Architecture (Freeze-Omni style):
- Pre-NN: Linear projection + half-depth transformer layers to transform LLM hidden states
- AR decoder (causal) generates semantic codebook 0 autoregressively
- Depformer predicts acoustic codebooks 1-7 conditioned on codebook 0
- Mimi decoder converts all 8 codebooks to audio waveform

Streaming Support (Moshi-style):
- StreamingState holds generation state between steps
- step() method generates one audio frame at a time
- Mimi decoder runs in streaming mode for low-latency output
"""

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from .modules.ar_decoder import CodecARDecoder
from .modules.depformer import Depformer

logger = logging.getLogger(__name__)


class PreNN(nn.Module):
    """Pre-NN projection from LLM hidden states to AudioHead hidden dim (Freeze-Omni style).

    Replaces a simple linear projection with:
    1. Linear projection from LLM dim to hidden dim
    2. Half-depth Llama transformer layers with bidirectional attention

    This gives the model multiple layers of self-attention to transform and
    contextualize the LLM representations before codec generation.

    Args:
        llm_dim: LLM hidden state dimension (e.g., 3072 for SmolLM3-3B)
        hidden_dim: AudioHead hidden dimension (e.g., 256)
        num_layers: Number of transformer layers (default: AR_LAYERS // 2 = 3)
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        llm_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        intermediate_size: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim

        # Linear projection from LLM dim to hidden dim
        self.proj = nn.Linear(llm_dim, hidden_dim, bias=False)

        # Llama transformer layers for contextual processing
        config = LlamaConfig(
            hidden_size=hidden_dim,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=4096,
            attention_dropout=dropout,
            _attn_implementation="sdpa",
        )

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(num_layers)]
        )
        self.norm = LlamaRMSNorm(hidden_dim, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project and contextualize LLM hidden states.

        Args:
            hidden_states: LLM hidden states [batch, seq_len, llm_dim]
            mask: Boolean mask [batch, seq_len] where True = valid, False = padding

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim]
        """
        # Project from LLM dim to hidden dim
        hidden_states = self.proj(hidden_states)

        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Build bidirectional attention mask (Freeze-Omni Pre-NN uses bidirectional)
        if mask is not None:
            # [batch, seq_len] -> [batch, 1, seq_len, seq_len] bidirectional mask
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, -1, seq_len, -1)  # [batch, 1, seq_len, seq_len]
            attn_mask = torch.where(
                attn_mask,
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype),
            )
        else:
            attn_mask = None

        # Position IDs and rotary embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Forward through transformer layers
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

        return self.norm(hidden_states)


@dataclass
class StreamingState:
    """State for streaming audio generation (Moshi-style).

    Maintains all state needed between step() calls for low-latency streaming.
    Includes delay-based cache for proper multi-codebook AR modeling.
    """

    # Generation context
    context: torch.Tensor  # [1, context_len, hidden_dim]
    context_mask: Optional[torch.Tensor]  # [1, context_len]

    # AR decoder state
    ar_generator: Optional[Iterator] = None  # AR decoder generator

    # Generated tokens buffer
    semantic_tokens: list[int] = field(default_factory=list)
    ar_hidden_states: list[torch.Tensor] = field(default_factory=list)

    # Delay-based cache (Moshi-style)
    # Cache stores generated tokens with delay alignment
    # Shape: [batch, num_codebooks, max_delay + buffer_size]
    delay_cache: Optional[torch.Tensor] = None
    delays: Optional[torch.Tensor] = None  # Per-codebook delays
    offset: int = 0  # Current generation offset (AR position)
    max_delay: int = 0  # Maximum delay across all codebooks

    # Special token for ungenerated positions
    ungenerated_token: int = -1

    # Streaming parameters
    chunk_size: int = 1  # Tokens per audio chunk (1 = lowest latency)
    tokens_generated: int = 0
    finished: bool = False

    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def write_to_cache(self, codebook_idx: int, token: int):
        """Write a token to the delay cache at current offset.

        Since Depformer handles delays internally (delays inputs, outputs time-aligned
        tokens), all codebooks at step t represent audio time t. No additional delay
        adjustment needed during cache write.
        """
        if self.delay_cache is None:
            return
        cache_size = self.delay_cache.shape[2]
        write_pos = self.offset % cache_size
        self.delay_cache[0, codebook_idx, write_pos] = token

    def read_from_cache(self) -> Optional[torch.Tensor]:
        """Read aligned tokens from cache.

        Since Depformer outputs are already time-aligned, we read directly from
        the current offset position. All codebooks at a given position represent
        the same audio time.
        """
        if self.delay_cache is None:
            return None
        # Read from current position (all codebooks are time-aligned)
        cache_size = self.delay_cache.shape[2]
        read_pos = self.offset % cache_size
        return self.delay_cache[:, :, read_pos]  # [1, num_codebooks]

    def advance(self):
        """Advance the offset after generating all codebooks for this step."""
        self.offset += 1


class AudioHead(nn.Module):
    """AR codec head: LLM embeddings -> Mimi codes -> audio.

    Architecture (Freeze-Omni style):
        - Input: Already-projected embeddings at hidden_dim (Pre-NN projects + contextualizes)
        - Concatenate conditioning + text embeddings as combined context
        - ar_decoder: Causal transformer layers
        - depformer: 4-layer transformer for acoustic codebooks 1-7
        - Mimi decoder (frozen) for codes -> audio

    Args:
        config: ASRConfig with:
            - llm_dim: LLM embedding dimension (default: 2048 for SmolLM3-3B)
        llm_dim: Override for LLM dimension (takes precedence over config)
    """

    # Architecture dimensions (matching Freeze-Omni defaults)
    # Freeze-Omni: hidden=256, intermediate=1024, heads=4, layers=6
    HIDDEN_DIM = 256  # Transformer hidden dimension
    INTERMEDIATE_DIM = 1024  # FFN intermediate dimension (4x hidden)
    NUM_HEADS = 4  # Attention heads (64 dim per head)

    # AR Decoder config (for semantic codebook 0)
    # No Pre-NN for simpler gradient flow
    AR_LAYERS = 6
    VOCAB_SIZE = 2048  # Mimi codebook size

    # Depformer config (for acoustic codebooks 1-7) - smaller than AR decoder
    DEPFORMER_DIM = 128
    DEPFORMER_LAYERS = 4
    DEPFORMER_HEADS = 4  # 32 dim per head
    DEPFORMER_INTERMEDIATE = 512  # 4x hidden
    NUM_ACOUSTIC_CODEBOOKS = 7  # Codebooks 1-7

    # Loss weighting (equal weights following Moshi's approach)
    SEMANTIC_LOSS_WEIGHT = 1.0  # Weight for codebook 0 loss
    ACOUSTIC_LOSS_WEIGHT = 1.0  # Weight for codebooks 1-7 loss

    # Generation defaults
    DEFAULT_MAX_TOKENS = 500
    DEFAULT_TOP_K = 50
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_REPETITION_PENALTY = 1.1

    # Moshi-style delays for multi-codebook AR (audio-time alignment)
    # delays[k] = how many AR steps codebook k is delayed
    # Semantic (codebook 0): delay 0 (generated first, sets the content)
    # Acoustic (codebooks 1-7): delay 1 (can condition on semantic at same audio time)
    # This flat delay pattern enables parallel acoustic generation after semantic
    DELAYS = [0, 1, 1, 1, 1, 1, 1, 1]  # [semantic, acoustic_1, ..., acoustic_7]

    # Cache buffer size for streaming (must be > max_delay)
    DELAY_CACHE_BUFFER = 4

    # Dropout rate (Freeze-Omni style)
    DROPOUT_RATE = 0.1

    # Mimi codec constants
    MIMI_SAMPLE_RATE = 24000  # Expected sample rate for Mimi

    def __init__(self, config, llm_dim: int = None):
        super().__init__()
        self.llm_dim = llm_dim or getattr(config, "llm_dim", None) or 2048  # SmolLM3 native
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

        # Hidden state dropout (Freeze-Omni style)
        self.hidden_dropout = nn.Dropout(p=self.DROPOUT_RATE)

        # Shared embedding for all tokens (Freeze-Omni style)
        # No projection needed - caller projects LLM hidden states before passing in
        # vocab_size + 4 special tokens: BOS, SOS, EOS, PAD
        self.total_vocab = self.vocab_size + 4  # BOS=+0, SOS=+1, EOS=+2, PAD=+3
        self.embedding = nn.Embedding(
            self.total_vocab,
            self.hidden_dim,  # Same dim as input - no projection needed
            padding_idx=self.vocab_size + 3,  # PAD token
        )

        # AR Decoder: Generate semantic codebook 0 autoregressively
        self.ar_decoder = CodecARDecoder(
            hidden_size=self.hidden_dim,
            num_layers=self.AR_LAYERS,
            num_heads=self.NUM_HEADS,
            intermediate_size=self.INTERMEDIATE_DIM,
            vocab_size=self.vocab_size,
            embedding=self.embedding,
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

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save state dict, excluding tied ar_decoder.embedding (shares with self.embedding)."""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Remove the tied weight - it will be restored via _tie_weights on load
        tied_key = prefix + "ar_decoder.embedding.weight"
        if tied_key in state:
            del state[tied_key]
        return state

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Load state dict, handling tied embedding weight."""
        # The ar_decoder.embedding.weight is tied to self.embedding.weight
        # so it won't be in the saved state_dict - don't report it as missing
        tied_key = prefix + "ar_decoder.embedding.weight"
        if tied_key in missing_keys:
            missing_keys.remove(tied_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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
        text_embeddings: torch.Tensor,
        codec_targets: Optional[torch.Tensor] = None,
        codec_lengths: Optional[torch.Tensor] = None,
        embeddings_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training or inference.

        Concatenation approach:
        - Concatenate conditioning embeddings + text embeddings
        - AR decoder handles via attention masking

        Args:
            embeddings: LLM hidden states [batch, seq_len, hidden_dim]
                Conditioning information (prosody/style) - already projected by caller
            text_embeddings: Text token embeddings [batch, text_len, hidden_dim]
                Linguistic content (what to say)
            codec_targets: Target Mimi codes for training.
                - [batch, 8, audio_seq_len] for full training (all codebooks)
                - [batch, audio_seq_len] for legacy (codebook 0 only)
            codec_lengths: Actual audio lengths per sample [batch]
            embeddings_mask: Mask for embeddings [batch, seq_len]
            text_mask: Mask for text_embeddings [batch, text_len]

        Returns:
            Training: scalar cross-entropy loss (weighted combination of all codebooks)
            Inference: tuple of (generated codes [batch, 8, gen_len], empty tensor)
        """
        # Simple concatenation approach (Freeze-Omni style)
        # Concatenate conditioning + text, let attention masking handle separation
        # Input already projected by caller - no projection needed here
        context = torch.cat([embeddings, text_embeddings], dim=1)
        context = self.hidden_dropout(context)

        # Combine masks
        if embeddings_mask is not None and text_mask is not None:
            context_mask = torch.cat([embeddings_mask, text_mask], dim=1)
        elif embeddings_mask is not None:
            context_mask = embeddings_mask
        elif text_mask is not None:
            context_mask = text_mask
        else:
            context_mask = None

        if codec_targets is not None:
            return self._compute_loss(context, context_mask, codec_targets, codec_lengths)
        return self._generate(context, context_mask)

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
            context: Combined context [batch, context_len, hidden_dim]
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
            context: Combined context [batch, context_len, hidden_dim]
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

        # Combine all codebooks: [1, 8, seq_len]
        # Depformer already handles delays internally (delays inputs, outputs time-aligned tokens)
        # No additional undelaying needed - outputs are already aligned to audio time
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

    # =========================================================================
    # Streaming Generation (Moshi-style)
    # =========================================================================

    def start_streaming(
        self,
        embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        embeddings_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 1,
    ) -> StreamingState:
        """Initialize streaming audio generation.

        Call this once with the LLM embeddings, then call step() repeatedly
        to generate audio chunks with minimal latency.

        Concatenation approach (Freeze-Omni style):
        - Concatenate conditioning + text embeddings
        - Project and process as combined context

        Args:
            embeddings: LLM hidden states [batch, seq_len, llm_dim]
                Conditioning information (prosody/style)
            text_embeddings: Text token embeddings [batch, text_len, llm_dim]
                Linguistic content (what to say)
            embeddings_mask: Mask for embeddings [batch, seq_len]
            text_mask: Mask for text embeddings [batch, text_len]
            chunk_size: Number of tokens per audio chunk (1 = lowest latency)

        Returns:
            StreamingState to pass to step() calls
        """
        # Simple concatenation approach (Freeze-Omni style)
        # Input already projected by caller - no projection needed here
        context = torch.cat([embeddings, text_embeddings], dim=1)
        context = self.hidden_dropout(context)

        # Combine masks
        if embeddings_mask is not None and text_mask is not None:
            context_mask = torch.cat([embeddings_mask, text_mask], dim=1)
        elif embeddings_mask is not None:
            context_mask = embeddings_mask
        elif text_mask is not None:
            context_mask = text_mask
        else:
            context_mask = None

        device = context.device

        # Initialize Moshi-style delays
        delays = torch.tensor(self.DELAYS, dtype=torch.long, device=device)
        max_delay = int(delays.max().item())

        # Initialize delay cache: [1, num_codebooks, max_delay + buffer]
        cache_size = max_delay + self.DELAY_CACHE_BUFFER
        delay_cache = torch.full(
            (1, self.num_codebooks, cache_size),
            -1,  # Ungenerated token marker
            dtype=torch.long,
            device=device,
        )

        # Create streaming state with delay cache
        state = StreamingState(
            context=context,
            context_mask=context_mask,
            chunk_size=chunk_size,
            device=device,
            delay_cache=delay_cache,
            delays=delays,
            max_delay=max_delay,
            ungenerated_token=-1,
        )

        # Initialize AR decoder generator
        state.ar_generator = self.ar_decoder.generate(
            context=context,
            context_mask=context_mask,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            return_hidden=True,
        )

        # Load Mimi if not already loaded
        if self.mimi is None:
            self.load_mimi_decoder(device=device)

        return state

    def step(self, state: StreamingState) -> Optional[torch.Tensor]:
        """Generate one chunk of audio (Moshi-style delay-based streaming).

        Uses delay-based cache for proper multi-codebook AR modeling.
        Each codebook has its own delay, allowing acoustic codebooks to
        condition on semantic tokens at the same audio time.

        Call this repeatedly after start_streaming() to generate audio
        with minimal latency. Returns None until enough tokens are generated
        to produce aligned output (offset > max_delay).

        Args:
            state: StreamingState from start_streaming()

        Returns:
            Audio waveform chunk [1, samples] or None if not ready/finished
        """
        if state.finished:
            # Flush remaining tokens from delay cache
            return self._flush_delay_cache(state)

        # Generate chunk_size AR steps
        output_codes_list = []

        for _ in range(state.chunk_size):
            try:
                # Step 1: Generate semantic token (codebook 0) from AR decoder
                semantic_token, hidden = next(state.ar_generator)
                state.semantic_tokens.append(semantic_token)
                state.ar_hidden_states.append(hidden)
                state.tokens_generated += 1

                # Step 2: Write semantic token to delay cache
                state.write_to_cache(0, semantic_token)

                # Step 3: Generate acoustic tokens (codebooks 1-7) using Depformer
                # Depformer needs hidden state from AR decoder
                ar_hidden = hidden  # [1, 1, hidden_dim]
                semantic_tensor = torch.tensor([[semantic_token]], device=state.device)

                acoustic = self.depformer.generate_batch(
                    main_hidden=ar_hidden,
                    semantic_tokens=semantic_tensor,
                    temperature=self.temperature,
                    top_k=self.top_k,
                )  # [1, 7, 1]

                # Step 4: Write acoustic tokens to delay cache
                for cb_idx in range(self.NUM_ACOUSTIC_CODEBOOKS):
                    token = int(acoustic[0, cb_idx, 0].item())
                    state.write_to_cache(cb_idx + 1, token)  # +1 for offset past semantic

                # Step 5: Try to read aligned output from cache
                aligned_codes = state.read_from_cache()
                if aligned_codes is not None:
                    output_codes_list.append(aligned_codes)

                # Step 6: Advance offset for next step
                state.advance()

            except StopIteration:
                state.finished = True
                break

        if not output_codes_list:
            return None

        # Stack aligned codes: [1, num_codebooks, num_frames]
        codes = torch.stack(output_codes_list, dim=2)

        # Decode to audio
        with torch.no_grad():
            output = self.mimi.decode(codes)
            return output.audio_values.squeeze(1)  # [1, samples]

    def _flush_delay_cache(self, state: StreamingState) -> Optional[torch.Tensor]:
        """Flush remaining tokens from delay cache after generation ends.

        Since Depformer outputs are time-aligned and we output immediately in step(),
        there are no remaining frames to flush. This method exists for API compatibility.

        Args:
            state: StreamingState (unused, kept for API compatibility)
        """
        del state  # Unused, kept for API compatibility
        return None

    def stop_streaming(self, state: StreamingState) -> None:
        """Clean up streaming state.

        Call this when done with streaming generation to free resources.

        Args:
            state: StreamingState to clean up
        """
        state.ar_generator = None
        state.semantic_tokens.clear()
        state.ar_hidden_states.clear()
        state.delay_cache = None
        state.delays = None
        state.finished = True

    def generate_streaming(
        self,
        embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        embeddings_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 1,
    ) -> Iterator[torch.Tensor]:
        """Generate audio with streaming output (convenience wrapper).

        Yields audio chunks as they're generated for low-latency output.

        Concatenation approach (Freeze-Omni style):
        - Concatenate conditioning + text embeddings
        - Project and process as combined context

        Args:
            embeddings: LLM hidden states [batch, seq_len, llm_dim]
                Conditioning information (prosody/style)
            text_embeddings: Text token embeddings [batch, text_len, llm_dim]
                Linguistic content (what to say)
            embeddings_mask: Mask for embeddings
            text_mask: Mask for text embeddings
            chunk_size: Tokens per chunk (1 = lowest latency, higher = better quality)

        Yields:
            Audio waveform chunks [1, samples]
        """
        state = self.start_streaming(
            embeddings=embeddings,
            text_embeddings=text_embeddings,
            embeddings_mask=embeddings_mask,
            text_mask=text_mask,
            chunk_size=chunk_size,
        )

        try:
            while not state.finished:
                audio_chunk = self.step(state)
                if audio_chunk is not None:
                    yield audio_chunk
        finally:
            self.stop_streaming(state)
