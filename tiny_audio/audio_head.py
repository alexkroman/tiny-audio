"""Autoregressive audio head for speech-to-speech.

Generates audio from LLM embeddings via discrete codec tokens:
  LLM embeddings -> Pre-NN -> AR Decoder -> Depformer -> Mimi codes -> audio

Architecture:
- Pre-NN transformer (3 layers, bidirectional) processes LLM hidden states
- AR decoder (6 layers, causal) generates semantic codebook 0 autoregressively
- Depformer (4 layers) predicts acoustic codebooks 1-7 conditioned on codebook 0
- Mimi decoder converts all 8 codebooks to audio waveform

Optional Prefix Bridge (Freeze-Omni style):
- PrefixBridge transforms LLM hidden states into KV cache entries
- Enables efficient fine-tuning by freezing main components
- Only prefix bridge layers are trained, bridging text→audio modality gap

Streaming Support (Moshi-style):
- StreamingState holds generation state between steps
- step() method generates one audio frame at a time
- Mimi decoder runs in streaming mode for low-latency output
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

import torch
import torch.nn as nn

from .modules.ar_decoder import CodecARDecoder, PreNN
from .modules.depformer import Depformer
from .modules.prefix_bridge import PrefixBridge

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)


@dataclass
class StreamingState:
    """State for streaming audio generation (Moshi-style).

    Maintains all state needed between step() calls for low-latency streaming.
    """

    # Generation context (from Pre-NN)
    context: torch.Tensor  # [1, context_len, hidden_dim]
    context_mask: Optional[torch.Tensor]  # [1, context_len]

    # AR decoder state
    ar_generator: Optional[Iterator] = None  # AR decoder generator
    ar_kv_cache: Optional["DynamicCache"] = None  # KV cache for AR decoder

    # Prefix bridge KV cache (optional)
    prefix_kv_cache: Optional["DynamicCache"] = None

    # Generated tokens buffer
    semantic_tokens: list[int] = field(default_factory=list)
    ar_hidden_states: list[torch.Tensor] = field(default_factory=list)

    # Streaming parameters
    chunk_size: int = 1  # Tokens per audio chunk (1 = lowest latency)
    tokens_generated: int = 0
    finished: bool = False

    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


class AudioHead(nn.Module):
    """AR codec head: LLM embeddings -> Mimi codes -> audio.

    Architecture:
        - input_proj: Projects LLM embeddings to hidden_dim
        - pre_nn: 3-layer bidirectional transformer for context processing
        - ar_decoder: 6-layer causal transformer for semantic codebook 0
        - depformer: 4-layer transformer for acoustic codebooks 1-7
        - Mimi decoder (frozen) for codes -> audio

    Optional Prefix Bridge (Freeze-Omni style KV-cache fine-tuning):
        - prefix_bridge: Learnable transformer layers that transform LLM hidden
          states into KV cache entries for the AR decoder
        - When enabled, main components can be frozen and only prefix bridge
          is trained, enabling efficient transfer from text to audio space

    Args:
        config: ASRConfig with:
            - llm_dim: LLM embedding dimension (default: 2048 for SmolLM3-3B)
            - use_prefix_bridge: Enable prefix bridge for KV-cache fine-tuning
        llm_dim: Override for LLM dimension (takes precedence over config)
        use_prefix_bridge: Override for prefix bridge flag
    """

    # Architecture dimensions (scaled for SmolLM3-3B following Freeze-Omni ratios)
    # Freeze-Omni uses ~0.22x LLM hidden dim (896 for 4096-dim Qwen2-7B)
    # For SmolLM3-3B (2048 hidden): 2048 * 0.25 = 512
    HIDDEN_DIM = 512  # Transformer hidden dimension
    INTERMEDIATE_DIM = 2048  # FFN intermediate dimension (4x hidden)
    NUM_HEADS = 8  # Attention heads (64 dim per head)

    # Pre-NN config (Freeze-Omni doesn't have this, keep minimal)
    PRE_NN_LAYERS = 2

    # AR Decoder config (for semantic codebook 0) - matches Freeze-Omni's 4 layers
    AR_LAYERS = 4
    VOCAB_SIZE = 2048  # Mimi codebook size

    # Depformer config (for acoustic codebooks 1-7) - scaled down proportionally
    DEPFORMER_DIM = 256
    DEPFORMER_LAYERS = 4
    DEPFORMER_HEADS = 4  # 64 dim per head
    DEPFORMER_INTERMEDIATE = 1024  # 4x hidden
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

    def __init__(self, config, llm_dim: int = None, use_prefix_bridge: bool = None):
        super().__init__()
        self.llm_dim = llm_dim or getattr(config, "llm_dim", None) or 2048  # SmolLM3 native
        self.hidden_dim = self.HIDDEN_DIM
        self.vocab_size = self.VOCAB_SIZE
        self.num_codebooks = 1 + self.NUM_ACOUSTIC_CODEBOOKS  # 8 total

        # Prefix bridge flag (Freeze-Omni style KV-cache fine-tuning)
        self.use_prefix_bridge = (
            use_prefix_bridge
            if use_prefix_bridge is not None
            else getattr(config, "use_prefix_bridge", False)
        )

        # Generation parameters from config
        self.max_tokens = getattr(config, "max_audio_tokens", self.DEFAULT_MAX_TOKENS)
        self.top_k = getattr(config, "audio_top_k", self.DEFAULT_TOP_K)
        self.temperature = getattr(config, "audio_temperature", self.DEFAULT_TEMPERATURE)
        self.repetition_penalty = getattr(
            config, "audio_repetition_penalty", self.DEFAULT_REPETITION_PENALTY
        )

        # Input projection: LLM dim -> hidden dim (identity if dims match)
        if self.llm_dim != self.hidden_dim:
            self.input_proj = nn.Linear(self.llm_dim, self.hidden_dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        # Shared embedding for all tokens (Freeze-Omni style)
        # vocab_size + 4 special tokens: BOS, SOS, EOS, PAD
        self.total_vocab = self.vocab_size + 4  # BOS=+0, SOS=+1, EOS=+2, PAD=+3
        self.embedding = nn.Embedding(
            self.total_vocab,
            self.hidden_dim,
            padding_idx=self.vocab_size + 3,  # PAD token
        )

        # Pre-NN: Process LLM hidden states with bidirectional attention
        self.pre_nn = PreNN(
            hidden_size=self.hidden_dim,
            num_layers=self.PRE_NN_LAYERS,
            num_heads=self.NUM_HEADS,
            intermediate_size=self.INTERMEDIATE_DIM,
        )

        # AR Decoder: Generate semantic codebook 0 autoregressively
        # Uses shared embedding from AudioHead
        self.ar_decoder = CodecARDecoder(
            hidden_size=self.hidden_dim,
            num_layers=self.AR_LAYERS,
            num_heads=self.NUM_HEADS,
            intermediate_size=self.INTERMEDIATE_DIM,
            vocab_size=self.vocab_size,
            embedding=self.embedding,  # Share embedding
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

        # Optional Prefix Bridge for KV-cache fine-tuning (Freeze-Omni style)
        # Transforms LLM hidden states into KV cache entries that condition
        # the AR decoder, enabling efficient text→audio transfer learning
        if self.use_prefix_bridge:
            self.prefix_bridge = PrefixBridge(
                hidden_size=self.hidden_dim,
                num_layers=self.AR_LAYERS,  # Match AR decoder layers
                num_heads=self.NUM_HEADS,
                intermediate_size=self.INTERMEDIATE_DIM,
                input_dim=self.hidden_dim,  # After input_proj
            )
        else:
            self.prefix_bridge = None

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

    def enable_prefix_tuning(self, freeze_main_components: bool = True):
        """Enable prefix bridge tuning mode.

        When enabled:
        - Prefix bridge layers are trainable (if present)
        - Main components (input_proj, pre_nn, ar_decoder, depformer) are frozen

        This enables efficient fine-tuning where only the prefix bridge
        learns to transfer text-space hidden states to audio decoder space.

        Args:
            freeze_main_components: If True, freeze all non-prefix components
        """
        if self.prefix_bridge is None:
            raise ValueError(
                "Cannot enable prefix tuning without prefix_bridge. "
                "Initialize AudioHead with use_prefix_bridge=True"
            )

        if freeze_main_components:
            # Freeze main components
            self.input_proj.requires_grad_(False)
            self.pre_nn.requires_grad_(False)
            self.ar_decoder.requires_grad_(False)
            self.depformer.requires_grad_(False)

            # Set to eval mode to disable dropout etc.
            self.input_proj.eval()
            self.pre_nn.eval()
            self.ar_decoder.eval()
            self.depformer.eval()

            logger.info("Prefix tuning enabled: froze input_proj, pre_nn, ar_decoder, depformer")

        # Ensure prefix bridge is trainable
        self.prefix_bridge.requires_grad_(True)
        self.prefix_bridge.train()

        logger.info("Prefix bridge layers are trainable")

    def disable_prefix_tuning(self):
        """Disable prefix tuning mode, making all components trainable."""
        self.input_proj.requires_grad_(True)
        self.pre_nn.requires_grad_(True)
        self.ar_decoder.requires_grad_(True)
        self.depformer.requires_grad_(True)

        if self.prefix_bridge is not None:
            self.prefix_bridge.requires_grad_(True)

        logger.info("Prefix tuning disabled: all components are trainable")

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
        codec_targets: Optional[torch.Tensor] = None,
        codec_lengths: Optional[torch.Tensor] = None,
        embeddings_mask: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training or inference.

        Dual-path processing (Freeze-Omni style):
        - Path 1 (Context): text_embeddings → Pre-NN → AR decoder context
          Provides linguistic content (what to say)
        - Path 2 (Conditioning): embeddings → Prefix Bridge → KV cache
          Provides prosodic/stylistic conditioning (how to say it)

        If text_embeddings is not provided, falls back to single-path mode
        where embeddings is used for both paths.

        Args:
            embeddings: LLM hidden states [batch, seq_len, llm_dim]
                Used for prefix bridge conditioning (prosody/style)
            codec_targets: Target Mimi codes for training.
                - [batch, 8, audio_seq_len] for full training (all codebooks)
                - [batch, audio_seq_len] for legacy (codebook 0 only)
            codec_lengths: Actual audio lengths per sample [batch]
            embeddings_mask: Mask for embeddings [batch, seq_len]
            text_embeddings: Optional text token embeddings [batch, text_len, llm_dim]
                Used for Pre-NN context (linguistic content)
                If None, uses embeddings for context instead (single-path mode)
            text_mask: Optional mask for text_embeddings [batch, text_len]

        Returns:
            Training: scalar cross-entropy loss (weighted combination of all codebooks)
            Inference: tuple of (generated codes [batch, 8, gen_len], empty tensor)
        """
        # Dual-path processing:
        # Path 1: Text embeddings → Pre-NN → context (what to say)
        # Path 2: Hidden states → Prefix Bridge → KV cache (how to say it)

        if text_embeddings is not None:
            # Dual-path mode: separate inputs for context and conditioning
            context_input = text_embeddings
            context_mask = text_mask
            conditioning_input = embeddings
            conditioning_mask = embeddings_mask
        else:
            # Single-path mode: same input for both (backward compatible)
            context_input = embeddings
            context_mask = embeddings_mask
            conditioning_input = embeddings
            conditioning_mask = embeddings_mask

        # Path 1: Project and process through Pre-NN for context
        hidden = self.input_proj(context_input)
        context = self.pre_nn(hidden, attention_mask=context_mask)

        # Path 2: Compute prefix KV cache from conditioning input
        prefix_kv_cache = None
        if self.prefix_bridge is not None:
            conditioning_hidden = self.input_proj(conditioning_input)
            _, prefix_kv_cache = self.prefix_bridge(
                hidden_states=conditioning_hidden,
                attention_mask=conditioning_mask,
            )

        if codec_targets is not None:
            return self._compute_loss(
                context, context_mask, codec_targets, codec_lengths, prefix_kv_cache
            )
        return self._generate(context, context_mask, prefix_kv_cache)

    def _compute_loss(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor],
        prefix_kv_cache: Optional["DynamicCache"] = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for codec token prediction.

        Trains both AR decoder (codebook 0) and Depformer (codebooks 1-7).
        Uses equal weighting following Moshi's approach.

        When prefix_kv_cache is provided (from prefix bridge):
        - AR decoder attends to prefix KV cache + context
        - Enables text→audio modality transfer via KV conditioning

        Args:
            context: Pre-NN output [batch, context_len, hidden_dim]
            context_mask: Mask for context [batch, context_len]
            targets: Target codec tokens [batch, 8, target_len] for all codebooks
            lengths: Actual target lengths [batch]
            prefix_kv_cache: Optional pre-computed KV cache from PrefixBridge

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
        # Pass prefix_kv_cache for text→audio modality bridging
        _, semantic_loss, ar_hidden = self.ar_decoder(
            context=context,
            context_mask=context_mask,
            target_ids=semantic_targets,
            target_mask=target_mask,
            return_hidden=True,
            prefix_kv_cache=prefix_kv_cache,
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
        prefix_kv_cache: Optional["DynamicCache"] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate codec tokens via AR decoding + Depformer.

        Uses AR decoder hidden states for Depformer conditioning, providing
        position-specific context at each generated timestep.

        When prefix_kv_cache is provided (from prefix bridge):
        - AR decoder attends to prefix KV cache during generation
        - Enables text→audio modality transfer via KV conditioning

        Args:
            context: Pre-NN output [batch, context_len, hidden_dim]
            context_mask: Mask for context [batch, context_len]
            prefix_kv_cache: Optional pre-computed KV cache from PrefixBridge

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
            prefix_kv_cache=prefix_kv_cache,
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

    # =========================================================================
    # Streaming Generation (Moshi-style)
    # =========================================================================

    def start_streaming(
        self,
        embeddings: torch.Tensor,
        embeddings_mask: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 1,
    ) -> StreamingState:
        """Initialize streaming audio generation.

        Call this once with the LLM embeddings, then call step() repeatedly
        to generate audio chunks with minimal latency.

        Args:
            embeddings: LLM hidden states [batch, seq_len, llm_dim]
            embeddings_mask: Mask for embeddings [batch, seq_len]
            text_embeddings: Optional text embeddings for dual-path
            text_mask: Optional mask for text embeddings
            chunk_size: Number of tokens per audio chunk (1 = lowest latency)

        Returns:
            StreamingState to pass to step() calls
        """
        # Dual-path processing (same as forward())
        if text_embeddings is not None:
            context_input = text_embeddings
            context_mask = text_mask
            conditioning_input = embeddings
            conditioning_mask = embeddings_mask
        else:
            context_input = embeddings
            context_mask = embeddings_mask
            conditioning_input = embeddings
            conditioning_mask = embeddings_mask

        # Path 1: Project and process through Pre-NN for context
        hidden = self.input_proj(context_input)
        context = self.pre_nn(hidden, attention_mask=context_mask)

        # Path 2: Compute prefix KV cache from conditioning input
        prefix_kv_cache = None
        if self.prefix_bridge is not None:
            conditioning_hidden = self.input_proj(conditioning_input)
            _, prefix_kv_cache = self.prefix_bridge(
                hidden_states=conditioning_hidden,
                attention_mask=conditioning_mask,
            )

        device = context.device

        # Create streaming state
        state = StreamingState(
            context=context,
            context_mask=context_mask,
            prefix_kv_cache=prefix_kv_cache,
            chunk_size=chunk_size,
            device=device,
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
            prefix_kv_cache=prefix_kv_cache,
        )

        # Load Mimi if not already loaded
        if self.mimi is None:
            self.load_mimi_decoder(device=device)

        return state

    def step(self, state: StreamingState) -> Optional[torch.Tensor]:
        """Generate one chunk of audio (Moshi-style streaming).

        Call this repeatedly after start_streaming() to generate audio
        with minimal latency. Returns None when generation is complete.

        Args:
            state: StreamingState from start_streaming()

        Returns:
            Audio waveform chunk [1, samples] or None if finished
        """
        if state.finished:
            return None

        # Generate chunk_size tokens
        tokens_this_chunk = []
        hidden_this_chunk = []

        for _ in range(state.chunk_size):
            try:
                token, hidden = next(state.ar_generator)
                tokens_this_chunk.append(token)
                hidden_this_chunk.append(hidden)
                state.semantic_tokens.append(token)
                state.ar_hidden_states.append(hidden)
                state.tokens_generated += 1
            except StopIteration:
                state.finished = True
                break

        if not tokens_this_chunk:
            return None

        # Convert to tensors
        semantic = torch.tensor([tokens_this_chunk], device=state.device)  # [1, chunk]
        ar_hidden = torch.cat(hidden_this_chunk, dim=1)  # [1, chunk, hidden]

        # Generate acoustic codebooks for this chunk
        acoustic = self.depformer.generate_batch(
            main_hidden=ar_hidden,
            semantic_tokens=semantic,
            temperature=self.temperature,
            top_k=self.top_k,
        )  # [1, 7, chunk]

        # Combine: [1, 8, chunk]
        codes = torch.cat([semantic.unsqueeze(1), acoustic], dim=1)

        # Decode to audio
        with torch.no_grad():
            output = self.mimi.decode(codes)
            return output.audio_values.squeeze(1)  # [1, samples]

    def stop_streaming(self, state: StreamingState) -> None:
        """Clean up streaming state.

        Call this when done with streaming generation to free resources.

        Args:
            state: StreamingState to clean up
        """
        state.ar_generator = None
        state.ar_kv_cache = None
        state.prefix_kv_cache = None
        state.semantic_tokens.clear()
        state.ar_hidden_states.clear()
        state.finished = True

    def generate_streaming(
        self,
        embeddings: torch.Tensor,
        embeddings_mask: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 1,
    ) -> Iterator[torch.Tensor]:
        """Generate audio with streaming output (convenience wrapper).

        Yields audio chunks as they're generated for low-latency output.

        Args:
            embeddings: LLM hidden states [batch, seq_len, llm_dim]
            embeddings_mask: Mask for embeddings
            text_embeddings: Optional text embeddings for dual-path
            text_mask: Optional mask for text embeddings
            chunk_size: Tokens per chunk (1 = lowest latency, higher = better quality)

        Yields:
            Audio waveform chunks [1, samples]
        """
        state = self.start_streaming(
            embeddings=embeddings,
            embeddings_mask=embeddings_mask,
            text_embeddings=text_embeddings,
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
