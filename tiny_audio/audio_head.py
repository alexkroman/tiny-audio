"""Audio Head module using CosyVoice 3 for speech token generation.

This module uses CosyVoice 3's pretrained LLM as a decoder, with a trainable
bridge that maps the main LLM's hidden states to CosyVoice's embedding space.

CosyVoice 3 advantages:
- S3 Tokens: Supervised semantic tokens trained on ASR, emotion, speaker analysis
- Dual-Resolution: 5Hz backbone + 25Hz refined head
- Efficient: 0.5B model (~4GB VRAM) outperforms larger models

Architecture:
    SmolLM3 Hidden States → Bridge (trainable) → CosyVoice LLM → Speech Tokens
                                                                      ↓
                                                              Flow Matching (CFM)
                                                                      ↓
                                                                   Audio

Distillation Training:
    No pre-computed codes needed! The bridge learns to match CosyVoice's text encoder:

    Teacher path: Text Response → CosyVoice Tokenizer → CosyVoice Embeddings
    Student path: LLM Hidden States → Bridge → Projected Embeddings
    Loss: MSE + cosine similarity between teacher and student embeddings
"""

import torch
import torch.nn as nn
from torch.nn import functional as nnf


class AudioHead(nn.Module):
    """CosyVoice 3-based Audio Head for speech generation.

    Uses distillation training: the bridge learns to produce embeddings that
    match CosyVoice's text encoder output, without needing pre-computed codes.
    """

    COSY_MODEL_ID = "FunAudioLLM/CosyVoice2-0.5B"

    def __init__(self, config):
        """Initialize AudioHead with CosyVoice bridge.

        Args:
            config: ASRConfig with audio head parameters:
                - llm_dim: Main LLM hidden dimension (default: 1536)
                - audio_head_hidden_dim: Bridge hidden dimension (default: 512)
                - freeze_cosy_llm: Whether to freeze CosyVoice LLM (default: True)
                - distillation_loss_weight: Weight for distillation loss (default: 1.0)
        """
        super().__init__()

        llm_dim = getattr(config, "llm_dim", 1536)
        hidden_dim = getattr(config, "audio_head_hidden_dim", 512)
        freeze_cosy = getattr(config, "freeze_cosy_llm", True)
        self.distillation_loss_weight = getattr(config, "distillation_loss_weight", 1.0)

        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim

        # Lazy load CosyVoice to avoid import issues during config
        self._cosy_llm = None
        self._cosy_tokenizer = None
        self._cosy_dim = None
        self._freeze_cosy = freeze_cosy

        # Bridge will be initialized after we know cosy_dim
        self._bridge = None

    def _load_cosy_model(self):
        """Lazy load CosyVoice LLM and tokenizer."""
        if self._cosy_llm is not None:
            return

        print(f"Loading CosyVoice from {self.COSY_MODEL_ID}...")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        # Load config first to get hidden size
        cosy_config = AutoConfig.from_pretrained(self.COSY_MODEL_ID, trust_remote_code=True)
        self._cosy_dim = cosy_config.hidden_size

        # Load tokenizer for distillation training
        self._cosy_tokenizer = AutoTokenizer.from_pretrained(
            self.COSY_MODEL_ID,
            trust_remote_code=True,
        )

        # Load model
        self._cosy_llm = AutoModelForCausalLM.from_pretrained(
            self.COSY_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        if self._freeze_cosy:
            self._cosy_llm.requires_grad_(False)
            print("CosyVoice LLM frozen")

        # Initialize bridge now that we know dimensions
        self._bridge = nn.Sequential(
            nn.LayerNorm(self.llm_dim),
            nn.Linear(self.llm_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self._cosy_dim),
        )

        # Move bridge to same device/dtype as cosy model
        device = next(self._cosy_llm.parameters()).device
        dtype = next(self._cosy_llm.parameters()).dtype
        self._bridge = self._bridge.to(device=device, dtype=dtype)

        print(f"CosyVoice loaded: {self._cosy_dim}d, bridge: {self.llm_dim} -> {self._cosy_dim}")

    def to(self, *args, **kwargs):
        """Override to handle lazy-loaded modules."""
        result = super().to(*args, **kwargs)
        if self._cosy_llm is not None:
            self._cosy_llm = self._cosy_llm.to(*args, **kwargs)
        if self._bridge is not None:
            self._bridge = self._bridge.to(*args, **kwargs)
        return result

    @property
    def bridge(self) -> nn.Sequential:
        """Get the bridge module, loading if necessary."""
        if self._bridge is None:
            self._load_cosy_model()
        assert self._bridge is not None  # Narrowing for type checker
        return self._bridge

    @property
    def cosy_llm(self):
        """Get CosyVoice LLM, loading if necessary."""
        if self._cosy_llm is None:
            self._load_cosy_model()
        return self._cosy_llm

    @property
    def cosy_tokenizer(self):
        """Get CosyVoice tokenizer, loading if necessary."""
        if self._cosy_tokenizer is None:
            self._load_cosy_model()
        return self._cosy_tokenizer

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_targets: list[str] | None = None,
    ) -> torch.Tensor:
        """Forward pass for training (distillation) or inference.

        Training uses distillation: the bridge learns to produce embeddings
        that match CosyVoice's text encoder output for the text response.
        No pre-computed speech codes needed!

        Args:
            hidden_states: Main LLM hidden states (batch, seq_len, llm_dim)
            text_targets: Text responses for distillation training (list of strings)

        Returns:
            If text_targets is provided: scalar distillation loss
            If text_targets is None: predicted speech tokens (batch, token_len)
        """
        # Ensure models are loaded
        self._load_cosy_model()

        # Project hidden states to CosyVoice embedding space
        projected = self.bridge(hidden_states)  # (batch, seq, cosy_dim)

        # Training mode: distillation
        if text_targets is not None:
            return self._forward_distillation(projected, text_targets)

        # Inference mode
        return self._forward_inference(projected)

    def _forward_distillation(
        self,
        projected_states: torch.Tensor,
        text_targets: list[str],
    ) -> torch.Tensor:
        """Distillation training: match CosyVoice's text encoder embeddings.

        No pre-computed codes needed! The bridge learns to produce embeddings
        that match what CosyVoice's text encoder would produce for the response.

        Args:
            projected_states: Bridge output (batch, seq_len, cosy_dim)
            text_targets: List of text responses to distill from

        Returns:
            Scalar distillation loss (MSE + cosine similarity)
        """
        device = projected_states.device
        dtype = projected_states.dtype

        # Get teacher embeddings from CosyVoice's text encoder
        with torch.no_grad():
            # Tokenize text targets
            encoded = self.cosy_tokenizer(
                text_targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Get embeddings from CosyVoice's embedding layer
            # This is the target our bridge should learn to produce
            teacher_embeds = self.cosy_llm.get_input_embeddings()(encoded.input_ids)
            teacher_embeds = teacher_embeds.to(dtype)  # (batch, text_len, cosy_dim)

        # Align sequence lengths between student and teacher
        student_len = projected_states.shape[1]
        teacher_len = teacher_embeds.shape[1]

        if student_len > teacher_len:
            # Pool student states to match teacher length
            projected_states = nnf.adaptive_avg_pool1d(
                projected_states.transpose(1, 2), teacher_len
            ).transpose(1, 2)
        elif teacher_len > student_len:
            # Pool teacher states to match student length
            teacher_embeds = nnf.adaptive_avg_pool1d(
                teacher_embeds.transpose(1, 2), student_len
            ).transpose(1, 2)

        # Create attention mask for valid positions
        attention_mask = encoded.attention_mask
        if attention_mask.shape[1] != projected_states.shape[1]:
            # Interpolate mask to match aligned length
            attention_mask = (
                nnf.interpolate(
                    attention_mask.unsqueeze(1).float(),
                    size=projected_states.shape[1],
                    mode="nearest",
                )
                .squeeze(1)
                .bool()
            )

        # MSE loss on embeddings (masked)
        mask = attention_mask.unsqueeze(-1).expand_as(projected_states)
        mse_loss = nnf.mse_loss(
            projected_states[mask],
            teacher_embeds[mask],
            reduction="mean",
        )

        # Cosine similarity loss for direction alignment
        student_norm = nnf.normalize(projected_states, p=2, dim=-1)
        teacher_norm = nnf.normalize(teacher_embeds, p=2, dim=-1)
        cosine_loss = 1.0 - (student_norm * teacher_norm).sum(dim=-1)
        cosine_loss = (cosine_loss * attention_mask.float()).sum() / attention_mask.sum()

        # Combined loss
        total_loss = mse_loss + 0.1 * cosine_loss

        return total_loss * self.distillation_loss_weight

    def _forward_inference(
        self,
        projected_states: torch.Tensor,
        max_new_tokens: int = 512,
    ) -> torch.Tensor:
        """Inference forward pass - generate speech tokens.

        Args:
            projected_states: (batch, seq_len, cosy_dim) projected embeddings
            max_new_tokens: Maximum tokens to generate

        Returns:
            (batch, token_len) generated speech tokens
        """
        # Generate speech tokens using CosyVoice LLM
        with torch.no_grad():
            return self.cosy_llm.generate(
                inputs_embeds=projected_states,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

    def get_output_length(self, input_length: int) -> int:
        """Estimate output speech token count.

        CosyVoice uses ~25Hz token rate for speech.
        Assuming ~5 LLM tokens per second input, ratio is ~5x.

        Args:
            input_length: Number of input LLM hidden states

        Returns:
            Estimated number of speech tokens
        """
        return int(input_length * 5)

    def state_dict(self, *args, **kwargs):
        """Return only the bridge weights (CosyVoice is pretrained)."""
        # Ignore args/kwargs - we only save bridge weights
        del args, kwargs
        if self._bridge is None:
            return {}
        return {f"bridge.{k}": v for k, v in self._bridge.state_dict().items()}

    def load_state_dict(self, state_dict, strict=True):
        """Load only bridge weights."""
        # Ensure models are loaded first
        self._load_cosy_model()

        # Extract bridge weights
        bridge_state = {
            k.replace("bridge.", ""): v for k, v in state_dict.items() if k.startswith("bridge.")
        }

        if bridge_state:
            self._bridge.load_state_dict(bridge_state, strict=strict)
