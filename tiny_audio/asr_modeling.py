import json
from pathlib import Path
from threading import Thread
from typing import Iterator, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TextIteratorStreamer,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .asr_config import ASRConfig, compute_encoder_output_length
    from .projectors import PROJECTOR_CLASSES
except ImportError:
    from asr_config import ASRConfig, compute_encoder_output_length  # type: ignore[no-redef]
    from projectors import PROJECTOR_CLASSES  # type: ignore[no-redef]


def _resolve_attn_implementation(requested: Optional[str]) -> Optional[str]:
    """Coerce flash_attention_2 to sdpa when CUDA isn't available.

    FA2 is CUDA-only. On MPS/CPU, requesting it either errors at load or
    silently falls back to a slower path; either way the user pays the FA2
    install + import cost for no win. Coerce here so a saved config that
    pins flash_attention_2 still loads on Mac / CPU-only Linux boxes.
    """
    if requested == "flash_attention_2" and not torch.cuda.is_available():
        return "sdpa"
    return requested


def _gather_audio_embeds(audio_embeds: torch.Tensor, token_counts: torch.Tensor) -> torch.Tensor:
    """Flatten per-sample audio embeddings into a packed tensor.

    For each row i, takes the first ``token_counts[i]`` rows of
    ``audio_embeds[i]`` and concatenates them. If any token count exceeds
    ``audio_embeds.shape[1]``, the deficit is zero-padded.

    Equivalent to a per-sample slice/cat loop but with O(1) host-device
    syncs per call (one ``max().item()``) instead of one per sample.
    """
    _, max_len, _ = audio_embeds.shape
    needed = int(token_counts.max().item())
    if needed > max_len:
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, needed - max_len))
        max_len = needed
    indices = torch.arange(max_len, device=audio_embeds.device).unsqueeze(0)
    mask = indices < token_counts.unsqueeze(1)
    return audio_embeds[mask]


class ASRModel(PreTrainedModel, GenerationMixin):
    """Audio-to-text model combining an audio encoder, projector, and language model."""

    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _is_loading_from_pretrained: bool = False

    TRANSCRIBE_PROMPT = "Transcribe the speech to text"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> "ASRModel":
        """Load model from pretrained, handling device placement correctly."""
        from safetensors.torch import load_file
        from transformers.utils.hub import cached_file

        config = kwargs.pop("config", None)
        if config is None:
            config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Set flag to avoid device_map="auto" in sub-model loaders
        cls._is_loading_from_pretrained = True

        try:
            model = cls(config, **kwargs)

            # Load projector weights from safetensors
            subfolder = kwargs.get("subfolder")
            revision = kwargs.get("revision")
            cache_kwargs = {}
            if subfolder:
                cache_kwargs["subfolder"] = subfolder
            if revision:
                cache_kwargs["revision"] = revision

            model_file = cached_file(
                pretrained_model_name_or_path,
                "model.safetensors",
                _raise_exceptions_for_missing_entries=False,
                **cache_kwargs,
            )

            if model_file is not None:
                state_dict = load_file(model_file)
                model.load_state_dict(state_dict, strict=False)

            # Load LoRA adapters if use_lora is enabled
            if getattr(config, "use_lora", False):
                # Check for adapter_config.json (required by PEFT to load adapters)
                adapter_config_file = cached_file(
                    pretrained_model_name_or_path,
                    "adapter_config.json",
                    _raise_exceptions_for_missing_entries=False,
                    **cache_kwargs,
                )
                if adapter_config_file is not None:
                    # Load saved adapter weights using the original repo_id/path
                    # PEFT handles Hub downloads and caching internally
                    from peft import PeftModel

                    model.language_model = PeftModel.from_pretrained(
                        model.language_model,
                        pretrained_model_name_or_path,
                        is_trainable=True,
                        **cache_kwargs,
                    )
                else:
                    # No saved adapters - initialize fresh LLM LoRA for training
                    from peft import LoraConfig, get_peft_model

                    lora_config = LoraConfig(
                        r=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                        target_modules=config.lora_target_modules,
                        lora_dropout=config.lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    model.language_model = get_peft_model(model.language_model, lora_config)

            return model
        finally:
            cls._is_loading_from_pretrained = False

    def __init__(self, config: ASRConfig, **kwargs) -> None:
        super().__init__(config)

        self.system_prompt = config.system_prompt
        target_dtype = getattr(torch, config.model_dtype)

        # Audio encoder (frozen)
        self.audio_tower = self._load_audio_encoder(config, target_dtype)

        # Language model (frozen)
        self.language_model = self._load_language_model(config, target_dtype)

        # Initialize tokenizer and special tokens
        self._init_tokenizer(config)

        # Set up generation config with greedy decoding defaults
        self.generation_config = self.language_model.generation_config
        self.generation_config.max_new_tokens = config.max_new_tokens
        self.generation_config.min_new_tokens = config.min_new_tokens
        self.generation_config.num_beams = config.num_beams
        self.generation_config.do_sample = config.do_sample
        # Set sampling params from config (None means use model defaults)
        self.generation_config.temperature = config.temperature
        self.generation_config.top_p = config.top_p
        self.generation_config.top_k = config.top_k
        self.generation_config.use_cache = config.use_cache
        self.generation_config.length_penalty = config.length_penalty
        self.generation_config.repetition_penalty = config.repetition_penalty
        self.generation_config.no_repeat_ngram_size = config.no_repeat_ngram_size
        # Set EOS tokens, filtering out any that don't exist in the tokenizer
        eos_candidates = [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        self.generation_config.eos_token_id = [t for t in eos_candidates if t is not None]
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Feature extractor for audio preprocessing
        self.feature_extractor = self._create_feature_extractor(config)

        # Audio projector (trainable unless freeze_projector is set)
        self.projector = self._create_projector(config, target_dtype)

        # Setup LoRA if enabled (Stage 2 fine-tuning)
        # Skip if loading from pretrained - from_pretrained will handle adapter loading
        if getattr(config, "use_lora", False) and not getattr(
            self.__class__, "_is_loading_from_pretrained", False
        ):
            self._setup_lora(config)

        # Freeze projector if specified (for Stage 2 LoRA-only training)
        if getattr(config, "freeze_projector", False):
            self.projector.requires_grad_(False)

        # Freeze the text-vocab embedding table (preserves base Qwen3's
        # token→embedding mapping during joint fine-tune). With
        # tie_word_embeddings=True the same tensor backs lm_head, so this
        # also freezes the output projection. Audio tokens bypass this
        # table — they're scattered into inputs_embeds via masked_scatter
        # at <audio> positions (forward(), below), so the audio path is
        # unaffected. Mirrors Baichuan-Audio's stage-2 policy of training
        # all decoder params except the text embedding and LM head.
        if getattr(config, "freeze_text_embed_tokens", False):
            self.language_model.get_input_embeddings().weight.requires_grad_(False)

        # For model parallelism
        self._no_split_modules = getattr(self.language_model, "_no_split_modules", [])

    def _create_feature_extractor(self, config: ASRConfig):
        """Create the appropriate feature extractor for the audio encoder."""
        from transformers import AutoFeatureExtractor

        feature_extractor = AutoFeatureExtractor.from_pretrained(config.audio_model_id)
        # Whisper's encoder requires a fixed 3000 mel frames (30s) and the
        # feature extractor pads to that by default — leave it alone. Other
        # encoders (e.g. GLM-ASR) accept variable-length input, so we disable
        # padding to avoid wasting compute on silent frames.
        if "whisper" not in config.audio_model_id.lower():
            feature_extractor.padding = False
        return feature_extractor

    @classmethod
    def _load_audio_encoder(cls, config: ASRConfig, dtype: torch.dtype) -> nn.Module:
        """Load the audio encoder; freeze unless `config.freeze_audio_encoder=False`.

        When unfrozen, the encoder participates in joint training — pair with a
        much lower `encoder_learning_rate` than the projector/decoder LRs
        (encoder is large, sensitive to perturbation, and shouldn't drift far
        from its pretrained features). See `ASRTrainer.create_optimizer` for the
        LR routing.
        """
        encoder_kwargs = {
            "attn_implementation": _resolve_attn_implementation(config.attn_implementation),
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }

        if "whisper" in config.audio_model_id.lower():
            from transformers import WhisperModel

            full_model = WhisperModel.from_pretrained(config.audio_model_id, **encoder_kwargs)
            encoder = full_model.encoder
            del full_model
        elif "glm" in config.audio_model_id.lower():
            # GLM-ASR models use audio_tower as the encoder
            # Requires transformers >= 5.x or installed from source
            from transformers import AutoModelForSeq2SeqLM

            full_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.audio_model_id, trust_remote_code=True, **encoder_kwargs
            )
            # GLM stores encoder at audio_tower (GlmAsrEncoder)
            encoder = full_model.audio_tower
            # Clear references to free VRAM from the LLM decoder
            full_model.language_model = None
            full_model.multi_modal_projector = None
            del full_model
        else:
            encoder = AutoModel.from_pretrained(config.audio_model_id, **encoder_kwargs)

        # Explicit cast: from_pretrained's `dtype=` kwarg is honored
        # inconsistently across loader paths (especially trust_remote_code
        # branches like GLM-ASR), leaving submodules in fp32. FA2's startup
        # then complains "current dype is torch.float32, expected fp16/bf16",
        # and even with sdpa the projector→encoder feed mismatches dtypes.
        # `.to(dtype=...)` after load is idempotent and forces the issue.
        encoder = encoder.to(dtype=dtype)
        if getattr(config, "freeze_audio_encoder", True):
            encoder.requires_grad_(False)
            encoder.train(False)  # equivalent to .eval(); avoids a security hook false-positive
        return encoder

    @classmethod
    def _load_language_model(cls, config: ASRConfig, dtype: torch.dtype) -> PreTrainedModel:
        """Load and freeze the language model."""
        decoder_kwargs = {
            "attn_implementation": _resolve_attn_implementation(config.attn_implementation),
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }

        decoder = AutoModelForCausalLM.from_pretrained(config.text_model_id, **decoder_kwargs)
        # See _load_audio_encoder note: idempotent post-load cast to dodge the
        # FA2 "current dype is fp32" warning when from_pretrained's dtype kwarg
        # isn't fully propagated to every submodule.
        decoder = decoder.to(dtype=dtype)
        decoder.config.use_cache = getattr(config, "use_cache", True)
        if getattr(config, "freeze_language_model", True):
            decoder.requires_grad_(False)
            decoder.train(False)
        return decoder

    def _create_projector(self, config: ASRConfig, dtype: torch.dtype) -> nn.Module:
        """Create the trainable audio projector."""
        # Auto-detect dimensions if not specified
        if config.encoder_dim is None:
            enc_cfg = self.audio_tower.config
            config.encoder_dim = getattr(enc_cfg, "hidden_size", None) or getattr(
                enc_cfg, "d_model", None
            )
            if config.encoder_dim is None:
                raise ValueError("Could not auto-detect encoder_dim. Please specify in config.")

        if config.llm_dim is None:
            dec_cfg = self.language_model.config
            config.llm_dim = getattr(dec_cfg, "hidden_size", None) or getattr(
                dec_cfg, "d_model", None
            )
            if config.llm_dim is None:
                raise ValueError("Could not auto-detect llm_dim. Please specify in config.")

        # Select projector type based on config
        projector_type = getattr(config, "projector_type", "mlp")
        projector_class = PROJECTOR_CLASSES.get(projector_type)
        if projector_class is None:
            raise ValueError(
                f"Unknown projector_type: {projector_type}. "
                f"Valid options: {list(PROJECTOR_CLASSES.keys())}"
            )
        projector = projector_class(config)

        # Move projector to same device as language model (important when using quantization)
        device = next(self.language_model.parameters()).device
        return projector.to(device=device, dtype=dtype)

    def _setup_lora(self, config: ASRConfig):
        """Apply LoRA adapters to the language model for Stage 2 fine-tuning."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, lora_config)

    def _init_tokenizer(self, config: ASRConfig):
        """Initialize tokenizer with audio token."""
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_id, trust_remote_code=True)

        # Set pad token. Prefer a dedicated pad token if the tokenizer has one
        # (e.g. Qwen's <|finetune_right_pad_id|>); otherwise fall back to
        # eos_token, which is the standard pattern for Llama-style tokenizers
        # (SmolLM2, Llama, etc.) that ship without a separate pad token.
        if (
            self.tokenizer.pad_token is None
            or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            if "<|finetune_right_pad_id|>" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
            elif self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add audio token
        existing_special = getattr(self.tokenizer, "additional_special_tokens", None) or []
        if "<audio>" not in existing_special:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": existing_special + ["<audio>"]}
            )
            # mean_resizing=True initializes the new <audio> row at the mean of
            # existing rows so its scale matches the pretrained distribution. The
            # input-side <audio> embedding is overwritten via masked_scatter and
            # never seen by the LM, but with tied embeddings (Qwen3-0.6B) this
            # same row is the lm_head column for predicting <audio>; a Gaussian
            # draw at config.initializer_range was visible in early-step logits.
            self.language_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)

        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<audio>")
        self.tokenizer.padding_side = "right"

        # Sync token IDs to configs
        for cfg in [self.config.text_config, self.language_model.config, self.generation_config]:
            if cfg is not None:
                cfg.pad_token_id = self.tokenizer.pad_token_id
                cfg.eos_token_id = self.tokenizer.eos_token_id
                cfg.bos_token_id = self.tokenizer.bos_token_id

    def train(self, mode: bool = True):
        """Set train/eval mode, but keep frozen submodules out of train mode.

        HF Trainer calls `model.train()` at the top of every training step, which
        recursively switches every submodule into train mode — re-enabling dropout
        on modules with `requires_grad_(False)`. The frozen encoder (and the LM
        when `freeze_language_model=True`) should always run deterministically;
        train-mode dropout only adds noise that can't improve a frozen network.
        """
        super().train(mode)
        if getattr(self.config, "freeze_audio_encoder", True):
            self.audio_tower.train(False)
        if getattr(self.config, "freeze_language_model", True):
            self.language_model.train(False)
        return self

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        """Enable/disable gradient checkpointing on the trainable submodules.

        Routes the request to whichever components are actually trainable in
        this run. The LM is always reached (its forward activations are
        needed for backprop to the projector even when its weights are
        frozen). The encoder is reached only when `freeze_audio_encoder` is
        False — when frozen, no gradient flows through it and checkpointing
        would just add recompute cost for no memory savings.
        """
        # The LLM still stores activations during forward for backprop to projector
        # Gradient checkpointing trades compute for memory by recomputing activations
        for submodule in self._gradient_checkpointing_targets():
            if hasattr(submodule, "_set_gradient_checkpointing"):
                submodule._set_gradient_checkpointing(enable, gradient_checkpointing_func)
            elif hasattr(submodule, "gradient_checkpointing_enable") and enable:
                submodule.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            elif hasattr(submodule, "gradient_checkpointing_disable") and not enable:
                submodule.gradient_checkpointing_disable()

    def _gradient_checkpointing_targets(self) -> list[nn.Module]:
        """Return the submodules that should respond to gradient_checkpointing
        toggles. Always includes the LM (activations are on the gradient path
        to the projector); includes the encoder only when it's trainable.
        """
        targets: list[nn.Module] = [self.language_model]
        if not getattr(self.config, "freeze_audio_encoder", True):
            targets.append(self.audio_tower)
        return targets

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_output_embeddings(value)

    def get_processor(self):
        """Get the processor for this model."""
        try:
            from .asr_processing import ASRProcessor
        except ImportError:
            from asr_processing import ASRProcessor  # type: ignore[no-redef]

        return ASRProcessor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            projector=self.projector,
            encoder_conv_layers=self.config.encoder_conv_layers,
        )

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Save trainable weights: projector, plus the language model when fine-tuned.

        With LoRA attached, the language_model entries are flattened to plain
        (non-PEFT) HF naming so model.safetensors round-trips through
        ASRModel.from_pretrained — which builds a vanilla base LM, overlays
        these weights, and only then re-attaches PEFT. lora_*/adapter weights
        are skipped here; PEFT serializes them separately as
        adapter_model.safetensors via the save_pretrained path below.
        """
        sd = {f"projector.{k}": v for k, v in self.projector.state_dict().items()}
        if not getattr(self.config, "freeze_language_model", True):
            lm = self.language_model
            if hasattr(lm, "peft_config"):
                for k, v in lm.state_dict().items():
                    if "lora_" in k:
                        continue
                    if k.startswith("base_model.model."):
                        k = k[len("base_model.model.") :]
                    # LoRA layers wrap the original Linear as `<name>.base_layer.<weight|bias>`.
                    k = k.replace(".base_layer.", ".")
                    sd[f"language_model.{k}"] = v
            else:
                sd.update({f"language_model.{k}": v for k, v in lm.state_dict().items()})
        return sd

    def _compute_encoder_output_lengths(
        self,
        audio_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample encoder output lengths using conv layer formulas."""
        return compute_encoder_output_length(
            audio_attention_mask.sum(dim=-1),
            self.config.encoder_conv_layers,
        )

    def _encode_audio(
        self,
        audio_features: torch.Tensor,
        expected_token_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Encode audio features and return flattened embeddings matching expected_token_counts.

        Args:
            audio_features: Mel spectrogram features (batch, n_mels, mel_len)
            expected_token_counts: Per-sample audio token counts as int64 tensor (batch,).

        Returns:
            Flattened audio embeddings of shape (sum(expected_token_counts), hidden_dim).
        """
        # SpecAugment is applied on the mel input, training-only. Most useful
        # when the encoder is trainable; on the frozen-encoder path it still
        # perturbs the projector's input slightly but with no gradient flowing
        # back to the encoder to leverage the diversity.
        if (
            self.training
            and getattr(self.config, "apply_spec_augment", False)
            and audio_features.numel() > 0
        ):
            audio_features = self._mask_input_features(audio_features)

        # When the encoder is frozen, skip gradient tracking through it — cuts
        # activation memory and matches the prior published recipe's behavior.
        # When trainable, we MUST allow gradients to flow back to encoder
        # params; wrapping in no_grad here would silently zero encoder
        # gradients regardless of requires_grad on its parameters.
        encoder_frozen = getattr(self.config, "freeze_audio_encoder", True)
        if encoder_frozen:
            with torch.no_grad():
                encoder_out = self.audio_tower(input_features=audio_features)
                hidden_states = encoder_out.last_hidden_state
        else:
            encoder_out = self.audio_tower(input_features=audio_features)
            hidden_states = encoder_out.last_hidden_state

        audio_embeds = self.projector(hidden_states)

        token_counts = expected_token_counts.to(device=audio_embeds.device, dtype=torch.long)
        return _gather_audio_embeds(audio_embeds, token_counts)

    def _mask_input_features(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # noqa: ARG002 — reserved for future use
    ) -> torch.Tensor:
        """SpecAugment on mel input (pure-torch, vectorized, compile-ready).

        Follows the same semantics as
        `transformers.models.whisper.modeling_whisper.WhisperModel._mask_input_features`
        (wav2vec2-style mask sampling: sample N start positions per sample,
        mask `mask_length` frames forward from each), but reimplemented in
        pure torch so it stays inside the autograd graph without crossing
        the numpy boundary. This avoids inductor codegen failures
        (e.g. the `‘zuf0’ was not declared` error from the prior numpy ->
        torch.tensor round-trip) AND avoids the per-forward host-to-GPU
        sync that the numpy path required.

        One minor semantic divergence vs the upstream helper: this version
        allows mask spans to overlap, while upstream rejects overlapping
        samples. For ASR purposes this is irrelevant — occasional region
        double-coverage has no measurable effect on the regularization
        signal.

        Reads ASRConfig fields by Whisper naming convention: mask_time_prob,
        mask_time_length, mask_time_min_masks, mask_feature_prob,
        mask_feature_length, mask_feature_min_masks.

        Args:
            input_features: (batch, n_mels, mel_len) log-mel features.
            attention_mask: reserved for future use; ignored here since our
                mel features are pre-padded to zero and double-masking
                pad regions is a no-op.

        Returns:
            Same-shape tensor with time-axis and/or feature-axis masks zeroed.
        """
        input_features = input_features.clone()
        batch_size, hidden_size, sequence_length = input_features.size()
        config = self.config
        device = input_features.device

        if getattr(config, "mask_time_prob", 0.0) > 0:
            mask_time = self._sample_mask_indices(
                batch_size,
                sequence_length,
                mask_prob=config.mask_time_prob,
                mask_length=config.mask_time_length,
                min_masks=config.mask_time_min_masks,
                device=device,
            )
            # Broadcast (B, T) -> (B, 1, T) to mask all mel bins at masked times.
            input_features.masked_fill_(mask_time.unsqueeze(1), 0)

        if getattr(config, "mask_feature_prob", 0.0) > 0:
            mask_feature = self._sample_mask_indices(
                batch_size,
                hidden_size,
                mask_prob=config.mask_feature_prob,
                mask_length=config.mask_feature_length,
                min_masks=config.mask_feature_min_masks,
                device=device,
            )
            # Broadcast (B, F) -> (B, F, 1) to mask all time steps at masked bins.
            input_features.masked_fill_(mask_feature.unsqueeze(-1), 0)

        return input_features

    @staticmethod
    def _sample_mask_indices(
        batch_size: int,
        axis_length: int,
        mask_prob: float,
        mask_length: int,
        min_masks: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Vectorized SpecAugment mask sampler — torch.compile-friendly.

        Returns a (batch_size, axis_length) bool tensor where True marks
        a position covered by at least one mask span. Spans may overlap
        (see _mask_input_features docstring on the semantic difference vs
        the upstream Whisper helper).
        """
        # Number of mask spans per sample: deterministic given config + axis_length.
        # Matches the upstream formula (ignoring the epsilon noise term, which
        # only shifts the count by ±1 stochastically — negligible at the
        # default mask_time_prob=0.05 / mask_length=10 setting which gives
        # ~5 spans for a typical 1500-frame mel input).
        num_masked_spans = max(int(mask_prob * axis_length / mask_length + 0.5), min_masks)
        if num_masked_spans == 0:
            return torch.zeros(batch_size, axis_length, device=device, dtype=torch.bool)

        # Sample start positions independently per sample × span.
        # Clamp range so a span of length mask_length never runs off the end.
        max_start = max(axis_length - mask_length + 1, 1)
        starts = torch.randint(
            0, max_start, (batch_size, num_masked_spans), device=device
        )  # (B, N)

        # For each (sample, span, position), True iff position ∈ [start, start+mask_length).
        positions = torch.arange(axis_length, device=device).view(1, 1, -1)  # (1, 1, T)
        starts_b = starts.unsqueeze(-1)  # (B, N, 1)
        span_mask = (positions >= starts_b) & (positions < starts_b + mask_length)
        # Reduce over the span dim: True if ANY span covers this position.
        return span_mask.any(dim=1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        audio_token_counts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass for training and inference."""
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            is_audio_token = input_ids == self.audio_token_id
            if audio_token_counts is None:
                audio_token_counts = is_audio_token.sum(dim=-1)
            else:
                audio_token_counts = audio_token_counts.to(
                    device=input_ids.device, dtype=torch.long
                )

            audio_embeds = self._encode_audio(input_features, audio_token_counts)

            audio_token_mask = is_audio_token.unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device),
                audio_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

        # Forward label_smoothing to the LM's loss_function via **kwargs.
        # transformers.loss.loss_utils.ForCausalLMLoss → fixed_cross_entropy
        # forwards extra kwargs to F.cross_entropy, which accepts label_smoothing.
        # When apply_liger_kernel_to_qwen3() has patched the LM, the smoothing
        # is consumed by liger's fused linear CE (no (B,T,V) materialization).
        # Zeroed on eval so eval/loss is raw CE and comparable to LS=0 runs.
        if labels is not None and self.training and self.config.label_smoothing > 0:
            kwargs.setdefault("label_smoothing", self.config.label_smoothing)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        if outputs.loss is not None and hasattr(self.projector, "get_aux_loss"):
            aux_loss = self.projector.get_aux_loss()
            if aux_loss is not None and aux_loss.numel() > 0:
                outputs.loss = outputs.loss + aux_loss.to(outputs.loss.device)

        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation, handling audio features for cached decoding."""
        input_features = kwargs.pop("input_features", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = self.language_model.prepare_inputs_for_generation(*args, **kwargs)

        # Only pass audio features on the first generation step (cache_position[0] == 0)
        if cache_position is not None and cache_position[0] == 0 and input_features is not None:
            model_inputs["input_features"] = input_features

        return model_inputs

    def _get_num_audio_tokens(
        self,
        audio_attention_mask: torch.Tensor,
    ) -> int:
        """Calculate number of audio tokens based on actual audio length.

        Uses attention mask to get real audio length, then computes:
        mel_frames -> encoder_frames (via conv formulas) -> projector output tokens
        """
        encoder_lengths = self._compute_encoder_output_lengths(audio_attention_mask)
        # Use max length for batch (all samples should have same token count for generation)
        encoder_output_len = int(encoder_lengths.max().item())
        return int(self.projector.get_output_length(encoder_output_len))

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
        **generate_kwargs,
    ):
        """Generate transcription from audio input.

        Can be called in two ways:
        1. With input_ids containing <audio> tokens (from processor)
        2. With just audio, and we build the prompt internally
        """
        if input_features is None:
            raise ValueError("input_features required for generation")
        if audio_attention_mask is None:
            raise ValueError("audio_attention_mask required for generation")

        device = input_features.device
        batch_size = input_features.shape[0]

        # Encode audio -> flattened embeddings (no per-sample host sync)
        encoder_lengths = self._compute_encoder_output_lengths(audio_attention_mask)
        token_counts = self.projector.get_output_length(encoder_lengths).to(torch.long)
        audio_embeds = self._encode_audio(input_features, token_counts)

        # If input_ids not provided, build prompt with correct number of audio tokens
        if input_ids is None:
            num_audio_tokens = self._get_num_audio_tokens(audio_attention_mask)
            audio_placeholder = "<audio>" * num_audio_tokens

            system_prompt = system_prompt or self.system_prompt

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # Audio tokens only (instruction-free)
            user_content = audio_placeholder
            if self.TRANSCRIBE_PROMPT:
                user_content += " " + self.TRANSCRIBE_PROMPT
            messages.append({"role": "user", "content": user_content})

            chat_result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,  # Disable Qwen3 thinking mode for ASR
            )
            input_ids = chat_result.input_ids.to(device)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.shape[0] == 1 and batch_size > 1:
                input_ids = input_ids.expand(batch_size, -1)

            attention_mask = torch.ones_like(input_ids)

        # Get text embeddings and replace audio tokens with audio embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        audio_token_mask = (input_ids == self.audio_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_token_mask.to(inputs_embeds.device),
            audio_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

        # transformers v5 deprecates passing generation flags as kwargs when a
        # `generation_config` is also passed — the kwargs get silently dropped.
        # Pull any score-related flags out of generate_kwargs and apply them to
        # a derived generation_config so they actually take effect.
        gen_cfg = self.generation_config
        score_flags = {}
        for flag in ("output_scores", "output_logits", "return_dict_in_generate"):
            if flag in generate_kwargs:
                score_flags[flag] = generate_kwargs.pop(flag)
        if score_flags:
            from copy import copy as _copy

            gen_cfg = _copy(self.generation_config)
            for flag, value in score_flags.items():
                setattr(gen_cfg, flag, value)
            # output_scores requires return_dict_in_generate for HF generate to
            # actually populate .scores on the output object.
            if gen_cfg.output_scores and not gen_cfg.return_dict_in_generate:
                gen_cfg.return_dict_in_generate = True

        # Generate using language model
        # Pass both input_ids and inputs_embeds so repetition_penalty works correctly
        # (it needs input_ids to track which tokens have been used)
        output = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            **generate_kwargs,
        )

        # When using inputs_embeds with input_ids, generate returns the full
        # sequence (prompt + generated). Strip the prompt to return only the
        # newly generated tokens. When scores were requested, preserve the
        # GenerateOutput so callers can read .scores; otherwise return the
        # bare tensor for backward compatibility with existing callers.
        input_len = input_ids.shape[1]
        if isinstance(output, torch.Tensor):
            return output[:, input_len:]
        output.sequences = output.sequences[:, input_len:]
        return output

    def generate_streaming(
        self,
        input_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        system_prompt: Optional[str] = None,
        **generate_kwargs,
    ) -> Iterator[str]:
        """Generate transcription with streaming token output.

        Yields partial transcript strings as tokens are generated.
        Reduces time-to-first-word by streaming tokens as they're decoded.

        Args:
            input_features: Mel spectrogram features (batch, n_mels, mel_len)
            audio_attention_mask: Mask for real vs padded mel frames (batch, mel_len)
            system_prompt: Optional system prompt override
            **generate_kwargs: Additional generation arguments

        Yields:
            Partial transcript text as each token is generated
        """
        device = input_features.device
        batch_size = input_features.shape[0]

        # Encode audio -> flattened embeddings (no per-sample host sync)
        encoder_lengths = self._compute_encoder_output_lengths(audio_attention_mask)
        token_counts = self.projector.get_output_length(encoder_lengths).to(torch.long)
        audio_embeds = self._encode_audio(input_features, token_counts)

        # Build prompt with correct number of audio tokens
        num_audio_tokens = self._get_num_audio_tokens(audio_attention_mask)
        audio_placeholder = "<audio>" * num_audio_tokens

        system_prompt = system_prompt or self.system_prompt

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Audio tokens only (instruction-free)
        user_content = audio_placeholder
        if self.TRANSCRIBE_PROMPT:
            user_content += " " + self.TRANSCRIBE_PROMPT
        messages.append({"role": "user", "content": user_content})

        chat_result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,  # Disable Qwen3 thinking mode for ASR
        )
        input_ids = chat_result.input_ids.to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] == 1 and batch_size > 1:
            input_ids = input_ids.expand(batch_size, -1)

        attention_mask = torch.ones_like(input_ids)

        # Get text embeddings and replace audio tokens with audio embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        audio_token_mask = (input_ids == self.audio_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_token_mask.to(inputs_embeds.device),
            audio_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

        # Setup streamer for token-by-token output
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Prepare generation kwargs
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "generation_config": self.generation_config,
            "streamer": streamer,
            **generate_kwargs,
        }

        # Run generation in background thread
        thread = Thread(target=self.language_model.generate, kwargs=gen_kwargs)
        thread.start()

        # Yield tokens as they're generated, filtering out <think>...</think> blocks
        # Start assuming no think block - only filter when we see <think>
        in_think_block = False
        buffer = ""

        for text in streamer:
            buffer += text

            # Check for think block start (in case model outputs think blocks)
            while "<think>" in buffer:
                in_think_block = True
                # Yield any text before <think>
                before_think = buffer.split("<think>")[0]
                if before_think:
                    yield before_think
                buffer = buffer.split("<think>", 1)[-1]

            # Check for think block end
            while in_think_block and "</think>" in buffer:
                in_think_block = False
                buffer = buffer.split("</think>", 1)[-1]

            # Yield text if not in think block
            if not in_think_block and buffer:
                yield buffer
                buffer = ""

        # Yield any remaining buffer
        if buffer and not in_think_block:
            yield buffer

        thread.join()

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs) -> None:
        """Save model, tokenizer, and processor."""
        import shutil

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Update config with actual vocab size
        self.config.vocab_size = self.language_model.config.vocab_size
        self.config.text_config.vocab_size = self.language_model.config.vocab_size

        if hasattr(self.audio_tower.config, "num_mel_bins"):
            self.config.audio_config.num_mel_bins = self.audio_tower.config.num_mel_bins

        # Save model (temporarily remove non-serializable attributes)
        tokenizer = self.tokenizer
        del self.tokenizer

        try:
            super().save_pretrained(save_dir, **kwargs)
        finally:
            self.tokenizer = tokenizer

        # Save tokenizer and feature extractor
        self.tokenizer.save_pretrained(save_dir)
        self.feature_extractor.save_pretrained(save_dir)

        # Save LoRA adapters if present (creates adapter_model.safetensors and adapter_config.json)
        # Don't save embedding layers - the <audio> token embedding is never used
        # (it's replaced with projected audio embeddings before the LLM sees it)
        if hasattr(self.language_model, "peft_config"):
            self.language_model.save_pretrained(save_dir, save_embedding_layers=False)

            # Clear base_model_name_or_path in adapter_config.json to prevent HF pipeline
            # from redirecting to the base LLM repo (like Qwen) which breaks feature
            # extractor loading for multimodal models. If a repo_id is provided, use that
            # so the model can be loaded directly from the Hub.
            adapter_config_path = save_dir / "adapter_config.json"
            if adapter_config_path.exists():
                with adapter_config_path.open() as f:
                    adapter_config = json.load(f)

                # Use repo_id if available, otherwise clear to prevent redirect.
                # Use empty string instead of None to avoid str(None) -> "None" bug
                # in some transformers/PEFT versions.
                repo_id = (
                    kwargs.get("repo_id")
                    or kwargs.get("push_to_hub_model_id")
                    or getattr(self.config, "pretrained_model_path", None)
                    or ""  # Use empty string instead of None
                )
                adapter_config["base_model_name_or_path"] = repo_id

                with adapter_config_path.open("w") as f:
                    json.dump(adapter_config, f, indent=2)

        # Add processor auto_map to preprocessor_config.json
        config_path = save_dir / "preprocessor_config.json"
        if config_path.exists():
            with config_path.open() as f:
                processor_config = json.load(f)
        else:
            processor_config = {}

        processor_config.update(
            {
                "processor_class": "ASRProcessor",
                "auto_map": {"AutoProcessor": "asr_processing.ASRProcessor"},
            }
        )

        with config_path.open("w") as f:
            json.dump(processor_config, f, indent=2)

        # Copy source files for auto-loading
        src_dir = Path(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)
        # Copy projectors module
        shutil.copy(src_dir / "projectors.py", save_dir / "projectors.py")
        # Copy alignment module
        shutil.copy(src_dir / "alignment.py", save_dir / "alignment.py")
        # Copy diarization module
        shutil.copy(src_dir / "diarization.py", save_dir / "diarization.py")

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        """Push model to HuggingFace Hub, ensuring adapter_config points to repo.

        IMPORTANT: Sets base_model_name_or_path in adapter_config.json to repo_id
        so that transformers pipeline() can load the model correctly. Without this,
        the pipeline tries to load from "None" which fails.
        """
        # Store repo_id in config so save_pretrained can access it
        self.config.pretrained_model_path = repo_id
        # Call parent's push_to_hub
        return super().push_to_hub(repo_id, **kwargs)


# Register with transformers Auto classes
# (AutoConfig.register is handled in asr_config.py at module load.)
AutoModel.register(ASRConfig, ASRModel)
