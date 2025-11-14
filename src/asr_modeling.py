from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    Wav2Vec2FeatureExtractor,
    WhisperFeatureExtractor,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)

try:
    from .asr_config import ASRConfig
except ImportError:
    from asr_config import ASRConfig  # type: ignore[no-redef]


class AudioProjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        in_dim = config.encoder_dim
        out_dim = config.llm_dim

        # Whisper already does 2x (conv stride), this adds another 2x = 4x total
        pool_stride = getattr(config, "projector_pool_stride", 2)
        self.avg_pooler = nn.AvgPool1d(pool_stride, stride=pool_stride) if pool_stride > 1 else None

        self.proj = nn.Linear(in_dim, out_dim, bias=False)

        dropout_rate = getattr(config, "projector_dropout", 0.05)
        self.dropout = nn.Dropout(dropout_rate)

        # Xavier init for stable gradient flow
        with torch.no_grad():
            nn.init.xavier_normal_(self.proj.weight, gain=1.0)

    def forward(self, x):
        # x: [batch, seq_len, dim] - already normalized by Whisper's LayerNorm
        batch_size, seq_len, dim = x.size()

        # Optional: Apply average pooling for temporal downsampling
        if self.avg_pooler is not None:
            x = x.permute(0, 2, 1)  # [batch, dim, seq_len]
            x = self.avg_pooler(x)   # [batch, dim, seq_len//pool_stride]
            x = x.permute(0, 2, 1)  # [batch, seq_len//pool_stride, dim]

        # Project to LLM dimension
        x = self.proj(x)

        # Regularization (LLM's first layer will normalize before attention)
        return self.dropout(x)


class ASRModel(PreTrainedModel):
    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_save = ["encoder", "decoder.base_model"]
    _is_loading_from_pretrained: bool = False
    _pretrained_model_path: Optional[str] = None

    # Task to prompt mapping for generation
    TASK_PROMPTS = {
        "transcribe": "Transcribe: <audio>",
        "continue": "Continue: <audio>",
        "describe": "Describe: <audio>",
        "emotion": "Emotion: <audio>",
    }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        import json

        from safetensors.torch import load_file
        from transformers.utils.hub import cached_file

        config = kwargs.pop("config", None)
        if config is None:
            config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # IMPORTANT: Load feature extractor from audio model for correct mel bin configuration
        is_whisper = "whisper" in config.audio_model_id.lower()
        if is_whisper:
            from transformers import WhisperConfig

            encoder_config = WhisperConfig.from_pretrained(config.audio_model_id)
            num_mel_bins = encoder_config.num_mel_bins
            kwargs["feature_extractor"] = WhisperFeatureExtractor.from_pretrained(
                config.audio_model_id,
                feature_size=num_mel_bins,
            )
        else:
            kwargs["feature_extractor"] = Wav2Vec2FeatureExtractor.from_pretrained(
                config.audio_model_id
            )

        cls._is_loading_from_pretrained = True
        cls._pretrained_model_path = pretrained_model_name_or_path

        subfolder = kwargs.get("subfolder")
        revision = kwargs.get("revision")
        cache_kwargs = {}
        if subfolder:
            cache_kwargs["subfolder"] = subfolder
        if revision:
            cache_kwargs["revision"] = revision

        try:

            def load_cached_file(filename, load_fn=None):
                """Load a file from the model directory.

                Args:
                    filename: Name of file to load
                    load_fn: Function to apply to loaded path (default: load_file for safetensors)
                """

                if load_fn is None:
                    load_fn = load_file

                try:
                    file_path = cached_file(
                        pretrained_model_name_or_path,
                        filename,
                        _raise_exceptions_for_missing_entries=False,
                        **cache_kwargs,
                    )
                    if file_path:
                        return load_fn(file_path)
                except Exception:
                    pass

                return None

            def load_json(path):
                from pathlib import Path as PathlibPath

                with PathlibPath(path).open() as f:
                    return json.load(f)

            encoder_lora_config = load_cached_file("encoder_lora_config.json", load_json)
            decoder_lora_config = load_cached_file("decoder_lora_config.json", load_json)

            model = cls(
                config, peft_config=decoder_lora_config, encoder_lora_config=encoder_lora_config
            )

            encoder_state = load_cached_file("encoder.safetensors")
            decoder_state = load_cached_file("decoder.safetensors")
            projector_state = load_cached_file("projector.safetensors")

            if not projector_state:
                raise FileNotFoundError(
                    f"projector.safetensors not found in {pretrained_model_name_or_path}. "
                    "The repository may not have been trained yet."
                )
            sum(v.numel() for v in projector_state.values())
            model.projector.load_state_dict(projector_state, strict=True, assign=True)

            if encoder_lora_config:
                if not encoder_state:
                    raise FileNotFoundError(
                        f"encoder.safetensors not found in {pretrained_model_name_or_path}. "
                        "The repository may not have been trained yet."
                    )
                sum(v.numel() for v in encoder_state.values())
                model.encoder.load_state_dict(encoder_state, strict=False, assign=True)

            if decoder_lora_config and decoder_lora_config.get("r", 0) > 0:
                if not decoder_state:
                    raise FileNotFoundError(
                        f"decoder.safetensors not found in {pretrained_model_name_or_path}. "
                        "The repository may not have been trained yet."
                    )
                sum(v.numel() for v in decoder_state.values())
                model.decoder.load_state_dict(decoder_state, strict=False, assign=True)

            device = kwargs.get("device")
            if device is not None:
                model = model.to(device)

            return model
        finally:
            cls._is_loading_from_pretrained = False
            del cls._pretrained_model_path

    def __init__(self, config: ASRConfig, **kwargs):
        super().__init__(config)

        peft_config = kwargs.pop("peft_config", None)
        encoder_lora_config = kwargs.pop("encoder_lora_config", None)
        feature_extractor = kwargs.pop("feature_extractor", None)

        self.system_prompt = config.system_prompt
        self.peft_config = peft_config
        self.encoder_lora_config = encoder_lora_config

        self.encoder = self._create_encoder(config, encoder_lora_config)

        is_whisper = "whisper" in config.audio_model_id.lower() or (
            hasattr(self.encoder.config, "model_type")
            and "whisper" in self.encoder.config.model_type.lower()
        )

        if is_whisper:
            self.main_input_name = "input_features"
        else:
            self.main_input_name = "input_values"

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            if is_whisper:
                num_mel_bins = self.encoder.config.num_mel_bins
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    config.audio_model_id,
                    feature_size=num_mel_bins,  # Override feature_size to match model's mel bins
                )
            else:
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    config.audio_model_id
                )

        self.decoder = self._create_decoder(config, peft_config)
        self.generation_config = self.decoder.generation_config

        self._init_tokenizer()

        from types import SimpleNamespace

        # Auto-detect encoder_dim and llm_dim if not specified
        encoder_dim = config.encoder_dim
        if encoder_dim is None:
            if hasattr(self.encoder.config, "hidden_size"):
                encoder_dim = self.encoder.config.hidden_size
            elif hasattr(self.encoder.config, "d_model"):
                encoder_dim = self.encoder.config.d_model
            else:
                raise ValueError("Could not auto-detect encoder_dim. Please specify in config.")

        llm_dim = config.llm_dim
        if llm_dim is None:
            if hasattr(self.decoder.config, "hidden_size"):
                llm_dim = self.decoder.config.hidden_size
            elif hasattr(self.decoder.config, "d_model"):
                llm_dim = self.decoder.config.d_model
            else:
                raise ValueError("Could not auto-detect llm_dim. Please specify in config.")

        projector_config = SimpleNamespace(
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            projector_pool_stride=getattr(config, "projector_pool_stride", 2),
            projector_dropout=getattr(config, "projector_dropout", 0.05),
        )
        self.projector = AudioProjector(projector_config)

        self._no_split_modules = self.decoder._no_split_modules

    @staticmethod
    def _apply_lora(
        model, lora_config: dict, task_type, model_name: str = "model", default_dropout: float = 0.0
    ):
        """Apply LoRA adapters to a model (encoder or decoder).

        Args:
            model: The model to apply LoRA to
            lora_config: Dict with LoRA configuration (r, lora_alpha, target_modules, etc.)
            task_type: peft.TaskType (FEATURE_EXTRACTION for encoder, CAUSAL_LM for decoder)
            model_name: Name for logging purposes
            default_dropout: Default dropout value from config
        """
        if lora_config.get("r", 0) == 0:
            for param in model.parameters():
                param.requires_grad = False
            return model

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "PEFT library is required for LoRA fine-tuning. Install with: pip install peft"
            ) from None

        target_modules = lora_config.get("target_modules", ["q_proj", "k_proj"])
        if target_modules == "all-linear":
            target_modules = "all-linear"

        peft_config = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("lora_alpha", 8),
            target_modules=target_modules,
            lora_dropout=lora_config.get("lora_dropout", default_dropout),
            bias=lora_config.get("bias", "none"),
            task_type=task_type,
            modules_to_save=lora_config.get("modules_to_save"),
            init_lora_weights=True,
        )

        return get_peft_model(model, peft_config)

    @classmethod
    def _create_encoder(cls, config: ASRConfig, encoder_lora_config: Optional[dict] = None):
        """Create and configure the audio encoder.

        Args:
            config: Model configuration
            encoder_lora_config: Optional LoRA configuration for encoder

        Returns:
            Configured encoder model (potentially with LoRA)
        """
        target_dtype = getattr(torch, config.model_dtype)

        encoder_kwargs = {
            "attn_implementation": config.attn_implementation,
            "dtype": target_dtype,
            "low_cpu_mem_usage": True,
        }
        if not cls._is_loading_from_pretrained:
            encoder_kwargs["device_map"] = "auto"

        if "whisper" in config.audio_model_id.lower():
            from transformers import WhisperModel

            full_model = WhisperModel.from_pretrained(config.audio_model_id, **encoder_kwargs)
            encoder = full_model.encoder
            del full_model
        else:
            encoder = AutoModel.from_pretrained(config.audio_model_id, **encoder_kwargs)

        is_whisper = "whisper" in config.audio_model_id.lower() or (
            hasattr(encoder.config, "model_type") and "whisper" in encoder.config.model_type.lower()
        )
        is_wav2vec2 = (
            hasattr(encoder.config, "model_type")
            and "wav2vec2" in encoder.config.model_type.lower()
        )

        if is_whisper or is_wav2vec2:
            encoder.config.apply_spec_augment = True
            encoder.config.mask_time_prob = getattr(config, "mask_time_prob", 0.05)
            encoder.config.mask_time_length = getattr(config, "mask_time_length", 10)
            encoder.config.mask_feature_prob = getattr(config, "mask_feature_prob", 0.0)
            encoder.config.mask_feature_length = getattr(config, "mask_feature_length", 10)

        encoder.requires_grad_(False)

        # Wrap encoder forward to handle Whisper's input_features vs input_values
        original_forward = encoder.forward
        is_whisper = "whisper" in config.audio_model_id.lower() or (
            hasattr(encoder.config, "model_type") and "whisper" in encoder.config.model_type.lower()
        )
        input_key = "input_features" if is_whisper else "input_values"

        def safe_encoder_forward(self_encoder, input_values=None, **kwargs):
            # Catch and discard invalid kwargs like input_ids
            kwargs.pop("input_ids", None)
            return original_forward(**{input_key: input_values}, **kwargs)

        import types

        encoder.forward = types.MethodType(safe_encoder_forward, encoder)

        if encoder_lora_config and encoder_lora_config.get("r", 0) > 0:
            from peft import TaskType

            encoder = cls._apply_lora(
                encoder,
                encoder_lora_config,
                TaskType.FEATURE_EXTRACTION,
                "encoder",
                default_dropout=config.lora_default_dropout,
            )

            # Re-apply the safe_encoder_forward wrapper after PEFT wrapping
            # PEFT wrapping loses our custom forward, so we need to re-apply it
            encoder.forward = types.MethodType(safe_encoder_forward, encoder)

        return encoder

    @classmethod
    def _create_decoder(cls, config: ASRConfig, peft_config: Optional[dict] = None):
        """Create and configure the language model decoder.

        Args:
            config: Model configuration
            peft_config: Optional LoRA configuration for decoder

        Returns:
            Configured decoder model (potentially with LoRA)
        """
        target_dtype = getattr(torch, config.model_dtype)

        # When loading from pretrained, avoid device_map="auto" to prevent meta tensor issues
        decoder_kwargs = {
            "attn_implementation": config.attn_implementation,
            "dtype": target_dtype,
            "trust_remote_code": True,
        }
        # Don't use device_map="auto" as it can cause meta tensor issues with Trainer
        # The Trainer will handle device placement

        decoder = AutoModelForCausalLM.from_pretrained(config.text_model_id, **decoder_kwargs)

        # use_cache is now safe because we pre-expand audio tokens for consistent sequence length
        # Cache can be enabled/disabled via config.use_cache
        decoder.config.use_cache = config.use_cache

        decoder.requires_grad_(False)

        # Apply LoRA to decoder if configured
        if peft_config and peft_config.get("peft_method") == "lora":
            from peft import TaskType

            decoder = cls._apply_lora(
                decoder,
                peft_config,
                TaskType.CAUSAL_LM,
                "decoder",
                default_dropout=config.lora_default_dropout,
            )

        return decoder

    def _init_weights(self, module):
        """Initialize weights for trainable modules.

        Note: This is a no-op since:
        - AudioProjector self-initializes in its __init__
        - Encoder/decoder are loaded from pretrained weights
        """
        pass

    def can_generate(self) -> bool:
        """Return True to indicate this model supports generation.

        Required for Transformers 4.50+ where PreTrainedModel no longer
        inherits from GenerationMixin.
        """
        return True

    @property
    def _tied_weights_keys(self):
        """Return list of weight keys that should be tied.

        In this model, input and output embeddings of the decoder may be tied.
        """
        if hasattr(self.decoder, "_tied_weights_keys"):
            return [f"decoder.{k}" for k in self.decoder._tied_weights_keys]
        return []

    def _init_tokenizer(self):
        model_path = (
            self.__class__._pretrained_model_path
            if self._is_loading_from_pretrained
            else self.config.text_model_id
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Set pad_token if not already set to avoid warnings during generation
        # If pad_token is same as eos_token, we need a different token for padding
        if (
            self.tokenizer.pad_token is None
            or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ) and "<|finetune_right_pad_id|>" in self.tokenizer.get_vocab():
            # For SmolLM3, use the dedicated finetune_right_pad_id token
            self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        existing_special = self.tokenizer.additional_special_tokens or []

        # Add single audio token if not present
        if "<audio>" not in existing_special:
            special_tokens = {"additional_special_tokens": existing_special + ["<audio>"]}
            num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
            if num_added_tokens > 0:
                # Use mean_resizing=False since this is a structural token, not semantic
                self.decoder.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        current_embed_size = self.decoder.get_input_embeddings().weight.shape[0]
        expected_size = len(self.tokenizer)
        if current_embed_size != expected_size:
            self.decoder.resize_token_embeddings(expected_size, mean_resizing=False)

        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<audio>")

        self.tokenizer.padding_side = "right"

        for cfg in [self.config.text_config, self.decoder.config, self.generation_config]:
            if isinstance(cfg, dict):
                cfg["pad_token_id"] = self.tokenizer.pad_token_id
                cfg["eos_token_id"] = self.tokenizer.eos_token_id
                cfg["bos_token_id"] = self.tokenizer.bos_token_id
            else:
                cfg.pad_token_id = self.tokenizer.pad_token_id
                cfg.eos_token_id = self.tokenizer.eos_token_id
                cfg.bos_token_id = self.tokenizer.bos_token_id

    def get_processor(self):
        try:
            from .asr_processing import ASRProcessor
        except ImportError:
            from asr_processing import ASRProcessor  # type: ignore[no-redef]

        return ASRProcessor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

    def state_dict(self, *args, **kwargs):
        """Save trainable parameters in separate component files.

        Instead of one large file, we save:
        - encoder.safetensors: encoder LoRA adapters
        - decoder.safetensors: decoder LoRA adapters
        - projector.safetensors: projector weights

        This eliminates key-matching complexity and makes loading more reliable.
        """
        # Note: HuggingFace Trainer calls this method and expects a single dict.
        # We'll return the combined state dict here, but save_pretrained() will
        # split into separate files.
        return self._get_trainable_state_dict()

    def _get_trainable_state_dict(self):
        """Get all trainable parameters as a single state dict.

        This is used by Trainer for checkpointing during training.
        """
        state = {}

        # Get encoder trainable params (LoRA adapters)
        encoder_state = self.encoder.state_dict()
        encoder_trainable = {
            name for name, param in self.encoder.named_parameters() if param.requires_grad
        }
        for name, tensor in encoder_state.items():
            if name in encoder_trainable:
                state[f"encoder.{name}"] = tensor

        # Get decoder trainable params (LoRA adapters)
        decoder_state = self.decoder.state_dict()
        decoder_trainable = {
            name for name, param in self.decoder.named_parameters() if param.requires_grad
        }
        for name, tensor in decoder_state.items():
            if name in decoder_trainable:
                state[f"decoder.{name}"] = tensor

        # Get projector params (always trainable)
        projector_state = self.projector.state_dict()
        for name, tensor in projector_state.items():
            state[f"projector.{name}"] = tensor

        return state

    def get_input_embeddings(self):
        """Delegate to decoder for proper HF Trainer integration."""
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Delegate to decoder for proper HF Trainer integration."""
        self.decoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Delegate to decoder for proper HF Trainer integration."""
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, value):
        """Delegate to decoder for proper HF Trainer integration."""
        self.decoder.set_output_embeddings(value)

    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Masks extracted features along time axis and/or along feature axis according to SpecAugment.
        Follows Whisper's implementation exactly.
        """
        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.encoder.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.encoder.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.encoder.config.mask_time_prob,
                mask_length=self.encoder.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=getattr(self.encoder.config, "mask_time_min_masks", 2),
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.encoder.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.encoder.config.mask_feature_prob,
                mask_length=self.encoder.config.mask_feature_length,
                min_masks=getattr(self.encoder.config, "mask_feature_min_masks", 0),
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features

    def _encode_audio(
        self,
        input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure input is on encoder's device and has the right dtype
        encoder_device = next(self.encoder.parameters()).device
        encoder_dtype = next(self.encoder.parameters()).dtype
        # Clone to prevent user tensor reuse contamination
        input_values = input_values.clone().to(device=encoder_device, dtype=encoder_dtype)

        # Apply SpecAugment if Whisper encoder (masking happens on input_features)
        is_whisper = "whisper" in self.config.audio_model_id.lower() or (
            hasattr(self.encoder.config, "model_type") and "whisper" in self.encoder.config.model_type.lower()
        )
        if is_whisper and self.training:
            input_values = self._mask_input_features(input_values, attention_mask=audio_attention_mask)

        # Only pass explicit valid arguments to encoder
        # Never use **kwargs to prevent torch.compile from injecting decoder args like input_ids
        # Don't use no_grad if encoder has LoRA (needs gradients for training)
        if self.encoder_lora_config and self.encoder_lora_config.get("r", 0) > 0:
            audio_features = self.encoder(
                input_values=input_values,
                attention_mask=audio_attention_mask,
            ).last_hidden_state
        else:
            with torch.no_grad():
                audio_features = self.encoder(
                    input_values=input_values,
                    attention_mask=audio_attention_mask,
                ).last_hidden_state

        # Project audio features and ensure dtype matches decoder
        audio_embeds = self.projector(audio_features)

        # Convert to decoder's dtype if needed (e.g., bfloat16)
        decoder_dtype = next(self.decoder.parameters()).dtype
        if audio_embeds.dtype != decoder_dtype:
            audio_embeds = audio_embeds.to(dtype=decoder_dtype)

        return audio_embeds

    def _get_audio_expansion_details(
        self, input_ids: torch.Tensor, num_audio_tokens: int
    ) -> dict:
        """Calculate the positions and masks needed to expand audio tokens.

        This helper consolidates the common cumsum logic used by both
        _expand_audio_tokens and _expand_for_audio_tokens.

        Args:
            input_ids: Token IDs with single <audio> token per sample
            num_audio_tokens: Number of tokens each audio token expands to

        Returns:
            Dictionary containing:
            - new_seq_len: The total sequence length after expansion
            - new_start_positions: [batch, old_seq_len] tensor mapping old indices to new
            - audio_mask: [batch, old_seq_len] boolean mask for audio token positions
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Find audio token positions
        audio_mask = input_ids == self.audio_token_id

        # Validate: each sample must have exactly one audio token
        audio_counts = audio_mask.sum(dim=1)
        if not (audio_counts == 1).all():
            missing = (audio_counts == 0).any()
            multiple = (audio_counts > 1).any()
            if missing:
                raise ValueError("Some samples are missing audio token")
            if multiple:
                raise ValueError("Some samples have multiple audio tokens")

        # Create placeholder tensor: 1 for normal tokens, num_audio_tokens for audio token
        token_counts = torch.where(audio_mask, num_audio_tokens, 1)

        # Cumsum - 1 gives us the ENDING position of each token's expansion
        cumsum_counts = torch.cumsum(token_counts, dim=1)

        # The starting position of token i is cumsum[i-1]
        new_start_positions = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                cumsum_counts[:, :-1],
            ],
            dim=1,
        )

        # Calculate new sequence length
        new_seq_len = seq_len - 1 + num_audio_tokens

        return {
            "new_seq_len": new_seq_len,
            "new_start_positions": new_start_positions,
            "audio_mask": audio_mask,
        }

    def _expand_audio_tokens(self, input_ids: torch.Tensor, num_audio_tokens: int) -> torch.Tensor:
        """Expand single <audio> token into N copies to match projected audio length.

        Pre-expands audio tokens in input_ids so we can use simple masked_scatter
        instead of complex concatenation. This enables KV caching and torch.compile.

        Args:
            input_ids: Token IDs with single <audio> token per sample
            num_audio_tokens: Number of tokens to expand each <audio> into

        Returns:
            Expanded input_ids where each <audio> is replaced with N copies
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        details = self._get_audio_expansion_details(input_ids, num_audio_tokens)
        new_seq_len = details["new_seq_len"]
        new_start_positions = details["new_start_positions"]
        audio_mask = details["audio_mask"]

        # Create output tensor (filled with pad token initially)
        expanded = torch.full(
            (batch_size, new_seq_len),
            self.tokenizer.pad_token_id,
            dtype=input_ids.dtype,
            device=device,
        )

        # Scatter non-audio tokens to their new positions
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        non_audio_mask = ~audio_mask

        # Place non-audio tokens (they only occupy 1 position each)
        expanded[batch_indices[non_audio_mask], new_start_positions[non_audio_mask]] = input_ids[
            non_audio_mask
        ]

        # Fill audio token positions using vectorized indexing
        # Find where audio token starts in the expanded sequence
        audio_positions = audio_mask.int().argmax(dim=1)  # [batch_size]
        audio_new_start = new_start_positions[
            torch.arange(batch_size, device=device), audio_positions
        ]

        # Create indices for all audio token positions
        audio_token_indices = torch.arange(num_audio_tokens, device=device).unsqueeze(0)
        audio_positions_expanded = audio_new_start.unsqueeze(1) + audio_token_indices

        batch_idx_expanded = (
            torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_audio_tokens)
        )
        expanded[batch_idx_expanded, audio_positions_expanded] = self.audio_token_id

        return expanded

    def _expand_for_audio_tokens(
        self,
        input_ids: torch.Tensor,
        tensor_to_expand: torch.Tensor,
        num_audio_tokens: int,
        fill_value: Union[int, float],
    ) -> torch.Tensor:
        """Expand attention mask or labels to match audio token expansion.

        Args:
            input_ids: Original input_ids (used only for mapping)
            tensor_to_expand: Tensor to expand (attention_mask or labels)
            num_audio_tokens: Number of tokens each audio token expands to
            fill_value: Value to fill for audio token positions (1 for attn, -100 for labels)

        Returns:
            Expanded tensor matching the expanded sequence length
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        details = self._get_audio_expansion_details(input_ids, num_audio_tokens)
        new_seq_len = details["new_seq_len"]
        new_start_positions = details["new_start_positions"]
        audio_mask = details["audio_mask"]

        # Create output tensor
        expanded = torch.full(
            (batch_size, new_seq_len), fill_value, dtype=tensor_to_expand.dtype, device=device
        )

        # Scatter non-audio positions to their new positions
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        non_audio_mask = ~audio_mask

        # Place non-audio values
        expanded[batch_indices[non_audio_mask], new_start_positions[non_audio_mask]] = (
            tensor_to_expand[non_audio_mask]
        )

        # Audio token positions are already filled with fill_value
        # No need to explicitly set them again

        return expanded

    def _prepare_audio_inputs_embeds(
        self, expanded_input_ids: torch.Tensor, audio_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Prepare inputs_embeds by replacing audio token embeddings with actual audio embeddings.

        Args:
            expanded_input_ids: Input IDs with expanded audio tokens
            audio_embeds: Audio embeddings to inject

        Returns:
            inputs_embeds with audio embeddings injected
        """
        # Get text embeddings for expanded input_ids
        inputs_embeds = self.decoder.get_input_embeddings()(expanded_input_ids)

        # Simple masked scatter: replace audio token embeddings with actual audio embeddings
        special_audio_mask = (expanded_input_ids == self.audio_token_id).unsqueeze(-1)
        special_audio_mask = special_audio_mask.expand_as(inputs_embeds)
        audio_embeds_flat = audio_embeds.reshape(-1, audio_embeds.shape[-1])
        return inputs_embeds.masked_scatter(special_audio_mask, audio_embeds_flat)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,  # For Whisper
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        audio_inputs = input_values if input_values is not None else input_features
        if audio_inputs is not None:
            # During inference, the pipeline may call forward with only audio inputs
            # In that case, we should raise an error directing to use generate() instead
            if input_ids is None:
                raise ValueError(
                    "forward() requires both audio inputs and input_ids (for training). "
                    "For inference, use the generate() method instead, or use the pipeline "
                    "which will automatically call generate()."
                )

            # Extract audio-specific kwargs, don't pass input_ids to encoder
            audio_attention_mask = kwargs.pop("audio_attention_mask", None)

            # Remove any decoder-specific kwargs that shouldn't go to the encoder
            kwargs.pop("past_key_values", None)
            use_cache = kwargs.pop("use_cache", None)

            # Encode audio to get embeddings
            audio_embeds = self._encode_audio(
                input_values=audio_inputs,  # Will be mapped to input_features for Whisper by safe_encoder_forward
                audio_attention_mask=audio_attention_mask,
            )

            # Validate audio token ID before using it
            if self.audio_token_id is None:
                raise ValueError(f"Audio token not properly initialized: {self.audio_token_id}")

            vocab_size = self.decoder.get_input_embeddings().weight.shape[0]
            if self.audio_token_id >= vocab_size:
                raise ValueError(
                    f"Audio token ID out of range. ID: {self.audio_token_id}, Vocab size: {vocab_size}"
                )

            # Check that audio token exists
            if not (input_ids == self.audio_token_id).any():
                raise ValueError("Audio token <audio> must be present in input")

            # Expand audio tokens to match audio embedding length
            num_audio_tokens = audio_embeds.shape[1]
            expanded_input_ids = self._expand_audio_tokens(input_ids, num_audio_tokens)

            # Prepare inputs_embeds with audio embeddings injected
            inputs_embeds = self._prepare_audio_inputs_embeds(expanded_input_ids, audio_embeds)

            # Expand attention mask to match new sequence length (vectorized)
            if attention_mask is not None:
                full_attention_mask = self._expand_for_audio_tokens(
                    input_ids, attention_mask, num_audio_tokens, fill_value=1
                )
            else:
                full_attention_mask = None

            # Expand labels to match new sequence length (vectorized, mark audio tokens as -100)
            if labels is not None:
                labels = self._expand_for_audio_tokens(
                    input_ids, labels, num_audio_tokens, fill_value=-100
                )
        else:
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
            full_attention_mask = attention_mask
            use_cache = kwargs.pop("use_cache", None)

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
            use_cache=use_cache if use_cache is not None else False,
            **kwargs,
        )

    @torch.no_grad()
    def _generate_text_only(
        self,
        text_input: str,
        system_prompt: Optional[str] = None,
        **generate_kwargs,
    ):
        """
        Generate text from text input only (no audio).
        Useful for testing the LLM directly without audio encoding.
        """
        device = self.decoder.device

        if system_prompt is None:
            system_prompt = self.system_prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text_input})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(device)

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        generate_kwargs.setdefault("max_new_tokens", self.config.max_new_tokens)
        generate_kwargs.setdefault("min_new_tokens", self.config.min_new_tokens)
        generate_kwargs.setdefault("num_beams", self.config.num_beams)
        generate_kwargs.setdefault("do_sample", self.config.do_sample)

        # Only set sampling params if they exist in config (depends on do_sample)
        if hasattr(self.config, "temperature"):
            generate_kwargs.setdefault("temperature", self.config.temperature)
        if hasattr(self.config, "top_k"):
            generate_kwargs.setdefault("top_k", self.config.top_k)
        if hasattr(self.config, "top_p"):
            generate_kwargs.setdefault("top_p", self.config.top_p)

        generate_kwargs.setdefault("repetition_penalty", self.config.repetition_penalty)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        generate_kwargs.setdefault("eos_token_id", im_end_id)

        return self.decoder.generate(
            input_ids=input_ids,
            **generate_kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,  # For Whisper
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        task: Optional[str] = None,
        text_input: Optional[str] = None,  # For text-only mode
        **generate_kwargs,
    ) -> Union[
        torch.Tensor,
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
    ]:
        if task == "text" or text_input is not None:
            return self._generate_text_only(
                text_input=text_input or user_prompt, system_prompt=system_prompt, **generate_kwargs
            )

        audio_inputs = input_values if input_values is not None else input_features
        if audio_inputs is None:
            raise ValueError("input_values or input_features must be provided for generation")

        audio_embeds = self._encode_audio(audio_inputs)
        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device

        if system_prompt is None:
            system_prompt = self.system_prompt

        if user_prompt is None:
            user_prompt = self.TASK_PROMPTS.get(
                task, self.config.user_prompt or "Transcribe: <audio>"
            ) or "Transcribe: <audio>"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(device)

        if len(prompt_ids.shape) == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        if prompt_ids.shape[0] == 1 and batch_size > 1:
            prompt_ids = prompt_ids.expand(batch_size, -1)

        if not (prompt_ids == self.audio_token_id).any():
            raise ValueError("Audio token <audio> not found in prompt")

        # Expand audio tokens to match audio embedding length
        num_audio_tokens = audio_embeds.shape[1]
        expanded_prompt_ids = self._expand_audio_tokens(prompt_ids, num_audio_tokens)

        # Prepare inputs_embeds with audio embeddings injected
        inputs_embeds = self._prepare_audio_inputs_embeds(expanded_prompt_ids, audio_embeds)

        # Create attention mask for expanded sequence
        total_seq_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(batch_size, total_seq_len, dtype=torch.long, device=device)

        # Apply generation defaults from config
        generate_kwargs.setdefault("max_new_tokens", self.config.max_new_tokens)
        generate_kwargs.setdefault("min_new_tokens", self.config.min_new_tokens)
        generate_kwargs.setdefault("num_beams", self.config.num_beams)
        generate_kwargs.setdefault("do_sample", self.config.do_sample)

        # Only set sampling params if they exist in config (depends on do_sample)
        if hasattr(self.config, "temperature"):
            generate_kwargs.setdefault("temperature", self.config.temperature)
        if hasattr(self.config, "top_k"):
            generate_kwargs.setdefault("top_k", self.config.top_k)
        if hasattr(self.config, "top_p"):
            generate_kwargs.setdefault("top_p", self.config.top_p)

        generate_kwargs.setdefault("repetition_penalty", self.config.repetition_penalty)
        generate_kwargs.setdefault("length_penalty", self.config.length_penalty)
        generate_kwargs.setdefault("no_repeat_ngram_size", self.config.no_repeat_ngram_size)

        # Only set early_stopping if it exists in config (depends on num_beams)
        if hasattr(self.config, "early_stopping"):
            generate_kwargs.setdefault("early_stopping", self.config.early_stopping)

        # Enable cache now that we use inputs_embeds consistently
        generate_kwargs.setdefault("use_cache", True)
        generate_kwargs.setdefault(
            "eos_token_id", self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        )
        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        return self.decoder.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs
        )

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        import json
        import shutil
        from pathlib import Path as PathlibPath

        from safetensors.torch import save_file

        save_dir = PathlibPath(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        actual_vocab_size = self.decoder.config.vocab_size
        self.config.vocab_size = actual_vocab_size
        self.config.text_config.vocab_size = actual_vocab_size

        if hasattr(self.encoder.config, "num_mel_bins"):
            self.config.audio_config.num_mel_bins = self.encoder.config.num_mel_bins

        self.config.save_pretrained(save_dir)

        encoder_state = {
            name: param.data
            for name, param in self.encoder.named_parameters()
            if param.requires_grad
        }
        if encoder_state:
            save_file(encoder_state, save_dir / "encoder.safetensors")

        decoder_state = {
            name: param.data
            for name, param in self.decoder.named_parameters()
            if param.requires_grad
        }
        if decoder_state:
            save_file(decoder_state, save_dir / "decoder.safetensors")

        projector_state = self.projector.state_dict()
        if projector_state:
            save_file(projector_state, save_dir / "projector.safetensors")

        if (
            self.peft_config
            and self.peft_config.get("peft_method") == "lora"
            and self.peft_config.get("r", 0) > 0
        ):
            with (save_dir / "decoder_lora_config.json").open("w") as f:
                json.dump(self.peft_config, f, indent=2)

        if (
            hasattr(self, "encoder_lora_config")
            and self.encoder_lora_config is not None
            and isinstance(self.encoder_lora_config, dict)
            and self.encoder_lora_config.get("r", 0) > 0
        ):
            with (save_dir / "encoder_lora_config.json").open("w") as f:
                json.dump(self.encoder_lora_config, f, indent=2)

        self.tokenizer.save_pretrained(save_dir)

        # For Whisper models, ensure feature_size matches num_mel_bins from encoder config
        if hasattr(self.encoder.config, "num_mel_bins"):
            # For Whisper models, explicitly set the correct feature_size before saving
            num_mel_bins = self.encoder.config.num_mel_bins
            self.feature_extractor.feature_size = num_mel_bins
            self.feature_extractor.num_mel_bins = num_mel_bins  # Explicitly set num_mel_bins
            if hasattr(self.feature_extractor, "n_mels"):
                self.feature_extractor.n_mels = num_mel_bins
            self.feature_extractor.nb_max_frames = 3000  # Whisper's max frames

        self.get_processor().save_pretrained(save_dir)

        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)


AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)
