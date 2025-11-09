from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import numpy as np
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    Wav2Vec2FeatureExtractor,
    WhisperFeatureExtractor,
)
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.activations import ACT2FN

try:
    from .asr_config import ASRConfig
except ImportError:
    from asr_config import ASRConfig  # type: ignore[no-redef]


class AudioProjector(nn.Module):
    """Audio projector using Llama-style SwiGLU architecture.

    This follows the exact pattern of LlamaMLP but allows for dimension changes
    (input_dim != output_dim), which LlamaMLP doesn't support.
    """
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim
        hidden_dim = config.projector_hidden_dim

        # Pre-norm: normalize stacked encoder features (fixes broken normalization from concatenation)
        self.ln_pre = LlamaRMSNorm(in_dim)

        # SwiGLU layers following LlamaMLP architecture
        # Using the same naming convention as LlamaMLP for consistency
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=False)

        # Use the same activation function as Llama
        self.act_fn = ACT2FN["silu"]

        # Dropout for regularization (configurable, defaults to 0.05)
        dropout_rate = getattr(config, 'projector_dropout', 0.05)
        self.dropout = nn.Dropout(dropout_rate)

        # Post-norm: normalize output to match LLM's expected embedding distribution
        self.ln_post = LlamaRMSNorm(out_dim)

        # Initialize weights with small std to avoid exploding gradients
        # std value is configurable via config.projector_init_std
        init_std = getattr(config, 'projector_init_std', 0.02)
        with torch.no_grad():
            nn.init.normal_(self.gate_proj.weight, std=init_std)
            nn.init.normal_(self.up_proj.weight, std=init_std)
            nn.init.normal_(self.down_proj.weight, std=init_std)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        # Pad the sequence to be divisible by k instead of truncating
        remainder = seq_len % self.k
        if remainder:
            pad_len = self.k - remainder
            x = F.pad(x, (0, 0, 0, pad_len))

        # Stack frames (concatenation breaks encoder's normalization)
        x = x.contiguous().view(batch_size, -1, dim * self.k)

        # Re-normalize after stacking
        x = self.ln_pre(x)

        # Apply SwiGLU following LlamaMLP's exact pattern
        # down_proj(act_fn(gate_proj(x)) * up_proj(x))
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # Apply dropout after activation
        hidden_states = self.dropout(hidden_states)

        # Final projection
        output = self.down_proj(hidden_states)

        # Normalize before LLM to ensure stable input distribution
        return self.ln_post(output)


class ASRModel(PreTrainedModel):
    config_class = ASRConfig
    base_model_prefix = "model"
    # Note: main_input_name will be set dynamically in __init__ based on encoder type
    # Whisper uses "input_features", others use "input_values"
    main_input_name = "input_values"  # Default, will be overridden
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_save = ["encoder", "decoder.base_model"]
    _is_loading_from_pretrained: bool = False
    _pretrained_model_path: Optional[str] = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        import json

        from safetensors.torch import load_file
        from transformers.utils import cached_file

        # Check if config is already provided in kwargs
        config = kwargs.pop("config", None)
        if config is None:
            config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            # If config is provided, still need to load subfolder/revision info
            # but use the provided config's values
            pass

        # IMPORTANT: Always load feature extractor from the audio model, not from the saved checkpoint
        # This ensures we get the correct mel bin configuration (e.g., 128 for Whisper V3 Turbo)
        is_whisper = "whisper" in config.audio_model_id.lower()
        if is_whisper:
            num_mel_bins = getattr(config.audio_config, "num_mel_bins", 80)
            kwargs["feature_extractor"] = WhisperFeatureExtractor.from_pretrained(
                config.audio_model_id,
                feature_size=num_mel_bins
            )
        else:
            kwargs["feature_extractor"] = Wav2Vec2FeatureExtractor.from_pretrained(
                config.audio_model_id
            )

        cls._is_loading_from_pretrained = True
        cls._pretrained_model_path = pretrained_model_name_or_path

        # Extract subfolder/revision from kwargs for cached_file calls
        subfolder = kwargs.get("subfolder")
        revision = kwargs.get("revision")
        cache_kwargs = {}
        if subfolder:
            cache_kwargs["subfolder"] = subfolder
        if revision:
            cache_kwargs["revision"] = revision

        try:
            # Helper function to load file with fallback to last-checkpoint subfolder
            def load_cached_file(filename, load_fn=None):
                """Load a file with fallback to last-checkpoint subfolder.

                Args:
                    filename: Name of file to load
                    load_fn: Function to apply to loaded path (default: load_file for safetensors)
                """
                from pathlib import Path as PathlibPath

                if load_fn is None:
                    load_fn = load_file

                # Try main directory first
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

                # Fallback to last-checkpoint subfolder
                try:
                    fallback_kwargs = cache_kwargs.copy()
                    fallback_kwargs["subfolder"] = "last-checkpoint"
                    file_path = cached_file(
                        pretrained_model_name_or_path,
                        filename,
                        _raise_exceptions_for_missing_entries=False,
                        **fallback_kwargs,
                    )
                    if file_path:
                        print(f"Loading {filename} from last-checkpoint subfolder")
                        return load_fn(file_path)
                except Exception:
                    pass

                return None

            # Load LoRA configs (JSON files)
            def load_json(path):
                from pathlib import Path as PathlibPath
                with PathlibPath(path).open() as f:
                    return json.load(f)

            encoder_lora_config = load_cached_file("encoder_lora_config.json", load_json)
            decoder_lora_config = load_cached_file("decoder_lora_config.json", load_json)

            # Create model with LoRA configs
            model = cls(
                config, peft_config=decoder_lora_config, encoder_lora_config=encoder_lora_config
            )

            # Load component weights (safetensors files)
            encoder_state = load_cached_file("encoder.safetensors")
            decoder_state = load_cached_file("decoder.safetensors")
            projector_state = load_cached_file("projector.safetensors")

            # Load projector weights (required)
            if not projector_state:
                raise FileNotFoundError(
                    f"projector.safetensors not found in {pretrained_model_name_or_path}. "
                    "The repository may not have been trained yet."
                )
            total_params = sum(v.numel() for v in projector_state.values())
            model.projector.load_state_dict(projector_state, strict=True, assign=True)
            print(f"✓ Loaded projector weights ({total_params:,} parameters)")

            # Load encoder LoRA weights
            if encoder_lora_config:
                if not encoder_state:
                    raise FileNotFoundError(
                        f"encoder.safetensors not found in {pretrained_model_name_or_path}. "
                        "The repository may not have been trained yet."
                    )
                total_params = sum(v.numel() for v in encoder_state.values())
                model.encoder.load_state_dict(encoder_state, strict=False, assign=True)
                print(
                    f"✓ Loaded encoder LoRA (r={encoder_lora_config.get('r', 0)}, {total_params:,} parameters)"
                )

            # Load decoder LoRA weights
            if decoder_lora_config and decoder_lora_config.get("r", 0) > 0:
                if not decoder_state:
                    raise FileNotFoundError(
                        f"decoder.safetensors not found in {pretrained_model_name_or_path}. "
                        "The repository may not have been trained yet."
                    )
                total_params = sum(v.numel() for v in decoder_state.values())
                model.decoder.load_state_dict(decoder_state, strict=False, assign=True)
                print(
                    f"✓ Loaded decoder LoRA (r={decoder_lora_config.get('r', 0)}, {total_params:,} parameters)"
                )

            # Move model to device if specified (needed after loading weights from meta tensors)
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

        # Determine if this is a Whisper model
        is_whisper = "whisper" in config.audio_model_id.lower() or (hasattr(self.encoder.config, "model_type") and "whisper" in self.encoder.config.model_type.lower())

        # Set main_input_name based on encoder type
        # Whisper uses "input_features", others use "input_values"
        if is_whisper:
            self.main_input_name = "input_features"
        else:
            self.main_input_name = "input_values"

        # Use provided feature extractor or create one
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            # Use appropriate feature extractor based on model type
            if is_whisper:
                # For Whisper models, we need to ensure the feature extractor matches the encoder's mel bin configuration
                # Whisper Large V3 Turbo uses 128 mel bins instead of the standard 80
                num_mel_bins = getattr(config.audio_config, "num_mel_bins", 80)
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    config.audio_model_id,
                    feature_size=num_mel_bins  # Override feature_size to match model's mel bins
                )
            else:
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.audio_model_id)

        # Create decoder first (needed for tokenizer init)
        self.decoder = self._create_decoder(config, peft_config)
        self.generation_config = self.decoder.generation_config

        # Override generation_config with ASR-appropriate defaults
        self.generation_config.num_beams = config.num_beams
        self.generation_config.max_new_tokens = config.max_new_tokens
        self.generation_config.min_new_tokens = config.min_new_tokens
        self.generation_config.do_sample = config.do_sample
        # Force use_cache=False for this architecture (uses inputs_embeds)
        self.generation_config.use_cache = False

        # Only set sampling parameters if sampling is enabled
        if config.do_sample:
            self.generation_config.top_k = config.top_k
            self.generation_config.top_p = config.top_p
            self.generation_config.temperature = config.temperature
        else:
            # Remove sampling parameters when not sampling
            self.generation_config.top_k = None
            self.generation_config.top_p = None
            self.generation_config.temperature = None

        # Initialize tokenizer and resize embeddings after decoder is created
        self._init_tokenizer()

        from types import SimpleNamespace

        projector_config = SimpleNamespace(
            encoder_projector_ds_rate=config.audio_downsample_rate,
            encoder_dim=config.encoder_dim,
            llm_dim=config.llm_dim,
            projector_hidden_dim=config.projector_hidden_dim,
        )
        self.projector = AudioProjector(projector_config)

        self._no_split_modules = self.decoder._no_split_modules

    @staticmethod
    def _apply_lora(model, lora_config: dict, task_type, model_name: str = "model", default_dropout: float = 0.0):
        """Apply LoRA adapters to a model (encoder or decoder).

        Args:
            model: The model to apply LoRA to
            lora_config: Dict with LoRA configuration (r, lora_alpha, target_modules, etc.)
            task_type: peft.TaskType (FEATURE_EXTRACTION for encoder, CAUSAL_LM for decoder)
            model_name: Name for logging purposes
            default_dropout: Default dropout value from config
        """
        if lora_config.get("r", 0) == 0:
            # Freeze the model if r=0
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

        model = get_peft_model(model, peft_config)
        return model

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

        # Configure model loading kwargs
        encoder_kwargs = {
            "attn_implementation": config.attn_implementation,
            "torch_dtype": target_dtype,  # Use torch_dtype instead of dtype
            "low_cpu_mem_usage": True,
        }
        # Avoid device_map="auto" when loading from pretrained to prevent meta tensor issues
        if not cls._is_loading_from_pretrained:
            encoder_kwargs["device_map"] = "auto"

        # Check if this is a Whisper model and load only the encoder part
        if "whisper" in config.audio_model_id.lower():
            from transformers import WhisperModel
            full_model = WhisperModel.from_pretrained(config.audio_model_id, **encoder_kwargs)
            encoder = full_model.encoder
            # Clean up the full model to save memory
            del full_model
        else:
            encoder = AutoModel.from_pretrained(config.audio_model_id, **encoder_kwargs)

        # Enable SpecAugment for both Whisper and Wav2Vec2 models during training
        # Both models apply augmentation automatically in train() mode via their forward pass
        is_whisper = "whisper" in config.audio_model_id.lower() or (hasattr(encoder.config, "model_type") and "whisper" in encoder.config.model_type.lower())
        is_wav2vec2 = hasattr(encoder.config, "model_type") and "wav2vec2" in encoder.config.model_type.lower()

        # Configure SpecAugment for both Whisper and Wav2Vec2
        if is_whisper or is_wav2vec2:
            encoder.config.apply_spec_augment = True
            encoder.config.mask_time_prob = getattr(config, 'mask_time_prob', 0.05)
            encoder.config.mask_time_length = getattr(config, 'mask_time_length', 10)
            encoder.config.mask_feature_prob = getattr(config, 'mask_feature_prob', 0.0)
            encoder.config.mask_feature_length = getattr(config, 'mask_feature_length', 10)

        encoder.requires_grad_(False)

        # Wrap encoder forward BEFORE applying LoRA to filter invalid kwargs
        original_forward = encoder.forward

        def safe_encoder_forward(
            self_encoder,
            input_values=None,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,  # Catch and discard invalid kwargs like input_ids
        ):
            # Check if this is a Whisper model (expects input_features instead of input_values)
            # Note: config.audio_model_id is not accessible here, so we check the encoder config
            is_whisper_encoder = hasattr(self_encoder.config, "model_type") and "whisper" in self_encoder.config.model_type.lower()
            if is_whisper_encoder:
                return original_forward(
                    input_features=input_values,  # Map input_values to input_features for Whisper
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                return original_forward(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        import types
        encoder.forward = types.MethodType(safe_encoder_forward, encoder)

        # Apply LoRA to encoder if configured (after wrapping base forward)
        if encoder_lora_config and encoder_lora_config.get("r", 0) > 0:
            from peft import TaskType

            encoder = cls._apply_lora(
                encoder,
                encoder_lora_config,
                TaskType.FEATURE_EXTRACTION,
                "encoder",
                default_dropout=config.lora_default_dropout
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

        # Set use_cache=False on the decoder config to prevent cache during training
        # This is critical because we concatenate text+audio embeddings, which invalidates any cache
        decoder.config.use_cache = False

        decoder.requires_grad_(False)

        # Apply LoRA to decoder if configured
        if peft_config and peft_config.get("peft_method") == "lora":
            from peft import TaskType

            decoder = cls._apply_lora(
                decoder,
                peft_config,
                TaskType.CAUSAL_LM,
                "decoder",
                default_dropout=config.lora_default_dropout
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
        encoder_trainable = {name for name, param in self.encoder.named_parameters() if param.requires_grad}
        for name, tensor in encoder_state.items():
            if name in encoder_trainable:
                state[f"encoder.{name}"] = tensor

        # Get decoder trainable params (LoRA adapters)
        decoder_state = self.decoder.state_dict()
        decoder_trainable = {name for name, param in self.decoder.named_parameters() if param.requires_grad}
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

    def _encode_audio(
        self,
        input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure input is on encoder's device and has the right dtype
        encoder_device = next(self.encoder.parameters()).device
        encoder_dtype = next(self.encoder.parameters()).dtype
        input_values = input_values.to(device=encoder_device, dtype=encoder_dtype)

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

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,  # For Whisper
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Handle both input_values (Wav2Vec2/HuBERT) and input_features (Whisper)
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

            audio_embeds = self._encode_audio(
                input_values=audio_inputs,  # Will be mapped to input_features for Whisper by safe_encoder_forward
                audio_attention_mask=audio_attention_mask,
            )

            batch_size = input_ids.shape[0]
            audio_seq_len = audio_embeds.shape[1]

            # Validate audio token ID before using it
            if self.audio_token_id is None:
                raise ValueError(f"Audio token not properly initialized: {self.audio_token_id}")

            vocab_size = self.decoder.get_input_embeddings().weight.shape[0]
            if self.audio_token_id >= vocab_size:
                raise ValueError(
                    f"Audio token ID out of range. ID: {self.audio_token_id}, Vocab size: {vocab_size}"
                )

            # Find positions of <audio> token
            audio_token_positions = (input_ids == self.audio_token_id).nonzero(as_tuple=True)

            if len(audio_token_positions[0]) == 0:
                raise ValueError("Audio token <audio> must be present in input")

            text_embeds = self.decoder.get_input_embeddings()(input_ids)

            new_embeds = []
            new_labels: list[torch.Tensor] = [] if labels is not None else None
            new_attention = []

            for i in range(batch_size):
                # Find audio token position for this batch item
                token_mask = audio_token_positions[0] == i

                if not token_mask.any():
                    raise ValueError(f"Missing audio token in batch item {i}")

                audio_pos = audio_token_positions[1][token_mask][0].item()

                # Split embeddings: before audio token, audio embeddings, after audio token
                before_audio = text_embeds[i, :audio_pos]
                after_audio = text_embeds[i, audio_pos + 1 :]

                # Replace audio token with audio embeddings
                batch_embeds = torch.cat([before_audio, audio_embeds[i], after_audio], dim=0)
                new_embeds.append(batch_embeds)

                if labels is not None:
                    before_labels = labels[i, :audio_pos]
                    # Audio embeddings don't contribute to loss
                    audio_labels = torch.full(
                        (audio_seq_len,), -100, dtype=labels.dtype, device=labels.device
                    )
                    after_labels = labels[i, audio_pos + 1 :]
                    batch_labels = torch.cat([before_labels, audio_labels, after_labels], dim=0)
                    new_labels.append(batch_labels)

                if attention_mask is not None:
                    before_attn = attention_mask[i, :audio_pos]
                    audio_attn = torch.ones(
                        audio_seq_len, dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    after_attn = attention_mask[i, audio_pos + 1 :]
                    batch_attn = torch.cat([before_attn, audio_attn, after_attn], dim=0)
                    new_attention.append(batch_attn)

            inputs_embeds = torch.stack(new_embeds)
            if labels is not None:
                labels = torch.stack(new_labels)
            full_attention_mask = torch.stack(new_attention) if attention_mask is not None else None
        else:
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
            full_attention_mask = attention_mask

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
            use_cache=False,  # Always False when training with audio to prevent cache corruption
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

        # Tokenize the prompt
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(device)

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        # Set default generation parameters
        generate_kwargs.setdefault("max_new_tokens", self.config.max_new_tokens)
        generate_kwargs.setdefault("min_new_tokens", self.config.min_new_tokens)
        generate_kwargs.setdefault("num_beams", self.config.num_beams)
        generate_kwargs.setdefault("do_sample", self.config.do_sample)
        generate_kwargs.setdefault("temperature", self.config.temperature)
        generate_kwargs.setdefault("top_k", self.config.top_k)
        generate_kwargs.setdefault("top_p", self.config.top_p)
        generate_kwargs.setdefault("repetition_penalty", self.config.repetition_penalty)

        # Set eos token
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        generate_kwargs.setdefault("eos_token_id", im_end_id)

        # Generate using decoder only
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
        # Handle text-only mode (no audio)
        if task == "text" or text_input is not None:
            return self._generate_text_only(
                text_input=text_input or user_prompt,
                system_prompt=system_prompt,
                **generate_kwargs
            )

        # Handle both input_values (Wav2Vec2/HuBERT) and input_features (Whisper)
        audio_inputs = input_values if input_values is not None else input_features
        if audio_inputs is None:
            raise ValueError("input_values or input_features must be provided for generation")

        audio_embeds = self._encode_audio(audio_inputs)
        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device

        if system_prompt is None:
            system_prompt = self.system_prompt

        # Determine user prompt based on task
        if user_prompt is None:
            if task == "continue":
                user_prompt = "Continue: <audio>"
            elif task == "describe":
                user_prompt = "Describe: <audio>"
            elif task == "emotion":
                user_prompt = "Emotion: <audio>"
            else:
                # Default to transcribe
                user_prompt = self.config.user_prompt if self.config.user_prompt else "Transcribe: <audio>"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        # Print the prompt being used for debugging
        print(f"[Tiny Audio] Using prompt: {user_prompt}")

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

        # Find positions of <audio> token
        audio_token_positions = (prompt_ids == self.audio_token_id).nonzero(as_tuple=True)

        if len(audio_token_positions[0]) == 0:
            raise ValueError("Audio token <audio> not found in prompt")

        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids)

        new_embeds = []
        for i in range(batch_size):
            token_mask = audio_token_positions[0] == i

            if not token_mask.any():
                raise ValueError(f"Missing audio token in batch item {i}")

            audio_pos = audio_token_positions[1][token_mask][0].item()

            # Replace audio token with audio embeddings
            before_audio = prompt_embeds[i, :audio_pos]
            after_audio = prompt_embeds[i, audio_pos + 1 :]

            batch_embeds = torch.cat([before_audio, audio_embeds[i], after_audio], dim=0)
            new_embeds.append(batch_embeds)

        inputs_embeds = torch.stack(new_embeds)

        total_seq_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(batch_size, total_seq_len, dtype=torch.long, device=device)

        generate_kwargs.setdefault("max_new_tokens", self.config.max_new_tokens)
        generate_kwargs.setdefault("min_new_tokens", self.config.min_new_tokens)
        generate_kwargs.setdefault("num_beams", self.config.num_beams)
        generate_kwargs.setdefault("do_sample", self.config.do_sample)
        generate_kwargs.setdefault("temperature", self.config.temperature)
        generate_kwargs.setdefault("top_k", self.config.top_k)
        generate_kwargs.setdefault("top_p", self.config.top_p)
        generate_kwargs.setdefault("repetition_penalty", self.config.repetition_penalty)
        generate_kwargs.setdefault("length_penalty", self.config.length_penalty)
        generate_kwargs.setdefault("no_repeat_ngram_size", self.config.no_repeat_ngram_size)
        generate_kwargs.setdefault("early_stopping", self.config.early_stopping)
        # Always disable cache for this architecture because we use inputs_embeds (not input_ids)
        # with audio embeddings mixed in, which the cache system doesn't handle correctly
        generate_kwargs["use_cache"] = False

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        generate_kwargs.setdefault("eos_token_id", im_end_id)
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

        self.config.save_pretrained(save_dir)
        if hasattr(self, "generation_config") and self.generation_config is not None:
            self.generation_config.save_pretrained(save_dir)

        # Save only trainable parameters for encoder
        encoder_state = {
            name: param.data
            for name, param in self.encoder.named_parameters()
            if param.requires_grad
        }
        if encoder_state:
            save_file(encoder_state, save_dir / "encoder.safetensors")

        # Save only trainable parameters for decoder
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

        # Save encoder LoRA config (only if LoRA is actually used, r > 0)
        if (
            hasattr(self, "encoder_lora_config")
            and self.encoder_lora_config is not None
            and isinstance(self.encoder_lora_config, dict)
            and self.encoder_lora_config.get("r", 0) > 0
        ):
            with (save_dir / "encoder_lora_config.json").open("w") as f:
                json.dump(self.encoder_lora_config, f, indent=2)

        self.tokenizer.save_pretrained(save_dir)

        # Save feature extractor with correct configuration
        # For Whisper models, ensure feature_size matches num_mel_bins from encoder config
        self.feature_extractor.save_pretrained(save_dir)

        self.get_processor().save_pretrained(save_dir)

        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)


AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)
