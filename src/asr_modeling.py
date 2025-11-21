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
    TextIteratorStreamer,
    Wav2Vec2FeatureExtractor,
)
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


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=False, dropout_rate=0.05):
        super().__init__()
        # SwiGLU: (Swish(xW_gate) * xW_val) W_out
        # Memory optimization: Combined layer for W1 (Gate) and W2 (Value) helps parallelism
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        # 2025 Optimization: Low dropout for Audio to preserve phonemes
        # Applied to input rather than gated output for better regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply dropout to input (more effective for gated networks)
        x = self.dropout(x)

        # Fusing the two projections allows PyTorch to use optimized kernels
        w12_out = self.w12(x)

        # Split into gate and value
        # chunk is a view, so no extra memory allocation
        x_gate, x_val = w12_out.chunk(2, dim=-1)

        # F.silu is the Swish activation
        x = F.silu(x_gate) * x_val

        # Final projection
        x = self.w3(x)
        return x


class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = getattr(config, "projector_pool_stride", 5)
        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim
        hidden_dim = config.llm_dim
        dropout_rate = getattr(config, "projector_dropout", 0.05)
        self.noise_scale = getattr(config, "projector_input_noise", 0.01)

        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        # 1. Pre-Norm (Epsilon aligned to Llama-3)
        self.ln_pre = LlamaRMSNorm(in_dim, eps=1e-6)

        # 2. SwiGLU
        self.proj = SwiGLU(in_dim, hidden_dim, out_dim, dropout_rate=dropout_rate)

        # 3. Residual Connection
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = nn.Identity()

        # 4. Interface Guardrail
        self.ln_post = LlamaRMSNorm(out_dim, eps=1e-6)

        # 5. Output Scale
        # Init at 1.0 is safer than 2.0. Let the gradients drive it up if needed.
        self.output_scale = nn.Parameter(torch.ones([]) * 1.0)

        # --- OPTIMIZED INITIALIZATION ---
        with torch.no_grad():
            std = getattr(config, "projector_init_std", 0.02)
            
            # Norms start as identity
            self.ln_pre.weight.data.fill_(1.0)
            self.ln_post.weight.data.fill_(1.0)
            
            # SwiGLU (The "Correction") starts small
            nn.init.trunc_normal_(self.proj.w12.weight, std=std)
            nn.init.trunc_normal_(self.proj.w3.weight, std=std)
            
            # Residual (The "Shortcut") starts STRONG
            # This ensures signal flows through even if SwiGLU is random
            if isinstance(self.residual_proj, nn.Linear):
                # Orthogonal ensures the matrix rotates/maps features without shrinking them
                nn.init.orthogonal_(self.residual_proj.weight)
                # Scale slightly to account for the change in width (3840 -> 4096)
                self.residual_proj.weight.data.mul_(0.5) # Conservative starting gain

    def forward(self, x):
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise

        batch_size, seq_len, dim = x.size()

        remainder = seq_len % self.k
        if remainder:
            pad_len = self.k - remainder
            x = x.transpose(1, 2) # Pad expects [B, C, T]
            x = F.pad(x, (0, pad_len), mode='constant') 
            x = x.transpose(1, 2)

        x = x.contiguous().view(batch_size, -1, dim * self.k)

        # Residual Path
        residual = self.residual_proj(x)
        
        # Main Path
        x = self.ln_pre(x)
        x = self.proj(x)
        
        # Injection
        x = x + residual

        # Final Norm & Scale
        x = self.ln_post(x)
        x = x * self.output_scale

        return x

class ASRModel(PreTrainedModel):
    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _is_loading_from_pretrained: bool = False
    _pretrained_model_path: Optional[str] = None

    # Task to prompt mapping for generation
    TASK_PROMPTS = {
        "transcribe": "Transcribe: <audio>",
        "continue": "Continue: <audio>",
        "describe": "Describe: <audio>",
        "emotion": "Emotion: <audio>",
    }

    @staticmethod
    def _create_feature_extractor(audio_model_id: str, training_mode: bool = False):
        """Factory method to create the appropriate feature extractor."""
        is_whisper = "whisper" in audio_model_id.lower()
        if is_whisper:
            from transformers import WhisperConfig, WhisperFeatureExtractor

            encoder_config = WhisperConfig.from_pretrained(audio_model_id)
            num_mel_bins = encoder_config.num_mel_bins

            # Add SpecAugment parameters for training
            return WhisperFeatureExtractor.from_pretrained(
                audio_model_id,
                feature_size=num_mel_bins,
                # SpecAugment parameters - optimized for adapter training
                apply_spec_augment=training_mode,  # Only during training
                # Reduced Freq Mask (Standard is ~27)
                mask_feature_prob=0.05,       # Low prob of masking features
                mask_feature_length=15,       # Narrower frequency bands (15 vs 27)
                mask_feature_min_masks=0,     # Allow 0 masks
                # Reduced Time Mask (Standard is ~80-100)
                mask_time_prob=0.05,          # Low prob of masking time
                mask_time_length=20,          # Shorter time masks (200ms vs 800ms)
                mask_time_min_masks=1,        # At least 1 mask if triggered
            )
        return Wav2Vec2FeatureExtractor.from_pretrained(audio_model_id)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        from transformers import AutoFeatureExtractor

        config = kwargs.pop("config", None)
        if config is None:
            config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Load feature extractor from saved model directory
        kwargs["feature_extractor"] = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        cls._is_loading_from_pretrained = True
        cls._pretrained_model_path = pretrained_model_name_or_path

        try:
            from safetensors.torch import load_file
            from transformers.utils.hub import cached_file

            model = cls(config, **kwargs)

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

            if not model_file:
                raise FileNotFoundError(
                    f"model.safetensors not found in {pretrained_model_name_or_path}. "
                    "The repository may not have been trained yet."
                )

            state_dict = load_file(model_file)
            model.load_state_dict(state_dict, strict=False, assign=True)

            target_dtype = getattr(torch, config.model_dtype)
            model.projector = model.projector.to(dtype=target_dtype)

            device = kwargs.get("device")
            if device is not None:
                model = model.to(device)

            return model
        finally:
            cls._is_loading_from_pretrained = False
            del cls._pretrained_model_path

    def __init__(self, config: ASRConfig, **kwargs):
        super().__init__(config)

        feature_extractor = kwargs.pop("feature_extractor", None)

        self.system_prompt = config.system_prompt

        self.encoder = self._create_encoder(config)

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
            # Enable SpecAugment during training (can be controlled via config)
            training_mode = getattr(config, "use_specaugment", False)
            self.feature_extractor = self._create_feature_extractor(
                config.audio_model_id, training_mode=training_mode
            )

        self.decoder = self._create_decoder(config)
        self.generation_config = self.decoder.generation_config

        # Sync generation config with ASRConfig defaults
        config_params = [
            "max_new_tokens", "min_new_tokens", "num_beams", "do_sample",
            "temperature", "top_k", "top_p", "repetition_penalty",
            "length_penalty", "no_repeat_ngram_size", "early_stopping", "use_cache"
        ]
        for param in config_params:
            if hasattr(config, param) and getattr(config, param) is not None:
                setattr(self.generation_config, param, getattr(config, param))

        self._init_tokenizer()

        from types import SimpleNamespace

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

        # Pass config directly to AudioProjector, let it handle defaults
        projector_config = SimpleNamespace(
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            **{k: v for k, v in vars(config).items() if k.startswith('projector_')}
        )
        self.projector = AudioProjector(projector_config)

        target_dtype = getattr(torch, config.model_dtype)
        self.projector = self.projector.to(dtype=target_dtype)

        # Create loss function with label smoothing
        self.label_smoothing = getattr(config, "label_smoothing", 0.1)
        self.loss_fct = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.label_smoothing
        )

        self._no_split_modules = self.decoder._no_split_modules

    @classmethod
    def _create_encoder(cls, config: ASRConfig):
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

        original_forward = encoder.forward
        input_key = "input_features" if is_whisper else "input_values"

        def safe_encoder_forward(self_encoder, input_values=None, **kwargs):
            kwargs.pop("input_ids", None)
            return original_forward(**{input_key: input_values}, **kwargs)

        import types

        encoder.forward = types.MethodType(safe_encoder_forward, encoder)
        encoder.requires_grad_(False)

        return encoder

    @classmethod
    def _create_decoder(cls, config: ASRConfig):
        target_dtype = getattr(torch, config.model_dtype)

        decoder_kwargs = {
            "attn_implementation": config.attn_implementation,
            "dtype": target_dtype,
            "trust_remote_code": True,
        }

        decoder = AutoModelForCausalLM.from_pretrained(config.text_model_id, **decoder_kwargs)
        decoder.config.use_cache = config.use_cache
        decoder.requires_grad_(False)

        return decoder

    def _init_weights(self, module):
        pass

    def can_generate(self) -> bool:
        return True

    @property
    def _tied_weights_keys(self):
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

        if (
            self.tokenizer.pad_token is None
            or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ) and "<|finetune_right_pad_id|>" in self.tokenizer.get_vocab():
            self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        existing_special = self.tokenizer.additional_special_tokens or []

        if "<audio>" not in existing_special:
            special_tokens = {"additional_special_tokens": existing_special + ["<audio>"]}
            num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
            if num_added_tokens > 0:
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
        return self._get_trainable_state_dict()

    def _get_trainable_state_dict(self):
        state = {}

        projector_state = self.projector.state_dict()
        for name, tensor in projector_state.items():
            state[f"projector.{name}"] = tensor

        return state

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.decoder.set_output_embeddings(value)

    def _encode_audio(
        self,
        input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_device = next(self.encoder.parameters()).device

        # Optimization: Only move/cast if strictly necessary
        if input_values.device != encoder_device:
            input_values = input_values.to(encoder_device)

        # Whisper/Wav2Vec2 might expect float, but encoder is half.
        # Check to prevent double casting
        target_dtype = next(self.encoder.parameters()).dtype
        if input_values.dtype != target_dtype:
            input_values = input_values.to(dtype=target_dtype)

        with torch.no_grad():
            audio_features = self.encoder(
                input_values=input_values,
                attention_mask=audio_attention_mask,
            ).last_hidden_state

        audio_embeds = self.projector(audio_features)

        decoder_dtype = next(self.decoder.parameters()).dtype
        if audio_embeds.dtype != decoder_dtype:
            audio_embeds = audio_embeds.to(dtype=decoder_dtype)

        return audio_embeds

    def _get_audio_expansion_details(self, input_ids: torch.Tensor, num_audio_tokens: int) -> dict:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        audio_mask = input_ids == self.audio_token_id

        audio_counts = audio_mask.sum(dim=1)
        if not (audio_counts == 1).all():
            missing = (audio_counts == 0).any()
            multiple = (audio_counts > 1).any()
            if missing:
                raise ValueError("Some samples are missing audio token")
            if multiple:
                raise ValueError("Some samples have multiple audio tokens")

        token_counts = torch.where(audio_mask, num_audio_tokens, 1)
        cumsum_counts = torch.cumsum(token_counts, dim=1)
        new_start_positions = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                cumsum_counts[:, :-1],
            ],
            dim=1,
        )

        new_seq_len = seq_len - 1 + num_audio_tokens

        return {
            "new_seq_len": new_seq_len,
            "new_start_positions": new_start_positions,
            "audio_mask": audio_mask,
        }

    def _expand_tensor_for_audio(
        self,
        input_ids: torch.Tensor,
        tensor_to_expand: Optional[torch.Tensor],
        num_audio_tokens: int,
        fill_value: Optional[Union[int, float]] = None,
        audio_fill_value: Optional[Union[int, float]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        details = self._get_audio_expansion_details(input_ids, num_audio_tokens)
        new_seq_len = details["new_seq_len"]
        new_start_positions = details["new_start_positions"]
        audio_mask = details["audio_mask"]

        if tensor_to_expand is None:
            tensor_to_expand = input_ids
            fill_value = fill_value or self.tokenizer.pad_token_id
            audio_fill_value = audio_fill_value or self.audio_token_id
        else:
            if fill_value is None:
                raise ValueError("fill_value must be provided when expanding non-input_ids tensors")
            if audio_fill_value is None:
                audio_fill_value = fill_value

        assert tensor_to_expand is not None

        expanded = torch.full(
            (batch_size, new_seq_len),
            fill_value,
            dtype=tensor_to_expand.dtype,
            device=device,
        )

        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        non_audio_mask = ~audio_mask
        expanded[batch_indices[non_audio_mask], new_start_positions[non_audio_mask]] = (
            tensor_to_expand[non_audio_mask]
        )

        if audio_fill_value != fill_value:
            audio_positions = audio_mask.int().argmax(dim=1)
            audio_new_start = new_start_positions[
                torch.arange(batch_size, device=device), audio_positions
            ]
            audio_token_indices = torch.arange(num_audio_tokens, device=device).unsqueeze(0)
            audio_positions_expanded = audio_new_start.unsqueeze(1) + audio_token_indices
            batch_idx_expanded = (
                torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_audio_tokens)
            )
            expanded[batch_idx_expanded, audio_positions_expanded] = audio_fill_value

        return expanded

    def _expand_audio_tokens(self, input_ids: torch.Tensor, num_audio_tokens: int) -> torch.Tensor:
        return self._expand_tensor_for_audio(input_ids, None, num_audio_tokens)

    def _expand_for_audio_tokens(
        self,
        input_ids: torch.Tensor,
        tensor_to_expand: torch.Tensor,
        num_audio_tokens: int,
        fill_value: Union[int, float],
    ) -> torch.Tensor:
        return self._expand_tensor_for_audio(
            input_ids, tensor_to_expand, num_audio_tokens, fill_value
        )

    def _prepare_audio_inputs_embeds(
        self, expanded_input_ids: torch.Tensor, audio_embeds: torch.Tensor
    ) -> torch.Tensor:
        inputs_embeds = self.decoder.get_input_embeddings()(expanded_input_ids)
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
        num_items_in_batch: Optional[
            int
        ] = None,  # HF Trainer provides this for gradient accumulation
        **kwargs,
    ):
        audio_inputs = input_values if input_values is not None else input_features
        if audio_inputs is not None:
            if input_ids is None:
                raise ValueError(
                    "forward() requires both audio inputs and input_ids (for training). "
                    "For inference, use the generate() method instead, or use the pipeline "
                    "which will automatically call generate()."
                )

            audio_attention_mask = kwargs.pop("audio_attention_mask", None)

            kwargs.pop("past_key_values", None)
            use_cache = kwargs.pop("use_cache", None)

            audio_embeds = self._encode_audio(
                input_values=audio_inputs,  # Will be mapped to input_features for Whisper by safe_encoder_forward
                audio_attention_mask=audio_attention_mask,
            )

            if self.audio_token_id is None:
                raise ValueError(f"Audio token not properly initialized: {self.audio_token_id}")

            vocab_size = self.decoder.get_input_embeddings().weight.shape[0]
            if self.audio_token_id >= vocab_size:
                raise ValueError(
                    f"Audio token ID out of range. ID: {self.audio_token_id}, Vocab size: {vocab_size}"
                )

            if not (input_ids == self.audio_token_id).any():
                raise ValueError("Audio token <audio> must be present in input")

            num_audio_tokens = audio_embeds.shape[1]
            expanded_input_ids = self._expand_audio_tokens(input_ids, num_audio_tokens)

            inputs_embeds = self._prepare_audio_inputs_embeds(expanded_input_ids, audio_embeds)

            if attention_mask is not None:
                full_attention_mask = self._expand_for_audio_tokens(
                    input_ids, attention_mask, num_audio_tokens, fill_value=1
                )
            else:
                full_attention_mask = None

            if labels is not None:
                labels = self._expand_for_audio_tokens(
                    input_ids, labels, num_audio_tokens, fill_value=-100
                )
        else:
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
            full_attention_mask = attention_mask
            use_cache = kwargs.pop("use_cache", None)

        # Get decoder outputs
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=None,  # Never let decoder compute loss, we'll do it ourselves
            use_cache=use_cache if use_cache is not None else False,
            **kwargs,
        )

        # Compute loss if labels provided
        if labels is not None:
            # Apply label smoothing only during training
            if self.training:
                loss = self.loss_fct(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.view(-1)
                )
            else:
                # No label smoothing for validation - get true loss
                val_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = val_loss_fct(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.view(-1)
                )

            # Create new output with loss (outputs might be immutable)
            from transformers.modeling_outputs import CausalLMOutputWithPast
            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,  # For Whisper
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        task: Optional[str] = None,
        **generate_kwargs,
    ) -> Union[
        torch.Tensor,
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
    ]:
        audio_inputs = input_values if input_values is not None else input_features
        if audio_inputs is None:
            raise ValueError("input_values or input_features must be provided for generation")

        audio_embeds = self._encode_audio(audio_inputs)
        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device

        if system_prompt is None:
            system_prompt = self.system_prompt

        if user_prompt is None:
            user_prompt = (
                self.TASK_PROMPTS.get(task, self.config.user_prompt or "Transcribe: <audio>")
                or "Transcribe: <audio>"
            )

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

        num_audio_tokens = audio_embeds.shape[1]
        expanded_prompt_ids = self._expand_audio_tokens(prompt_ids, num_audio_tokens)
        inputs_embeds = self._prepare_audio_inputs_embeds(expanded_prompt_ids, audio_embeds)
        total_seq_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(batch_size, total_seq_len, dtype=torch.long, device=device)
        config_params = [
            "max_new_tokens",
            "min_new_tokens",
            "num_beams",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "length_penalty",
            "no_repeat_ngram_size",
            "early_stopping",
        ]
        for param in config_params:
            if hasattr(self.config, param) and getattr(self.config, param) is not None:
                generate_kwargs.setdefault(param, getattr(self.config, param))

        generate_kwargs.setdefault("use_cache", True)
        generate_kwargs.setdefault(
            "eos_token_id", self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        )
        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        prompt_length = expanded_prompt_ids.shape[1]

        generated_ids = self.decoder.generate(
            input_ids=expanded_prompt_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return generated_ids[:, prompt_length:]

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        import shutil
        from pathlib import Path as PathlibPath

        save_dir = PathlibPath(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        actual_vocab_size = self.decoder.config.vocab_size
        self.config.vocab_size = actual_vocab_size
        self.config.text_config.vocab_size = actual_vocab_size

        if hasattr(self.encoder.config, "num_mel_bins"):
            self.config.audio_config.num_mel_bins = self.encoder.config.num_mel_bins

        feature_extractor = self.feature_extractor
        tokenizer = self.tokenizer
        del self.feature_extractor
        del self.tokenizer

        try:
            super().save_pretrained(save_dir, **kwargs)
        finally:
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer

        self.tokenizer.save_pretrained(save_dir)

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
