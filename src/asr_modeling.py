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
    def __init__(self, in_features, hidden_features, out_features, bias=False, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_gate = self.act(self.w1(x))
        x_val = self.w2(x)
        x = x_gate * x_val
        x = self.dropout(x)
        return self.w3(x)


class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = getattr(config, "projector_pool_stride", 2)  # Downsampling rate
        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim
        hidden_dim = config.projector_hidden_dim
        if hidden_dim is None:
            hidden_dim = config.encoder_dim * 4

        dropout_rate = getattr(config, "projector_dropout", 0.0)

        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        self.ln_pre = LlamaRMSNorm(in_dim, eps=1e-6)
        self.proj = SwiGLU(in_dim, hidden_dim, out_dim, dropout=dropout_rate)
        self.ln_post = LlamaRMSNorm(out_dim, eps=1e-6)
        self.output_dropout = nn.Dropout(dropout_rate)

        with torch.no_grad():
            std = getattr(config, "projector_init_std", 0.02)
            self.ln_pre.weight.data.fill_(1.0)
            self.ln_post.weight.data.fill_(1.0)
            nn.init.normal_(self.proj.w1.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj.w2.weight, mean=0.0, std=std)
            nn.init.normal_(self.proj.w3.weight, mean=0.0, std=std)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        target_dtype = self.proj.w1.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        remainder = seq_len % self.k
        if remainder:
            pad_len = self.k - remainder
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.contiguous().view(batch_size, -1, dim * self.k)
        x = self.ln_pre(x)
        x = self.proj(x)
        x = self.ln_post(x)

        return self.output_dropout(x)


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
    def _create_feature_extractor(audio_model_id: str):
        """Factory method to create the appropriate feature extractor."""
        is_whisper = "whisper" in audio_model_id.lower()
        if is_whisper:
            from transformers import WhisperConfig, WhisperFeatureExtractor

            encoder_config = WhisperConfig.from_pretrained(audio_model_id)
            num_mel_bins = encoder_config.num_mel_bins
            return WhisperFeatureExtractor.from_pretrained(
                audio_model_id,
                feature_size=num_mel_bins,
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
            self.feature_extractor = self._create_feature_extractor(config.audio_model_id)

        self.decoder = self._create_decoder(config)
        self.generation_config = self.decoder.generation_config

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

        projector_config = SimpleNamespace(
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            projector_pool_stride=getattr(config, "projector_pool_stride", 2),
            projector_hidden_dim=getattr(config, "projector_hidden_dim", None),
            projector_init_std=getattr(config, "projector_init_std", 0.02),
            projector_dropout=getattr(config, "projector_dropout", 0.0),
        )
        self.projector = AudioProjector(projector_config)

        target_dtype = getattr(torch, config.model_dtype)
        self.projector = self.projector.to(dtype=target_dtype)

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
        encoder_dtype = next(self.encoder.parameters()).dtype
        input_values = input_values.clone().to(device=encoder_device, dtype=encoder_dtype)

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

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
            use_cache=use_cache if use_cache is not None else False,
            **kwargs,
        )

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
