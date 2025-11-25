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
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .asr_config import ASRConfig
from .asr_projector import MoEAudioProjector


class ASRModel(PreTrainedModel):
    """Audio-to-text model combining an audio encoder, projector, and language model."""

    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _is_loading_from_pretrained: bool = False
    _pretrained_model_path: Optional[str] = None

    TASK_PROMPTS = {
        "transcribe": "Transcribe: <audio>",
        "continue": "Continue: <audio>",
        "describe": "Describe: <audio>",
        "emotion": "Emotion: <audio>",
    }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model from pretrained, handling device placement correctly."""
        from safetensors.torch import load_file
        from transformers import AutoFeatureExtractor
        from transformers.utils.hub import cached_file

        config = kwargs.pop("config", None)
        if config is None:
            config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Set flag to avoid device_map="auto" in sub-model loaders
        cls._is_loading_from_pretrained = True
        cls._pretrained_model_path = pretrained_model_name_or_path

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

            return model
        finally:
            cls._is_loading_from_pretrained = False
            cls._pretrained_model_path = None

    def __init__(self, config: ASRConfig, **kwargs):
        super().__init__(config)

        self.system_prompt = config.system_prompt
        target_dtype = getattr(torch, config.model_dtype)

        # Audio encoder (frozen)
        self.audio_tower = self._load_audio_encoder(config, target_dtype)

        # Language model (frozen)
        self.language_model = self._load_language_model(config, target_dtype)
        self.generation_config = self.language_model.generation_config

        # Initialize tokenizer and special tokens
        self._init_tokenizer(config)

        # Feature extractor for audio preprocessing
        self.feature_extractor = self._create_feature_extractor(config)

        # Audio projector (trainable)
        self.projector = self._create_projector(config, target_dtype)

        # Loss function
        self.label_smoothing = getattr(config, "label_smoothing", 0.1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.label_smoothing)

        # For model parallelism
        self._no_split_modules = getattr(self.language_model, "_no_split_modules", [])

    def _create_feature_extractor(self, config: ASRConfig):
        """Create the appropriate feature extractor for the audio encoder."""
        from transformers import AutoFeatureExtractor

        return AutoFeatureExtractor.from_pretrained(config.audio_model_id)

    @classmethod
    def _load_audio_encoder(cls, config: ASRConfig, dtype: torch.dtype) -> nn.Module:
        """Load and freeze the audio encoder."""
        encoder_kwargs = {
            "attn_implementation": config.attn_implementation,
            "dtype": dtype,
        }
        # Only use device_map="auto" when NOT loading from pretrained
        # (avoids meta tensor conflicts during from_pretrained)
        if not cls._is_loading_from_pretrained:
            encoder_kwargs["device_map"] = "auto"

        if "whisper" in config.audio_model_id.lower():
            from transformers import WhisperModel

            full_model = WhisperModel.from_pretrained(config.audio_model_id, **encoder_kwargs)
            encoder = full_model.encoder
            del full_model
        else:
            encoder = AutoModel.from_pretrained(config.audio_model_id, **encoder_kwargs)

        encoder.requires_grad_(False)
        encoder.eval()
        return encoder

    @classmethod
    def _load_language_model(cls, config: ASRConfig, dtype: torch.dtype) -> PreTrainedModel:
        """Load and freeze the language model."""
        decoder_kwargs = {
            "attn_implementation": config.attn_implementation,
            "dtype": dtype,
            "trust_remote_code": True,
            "tie_word_embeddings": True,
        }
        # Only use device_map="auto" when NOT loading from pretrained
        if not cls._is_loading_from_pretrained:
            decoder_kwargs["device_map"] = "auto"

        decoder = AutoModelForCausalLM.from_pretrained(config.text_model_id, **decoder_kwargs)
        decoder.config.use_cache = getattr(config, "use_cache", True)
        decoder.requires_grad_(False)
        decoder.eval()
        return decoder

    def _create_projector(self, config: ASRConfig, dtype: torch.dtype) -> MoEAudioProjector:
        """Create the trainable audio projector."""
        # Auto-detect dimensions if not specified
        if config.encoder_dim is None:
            enc_cfg = self.audio_tower.config
            config.encoder_dim = getattr(enc_cfg, "hidden_size", None) or getattr(enc_cfg, "d_model", None)
            if config.encoder_dim is None:
                raise ValueError("Could not auto-detect encoder_dim. Please specify in config.")

        if config.llm_dim is None:
            dec_cfg = self.language_model.config
            config.llm_dim = getattr(dec_cfg, "hidden_size", None) or getattr(dec_cfg, "d_model", None)
            if config.llm_dim is None:
                raise ValueError("Could not auto-detect llm_dim. Please specify in config.")

        projector = MoEAudioProjector(config)
        return projector.to(dtype=dtype)

    def _init_tokenizer(self, config: ASRConfig):
        """Initialize tokenizer with audio token."""
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_id, trust_remote_code=True)

        # Set pad token
        if (
            self.tokenizer.pad_token is None
            or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ) and "<|finetune_right_pad_id|>" in self.tokenizer.get_vocab():
            self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        # Add audio token
        existing_special = self.tokenizer.additional_special_tokens or []
        if "<audio>" not in existing_special:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": existing_special + ["<audio>"]}
            )
            self.language_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<audio>")
        self.tokenizer.padding_side = "right"

        # Sync token IDs to configs
        for cfg in [self.config.text_config, self.language_model.config, self.generation_config]:
            if cfg is not None:
                cfg.pad_token_id = self.tokenizer.pad_token_id
                cfg.eos_token_id = self.tokenizer.eos_token_id
                cfg.bos_token_id = self.tokenizer.bos_token_id

    def _init_weights(self, module):
        """Weight initialization (projector weights are initialized in MoEAudioProjector)."""
        pass

    def can_generate(self) -> bool:
        return True

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.language_model.set_output_embeddings(value)

    def get_processor(self):
        """Get the processor for this model."""
        from .asr_processing import ASRProcessor

        return ASRProcessor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

    def state_dict(self, *args, **kwargs):
        """Only save trainable projector weights."""
        return {f"projector.{k}": v for k, v in self.projector.state_dict().items()}

    def _encode_audio(
        self,
        audio_features: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode audio and project to LLM embedding space."""
        with torch.no_grad():
            is_whisper = hasattr(self.audio_tower.config, "num_mel_bins")
            if is_whisper:
                encoder_out = self.audio_tower(
                    input_features=audio_features, attention_mask=audio_attention_mask
                )
            else:
                encoder_out = self.audio_tower(
                    input_values=audio_features, attention_mask=audio_attention_mask
                )
            hidden_states = encoder_out.last_hidden_state

        audio_embeds, aux_loss = self.projector(hidden_states)

        # Create attention mask for projected audio
        if audio_attention_mask is not None:
            mask_float = audio_attention_mask.unsqueeze(1).float()
            pooled_mask = F.interpolate(mask_float, size=audio_embeds.shape[1], mode="nearest")
            audio_mask = pooled_mask.squeeze(1).long()
        else:
            audio_mask = torch.ones(audio_embeds.shape[:2], device=audio_embeds.device, dtype=torch.long)

        return audio_embeds, aux_loss, audio_mask

    def _merge_audio_features(
        self,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Merge audio embeddings into text embeddings at <audio> token positions.

        Returns: (inputs_embeds, attention_mask, labels)
        """
        batch_size, seq_len = input_ids.shape
        num_audio_tokens = audio_embeds.shape[1]
        device = input_ids.device

        # Find audio token position in each sequence
        audio_positions = (input_ids == self.audio_token_id).int().argmax(dim=1)

        # Calculate new sequence length (replace 1 audio token with num_audio_tokens)
        new_seq_len = seq_len - 1 + num_audio_tokens

        # Get text embeddings
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Build merged embeddings
        merged_embeds = torch.zeros(
            batch_size, new_seq_len, text_embeds.shape[-1],
            device=device, dtype=text_embeds.dtype
        )
        merged_attention = torch.ones(batch_size, new_seq_len, device=device, dtype=torch.long)
        merged_labels = None
        if labels is not None:
            merged_labels = torch.full(
                (batch_size, new_seq_len), -100, device=device, dtype=labels.dtype
            )

        for i in range(batch_size):
            pos = audio_positions[i].item()

            # Before audio token
            if pos > 0:
                merged_embeds[i, :pos] = text_embeds[i, :pos]
                if attention_mask is not None:
                    merged_attention[i, :pos] = attention_mask[i, :pos]
                if merged_labels is not None and labels is not None:
                    merged_labels[i, :pos] = labels[i, :pos]

            # Audio embeddings
            audio_end = pos + num_audio_tokens
            merged_embeds[i, pos:audio_end] = audio_embeds[i]
            if audio_mask is not None:
                merged_attention[i, pos:audio_end] = audio_mask[i]

            # After audio token
            remaining = seq_len - pos - 1
            if remaining > 0:
                merged_embeds[i, audio_end:audio_end + remaining] = text_embeds[i, pos + 1:]
                if attention_mask is not None:
                    merged_attention[i, audio_end:audio_end + remaining] = attention_mask[i, pos + 1:]
                if merged_labels is not None and labels is not None:
                    merged_labels[i, audio_end:audio_end + remaining] = labels[i, pos + 1:]

        return merged_embeds, merged_attention, merged_labels

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass for training and inference."""
        # Accept either input_values (wav2vec2) or input_features (whisper)
        audio_inputs = input_features if input_features is not None else input_values

        if audio_inputs is not None:
            if input_ids is None:
                raise ValueError("input_ids required when audio is provided")

            # Encode audio
            audio_embeds, aux_loss, audio_mask = self._encode_audio(
                audio_inputs, audio_attention_mask
            )

            # Merge audio with text
            inputs_embeds, full_attention_mask, labels = self._merge_audio_features(
                input_ids, audio_embeds, attention_mask, labels, audio_mask
            )
        else:
            # Text-only forward
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            full_attention_mask = attention_mask
            aux_loss = torch.tensor(0.0, device=input_ids.device)

        # Run through language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            **kwargs,
        )

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss + aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        task: Optional[str] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate text from audio input."""
        audio_inputs = input_features if input_features is not None else input_values
        if audio_inputs is None:
            raise ValueError("input_values or input_features required for generation")

        device = audio_inputs.device
        batch_size = audio_inputs.shape[0]

        # Encode audio
        audio_embeds, _, audio_mask = self._encode_audio(audio_inputs, audio_attention_mask)

        # Build prompt
        system_prompt = system_prompt or self.system_prompt
        user_prompt = user_prompt or self.TASK_PROMPTS.get(task, self.config.user_prompt or "Transcribe: <audio>")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(device)

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if prompt_ids.shape[0] == 1 and batch_size > 1:
            prompt_ids = prompt_ids.expand(batch_size, -1)

        if not (prompt_ids == self.audio_token_id).any():
            raise ValueError("Audio token <audio> not found in prompt")

        # Merge audio with prompt
        inputs_embeds, attention_mask, _ = self._merge_audio_features(
            prompt_ids, audio_embeds, audio_mask=audio_mask
        )

        prompt_length = inputs_embeds.shape[1]

        # Set generation defaults
        generate_kwargs.setdefault("max_new_tokens", getattr(self.config, "max_new_tokens", 128))
        generate_kwargs.setdefault("use_cache", True)
        generate_kwargs.setdefault("eos_token_id", self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        # Generate (type ignore needed as generate() has complex return type)
        output = self.language_model.generate(  # type: ignore[operator]
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # Return only generated tokens (strip prompt)
        if isinstance(output, torch.Tensor):
            return output[:, prompt_length:]
        # Handle GenerateOutput types that have sequences attribute
        return output.sequences[:, prompt_length:]

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Save model, tokenizer, and processor."""
        import shutil
        from pathlib import Path as PathlibPath

        save_dir = PathlibPath(save_directory)
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

        # Save tokenizer and processor
        self.tokenizer.save_pretrained(save_dir)
        self.get_processor().save_pretrained(save_dir)

        # Copy source files for auto-loading
        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)


# Register with transformers Auto classes
AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)
