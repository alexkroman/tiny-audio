import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .asr_config import ASRConfig
    from .projectors import PROJECTOR_CLASSES
except ImportError:
    from asr_config import ASRConfig  # type: ignore[no-redef]
    from projectors import PROJECTOR_CLASSES  # type: ignore[no-redef]


class ASRModel(PreTrainedModel, GenerationMixin):
    """Audio-to-text model combining an audio encoder, projector, and language model."""

    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _is_loading_from_pretrained: bool = False
    _pretrained_model_path: Optional[str] = None

    TRANSCRIBE_PROMPT = "Transcribe: "

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model from pretrained, handling device placement correctly."""
        from pathlib import Path

        from safetensors.torch import load_file
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

            # Load LoRA adapter if present
            adapter_config = cached_file(
                pretrained_model_name_or_path,
                "adapter_config.json",
                _raise_exceptions_for_missing_entries=False,
                **cache_kwargs,
            )
            if adapter_config is not None:
                from peft import PeftModel

                # Get adapter directory (parent of adapter_config.json)
                adapter_path = str(Path(adapter_config).parent)
                model.language_model = PeftModel.from_pretrained(
                    model.language_model, adapter_path, is_trainable=False
                )

            return model
        finally:
            cls._is_loading_from_pretrained = False
            cls._pretrained_model_path = None

    def __init__(self, config: ASRConfig, **kwargs):
        super().__init__(config)

        self.system_prompt = config.system_prompt
        self.encoder_stride = config.encoder_stride
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
        self.generation_config.num_beams = config.num_beams
        self.generation_config.do_sample = False
        # Clear sampling params (inherited from LLM) since we use greedy decoding
        self.generation_config.temperature = None
        self.generation_config.top_p = None
        self.generation_config.top_k = None
        self.generation_config.use_cache = config.use_cache
        self.generation_config.length_penalty = config.length_penalty
        self.generation_config.repetition_penalty = config.repetition_penalty
        self.generation_config.no_repeat_ngram_size = config.no_repeat_ngram_size
        self.generation_config.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Feature extractor for audio preprocessing
        self.feature_extractor = self._create_feature_extractor(config)

        # Audio projector (trainable)
        self.projector = self._create_projector(config, target_dtype)

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
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
            "trust_remote_code": True,
            "tie_word_embeddings": False,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }

        decoder = AutoModelForCausalLM.from_pretrained(config.text_model_id, **decoder_kwargs)
        decoder.config.use_cache = getattr(config, "use_cache", True)
        decoder.requires_grad_(False)
        decoder.eval()
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
        existing_special = getattr(self.tokenizer, "additional_special_tokens", None) or []
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

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        """Enable/disable gradient checkpointing for the language model."""
        # The LLM still stores activations during forward for backprop to projector
        # Gradient checkpointing trades compute for memory by recomputing activations
        if hasattr(self.language_model, "_set_gradient_checkpointing"):
            self.language_model._set_gradient_checkpointing(enable, gradient_checkpointing_func)
        elif hasattr(self.language_model, "gradient_checkpointing_enable") and enable:
            self.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        elif hasattr(self.language_model, "gradient_checkpointing_disable") and not enable:
            self.language_model.gradient_checkpointing_disable()

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
        try:
            from .asr_processing import ASRProcessor
        except ImportError:
            from asr_processing import ASRProcessor  # type: ignore[no-redef]

        return ASRProcessor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            projector=self.projector,
            encoder_stride=self.encoder_stride,
            encoder_conv_layers=self.config.encoder_conv_layers,
        )

    def state_dict(self, *args, **kwargs):
        """Only save trainable projector weights."""
        return {f"projector.{k}": v for k, v in self.projector.state_dict().items()}

    def _compute_encoder_output_lengths(
        self,
        audio_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample encoder output lengths using conv layer formulas.

        Args:
            audio_attention_mask: Mask indicating real vs padded mel frames (batch, mel_len)

        Returns:
            Tensor of encoder output lengths per sample (batch,)
        """
        # Get mel frame lengths from attention mask
        lengths = audio_attention_mask.sum(dim=-1)

        # Apply conv layer formulas: output = (input + 2*pad - (kernel-1) - 1) // stride + 1
        for padding, kernel_size, stride in self.config.encoder_conv_layers:
            lengths = (lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        return lengths

    def _encode_audio(
        self,
        audio_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode audio and project to LLM embedding space.

        Args:
            audio_features: Mel spectrogram features (batch, n_mels, mel_len)
            audio_attention_mask: Mask indicating real vs padded mel frames (batch, mel_len)

        Returns:
            Flattened audio embeddings of shape (total_audio_tokens, hidden_dim).
        """
        with torch.no_grad():
            encoder_out = self.audio_tower(input_features=audio_features)
            hidden_states = encoder_out.last_hidden_state

        # Compute per-sample encoder output lengths using conv formulas
        encoder_lengths = self._compute_encoder_output_lengths(audio_attention_mask)

        # Project to LLM space
        audio_embeds = self.projector(hidden_states)

        # Compute per-sample projector output lengths
        projector_lengths = torch.tensor(
            [self.projector.get_output_length(int(length.item())) for length in encoder_lengths],
            device=audio_embeds.device,
        )

        # Create valid mask for variable-length samples and extract only real embeddings
        max_len = audio_embeds.shape[1]
        valid_mask = (
            torch.arange(max_len, device=audio_embeds.device)[None, :] < projector_lengths[:, None]
        )
        return audio_embeds[valid_mask]

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
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass for training and inference."""
        # Get text embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            # Encode audio -> flattened (total_audio_tokens, hidden_dim)
            audio_embeds = self._encode_audio(input_features, audio_attention_mask)

            # Replace <audio> token placeholders with audio embeddings using masked_scatter
            audio_token_mask = (input_ids == self.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device),
                audio_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

        # Run through language model (let it compute loss if labels provided)
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

        # Add auxiliary loss from MoE projectors if available
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
    ) -> torch.Tensor:
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

        # Encode audio -> flattened embeddings
        audio_embeds = self._encode_audio(input_features, audio_attention_mask)

        # If input_ids not provided, build prompt with correct number of audio tokens
        if input_ids is None:
            num_audio_tokens = self._get_num_audio_tokens(audio_attention_mask)
            audio_placeholder = "<audio>" * num_audio_tokens

            system_prompt = system_prompt or self.system_prompt

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": self.TRANSCRIBE_PROMPT + audio_placeholder})

            chat_result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
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

        # Generate using language model
        output = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            **generate_kwargs,
        )

        # When using inputs_embeds without input_ids, generate returns only new tokens
        if isinstance(output, torch.Tensor):
            return output
        return output.sequences

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

        # Save tokenizer and feature extractor
        self.tokenizer.save_pretrained(save_dir)
        self.feature_extractor.save_pretrained(save_dir)

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
        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)
        # Copy projectors module
        shutil.copy(src_dir / "projectors.py", save_dir / "projectors.py")


# Register with transformers Auto classes
AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)
