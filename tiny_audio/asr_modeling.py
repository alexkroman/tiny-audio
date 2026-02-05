import json
from pathlib import Path
from threading import Thread
from typing import Iterator, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TextIteratorStreamer,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .asr_config import ASRConfig
    from .projectors import PROJECTOR_CLASSES
except ImportError:
    from asr_config import ASRConfig  # type: ignore[no-redef]
    from projectors import PROJECTOR_CLASSES  # type: ignore[no-redef]


from torchaudio.transforms import SpecAugment


class ASRModel(PreTrainedModel, GenerationMixin):
    """Audio-to-text model combining an audio encoder, projector, and language model."""

    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _is_loading_from_pretrained: bool = False
    _pretrained_model_path: Optional[str] = None

    TRANSCRIBE_PROMPT = ""

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
            cls._pretrained_model_path = None

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

        # SpecAugment for data augmentation during training
        if getattr(config, "use_specaugment", False):
            self.spec_augment = SpecAugment(
                n_time_masks=config.num_time_masks,
                time_mask_param=config.time_mask_length,
                n_freq_masks=config.num_freq_masks,
                freq_mask_param=config.freq_mask_length,
            )
        else:
            self.spec_augment = None

        # Audio head for S2S (flow matching)
        if getattr(config, "use_audio_head", False):
            from .audio_head import AudioHead

            device = next(self.language_model.parameters()).device
            llm_dim = self.language_model.config.hidden_size

            self.audio_head = AudioHead(config, llm_dim=llm_dim).to(
                device=device, dtype=target_dtype
            )

            if getattr(config, "freeze_audio_head", False):
                self.audio_head.requires_grad_(False)
        else:
            self.audio_head = None

        # For model parallelism
        self._no_split_modules = getattr(self.language_model, "_no_split_modules", [])

    def _create_feature_extractor(self, config: ASRConfig):
        """Create the appropriate feature extractor for the audio encoder."""
        from transformers import AutoFeatureExtractor

        feature_extractor = AutoFeatureExtractor.from_pretrained(config.audio_model_id)
        # Disable padding by default - use actual audio length
        feature_extractor.padding = False
        return feature_extractor

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

    def _init_weights(self, _module):
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
        """Save trainable weights (projector + audio_head if present)."""
        state = {f"projector.{k}": v for k, v in self.projector.state_dict().items()}
        if self.audio_head is not None:
            state.update({f"audio_head.{k}": v for k, v in self.audio_head.state_dict().items()})
        return state

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
        expected_token_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode audio and project to LLM embedding space.

        Args:
            audio_features: Mel spectrogram features (batch, n_mels, mel_len)
            audio_attention_mask: Mask indicating real vs padded mel frames (batch, mel_len)
            expected_token_counts: Expected number of audio tokens per sample from input_ids.
                If provided, output will match these counts exactly (padding/truncating as needed).

        Returns:
            Flattened audio embeddings of shape (total_audio_tokens, hidden_dim).
        """
        with torch.no_grad():
            encoder_out = self.audio_tower(input_features=audio_features)
            hidden_states = encoder_out.last_hidden_state

        # Project to LLM space
        audio_embeds = self.projector(hidden_states)

        # Use expected token counts if provided (from input_ids), otherwise compute from audio
        if expected_token_counts is not None:
            token_counts = expected_token_counts
        else:
            # Compute per-sample encoder output lengths using conv formulas
            encoder_lengths = self._compute_encoder_output_lengths(audio_attention_mask)
            token_counts = torch.tensor(
                [
                    self.projector.get_output_length(int(length.item()))
                    for length in encoder_lengths
                ],
                device=audio_embeds.device,
            )

        # Extract embeddings matching expected token counts per sample
        batch_size = audio_embeds.shape[0]
        hidden_dim = audio_embeds.shape[2]

        result_embeds = []
        for i in range(batch_size):
            count = int(token_counts[i].item())
            sample_embeds = audio_embeds[i, :count, :]  # Take first 'count' embeddings
            # Pad with zeros if we don't have enough embeddings
            if sample_embeds.shape[0] < count:
                padding = torch.zeros(
                    count - sample_embeds.shape[0],
                    hidden_dim,
                    device=audio_embeds.device,
                    dtype=audio_embeds.dtype,
                )
                sample_embeds = torch.cat([sample_embeds, padding], dim=0)
            result_embeds.append(sample_embeds)

        return torch.cat(result_embeds, dim=0)

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
        codec_targets: Optional[torch.Tensor] = None,
        codec_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass for training and inference."""
        # Get text embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            # Apply SpecAugment during training if enabled
            if self.training and self.spec_augment is not None:
                input_features = self.spec_augment(input_features)

            # Count expected audio tokens from input_ids (ground truth from collator)
            audio_token_counts = (input_ids == self.audio_token_id).sum(dim=-1)

            # Encode audio -> flattened (total_audio_tokens, hidden_dim)
            audio_embeds = self._encode_audio(
                input_features, audio_attention_mask, audio_token_counts
            )

            # Replace <audio> token placeholders with audio embeddings using masked_scatter
            audio_token_mask = (input_ids == self.audio_token_id).unsqueeze(-1)

            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device),
                audio_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

        # Request hidden states if training audio head with codec targets
        if self.audio_head is not None and codec_targets is not None:
            kwargs["output_hidden_states"] = True

        # Remove TRL-specific keys that shouldn't go to the LLM
        kwargs.pop("prompts", None)
        kwargs.pop("prompt_attention_mask", None)

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

        # Compute audio head loss if training S2S with codec targets
        if self.audio_head is not None and codec_targets is not None:
            # Use token embeddings instead of hidden states for conditioning
            # This is simpler and more similar to how pocket-tts conditions on text
            # Note: input_ids is a direct argument to forward(), not in kwargs
            if input_ids is None:
                raise ValueError("input_ids required for audio head training")

            # Get embeddings for all tokens
            all_embeddings = self.language_model.get_input_embeddings()(input_ids)

            # Extract only assistant-position embeddings using assistant_mask
            # This mask identifies text output positions (where LLM generates response)
            assistant_mask = kwargs.get("assistant_mask")
            if assistant_mask is not None:
                batch_size = all_embeddings.shape[0]

                # Extract assistant embeddings for each sample
                assistant_emb_list = []
                for i in range(batch_size):
                    mask_i = assistant_mask[i]  # [seq_len]
                    emb_i = all_embeddings[i][mask_i]  # [num_assistant_tokens, embed_dim]
                    assistant_emb_list.append(emb_i)

                # Pad sequences
                embeddings = torch.nn.utils.rnn.pad_sequence(
                    assistant_emb_list, batch_first=True, padding_value=0.0
                )
            else:
                embeddings = all_embeddings

            # Compute loss: embeddings condition the AR decoder to generate codec_targets
            audio_head_loss = self.audio_head(
                embeddings,
                codec_targets=codec_targets,
                codec_lengths=codec_lengths,
            )

            # Combine with LLM loss if present (e.g., joint ASR+S2S training)
            if outputs.loss is not None:
                total_loss = outputs.loss + audio_head_loss
            else:
                total_loss = audio_head_loss

            # Return new output object (direct assignment doesn't work with Accelerator/DDP)
            from transformers.modeling_outputs import CausalLMOutputWithPast

            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

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
                enable_thinking=getattr(self.config, "enable_thinking", False),
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
        # Pass both input_ids and inputs_embeds so repetition_penalty works correctly
        # (it needs input_ids to track which tokens have been used)
        output = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            **generate_kwargs,
        )

        # When using inputs_embeds with input_ids, generate returns full sequence
        # Strip the input tokens to return only generated tokens
        sequences = output if isinstance(output, torch.Tensor) else output.sequences
        input_len = input_ids.shape[1]
        return sequences[:, input_len:]

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

        # Encode audio -> flattened embeddings
        audio_embeds = self._encode_audio(input_features, audio_attention_mask)

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
            enable_thinking=getattr(self.config, "enable_thinking", False),
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

    @torch.no_grad()
    def generate_text_only(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> str:
        """Generate text using only the LLM (no audio encoding).

        Used for SIFT-style response generation from metadata prompts.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            max_new_tokens: Maximum tokens to generate
            **generate_kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        device = next(self.language_model.parameters()).device

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=getattr(self.config, "enable_thinking", False),
        ).to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        attention_mask = torch.ones_like(input_ids)

        # Generate using language model directly
        output = self.language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )

        # Decode only the new tokens
        new_tokens = output[0, input_ids.shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def _process_audio(
        self,
        audio,
        sampling_rate: int = 16000,
    ) -> dict[str, torch.Tensor]:
        """Process raw audio waveform to model inputs."""
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Get mel features from feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )

        device = next(self.language_model.parameters()).device
        return {
            "input_features": inputs["input_features"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
        }

    @torch.no_grad()
    def generate_with_audio(
        self,
        audio,
        sampling_rate: int = 16000,
        **generate_kwargs,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Generate text and audio for Speech-to-Speech.

        Uses LLM token embeddings (not hidden states) to condition the flow model.
        This is simpler and more similar to how pocket-tts conditions on text.

        Args:
            audio: Raw audio waveform (numpy array or tensor) at given sampling_rate
            sampling_rate: Audio sampling rate (default 16kHz)
            **generate_kwargs: Additional generation arguments

        Returns:
            Dict with:
                - text: Decoded text strings (list of str)
                - audio: Audio waveform at 24kHz (batch, samples)
        """
        if self.audio_head is None:
            raise ValueError("Audio head not configured. Set use_audio_head=True in config.")

        inputs = self._process_audio(audio, sampling_rate)
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]

        device = input_features.device
        batch_size = input_features.shape[0]

        # Encode audio -> flattened embeddings
        audio_embeds = self._encode_audio(input_features, attention_mask)

        # Build prompt with correct number of audio tokens
        num_audio_tokens = self._get_num_audio_tokens(attention_mask)
        audio_placeholder = "<audio>" * num_audio_tokens

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_content = audio_placeholder
        if self.TRANSCRIBE_PROMPT:
            user_content += " " + self.TRANSCRIBE_PROMPT
        messages.append({"role": "user", "content": user_content})

        chat_result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=getattr(self.config, "enable_thinking", False),
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

        # Generate text response
        output = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            **generate_kwargs,
        )

        # Extract generated text
        text_ids = output[:, input_ids.shape[1] :]
        text = self.tokenizer.batch_decode(text_ids, skip_special_tokens=True)

        # Get embeddings for the generated text tokens
        # This is simpler than hidden states - just a lookup table
        embeddings = self.language_model.get_input_embeddings()(text_ids)

        # Generate Mimi codec codes from text embeddings via AR decoder
        codes, _ = self.audio_head(embeddings)

        # Load Mimi decoder if not already loaded
        if self.audio_head.mimi is None:
            self.audio_head.load_mimi_decoder(device=device)

        # Decode codes to audio waveform
        audio = self.audio_head.decode_to_audio(codes)

        return {
            "text": text,
            "audio": audio,
        }

    def generate_speech(
        self,
        text: str,
        system_prompt: str | None = None,
        **generate_kwargs,
    ) -> dict[str, torch.Tensor | str]:
        """Generate speech from text (Text-to-Speech).

        Args:
            text: Input text to speak
            system_prompt: Optional system prompt (defaults to self.system_prompt)
            **generate_kwargs: Additional generation arguments

        Returns:
            Dict with:
                - text: Generated response text (str)
                - audio: Audio waveform at 24kHz (batch, samples)
        """
        if self.audio_head is None:
            raise ValueError("Audio head not configured. Set use_audio_head=True in config.")

        device = next(self.language_model.parameters()).device

        # Build chat messages
        messages: list[dict[str, str]] = []
        prompt = system_prompt or self.system_prompt
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": text})

        # Tokenize
        chat_result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=getattr(self.config, "enable_thinking", False),
        )
        input_ids = chat_result.input_ids.to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Generate text response
        output = self.language_model.generate(
            input_ids,
            generation_config=self.generation_config,
            **generate_kwargs,
        )

        # Extract generated text
        text_ids = output[:, input_ids.shape[1] :]
        response_text: str = self.tokenizer.decode(text_ids[0], skip_special_tokens=True)

        # Get embeddings for the generated text tokens
        embeddings = self.language_model.get_input_embeddings()(text_ids)

        # Generate Mimi codec codes from text embeddings via AR decoder
        codes, _ = self.audio_head(embeddings)

        # Load Mimi decoder if not already loaded
        if self.audio_head.mimi is None:
            self.audio_head.load_mimi_decoder(device=device)

        # Decode codes to audio waveform
        audio = self.audio_head.decode_to_audio(codes)

        return {
            "text": response_text,
            "audio": audio,
        }

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs) -> None:
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
        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)
        # Copy projectors module
        shutil.copy(src_dir / "projectors.py", save_dir / "projectors.py")
        # Copy alignment module
        shutil.copy(src_dir / "alignment.py", save_dir / "alignment.py")
        # Copy diarization module
        shutil.copy(src_dir / "diarization.py", save_dir / "diarization.py")
        # Copy audio head for S2S
        audio_head_path = src_dir / "audio_head.py"
        if audio_head_path.exists():
            shutil.copy(audio_head_path, save_dir / "audio_head.py")
        # Copy modules directory (for audio head dependencies)
        modules_dir = src_dir / "modules"
        if modules_dir.exists():
            save_modules_dir = save_dir / "modules"
            save_modules_dir.mkdir(exist_ok=True)
            for module_file in modules_dir.glob("*.py"):
                shutil.copy(module_file, save_modules_dir / module_file.name)

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

    def create_or_update_model_card(self, output_dir: Union[str, Path]) -> None:
        """No-op for model card creation - we use MODEL_CARD.md in repo instead."""
        pass


# Register with transformers Auto classes
AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)
