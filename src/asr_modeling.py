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
from transformers.models.llama.modeling_llama import LlamaRMSNorm

try:
    from .asr_config import ASRConfig
except ImportError:
    from asr_config import ASRConfig  # type: ignore[no-redef]


class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        hidden_dim = config.projector_hidden_dim
        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim

        # Pre-norm: normalize stacked encoder features (fixes broken normalization from concatenation)
        self.ln_pre = LlamaRMSNorm(in_dim)

        # SwiGLU layers, following the Llama architecture
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=False)

        # Post-norm: normalize output to match LLM's expected embedding distribution
        self.ln_post = LlamaRMSNorm(out_dim)

        with torch.no_grad():
            nn.init.normal_(self.gate_proj.weight, std=0.02)
            nn.init.normal_(self.up_proj.weight, std=0.02)
            nn.init.normal_(self.down_proj.weight, std=0.02)

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

        # SwiGLU projection
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gated_output = F.silu(gate) * up

        # Project to LLM dimension
        output = self.down_proj(gated_output)

        # Normalize before LLM to ensure stable input distribution
        return self.ln_post(output)


class ASRModel(PreTrainedModel):
    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
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

        cls._is_loading_from_pretrained = True
        cls._pretrained_model_path = pretrained_model_name_or_path

        try:
            # Load decoder LoRA config if it exists
            peft_config = None
            try:
                peft_config_file = cached_file(
                    pretrained_model_name_or_path,
                    "peft_config.json",
                    _raise_exceptions_for_missing_entries=False,
                )
                if peft_config_file:
                    from pathlib import Path as PathlibPath

                    with PathlibPath(peft_config_file).open() as f:
                        peft_config = json.load(f)
            except Exception:
                pass

            # Load encoder LoRA config if it exists
            encoder_lora_config = None
            try:
                encoder_lora_config_file = cached_file(
                    pretrained_model_name_or_path,
                    "encoder_lora_config.json",
                    _raise_exceptions_for_missing_entries=False,
                )
                if encoder_lora_config_file:
                    from pathlib import Path as PathlibPath

                    with PathlibPath(encoder_lora_config_file).open() as f:
                        encoder_lora_config = json.load(f)
            except Exception:
                pass

            model = cls(config, peft_config=peft_config, encoder_lora_config=encoder_lora_config)

            model_path = cached_file(
                pretrained_model_name_or_path,
                "model.safetensors",
            )
            model_state = load_file(model_path)
            projector_state = {
                k.replace("projector.", ""): v
                for k, v in model_state.items()
                if k.startswith("projector.")
            }

            if projector_state:
                model.projector.load_state_dict(projector_state, strict=True)

            # Load decoder LoRA adapters if config exists
            if peft_config:
                adapter_file = cached_file(
                    pretrained_model_name_or_path,
                    "adapter_model.safetensors",
                    _raise_exceptions_for_missing_entries=False,
                )

                if adapter_file:
                    from peft import PeftModel

                    print(f"Loading decoder LoRA adapters from {pretrained_model_name_or_path}")
                    model.decoder = PeftModel.from_pretrained(
                        model.decoder,
                        pretrained_model_name_or_path,
                        is_trainable=True,  # Keep adapters trainable for continued training
                    )
                else:
                    print("No decoder LoRA adapters found, initializing fresh LoRA weights")

            # Load encoder LoRA adapters if config exists
            if encoder_lora_config:
                try:
                    from peft import PeftModel

                    print(f"Loading encoder LoRA adapters from {pretrained_model_name_or_path}")
                    model.encoder = PeftModel.from_pretrained(
                        model.encoder,
                        pretrained_model_name_or_path,
                        subfolder="encoder_adapter",
                        is_trainable=True,
                    )
                except Exception as e:
                    print(f"No encoder LoRA adapters found ({e}), will initialize fresh if configured")

            return model
        finally:
            cls._is_loading_from_pretrained = False
            del cls._pretrained_model_path

    def __init__(self, config: ASRConfig, **kwargs):
        super().__init__(config)

        peft_config = kwargs.pop("peft_config", None)
        encoder_lora_config = kwargs.pop("encoder_lora_config", None)

        self.system_prompt = config.system_prompt
        self.peft_config = peft_config
        self.encoder_lora_config = encoder_lora_config

        self.encoder = self._create_encoder(config, encoder_lora_config)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.audio_model_id)

        # Create decoder first (needed for tokenizer init)
        self.decoder = self._create_decoder(config, peft_config)
        self.generation_config = self.decoder.generation_config
        self.generation_config.num_beams = config.num_beams

        # Initialize tokenizer and resize embeddings after decoder is created
        self._init_tokenizer()

        from types import SimpleNamespace

        if config.encoder_dim is None or config.llm_dim is None:
            raise ValueError(
                "encoder_dim and llm_dim must be specified in config. "
                "These dimensions should match the checkpoint weights. "
                "If loading an old model, please update the config file on HuggingFace Hub."
            )

        projector_config = SimpleNamespace(
            encoder_projector_ds_rate=config.audio_downsample_rate,
            encoder_dim=config.encoder_dim,
            llm_dim=config.llm_dim,
            projector_hidden_dim=config.projector_hidden_dim,
        )
        self.projector: AudioProjector = AudioProjector(projector_config)

        decoder_dtype = next(self.decoder.parameters()).dtype
        self.projector.to(dtype=decoder_dtype)

        self._no_split_modules = self.decoder._no_split_modules

    @staticmethod
    def _apply_lora(model, lora_config: dict, task_type, model_name: str = "model"):
        """Apply LoRA adapters to a model (encoder or decoder).

        Args:
            model: The model to apply LoRA to
            lora_config: Dict with LoRA configuration (r, lora_alpha, target_modules, etc.)
            task_type: peft.TaskType (FEATURE_EXTRACTION for encoder, CAUSAL_LM for decoder)
            model_name: Name for logging purposes
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
            lora_dropout=lora_config.get("lora_dropout", 0.0),
            bias=lora_config.get("bias", "none"),
            task_type=task_type,
            modules_to_save=lora_config.get("modules_to_save"),
            init_lora_weights=True,
        )

        print(f"Applying LoRA to {model_name} with r={peft_config.r}, alpha={peft_config.lora_alpha}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
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
        import types

        target_dtype = getattr(torch, config.model_dtype)

        encoder = AutoModel.from_pretrained(
            config.audio_model_id,
            attn_implementation=config.attn_implementation,
            dtype=target_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
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
            **kwargs  # Catch and discard invalid kwargs like input_ids
        ):
            return original_forward(
                input_values=input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        encoder.forward = types.MethodType(safe_encoder_forward, encoder)

        # Apply LoRA to encoder if configured (after wrapping base forward)
        if encoder_lora_config and encoder_lora_config.get("r", 0) > 0:
            from peft import TaskType
            encoder = cls._apply_lora(encoder, encoder_lora_config, TaskType.FEATURE_EXTRACTION, "encoder")

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

        decoder = AutoModelForCausalLM.from_pretrained(
            config.text_model_id,
            attn_implementation=config.attn_implementation,
            dtype=target_dtype,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        decoder.requires_grad_(False)

        # Apply LoRA to decoder if configured
        if peft_config and peft_config.get("peft_method") == "lora":
            from peft import TaskType
            decoder = cls._apply_lora(decoder, peft_config, TaskType.CAUSAL_LM, "decoder")

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

    def _set_gradient_checkpointing(self, module, value=False):
        """Control gradient checkpointing for the model.

        Only apply gradient checkpointing to the decoder, not the encoder.
        The encoder either has no gradients (frozen) or has LoRA adapters
        which handle gradients differently.
        """
        if hasattr(module, "gradient_checkpointing_enable"):
            if value:
                # Only enable gradient checkpointing on the decoder
                if module is self.decoder or (hasattr(self, 'decoder') and module is self.decoder.base_model):
                    module.gradient_checkpointing_enable()
            else:
                if hasattr(module, "gradient_checkpointing_disable"):
                    module.gradient_checkpointing_disable()

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
                print(
                    f"Added {num_added_tokens} audio token, vocab size now: {len(self.tokenizer)}"
                )

        current_embed_size = self.decoder.get_input_embeddings().weight.shape[0]
        expected_size = len(self.tokenizer)
        if current_embed_size != expected_size:
            print(f"Resizing embeddings from {current_embed_size} to {expected_size}")
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
        """Only save trainable parameters for efficient checkpointing.

        This prevents saving frozen encoder/decoder weights in training checkpoints.
        LoRA adapters are saved separately by PEFT's integration with Trainer.
        """
        full_state = super().state_dict(*args, **kwargs)
        return self.diff_state_dict(full_state)

    def diff_state_dict(self, state_dict=None):
        """Filter state dict to only include trainable parameters.

        This ensures minimal checkpoint size by only saving what's being trained.
        """
        if state_dict is None:
            state_dict = super().state_dict()

        # Get all trainable parameter names
        trainable_params = {k for k, v in self.named_parameters() if v.requires_grad}

        # Only keep trainable parameters
        return {k: v for k, v in state_dict.items() if k in trainable_params}

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
        # Only pass explicit valid arguments to encoder
        # Never use **kwargs to prevent torch.compile from injecting decoder args like input_ids
        # Don't use no_grad if encoder has LoRA (needs gradients for training)
        if self.encoder_lora_config and self.encoder_lora_config.get("r", 0) > 0:
            audio_features = self.encoder(
                input_values=input_values.to(self.encoder.dtype),
                attention_mask=audio_attention_mask,
            ).last_hidden_state
        else:
            with torch.no_grad():
                audio_features = self.encoder(
                    input_values=input_values.to(self.encoder.dtype),
                    attention_mask=audio_attention_mask,
                ).last_hidden_state

        return self.projector(audio_features)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_values is not None:
            # Extract audio-specific kwargs, don't pass input_ids to encoder
            audio_attention_mask = kwargs.pop("audio_attention_mask", None)

            # Remove any decoder-specific kwargs that shouldn't go to the encoder
            kwargs.pop("past_key_values", None)
            kwargs.pop("use_cache", None)

            audio_embeds = self._encode_audio(
                input_values=input_values,
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
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_values: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
        **generate_kwargs,
    ) -> Union[
        torch.Tensor,
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
    ]:
        if input_values is None:
            raise ValueError("input_values must be provided for generation")

        audio_embeds = self._encode_audio(input_values)
        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device

        if system_prompt is None:
            system_prompt = self.system_prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": "Repeat the following text, without any explanation: <audio>",
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

        generate_kwargs.setdefault("max_new_tokens", 150)
        generate_kwargs.setdefault("num_beams", self.config.num_beams)
        generate_kwargs.setdefault("do_sample", False)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        generate_kwargs.setdefault("eos_token_id", im_end_id)
        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        return self.decoder.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs
        )

    def merge_and_unload(self):
        """Merge LoRA weights into base model and remove PEFT wrappers.

        This creates cleaner model saves and enables faster inference without PEFT.
        """
        try:
            from peft import PeftModel
        except ImportError:
            return

        # Merge decoder LoRA
        if isinstance(self.decoder, PeftModel):
            self.decoder = self.decoder.merge_and_unload()
            self.peft_config = None

        # Merge encoder LoRA
        if isinstance(self.encoder, PeftModel):
            self.encoder = self.encoder.merge_and_unload()
            self.encoder_lora_config = None

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

        # Use diff_state_dict to only save trainable parameters
        state_dict = self.diff_state_dict()
        if state_dict:
            save_file(state_dict, save_dir / "model.safetensors")

        # Save decoder LoRA if configured
        if self.peft_config and self.peft_config.get("peft_method") == "lora":
            if hasattr(self.decoder, "save_pretrained"):
                self.decoder.save_pretrained(save_dir)
            with (save_dir / "peft_config.json").open("w") as f:
                json.dump(self.peft_config, f, indent=2)

        # Save encoder LoRA if configured
        if self.encoder_lora_config and self.encoder_lora_config.get("r", 0) > 0:
            if hasattr(self.encoder, "save_pretrained"):
                encoder_adapter_dir = save_dir / "encoder_adapter"
                encoder_adapter_dir.mkdir(exist_ok=True)
                self.encoder.save_pretrained(encoder_adapter_dir)
            with (save_dir / "encoder_lora_config.json").open("w") as f:
                json.dump(self.encoder_lora_config, f, indent=2)

        self.tokenizer.save_pretrained(save_dir)
        self.feature_extractor.save_pretrained(save_dir)
        self.get_processor().save_pretrained(save_dir)

        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)


AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)
