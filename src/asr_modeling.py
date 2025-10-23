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


class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        hidden_dim = config.projector_hidden_dim
        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim

        # SwiGLU layers, following the Llama architecture
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=False)

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

        x = x.contiguous().view(batch_size, -1, dim * self.k)

        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gated_output = F.silu(gate) * up  # Swish activation on the gate pathway

        return self.down_proj(gated_output)


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

        config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        cls._is_loading_from_pretrained = True
        cls._pretrained_model_path = pretrained_model_name_or_path

        try:
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

            model = cls(config, peft_config=peft_config)

            projector_path = cached_file(
                pretrained_model_name_or_path,
                "projector.safetensors",
            )
            projector_state = load_file(projector_path)
            projector_state = {
                k.replace("projector.", ""): v
                for k, v in projector_state.items()
                if k.startswith("projector.")
            }

            if projector_state:
                model.projector.load_state_dict(projector_state, strict=True)

            if peft_config:
                adapter_file = cached_file(
                    pretrained_model_name_or_path,
                    "adapter_model.safetensors",
                    _raise_exceptions_for_missing_entries=False,
                )

                if adapter_file:
                    from peft import PeftModel

                    print(f"Loading LoRA adapters from {pretrained_model_name_or_path}")
                    model.decoder = PeftModel.from_pretrained(
                        model.decoder,
                        pretrained_model_name_or_path,
                        is_trainable=True,  # Keep adapters trainable for continued training
                    )
                else:
                    print("No LoRA adapters found, initializing fresh LoRA weights")

            return model
        finally:
            cls._is_loading_from_pretrained = False
            del cls._pretrained_model_path

    def __init__(self, config: ASRConfig, **kwargs):
        super().__init__(config)

        peft_config = kwargs.pop("peft_config", None)

        self.system_prompt = config.system_prompt
        self.peft_config = peft_config

        target_dtype = getattr(torch, config.model_dtype)

        self.encoder = AutoModel.from_pretrained(
            config.audio_model_id,
            attn_implementation=config.attn_implementation,
            dtype=target_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.encoder.requires_grad_(False)

        self.decoder = AutoModelForCausalLM.from_pretrained(
            config.text_model_id,
            attn_implementation=config.attn_implementation,
            dtype=target_dtype,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.decoder.requires_grad_(False)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.audio_model_id)

        self.generation_config = self.decoder.generation_config
        self.generation_config.num_beams = config.num_beams

        # Initialize tokenizer and resize embeddings BEFORE applying LoRA
        self._init_tokenizer()

        if peft_config and peft_config.get("peft_method") == "lora":
            self._apply_lora(peft_config)

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

    def _apply_lora(self, peft_config: dict):
        """Apply LoRA adapters to the decoder model."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError(
                "PEFT library is required for LoRA fine-tuning. Install with: pip install peft"
            ) from None

        target_modules = peft_config.get("target_modules", ["q_proj", "v_proj"])

        if target_modules == "all-linear":
            target_modules = "all-linear"

        # Note: We exclude embedding and lm_head layers to avoid issues with vocab resizing
        lora_config = LoraConfig(
            r=peft_config.get("r", 8),
            lora_alpha=peft_config.get("lora_alpha", 32),
            target_modules=target_modules,
            lora_dropout=peft_config.get("lora_dropout", 0.05),
            bias=peft_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=None,  # Don't save any modules, just apply LoRA
        )

        self.decoder = get_peft_model(self.decoder, lora_config)
        self.decoder.print_trainable_parameters()

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

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable/disable gradient checkpointing for the decoder."""
        if isinstance(module, type(self.decoder)):
            module.gradient_checkpointing_enable() if value else module.gradient_checkpointing_disable()

    def _init_tokenizer(self):
        model_path = (
            self.__class__._pretrained_model_path
            if self._is_loading_from_pretrained
            else self.config.text_model_id
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        existing_special = self.tokenizer.additional_special_tokens or []
        tokens_to_add = []
        if "<|audio_start|>" not in existing_special:
            tokens_to_add.append("<|audio_start|>")
        if "<|audio_end|>" not in existing_special:
            tokens_to_add.append("<|audio_end|>")

        if tokens_to_add:
            special_tokens = {"additional_special_tokens": existing_special + tokens_to_add}
            num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
            if num_added_tokens > 0:
                # Use mean_resizing=False since these are structural tokens, not semantic ones
                self.decoder.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
                print(
                    f"Added {num_added_tokens} special tokens, vocab size now: {len(self.tokenizer)}"
                )

        current_embed_size = self.decoder.get_input_embeddings().weight.shape[0]
        expected_size = len(self.tokenizer)
        if current_embed_size != expected_size:
            print(f"Resizing embeddings from {current_embed_size} to {expected_size}")
            self.decoder.resize_token_embeddings(expected_size, mean_resizing=False)

        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")

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
        """Only save trainable projector parameters for efficient checkpointing.

        This prevents saving frozen encoder/decoder weights in training checkpoints,
        reducing checkpoint size from ~10GB to <100MB.

        Note: LoRA adapters are saved separately by PEFT's integration with Trainer.
        """
        state = {}
        state.update({f"projector.{k}": v for k, v in self.projector.state_dict().items()})
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
            audio_attention_mask = kwargs.pop("audio_attention_mask", None)
            audio_embeds = self._encode_audio(
                input_values=input_values,
                audio_attention_mask=audio_attention_mask,
            )

            batch_size = input_ids.shape[0]
            audio_seq_len = audio_embeds.shape[1]

            # Validate audio token IDs before using them
            if self.audio_start_id is None or self.audio_end_id is None:
                raise ValueError(
                    f"Audio tokens not properly initialized. Start: {self.audio_start_id}, End: {self.audio_end_id}"
                )

            vocab_size = self.decoder.get_input_embeddings().weight.shape[0]
            if self.audio_start_id >= vocab_size or self.audio_end_id >= vocab_size:
                raise ValueError(
                    f"Audio token IDs out of range. Start: {self.audio_start_id}, End: {self.audio_end_id}, "
                    f"Vocab size: {vocab_size}"
                )

            audio_start_positions = (input_ids == self.audio_start_id).nonzero(as_tuple=True)
            audio_end_positions = (input_ids == self.audio_end_id).nonzero(as_tuple=True)

            if len(audio_start_positions[0]) == 0 or len(audio_end_positions[0]) == 0:
                raise ValueError(
                    "Audio boundary tokens <|audio_start|> and <|audio_end|> must be present"
                )

            text_embeds = self.decoder.get_input_embeddings()(input_ids)

            new_embeds = []
            new_labels: list[torch.Tensor] = [] if labels is not None else None
            new_attention = []

            for i in range(batch_size):
                start_mask = audio_start_positions[0] == i
                end_mask = audio_end_positions[0] == i

                if not start_mask.any() or not end_mask.any():
                    raise ValueError(f"Missing audio boundaries in batch item {i}")

                start_pos = audio_start_positions[1][start_mask][0].item()
                end_pos = audio_end_positions[1][end_mask][0].item()

                before_audio = text_embeds[i, : start_pos + 1]
                after_audio = text_embeds[i, end_pos:]

                batch_embeds = torch.cat([before_audio, audio_embeds[i], after_audio], dim=0)
                new_embeds.append(batch_embeds)

                if labels is not None:
                    before_labels = labels[i, : start_pos + 1]
                    audio_labels = torch.full(
                        (audio_seq_len,), -100, dtype=labels.dtype, device=labels.device
                    )
                    after_labels = labels[i, end_pos:]
                    batch_labels = torch.cat([before_labels, audio_labels, after_labels], dim=0)
                    new_labels.append(batch_labels)

                if attention_mask is not None:
                    before_attn = attention_mask[i, : start_pos + 1]
                    audio_attn = torch.ones(
                        audio_seq_len, dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    after_attn = attention_mask[i, end_pos:]
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
                "content": "Transcribe: <|audio_start|><|audio_end|>",
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

        audio_start_positions = (prompt_ids == self.audio_start_id).nonzero(as_tuple=True)
        audio_end_positions = (prompt_ids == self.audio_end_id).nonzero(as_tuple=True)

        if len(audio_start_positions[0]) == 0 or len(audio_end_positions[0]) == 0:
            raise ValueError("Audio boundary tokens not found in prompt")

        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids)

        new_embeds = []
        for i in range(batch_size):
            start_mask = audio_start_positions[0] == i
            end_mask = audio_end_positions[0] == i

            if not start_mask.any() or not end_mask.any():
                raise ValueError(f"Missing audio boundaries in batch item {i}")

            start_pos = audio_start_positions[1][start_mask][0].item()
            end_pos = audio_end_positions[1][end_mask][0].item()

            before_audio = prompt_embeds[i, : start_pos + 1]
            after_audio = prompt_embeds[i, end_pos:]

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

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
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

        projector_state = {"projector." + k: v for k, v in self.projector.state_dict().items()}
        save_file(projector_state, save_dir / "projector.safetensors")

        if self.peft_config and self.peft_config.get("peft_method") == "lora":
            if hasattr(self.decoder, "save_pretrained"):
                self.decoder.save_pretrained(save_dir)

            import json

            peft_config_path = save_dir / "peft_config.json"
            with peft_config_path.open("w") as f:
                json.dump(self.peft_config, f, indent=2)

        self.tokenizer.save_pretrained(save_dir)
        self.feature_extractor.save_pretrained(save_dir)
        self.get_processor().save_pretrained(save_dir)

        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)


AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)
