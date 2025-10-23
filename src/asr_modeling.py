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

        # Initialize weights
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

        # Reshape for temporal compression
        x = x.contiguous().view(batch_size, -1, dim * self.k)

        # Apply SwiGLU block
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gated_output = F.silu(gate) * up  # Swish activation on the gate pathway

        return self.down_proj(gated_output)


class ASRModel(PreTrainedModel):
    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _keys_to_ignore_on_save = ["encoder", "decoder.base_model"]
    _is_loading_from_pretrained: bool = False
    _pretrained_model_path: Optional[str] = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        import json
        from pathlib import Path as PathlibPath

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        cls._is_loading_from_pretrained = True
        cls._pretrained_model_path = pretrained_model_name_or_path

        try:
            # Check if PEFT config exists (for LoRA models)
            peft_config = None
            local_path = PathlibPath(pretrained_model_name_or_path)

            if local_path.exists():
                peft_config_path = local_path / "peft_config.json"
                if peft_config_path.exists():
                    with peft_config_path.open() as f:
                        peft_config = json.load(f)

            # Initialize model with PEFT config if found
            model = cls(config, peft_config=peft_config)

            # Load projector weights from dedicated file
            if local_path.exists():
                # Local path
                projector_path = local_path / "projector.safetensors"
                # Fallback for old format
                if not projector_path.exists():
                    projector_path = local_path / "model.safetensors"
                    if projector_path.exists():
                        print("Note: Loading from old format (model.safetensors)")
            else:
                # Download from HuggingFace Hub
                try:
                    projector_path = hf_hub_download(
                        repo_id=pretrained_model_name_or_path, filename="projector.safetensors"
                    )
                except Exception:
                    # Fallback for old format
                    projector_path = hf_hub_download(
                        repo_id=pretrained_model_name_or_path, filename="model.safetensors"
                    )
                    print("Note: Loading from old format (model.safetensors)")

            projector_state = load_file(projector_path)
            # Extract projector weights
            projector_state = {
                k.replace("projector.", ""): v
                for k, v in projector_state.items()
                if k.startswith("projector.")
            }

            # Load projector weights strictly
            if projector_state:
                model.projector.load_state_dict(projector_state, strict=True)

            # Load LoRA adapters using PEFT's standard method
            if peft_config:
                # Try to load adapter files
                adapter_exists = False

                if local_path.exists():
                    # Check local path
                    adapter_model_path = local_path / "adapter_model.safetensors"
                    if not adapter_model_path.exists():
                        adapter_model_path = local_path / "adapter_model.bin"
                    adapter_exists = adapter_model_path.exists()
                else:
                    # Check Hub for adapter files
                    try:
                        from huggingface_hub import list_repo_files

                        files = list_repo_files(pretrained_model_name_or_path)
                        adapter_exists = any(
                            f in files for f in ["adapter_model.safetensors", "adapter_model.bin"]
                        )
                    except Exception:
                        pass

                if adapter_exists:
                    from peft import PeftModel

                    print(f"Loading LoRA adapters from {pretrained_model_name_or_path}")
                    # Load LoRA adapters using PEFT's standard method
                    model.decoder = PeftModel.from_pretrained(
                        model.decoder,
                        pretrained_model_name_or_path if not local_path.exists() else local_path,
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

        # Extract peft_config from kwargs
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

        # Apply LoRA AFTER tokenizer initialization and embedding resize
        if peft_config and peft_config.get("peft_method") == "lora":
            self._apply_lora(peft_config)

        from types import SimpleNamespace

        # Use dimensions from config (required)
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

        # Handle target_modules configuration
        target_modules = peft_config.get("target_modules", ["q_proj", "v_proj"])

        # If "all-linear" is specified, let PEFT automatically find all linear layers
        if target_modules == "all-linear":
            # For recent PEFT versions, this targets all Linear layers
            target_modules = "all-linear"

        # Create LoRA configuration
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

        # Apply LoRA to decoder
        self.decoder = get_peft_model(self.decoder, lora_config)

        # Enable LoRA adapters for training
        self.decoder.print_trainable_parameters()

    def _init_tokenizer(self):
        model_path = (
            self.__class__._pretrained_model_path
            if self._is_loading_from_pretrained
            else self.config.text_model_id
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Add special audio boundary tokens if they don't exist
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
                # Resize model embeddings to account for new tokens
                # Use mean_resizing=False since these are structural tokens, not semantic ones
                self.decoder.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
                print(
                    f"Added {num_added_tokens} special tokens, vocab size now: {len(self.tokenizer)}"
                )

        # Always ensure embeddings match tokenizer size (important for loaded models)
        current_embed_size = self.decoder.get_input_embeddings().weight.shape[0]
        expected_size = len(self.tokenizer)
        if current_embed_size != expected_size:
            print(f"Resizing embeddings from {current_embed_size} to {expected_size}")
            self.decoder.resize_token_embeddings(expected_size, mean_resizing=False)

        # Store audio token IDs for easy access
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

            # Find positions of <|audio_start|> and <|audio_end|> tokens
            audio_start_positions = (input_ids == self.audio_start_id).nonzero(as_tuple=True)
            audio_end_positions = (input_ids == self.audio_end_id).nonzero(as_tuple=True)

            if len(audio_start_positions[0]) == 0 or len(audio_end_positions[0]) == 0:
                raise ValueError(
                    "Audio boundary tokens <|audio_start|> and <|audio_end|> must be present"
                )

            # Get text embeddings
            text_embeds = self.decoder.get_input_embeddings()(input_ids)

            # Build new embedding sequences with audio embeddings between the boundary tokens
            new_embeds = []
            new_labels: list[torch.Tensor] = [] if labels is not None else None
            new_attention = []

            for i in range(batch_size):
                # Find audio boundaries for this batch item
                start_mask = audio_start_positions[0] == i
                end_mask = audio_end_positions[0] == i

                if not start_mask.any() or not end_mask.any():
                    raise ValueError(f"Missing audio boundaries in batch item {i}")

                start_pos = audio_start_positions[1][start_mask][0].item()
                end_pos = audio_end_positions[1][end_mask][0].item()

                # Build sequence: [..., <|audio_start|>, audio_embeds, <|audio_end|>, ...]
                before_audio = text_embeds[i, : start_pos + 1]  # Include audio_start token
                after_audio = text_embeds[
                    i, end_pos:
                ]  # Include audio_end token and everything after

                batch_embeds = torch.cat([before_audio, audio_embeds[i], after_audio], dim=0)
                new_embeds.append(batch_embeds)

                # Handle labels if present
                if labels is not None:
                    before_labels = labels[i, : start_pos + 1]
                    # Audio embeddings are always masked
                    audio_labels = torch.full(
                        (audio_seq_len,), -100, dtype=labels.dtype, device=labels.device
                    )
                    after_labels = labels[i, end_pos:]
                    batch_labels = torch.cat([before_labels, audio_labels, after_labels], dim=0)
                    new_labels.append(batch_labels)

                # Handle attention mask
                if attention_mask is not None:
                    before_attn = attention_mask[i, : start_pos + 1]
                    audio_attn = torch.ones(
                        audio_seq_len, dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    after_attn = attention_mask[i, end_pos:]
                    batch_attn = torch.cat([before_attn, audio_attn, after_attn], dim=0)
                    new_attention.append(batch_attn)

            # Stack all batches
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

        # Use model's default system prompt if none provided
        if system_prompt is None:
            system_prompt = self.system_prompt

        # Apply chat template with audio boundary tokens
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

        # Expand to batch size if needed
        if prompt_ids.shape[0] == 1 and batch_size > 1:
            prompt_ids = prompt_ids.expand(batch_size, -1)

        # Find positions of audio boundary tokens
        audio_start_positions = (prompt_ids == self.audio_start_id).nonzero(as_tuple=True)
        audio_end_positions = (prompt_ids == self.audio_end_id).nonzero(as_tuple=True)

        if len(audio_start_positions[0]) == 0 or len(audio_end_positions[0]) == 0:
            raise ValueError("Audio boundary tokens not found in prompt")

        # Get text embeddings
        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids)

        # Insert audio embeddings between boundary tokens for each batch item
        new_embeds = []
        for i in range(batch_size):
            # Find audio boundaries for this batch item
            start_mask = audio_start_positions[0] == i
            end_mask = audio_end_positions[0] == i

            if not start_mask.any() or not end_mask.any():
                raise ValueError(f"Missing audio boundaries in batch item {i}")

            start_pos = audio_start_positions[1][start_mask][0].item()
            end_pos = audio_end_positions[1][end_mask][0].item()

            # Build sequence with audio embeddings between boundaries
            before_audio = prompt_embeds[i, : start_pos + 1]  # Include start token
            after_audio = prompt_embeds[i, end_pos:]  # Include end token

            batch_embeds = torch.cat([before_audio, audio_embeds[i], after_audio], dim=0)
            new_embeds.append(batch_embeds)

        inputs_embeds = torch.stack(new_embeds)

        # Create attention mask for full input
        total_seq_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(batch_size, total_seq_len, dtype=torch.long, device=device)

        generate_kwargs.setdefault(
            "max_new_tokens", 150
        )  # Increased from 120 to handle longest samples (~95 words)
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

        # Update config dimensions before saving
        self.config.text_config.vocab_size = self.decoder.config.vocab_size
        self.config.text_config.hidden_size = self.decoder.config.hidden_size
        self.config.vocab_size = self.decoder.config.vocab_size
        self.config.hidden_size = self.decoder.config.hidden_size
        self.config.pad_token_id = self.decoder.config.pad_token_id
        self.config.text_config.bos_token_id = self.tokenizer.bos_token_id
        self.config.text_config.eos_token_id = self.tokenizer.eos_token_id
        self.config.text_config.pad_token_id = self.tokenizer.pad_token_id

        if not hasattr(self.config, "encoder_dim") or self.config.encoder_dim is None:
            self.config.encoder_dim = self.encoder.config.hidden_size
        if not hasattr(self.config, "llm_dim") or self.config.llm_dim is None:
            self.config.llm_dim = self.decoder.config.hidden_size
        if (
            not hasattr(self.config, "projector_hidden_dim")
            or self.config.projector_hidden_dim is None
        ):
            self.config.projector_hidden_dim = self.projector.gate_proj.out_features

        # Call parent's save_pretrained to handle config saving
        super().save_pretrained(save_directory, **kwargs)

        # Save projector weights
        projector_state = {"projector." + k: v for k, v in self.projector.state_dict().items()}
        save_file(projector_state, save_dir / "projector.safetensors")

        # Save LoRA adapters
        if self.peft_config and self.peft_config.get("peft_method") == "lora":
            if hasattr(self.decoder, "save_pretrained"):
                self.decoder.save_pretrained(save_dir)

            import json

            peft_config_path = save_dir / "peft_config.json"
            with peft_config_path.open("w") as f:
                json.dump(self.peft_config, f, indent=2)

        # Save tokenizer and feature extractor
        self.tokenizer.save_pretrained(save_dir)
        self.feature_extractor.save_pretrained(save_dir)
        self.get_processor().save_pretrained(save_dir)

        # Copy source files
        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)


AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)
