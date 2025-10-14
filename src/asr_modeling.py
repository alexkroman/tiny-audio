from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Wav2Vec2FeatureExtractor,
)

try:
    from .asr_config import ASRConfig
except ImportError:
    from asr_config import ASRConfig


class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.mlp = nn.Sequential(
            nn.Linear(config.encoder_dim * self.k, config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.llm_dim),
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        if seq_len % self.k:
            x = x[:, : -(seq_len % self.k)]
        x = x.contiguous().view(batch_size, -1, dim * self.k)
        return self.mlp(x)


class ASRModel(nn.Module):
    config_class = ASRConfig
    main_input_name = "input_values"
    _supports_generate = True
    _is_loading_from_pretrained: bool = False
    _pretrained_model_path: Optional[str] = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        from pathlib import Path as PathlibPath
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        import os

        config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        cls._is_loading_from_pretrained = True
        cls._pretrained_model_path = pretrained_model_name_or_path

        try:
            model = cls(config, **kwargs)

            # Check if it's a local path or a Hugging Face model ID
            if os.path.exists(pretrained_model_name_or_path):
                # Local path
                projector_path = PathlibPath(pretrained_model_name_or_path) / "model.safetensors"
            else:
                # Hugging Face model ID - download the file
                projector_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="model.safetensors"
                )

            projector_state = load_file(projector_path)
            projector_state = {k.replace("projector.", ""): v for k, v in projector_state.items()}
            model.projector.load_state_dict(projector_state)
            return model
        finally:
            cls._is_loading_from_pretrained = False
            del cls._pretrained_model_path

    def __init__(self, config: Union[ASRConfig, dict], **kwargs):
        super().__init__()

        if isinstance(config, dict):
            config = ASRConfig(**config)

        self.config = config
        self.system_prompt = getattr(config, 'system_prompt', None)

        target_dtype = getattr(torch, config.model_dtype)

        self.encoder = AutoModel.from_pretrained(
            config.audio_model_id,
            attn_implementation=config.attn_implementation,
            dtype=target_dtype,
        )
        self.encoder.requires_grad_(False)

        self.decoder = AutoModelForCausalLM.from_pretrained(
            config.text_model_id,
            attn_implementation=config.attn_implementation,
            dtype=target_dtype,
            trust_remote_code=True,
        )
        self.decoder.requires_grad_(False)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.audio_model_id)

        self.generation_config = self.decoder.generation_config
        self.generation_config.num_beams = 4

        self._init_tokenizer()

        from types import SimpleNamespace

        audio_dim = self.encoder.config.hidden_size
        text_dim = self.decoder.config.hidden_size
        projector_config = SimpleNamespace(
            encoder_projector_ds_rate=config.audio_downsample_rate,
            encoder_dim=audio_dim,
            llm_dim=text_dim,
            projector_hidden_dim=config.projector_hidden_dim,
        )
        self.projector: AudioProjector = AudioProjector(projector_config)

        decoder_dtype = next(self.decoder.parameters()).dtype
        self.projector.to(dtype=decoder_dtype)

        self._no_split_modules = self.decoder._no_split_modules

    def _init_tokenizer(self):
        model_path = self.__class__._pretrained_model_path if self._is_loading_from_pretrained else self.config.text_model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Always ensure chat template is loaded for SmolLM3
        if not self.tokenizer.chat_template and "SmolLM3" in self.config.text_model_id:
            from huggingface_hub import hf_hub_download
            from pathlib import Path
            try:
                template_path = hf_hub_download(
                    repo_id="HuggingFaceTB/SmolLM3-3B",  # Always use the canonical repo
                    filename="chat_template.jinja"
                )
                self.tokenizer.chat_template = Path(template_path).read_text()
                print(f"Loaded chat template from {template_path}")
            except Exception as e:
                print(f"Warning: Could not load chat template: {e}")
                # Fallback to a simple ChatML template
                self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
                print("Using fallback ChatML template")

        self.tokenizer.padding_side = "right"

        if not self._is_loading_from_pretrained:
            if "SmolLM3" in self.config.text_model_id:
                self.tokenizer.bos_token = "<|begin_of_text|>"
                self.tokenizer.bos_token_id = 128000
                self.tokenizer.eos_token = "<|end_of_text|>"
                self.tokenizer.eos_token_id = 128001
                self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
                self.tokenizer.pad_token_id = 128004
            else:
                if self.tokenizer.bos_token_id is None:
                    self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        for cfg in [self.config.text_config, self.decoder.config, self.generation_config]:
            if isinstance(cfg, dict):
                cfg["pad_token_id"] = self.tokenizer.pad_token_id
                cfg["eos_token_id"] = self.tokenizer.eos_token_id
                cfg["bos_token_id"] = self.tokenizer.bos_token_id
            else:
                cfg.pad_token_id = self.tokenizer.pad_token_id
                cfg.eos_token_id = self.tokenizer.eos_token_id
                cfg.bos_token_id = self.tokenizer.bos_token_id

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.decoder.set_output_embeddings(new_embeddings)

    def tie_weights(self):
        if hasattr(self.decoder, "tie_weights"):
            self.decoder.tie_weights()

    def state_dict(self, *args, **kwargs):
        return {f"projector.{k}": v for k, v in self.projector.state_dict().items()}

    def load_state_dict(self, state_dict, strict=True):
        projector_state = {
            k.replace("projector.", ""): v
            for k, v in state_dict.items()
            if k.startswith("projector.")
        }
        if projector_state:
            self.projector.load_state_dict(projector_state, strict=strict)
        return self

    @property
    def device(self) -> torch.device:
        return self.decoder.device

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.dtype

    def can_generate(self) -> bool:
        return True

    def get_processor(self):
        try:
            from .asr_processing import ASRProcessor
        except ImportError:
            from asr_processing import ASRProcessor

        return ASRProcessor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        model_embeds = self.decoder.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        return model_embeds

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

            text_embeds = self.decoder.get_input_embeddings()(input_ids)

            inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

            batch_size = audio_embeds.shape[0]
            audio_seq_len = audio_embeds.shape[1]

            if attention_mask is None:
                attention_mask = torch.ones(
                    batch_size, input_ids.shape[1], dtype=torch.long, device=input_ids.device
                )

            audio_attention = torch.ones(
                batch_size, audio_seq_len, dtype=torch.long, device=input_ids.device
            )
            full_attention_mask = torch.cat([audio_attention, attention_mask], dim=1)

            if labels is not None:
                audio_labels = torch.full(
                    (batch_size, audio_seq_len), -100, dtype=torch.long, device=labels.device
                )
                labels = torch.cat([audio_labels, labels], dim=1)
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
        self, input_values: Optional[torch.Tensor] = None, system_prompt: Optional[str] = None, **generate_kwargs
    ) -> torch.Tensor:
        if input_values is None:
            raise ValueError("input_values must be provided for generation")

        audio_embeds = self._encode_audio(input_values)
        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device

        # Apply chat template to match training format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "Translate the audio to English."})

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        if len(prompt_ids.shape) == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([audio_embeds, prompt_embeds], dim=1)

        # Create attention mask for full input
        total_seq_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(batch_size, total_seq_len, dtype=torch.long, device=device)

        generate_kwargs.setdefault("max_new_tokens", 200)
        generate_kwargs.setdefault("num_beams", 4)
        generate_kwargs.setdefault("do_sample", False)
        generate_kwargs.setdefault("length_penalty", 1.0)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        generate_kwargs.setdefault("eos_token_id", im_end_id)
        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs
        )

        return outputs[:, 1:]

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        import shutil
        from pathlib import Path as PathlibPath

        from safetensors.torch import save_file

        save_dir = PathlibPath(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.config.text_config.vocab_size = self.decoder.config.vocab_size
        self.config.vocab_size = self.decoder.config.vocab_size
        self.config.pad_token_id = self.decoder.config.pad_token_id
        self.config.text_config.bos_token_id = self.tokenizer.bos_token_id
        self.config.text_config.eos_token_id = self.tokenizer.eos_token_id
        self.config.text_config.pad_token_id = self.tokenizer.pad_token_id

        if hasattr(self.config, 'system_prompt'):
            self.config.system_prompt = getattr(self.config, 'system_prompt', None)

        self.config.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        self.feature_extractor.save_pretrained(save_dir)

        self.get_processor().save_pretrained(save_dir)

        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)


AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)
