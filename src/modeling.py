"""ASR Model for Whisper-LLM integration (Refactored)"""

from typing import Optional, Union

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    WhisperFeatureExtractor,
    WhisperModel,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm


class ASRModelConfig(PretrainedConfig):
    model_type = "asr_model"

    def __init__(
        self,
        decoder_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        encoder_model_name="openai/whisper-small",
        lora_r=32,
        lora_alpha=64,
        lora_target_modules=None,
        lora_dropout=0.05,
        **kwargs,
    ):
        self.decoder_model_name = decoder_model_name
        self.encoder_model_name = encoder_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]
        self.lora_dropout = lora_dropout
        super().__init__(**kwargs)


class ASRModel(PreTrainedModel):
    config_class = ASRModelConfig
    base_model_prefix = "asr"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoder", "LLMDecoder", "AudioProjector"]
    main_input_name = "input_features"

    def __init__(self, config: Union[ASRModelConfig, dict]) -> None:
        if isinstance(config, dict):
            config = ASRModelConfig(**config)
        super().__init__(config)

        self.encoder = WhisperModel.from_pretrained(config.encoder_model_name)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        self.decoder = AutoModelForCausalLM.from_pretrained(
            config.decoder_model_name, use_cache=False
        )
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.decoder = get_peft_model(self.decoder, lora_config)

        text_dim = self.decoder.config.hidden_size
        audio_dim = self.encoder.config.d_model
        self.audio_projector = nn.Sequential(
            RMSNorm(audio_dim, eps=1e-6),
            nn.Linear(audio_dim, text_dim, bias=True),
            nn.GELU(),
            nn.Linear(text_dim, text_dim, bias=True),
        )
        self.audio_scale = nn.Parameter(torch.tensor(0.01))

        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.encoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._tokens_initialized = False
        self.add_audio_special_tokens()

    def add_audio_special_tokens(self):
        if self._tokens_initialized:
            return
        audio_tokens = [
            "<|audio_start|>",
            "<|audio_end|>",
            "<|audio_pad|>",
            "<|audio_sep|>",
            "<|audio_chunk|>",
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": audio_tokens})

        if hasattr(self.decoder.model, "resize_token_embeddings"):
            self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.audio_chunk_id = self.tokenizer.convert_tokens_to_ids("<|audio_chunk|>")
        self._tokens_initialized = True

    def _encode_audio(
        self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            input_features = input_features.to(self.encoder.dtype)
            audio_features = self.encoder.encoder(
                input_features, attention_mask=attention_mask
            ).last_hidden_state

        projected_features = self.audio_projector(audio_features) * self.audio_scale
        return projected_features.to(self.decoder.dtype)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_features is not None and input_ids is None:
            return self.generate(input_features, attention_mask=audio_attention_mask, **kwargs)

        if not (input_ids is not None and input_features is not None):
            raise ValueError("Both input_ids and input_features are required for training.")

        audio_chunk_mask = input_ids == self.audio_chunk_id
        if not torch.any(audio_chunk_mask):
            raise ValueError("The '<|audio_chunk|>' token was not found in the input_ids.")

        chunk_indices = torch.where(audio_chunk_mask)[1]

        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        audio_embeds = self._encode_audio(input_features, audio_attention_mask)

        batch_size, max_text_len, embed_dim = text_embeds.shape
        _, audio_len, _ = audio_embeds.shape
        new_len = max_text_len - 1 + audio_len

        inputs_embeds = torch.zeros(
            batch_size, new_len, embed_dim, device=text_embeds.device, dtype=text_embeds.dtype
        )
        new_labels = torch.full(
            (batch_size, new_len), -100, device=labels.device, dtype=labels.dtype
        )

        for i in range(batch_size):
            chunk_idx = chunk_indices[i]
            inputs_embeds[i, :chunk_idx] = text_embeds[i, :chunk_idx]
            new_labels[i, :chunk_idx] = labels[i, :chunk_idx]
            inputs_embeds[i, chunk_idx : chunk_idx + audio_len] = audio_embeds[i]
            inputs_embeds[i, chunk_idx + audio_len :] = text_embeds[i, chunk_idx + 1 :]
            new_labels[i, chunk_idx + audio_len :] = labels[i, chunk_idx + 1 :]

        original_lengths = attention_mask.sum(dim=1)
        new_lengths = original_lengths - 1 + audio_len
        new_attention_mask = (
            torch.arange(new_len, device=inputs_embeds.device)[None, :] < new_lengths[:, None]
        ).long()

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        prompt: str = "Please transcribe the following audio recording.\n<|audio_chunk|>",
        **kwargs,
    ) -> torch.LongTensor:
        audio_embeds = self._encode_audio(input_features, attention_mask=attention_mask)
        embed_layer = self.decoder.get_input_embeddings()

        messages = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(input_features.device)

        chunk_idx = torch.where(prompt_ids[0] == self.audio_chunk_id)[0]
        if len(chunk_idx) == 0:
            raise ValueError("'<|audio_chunk|>' token not found in prompt.")
        chunk_idx = chunk_idx[0].item()

        prompt_embeds = embed_layer(prompt_ids[0])

        initial_embeds = torch.cat(
            [
                prompt_embeds[:chunk_idx],
                audio_embeds[0],
                prompt_embeds[chunk_idx + 1 :],
            ],
            dim=0,
        ).unsqueeze(0)

        return self.decoder.generate(
            inputs_embeds=initial_embeds,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


AutoConfig.register("asr_model", ASRModelConfig)
AutoModel.register(ASRModelConfig, ASRModel)
