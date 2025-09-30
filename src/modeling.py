"""Simplified ASR Model (Refactored)"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
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

# --- Configuration Class ---


class ASRModelConfig(PretrainedConfig):
    model_type = "asr_model"

    def __init__(
        self,
        decoder_model_name: str = "HuggingFaceTB/SmolLM3-3B",
        encoder_model_name: str = "openai/whisper-small",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[list[str]] = None,
        lora_dropout: float = 0.05,
        compile_decoder: bool = False,
        compile_mode: str = "reduce-overhead",
        **kwargs,
    ):
        self.decoder_model_name = decoder_model_name
        self.encoder_model_name = encoder_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.lora_dropout = lora_dropout
        self.compile_decoder = compile_decoder
        self.compile_mode = compile_mode
        super().__init__(**kwargs)

        self.auto_map = {"AutoConfig": "modeling.ASRModelConfig", "AutoModel": "modeling.ASRModel"}


# --- Main ASR Model ---


class ASRModel(PreTrainedModel):
    config_class = ASRModelConfig
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _supports_generate = True

    def __init__(self, config: Union[ASRModelConfig, dict]):
        if isinstance(config, dict):
            config = ASRModelConfig(**config)
        super().__init__(config)

        # 1. Initialize Audio Encoder and Feature Extractor
        self.encoder = WhisperModel.from_pretrained(config.encoder_model_name)
        self.encoder.requires_grad_(False)  # Freeze encoder
        # Disable gradient checkpointing on frozen encoder to avoid warnings
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config.encoder_model_name
        )

        # 2. Initialize Text Decoder and Tokenizer
        self._init_decoder_and_tokenizer()

        # 3. Initialize layers to connect Encoder and Decoder
        self._init_projector()

        # 4. Apply LoRA for efficient fine-tuning
        self._init_lora()

    def _init_decoder_and_tokenizer(self):
        """Initializes the text decoder, tokenizer, and special tokens."""
        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.config.decoder_model_name,
            attn_implementation="sdpa",
        )
        self.generation_config = self.decoder.generation_config

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Resize embeddings for new tokens and sync configs
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        for cfg in [self.decoder.config, self.generation_config]:
            cfg.pad_token_id = self.tokenizer.pad_token_id
            cfg.eos_token_id = self.tokenizer.eos_token_id
            cfg.bos_token_id = getattr(self.tokenizer, "bos_token_id", self.tokenizer.eos_token_id)

        # Register token IDs as buffers for torch.compile
        self.register_buffer(
            "pad_token_id",
            torch.tensor(self.tokenizer.pad_token_id, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "eos_token_id",
            torch.tensor(self.tokenizer.eos_token_id, dtype=torch.long),
            persistent=False,
        )

    def _init_projector(self):
        """Initializes layers to project audio features to text space."""
        audio_dim = self.encoder.config.d_model
        text_dim = self.decoder.config.hidden_size
        self.audio_norm = RMSNorm(audio_dim)
        self.audio_proj = nn.Linear(audio_dim, text_dim, bias=True)

        # Initialize projection with small weights to avoid large gradients
        nn.init.normal_(self.audio_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.audio_proj.bias)

        # Downsample factor: reduces 50 Hz -> 25 Hz (2x reduction)
        self.downsample_factor = 2

    def _init_lora(self):
        """Applies LoRA to the decoder for efficient fine-tuning."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.decoder = get_peft_model(self.decoder, lora_config)

        # Enable gradient checkpointing on the decoder if supported
        if hasattr(self.decoder, "enable_input_require_grads"):
            self.decoder.enable_input_require_grads()

        # Compile decoder for better performance (skip projector compilation)
        if getattr(self.config, "compile_decoder", False):
            import torch

            compile_mode = getattr(self.config, "compile_mode", "reduce-overhead")
            print(f"🔧 Compiling decoder with mode: {compile_mode}")
            self.decoder = torch.compile(self.decoder, mode=compile_mode)

    def _encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encodes audio and projects it into the text embedding space."""
        with torch.no_grad():
            # Whisper encoder outputs hidden states directly
            audio_features = self.encoder.encoder(input_features).last_hidden_state

        # Project to text space
        projected = self.audio_proj(self.audio_norm(audio_features))

        # Downsample using reshape+mean (compile-friendly, no transpose issues)
        batch_size, seq_len, hidden_dim = projected.shape
        # Trim to multiple of downsample_factor
        trimmed_len = (seq_len // self.downsample_factor) * self.downsample_factor
        projected = projected[:, :trimmed_len, :]

        # Reshape and average: (B, T, D) -> (B, T/2, 2, D) -> (B, T/2, D)
        projected = projected.reshape(
            batch_size, trimmed_len // self.downsample_factor, self.downsample_factor, hidden_dim
        )
        projected = projected.mean(dim=2)  # Average over the downsample dimension

        return projected

    def _prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepares inputs by concatenating audio embeddings as a prefix to text embeddings.
        This is fully vectorized and torch.compile friendly.
        """
        batch_size = input_ids.shape[0]
        audio_seq_len = audio_embeds.shape[1]

        # Get text embeddings directly from base model (faster, avoids PEFT wrapper)
        text_embeds = self.decoder.base_model.model.get_input_embeddings()(input_ids)

        # Concatenate audio as prefix: [audio_embeds, text_embeds]
        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

        # Create attention mask for the combined sequence
        text_lengths = (input_ids != self.pad_token_id).sum(dim=1)
        total_lengths = audio_seq_len + text_lengths
        combined_seq_len = inputs_embeds.shape[1]
        # Use expand instead of [None, :] for better compilation
        attention_mask = torch.arange(combined_seq_len, device=input_ids.device).unsqueeze(
            0
        ).expand(batch_size, -1) < total_lengths.unsqueeze(1)

        # Prepare labels: audio tokens are ignored (-100), text tokens keep their labels
        combined_labels = None
        if labels is not None:
            audio_labels = torch.full(
                (batch_size, audio_seq_len), -100, device=labels.device, dtype=labels.dtype
            )
            combined_labels = torch.cat([audio_labels, labels], dim=1)

        return inputs_embeds, attention_mask.long(), combined_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # Accept but ignore
        **kwargs,
    ):
        """Forward pass for training, combining audio and text inputs."""
        # Ignore input attention_mask, we compute our own

        audio_embeds = self._encode_audio(input_features)
        inputs_embeds, attention_mask, combined_labels = self._prepare_inputs_embeds(
            input_ids, audio_embeds, labels
        )

        # Disable cache when using gradient checkpointing
        if self.training and getattr(self.config, "use_cache", True):
            kwargs["use_cache"] = False

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=combined_labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_features: torch.Tensor, **generate_kwargs) -> torch.LongTensor:
        """Generates text from audio features using audio embeddings as prefix."""
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        batch_size = input_features.shape[0]
        device = input_features.device

        # Ensure input_features match encoder's dtype
        encoder_dtype = next(self.encoder.parameters()).dtype
        input_features = input_features.to(dtype=encoder_dtype)

        # Encode audio features
        audio_embeds = self._encode_audio(input_features)
        audio_seq_len = audio_embeds.shape[1]

        # Create attention mask for audio tokens (all valid)
        attention_mask = torch.ones(batch_size, audio_seq_len, device=device, dtype=torch.long)

        generate_kwargs.setdefault("max_new_tokens", 448)
        generate_kwargs.setdefault("pad_token_id", self.pad_token_id.item())
        generate_kwargs.setdefault("eos_token_id", self.eos_token_id.item())
        generate_kwargs.setdefault("use_cache", True)
        generate_kwargs.setdefault("do_sample", False)
        generate_kwargs.setdefault("num_beams", 1)
        generate_kwargs.setdefault("repetition_penalty", 1)

        return self.decoder.generate(
            inputs_embeds=audio_embeds, attention_mask=attention_mask, **generate_kwargs
        )

    @torch.no_grad()
    def transcribe(
        self, audio: Union[np.ndarray, str], sampling_rate: int = 16000, **generate_kwargs
    ) -> str:
        """High-level convenience method to transcribe an audio file or array."""
        if isinstance(audio, str):
            import torchaudio

            audio, sr = torchaudio.load(audio)
            if sr != sampling_rate:
                audio = torchaudio.functional.resample(audio, sr, sampling_rate)
            audio = audio.squeeze(0).numpy()

        inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        device = next(self.decoder.parameters()).device
        input_features = inputs.input_features.to(device)

        generated_ids = self.generate(input_features, **generate_kwargs)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Saves the model, tokenizer, and feature extractor."""
        super().save_pretrained(save_directory, **kwargs)
        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)


# Register the custom model with AutoClasses for easy loading
AutoConfig.register("asr_model", ASRModelConfig)
AutoModel.register(ASRModelConfig, ASRModel)
