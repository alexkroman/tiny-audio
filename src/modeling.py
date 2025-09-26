"""Simplified ASR Model (Refactored)"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertModel,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm

# --- Helper Modules ---


class SwiGLU(nn.Module):
    """A SwiGLU activation layer, a variant of Gated Linear Units."""

    def __init__(self, hidden_dim: int, expansion_factor: float = 8 / 3):
        super().__init__()
        intermediate_dim = int(hidden_dim * expansion_factor)
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)  # Up projection
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)  # Down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


class AttentionPoolingHead(nn.Module):
    """Compresses audio features using attention with learnable query probes."""

    def __init__(self, hidden_dim: int, num_heads: int, num_probe_tokens: int = 128):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, num_probe_tokens, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Pre-normalization is used for stability, common in models like Llama.
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        # Attention layer with pre-norm
        attn_input = self.norm1(probe)
        attn_output, _ = self.attention(query=attn_input, key=hidden_state, value=hidden_state)
        residual1 = probe + attn_output

        # MLP layer with pre-norm
        mlp_input = self.norm2(residual1)
        mlp_output = self.mlp(mlp_input)

        return residual1 + mlp_output


# --- Configuration Class ---


class ASRModelConfig(PretrainedConfig):
    model_type = "asr_model"

    def __init__(
        self,
        decoder_model_name: str = "HuggingFaceTB/SmolLM3-3B",
        encoder_model_name: str = "facebook/w2v-bert-2.0",
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_target_modules: Optional[list[str]] = None,
        lora_dropout: float = 0.05,
        **kwargs,
    ):
        self.decoder_model_name = decoder_model_name
        self.encoder_model_name = encoder_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.lora_dropout = lora_dropout
        super().__init__(**kwargs)


# --- Main ASR Model ---


class ASRModel(PreTrainedModel):
    config_class = ASRModelConfig
    main_input_name = "input_features"
    supports_gradient_checkpointing = False
    _supports_generate = True

    def __init__(self, config: Union[ASRModelConfig, dict]):
        if isinstance(config, dict):
            config = ASRModelConfig(**config)
        super().__init__(config)

        # 1. Initialize Audio Encoder and Feature Extractor
        self.encoder = Wav2Vec2BertModel.from_pretrained(config.encoder_model_name)
        self.encoder.requires_grad_(False)  # Freeze encoder
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
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
        # Load decoder model
        self.decoder = AutoModelForCausalLM.from_pretrained(self.config.decoder_model_name)
        self.generation_config = self.decoder.generation_config

        # Load tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio_chunk|>"]})
        self.audio_chunk_id = self.tokenizer.convert_tokens_to_ids("<|audio_chunk|>")

        # Resize embeddings for new tokens and sync configs
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        for cfg in [self.decoder.config, self.generation_config]:
            cfg.pad_token_id = self.tokenizer.pad_token_id
            cfg.eos_token_id = self.tokenizer.eos_token_id
            cfg.bos_token_id = getattr(self.tokenizer, "bos_token_id", self.tokenizer.eos_token_id)

    def _init_projector(self):
        """Initializes layers to project and compress audio features."""
        audio_dim = self.encoder.config.hidden_size
        text_dim = self.decoder.config.hidden_size
        num_probe_tokens = 128

        self.audio_norm = RMSNorm(audio_dim)
        self.audio_to_inter = nn.Linear(audio_dim, text_dim, bias=False)
        self.audio_pooler = AttentionPoolingHead(
            text_dim, num_heads=8, num_probe_tokens=num_probe_tokens
        )
        self.audio_pos_embed = nn.Embedding(num_probe_tokens, text_dim)

    def _init_lora(self):
        """Applies LoRA to the decoder for efficient fine-tuning."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["lm_head"],  # Train output layer
        )
        self.decoder = get_peft_model(self.decoder, lora_config)

    def _encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encodes audio and projects it into the text embedding space."""
        with torch.no_grad():
            audio_features = self.encoder(input_features.to(self.encoder.dtype)).last_hidden_state

        # Project, pool, and add positional embeddings
        projected = self.audio_to_inter(self.audio_norm(audio_features))
        pooled = self.audio_pooler(projected)
        positions = torch.arange(pooled.size(1), device=pooled.device).unsqueeze(0)
        return (pooled + self.audio_pos_embed(positions)).to(self.decoder.dtype)

    def _splice_audio_embeddings(
        self, input_ids: torch.Tensor, audio_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replaces the audio placeholder token in input_ids with audio embeddings."""
        # Find the location of the placeholder token
        is_audio_chunk = input_ids == self.audio_chunk_id
        if not torch.any(is_audio_chunk):
            raise ValueError("The '<|audio_chunk|>' token is required in the input.")
        batch_indices, token_indices = torch.where(is_audio_chunk)
        if batch_indices.size(0) != input_ids.size(0):
            raise ValueError("Each sequence must contain exactly one '<|audio_chunk|>' token.")

        # Get text embeddings and prepare for splicing
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        spliced_embeds = []

        # Iterate through the batch to combine text and audio embeddings
        for i, chunk_idx in enumerate(token_indices):
            combined = torch.cat(
                [
                    text_embeds[i, :chunk_idx],
                    audio_embeds[i],
                    text_embeds[i, chunk_idx + 1 :],
                ],
                dim=0,
            )
            spliced_embeds.append(combined)

        # Pad the list of tensors into a single batch tensor
        inputs_embeds = nn.utils.rnn.pad_sequence(spliced_embeds, batch_first=True)
        # Create a matching attention mask
        attention_mask = (inputs_embeds.sum(dim=-1) != 0).long()
        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass for training, combining audio and text inputs."""
        # Remove attention_mask from kwargs since we'll create our own
        kwargs.pop("attention_mask", None)

        audio_embeds = self._encode_audio(input_features)
        inputs_embeds, new_attention_mask = self._splice_audio_embeddings(input_ids, audio_embeds)

        new_labels = None
        if labels is not None:
            # Splice labels similarly, inserting ignore_index (-100) for audio parts
            _, token_indices = torch.where(input_ids == self.audio_chunk_id)
            spliced_labels = []
            audio_labels = torch.full(
                (audio_embeds.shape[1],), -100, device=labels.device, dtype=labels.dtype
            )
            for i, chunk_idx in enumerate(token_indices):
                combined = torch.cat(
                    [
                        labels[i, :chunk_idx],
                        audio_labels,
                        labels[i, chunk_idx + 1 :],
                    ],
                    dim=0,
                )
                spliced_labels.append(combined)
            new_labels = nn.utils.rnn.pad_sequence(
                spliced_labels, batch_first=True, padding_value=-100
            )

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_features: torch.Tensor, **generate_kwargs) -> torch.LongTensor:
        """Generates text from audio features (for a single audio input)."""
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)
        if input_features.size(0) > 1:
            raise ValueError("generate() only supports a batch size of 1.")

        # Create a standard prompt with the audio placeholder
        prompt_template = [{"role": "user", "content": "<|audio_chunk|>"}]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_template, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(self.device)

        # Splice audio embeddings into the prompt
        audio_embeds = self._encode_audio(input_features)
        inputs_embeds, attention_mask = self._splice_audio_embeddings(prompt_ids, audio_embeds)

        # Set default generation parameters if not provided
        generate_kwargs.setdefault("max_new_tokens", 448)
        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generate_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        return self.decoder.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs
        )

    @torch.no_grad()
    def transcribe(
        self, audio: Union[np.ndarray, str], sampling_rate: int = 16000, **generate_kwargs
    ) -> str:
        """High-level convenience method to transcribe an audio file or array."""
        if isinstance(audio, str):
            import librosa

            audio, _ = librosa.load(audio, sr=sampling_rate)

        inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.decoder.dtype)

        generated_ids = self.generate(input_features, **generate_kwargs)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Saves the model, tokenizer, and feature extractor."""
        super().save_pretrained(save_directory, **kwargs)
        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)


# Register the custom model with AutoClasses for easy loading
AutoConfig.register("asr_model", ASRModelConfig)
AutoModelForSpeechSeq2Seq.register(ASRModelConfig, ASRModel)
