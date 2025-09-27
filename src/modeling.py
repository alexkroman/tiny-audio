"""Simplified ASR Model (Refactored)"""

from pathlib import Path
from typing import Optional, Union
import torch._dynamo as dynamo

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
        self.probe = nn.Parameter(torch.randn(1, num_probe_tokens, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = RMSNorm(hidden_dim, eps=1e-6)
        self.norm2 = RMSNorm(hidden_dim, eps=1e-6)
        self.mlp = SwiGLU(hidden_dim)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Pre-normalization is used for stability, common in models like Llama.
        batch_size = hidden_state.shape[0]
        probe = self.probe.expand(batch_size, -1, -1) # More memory-efficient

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
        # Load decoder model with SDPA for faster attention
        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.config.decoder_model_name,
            attn_implementation="sdpa",
        )
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

        self.register_buffer(
            'position_ids',
            torch.arange(num_probe_tokens, dtype=torch.long).unsqueeze(0)
        )

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
            audio_features = self.encoder(input_features).last_hidden_state

        projected = self.audio_to_inter(self.audio_norm(audio_features))
        projected = torch.clamp(projected, min=-10.0, max=10.0)
        pooled = self.audio_pooler(projected)
        pooled = torch.clamp(pooled, min=-10.0, max=10.0)

        return pooled + self.audio_pos_embed(self.position_ids)

    def _splice_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Splices audio embeddings into text embeddings in a fully vectorized,
        loop-free, and compiler-friendly manner.
        """
        # 1. Get shapes and base embeddings
        batch_size, seq_len = input_ids.shape
        audio_seq_len = audio_embeds.shape[1]
        device = input_ids.device
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        hidden_dim = text_embeds.shape[-1]
        
        # 2. Get text-only parts by masking out the audio chunk placeholder
        text_mask = (input_ids != self.audio_chunk_id)
        text_only_embeds = text_embeds[text_mask].view(batch_size, seq_len - 1, hidden_dim)
        
        chunk_token_indices = (input_ids == self.audio_chunk_id).to(torch.int8).argmax(dim=-1)
        
        # Create the pieces to assemble: the part before audio, the audio itself, and the part after
        before_audio = text_only_embeds.gather(1, chunk_token_indices.view(-1, 1, 1).expand(-1, -1, hidden_dim)) # This is a placeholder, we need to slice
        
        # Let's use a simpler, more direct `torch.cat` approach which is also out-of-place
        # and will be fixed by the compiler. This avoids the complexity of `where` and `gather`.
        final_embeds_list = []
        final_labels_list = [] if labels is not None else None

        # We fall back to a simple, traceable loop that uses out-of-place `torch.cat`
        # This is often easier for Dynamo to reason about than complex scatter/gather operations.
        for i in range(batch_size):
            idx = chunk_token_indices[i]
            
            # Slicing the text_only_embeds tensor which is already prepared
            before_embeds = text_only_embeds[i, :idx]
            after_embeds = text_only_embeds[i, idx:]
            
            spliced = torch.cat([before_embeds, audio_embeds[i], after_embeds], dim=0)
            final_embeds_list.append(spliced)
            
            if final_labels_list is not None:
                labels_only = labels[i, text_mask[i]].view(seq_len - 1)
                before_labels = labels_only[:idx]
                after_labels = labels_only[idx:]
                audio_pad = torch.full((audio_seq_len,), -100, device=device, dtype=labels.dtype)
                
                spliced_labels = torch.cat([before_labels, audio_pad, after_labels], dim=0)
                final_labels_list.append(spliced_labels)

        final_embeds = torch.stack(final_embeds_list, dim=0)
        final_labels = None
        if final_labels_list is not None:
            final_labels = torch.stack(final_labels_list, dim=0)

        # 5. Create the final attention mask
        original_lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
        new_lengths = original_lengths - 1 + audio_seq_len
        final_attention_mask = torch.arange(final_embeds.shape[1], device=device)[None, :] < new_lengths[:, None]
        
        return final_embeds, final_attention_mask.long(), final_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass for training, combining audio and text inputs."""
        kwargs.pop("attention_mask", None)

        audio_embeds = self._encode_audio(input_features)
        inputs_embeds, new_attention_mask, new_labels = self._splice_audio_embeddings(
            input_ids, audio_embeds, labels
        )
        
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_features: torch.Tensor, **generate_kwargs) -> torch.LongTensor:
        """Generates text from audio features."""
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        prompt_template = [{"role": "user", "content": "<|audio_chunk|>"}]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_template, tokenize=False, add_generation_prompt=True
        )
        device = input_features.device
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)

        batch_size = input_features.shape[0]
        if batch_size > 1:
            prompt_ids = prompt_ids.repeat(batch_size, 1)

        audio_embeds = self._encode_audio(input_features)
        inputs_embeds, attention_mask, _ = self._splice_audio_embeddings(prompt_ids, audio_embeds)

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
        device = next(self.decoder.parameters()).device
        input_features = inputs.input_features.to(device)

        generated_ids = self.generate(input_features, **generate_kwargs)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Saves the model, tokenizer, and feature extractor."""
        kwargs.setdefault("safe_serialization", False)
        super().save_pretrained(save_directory, **kwargs)
        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)


# Register the custom model with AutoClasses for easy loading
AutoConfig.register("asr_model", ASRModelConfig)
AutoModelForSpeechSeq2Seq.register(ASRModelConfig, ASRModel)