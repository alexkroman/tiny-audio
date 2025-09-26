"""Simplified ASR Model"""

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
    """SwiGLU activation function used in the projector."""

    def __init__(self, hidden_dim: int, expansion_factor: float = 8 / 3):
        super().__init__()
        intermediate_dim = int(hidden_dim * expansion_factor)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class AttentionPoolingHead(nn.Module):
    """Compresses audio features using attention with learnable probes."""

    def __init__(self, hidden_dim: int, num_heads: int, num_probe_tokens: int = 128):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, num_probe_tokens, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.layernorm1 = RMSNorm(hidden_dim, eps=1e-6)
        self.layernorm2 = RMSNorm(hidden_dim, eps=1e-6)
        self.mlp = SwiGLU(hidden_dim)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        # Pre-norm for attention
        attn_input = self.layernorm1(probe)  # Norm the query
        pooled_output, _ = self.attention(query=attn_input, key=hidden_state, value=hidden_state)

        # First residual connection
        residual1 = probe + pooled_output

        # Pre-norm for MLP
        mlp_input = self.layernorm2(residual1)
        mlp_output = self.mlp(mlp_input)

        # Second residual connection
        return residual1 + mlp_output


# --- Configuration Class ---


class ASRModelConfig(PretrainedConfig):
    model_type = "asr_model"

    def __init__(
        self,
        decoder_model_name: str = "HuggingFaceTB/SmolLM-1.6B-Instruct",
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
        self.lora_target_modules = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        ]
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

        # --- 1. Encoder (W2V-BERT) Setup ---
        self.encoder = Wav2Vec2BertModel.from_pretrained(config.encoder_model_name)
        self.encoder.requires_grad_(False)  # Freeze the audio encoder

        # --- 2. Decoder (LLM) Setup ---
        self._setup_decoder()

        # --- 3. Audio Projector Setup (Connects Encoder to Decoder) ---
        self._setup_projector()

        # --- 4. Tokenizer & Special Tokens ---
        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)
        self._setup_special_tokens()

        # --- 5. LoRA (PEFT) Setup ---
        self._setup_lora()

        # --- 6. Feature Extractor & Final Config Sync ---
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            config.encoder_model_name
        )
        self._sync_configs()

    def _setup_decoder(self):
        """Initializes the text decoder with optional Flash Attention."""
        decoder_config = AutoConfig.from_pretrained(self.config.decoder_model_name)
        decoder_config.sliding_window = 4096
        decoder_config.use_cache = True

        decoder_kwargs = {"config": decoder_config, "low_cpu_mem_usage": False}

        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.config.decoder_model_name, **decoder_kwargs
        )

        # Initialize generation config from decoder
        self.generation_config = self.decoder.generation_config

    def _setup_projector(self):
        """Initializes layers to project and compress audio features."""
        audio_dim = self.encoder.config.hidden_size  # W2V-BERT uses hidden_size instead of d_model
        text_dim = self.decoder.config.hidden_size
        num_probe_tokens = 128

        self.audio_norm = RMSNorm(audio_dim, eps=self.decoder.config.rms_norm_eps or 1e-6)
        self.audio_to_inter = nn.Linear(audio_dim, text_dim, bias=False)
        self.audio_pooler = AttentionPoolingHead(
            hidden_dim=text_dim, num_heads=8, num_probe_tokens=num_probe_tokens
        )
        self.audio_pos_embed = nn.Embedding(num_probe_tokens, text_dim)

    def _setup_special_tokens(self):
        """Adds and configures special tokens needed for audio."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio_chunk|>"]})

        self.decoder.resize_token_embeddings(len(self.tokenizer))
        self.audio_chunk_id = self.tokenizer.convert_tokens_to_ids("<|audio_chunk|>")

    def _setup_lora(self):
        """Applies LoRA to the decoder for efficient fine-tuning."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["lm_head"],  # Also train the output layer
        )
        self.decoder = get_peft_model(self.decoder, lora_config)

    def _sync_configs(self):
        """Ensures model, generation, and tokenizer configs are consistent."""
        self.decoder.config.pad_token_id = self.tokenizer.pad_token_id
        self.decoder.config.eos_token_id = self.tokenizer.eos_token_id
        self.decoder.config.bos_token_id = getattr(
            self.tokenizer, "bos_token_id", self.tokenizer.eos_token_id
        )

        # Update the separate generation config if it exists
        if hasattr(self, "generation_config") and self.generation_config is not None:
            self.generation_config.pad_token_id = self.decoder.config.pad_token_id
            self.generation_config.eos_token_id = self.decoder.config.eos_token_id
            self.generation_config.bos_token_id = self.decoder.config.bos_token_id

    def _encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encodes audio features and projects them into the text embedding space."""
        with torch.no_grad():
            # W2V-BERT expects input_values instead of input_features
            audio_features = self.encoder(input_features.to(self.encoder.dtype)).last_hidden_state

        audio_features = self.audio_norm(audio_features)
        projected_features = self.audio_to_inter(audio_features)
        pooled_features = self.audio_pooler(projected_features)

        positions = torch.arange(pooled_features.size(1), device=pooled_features.device).unsqueeze(
            0
        )
        pooled_features += self.audio_pos_embed(positions)

        return pooled_features.to(self.decoder.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass for training. Combines text and audio embeddings.
        This process replaces the special '<|audio_chunk|>' token in the input
        with the actual processed audio embeddings.
        """
        # Step 1: Get separate embeddings for text and audio
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        audio_embeds = self._encode_audio(input_features)

        # Step 2: Find the indices of the audio chunk token
        audio_chunk_mask = input_ids == self.audio_chunk_id
        if not torch.any(audio_chunk_mask):
            raise ValueError("Training requires '<|audio_chunk|>' token in `input_ids`.")

        # batch_indices will be [0, 1, 2, ...], token_indices are the positions of the chunk token
        batch_indices, token_indices = torch.where(audio_chunk_mask)

        # For simplicity, this vectorized version assumes one chunk token per example.
        if batch_indices.size(0) != input_ids.size(0):
            raise ValueError(
                "All examples in the batch must contain exactly one '<|audio_chunk|>' token."
            )

        # Step 3: Vectorized Splicing
        # This is more memory-efficient than creating a new large tensor upfront.
        # We create a list of tensors for each example and then stack them.
        final_embeds = []
        final_labels = []
        audio_len = audio_embeds.shape[1]

        for i in range(input_ids.size(0)):
            chunk_idx = token_indices[i]

            # Combine embeddings for this single example
            combined_embed = torch.cat(
                [
                    text_embeds[i, :chunk_idx],
                    audio_embeds[i],
                    text_embeds[i, chunk_idx + 1 :],
                ],
                dim=0,
            )
            final_embeds.append(combined_embed)

            # Combine labels for this single example
            # Labels for audio are implicitly -100
            if labels is not None:
                audio_labels = torch.full(
                    (audio_len,), -100, device=labels.device, dtype=labels.dtype
                )
                combined_label = torch.cat(
                    [
                        labels[i, :chunk_idx],
                        audio_labels,
                        labels[i, chunk_idx + 1 :],
                    ],
                    dim=0,
                )
                final_labels.append(combined_label)

        # Pad the sequences to the max length in the batch
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(final_embeds, batch_first=True)

        if labels is not None:
            new_labels = torch.nn.utils.rnn.pad_sequence(
                final_labels, batch_first=True, padding_value=-100
            )
        else:
            new_labels = None

        # Step 4: Create a new attention mask
        # A 1 for every real token (including audio), a 0 for padding.
        new_attention_mask = (inputs_embeds.sum(dim=-1) != 0).long()

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_features: torch.Tensor, **generate_kwargs) -> torch.LongTensor:
        """Generates text transcription from audio features.

        Note: This method only supports single audio input (batch size 1).
        For batch processing, use forward() method with appropriate batching logic.
        """
        # Ensure single input
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)
        if input_features.size(0) != 1:
            raise ValueError("generate() only supports single audio input (batch size 1)")

        # Create a standard prompt template for ASR
        prompt_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<|audio_chunk|>"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(self.device)

        # Find audio chunk position
        audio_chunk_positions = torch.where(prompt_ids[0] == self.audio_chunk_id)[0]
        if len(audio_chunk_positions) == 0:
            raise ValueError("No audio chunk token found in prompt")
        chunk_idx = int(audio_chunk_positions[0].item())

        # Get embeddings for the prompt and the audio
        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids)
        audio_embeds = self._encode_audio(input_features)

        # Splice audio embeddings into the prompt embeddings
        inputs_embeds = torch.cat(
            [
                prompt_embeds[0, :chunk_idx].unsqueeze(0),
                audio_embeds,
                prompt_embeds[0, chunk_idx + 1 :].unsqueeze(0),
            ],
            dim=1,
        )

        # Set default generation parameters
        generate_kwargs.setdefault("max_new_tokens", 448)
        generate_kwargs.setdefault("num_beams", 1)
        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generate_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        return self.decoder.generate(inputs_embeds=inputs_embeds, **generate_kwargs)

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        sampling_rate: int = 16000,
        **generate_kwargs,
    ) -> str:
        """Convenience method to transcribe an audio file or array.

        Note: This method only supports single audio input.
        For batch processing, implement your own batching logic using forward().
        """
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
