"""ASR Model"""

from pathlib import Path
from typing import Callable, Optional, Union

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
    WhisperFeatureExtractor,
    WhisperModel,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm


class ASRModelConfig(PretrainedConfig):
    """Configuration class for the ASRModel."""

    model_type = "asr_model"

    def __init__(
        self,
        decoder_model_name="HuggingFaceTB/SmolLM-1.6B-Instruct",
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
    _supports_generate = True  # Tell transformers this model supports generation
    _tokenizer: Optional[AutoTokenizer]
    _feature_extractor: Optional[WhisperFeatureExtractor]

    def can_generate(self) -> bool:
        """Override to tell transformers this model can generate."""
        return True

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer, loading it if necessary."""
        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.decoder_model_name)
        return self._tokenizer

    @property
    def feature_extractor(self) -> WhisperFeatureExtractor:
        """Get the feature extractor, loading it if necessary."""
        if not hasattr(self, "_feature_extractor") or self._feature_extractor is None:
            from transformers import AutoFeatureExtractor

            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.config.encoder_model_name
            )
        return self._feature_extractor

    def __init__(self, config: Union[ASRModelConfig, dict]) -> None:
        if isinstance(config, dict):
            config = ASRModelConfig(**config)
        super().__init__(config)

        # Initialize cached properties
        self._tokenizer = None
        self._feature_extractor = None

        self.encoder = WhisperModel.from_pretrained(config.encoder_model_name)
        self.encoder.requires_grad_(False)  # Freeze the encoder
        self.encoder.eval()

        # Load decoder config and enable sliding window attention for efficiency
        from transformers import AutoConfig

        decoder_config = AutoConfig.from_pretrained(config.decoder_model_name)
        # Set sliding window size for linear attention complexity
        decoder_config.sliding_window = 4096
        # Enable cache for generation
        decoder_config.use_cache = True

        self.decoder = AutoModelForCausalLM.from_pretrained(
            config.decoder_model_name,
            config=decoder_config,  # Pass the modified config with SWA
            low_cpu_mem_usage=False,
        )

        text_dim = self.decoder.config.hidden_size
        audio_dim = self.encoder.config.d_model

        # SwiGLU module for use in AttentionPoolingHead
        class SwiGLU(nn.Module):
            def __init__(self, hidden_dim, expansion_factor=8 / 3):
                super().__init__()
                intermediate_dim = int(hidden_dim * expansion_factor)
                self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        # AttentionPoolingHead for intelligent audio compression
        class AttentionPoolingHead(nn.Module):
            def __init__(self, hidden_dim, num_heads, num_probe_tokens=128):
                super().__init__()
                self.probe = nn.Parameter(torch.randn(1, num_probe_tokens, hidden_dim))
                self.attention = torch.nn.MultiheadAttention(
                    hidden_dim, num_heads, batch_first=True
                )
                self.layernorm = RMSNorm(hidden_dim, eps=1e-6)
                # Use SwiGLU to process the pooled output
                self.mlp = SwiGLU(hidden_dim)

            def forward(self, hidden_state):
                batch_size = hidden_state.shape[0]
                probe = self.probe.repeat(batch_size, 1, 1)

                # Probe attends to the audio features
                pooled_output = self.attention(query=probe, key=hidden_state, value=hidden_state)[0]

                residual = pooled_output
                pooled_output = self.layernorm(pooled_output)
                return residual + self.mlp(pooled_output)

        # Audio projector with attention pooling
        self.audio_norm = RMSNorm(audio_dim, eps=self.decoder.config.rms_norm_eps or 1e-6)
        # Project to text dimension first
        self.audio_to_inter = nn.Linear(audio_dim, text_dim, bias=False)
        self.audio_pooler = AttentionPoolingHead(hidden_dim=text_dim, num_heads=8)

        # Explicit positional embeddings for audio features
        audio_len = 128  # num_probe_tokens from AttentionPoolingHead
        self.audio_pos_embed = nn.Embedding(audio_len, text_dim)

        # Initialize tokenizer for special tokens
        tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer  # Store in cache

        self.add_audio_special_tokens()

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.decoder = get_peft_model(self.decoder, lora_config)

        # Initialize feature extractor
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(config.encoder_model_name)
        self.generation_config = self.decoder.generation_config

        # Set processing_class to avoid warning when saving with Trainer
        self.processing_class = "WhisperProcessor"

    def add_audio_special_tokens(self):
        """Adds special tokens required for audio processing to the tokenizer."""
        audio_tokens = ["<|audio_chunk|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": audio_tokens})

        if hasattr(self.decoder, "resize_token_embeddings"):
            self.decoder.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        self.audio_chunk_id = self.tokenizer.convert_tokens_to_ids("<|audio_chunk|>")

    def _encode_audio(
        self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encodes audio features and projects them into the text embedding space."""
        with torch.no_grad():
            input_features = input_features.to(self.encoder.dtype)
            audio_features = self.encoder.encoder(
                input_features, attention_mask=attention_mask
            ).last_hidden_state

        # Apply normalization
        audio_features = self.audio_norm(audio_features)

        # Project to text dimension BEFORE pooling
        projected = self.audio_to_inter(audio_features)

        # Pool the features using attention
        pooled_features = self.audio_pooler(projected)

        # Add positional embeddings
        positions = torch.arange(pooled_features.size(1), device=pooled_features.device).unsqueeze(
            0
        )
        pooled_features = pooled_features + self.audio_pos_embed(positions)

        # Convert to decoder dtype
        return pooled_features.to(self.decoder.dtype)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass for training and evaluation. For inference, use generate().
        """

        if not (input_ids is not None and input_features is not None):
            raise ValueError("Both `input_ids` and `input_features` are required for training.")

        audio_chunk_mask = input_ids == self.audio_chunk_id
        if not torch.any(audio_chunk_mask):
            raise ValueError(
                "The '<|audio_chunk|>' token was not found in `input_ids` for training."
            )

        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        audio_embeds = self._encode_audio(input_features, audio_attention_mask)

        batch_size, max_text_len, embed_dim = text_embeds.shape
        _, audio_len, _ = audio_embeds.shape  # audio_len is now num_probe_tokens (128)
        new_len = max_text_len - 1 + audio_len

        inputs_embeds = torch.zeros(
            batch_size, new_len, embed_dim, device=text_embeds.device, dtype=text_embeds.dtype
        )
        new_labels = torch.full(
            (batch_size, new_len), -100, device=labels.device, dtype=labels.dtype
        )

        # Vectorized approach for massive speedup
        # 1. Find indices for text and audio parts
        text_indices_mask = ~audio_chunk_mask & (input_ids != self.tokenizer.pad_token_id)

        # 2. Create position mapping - each audio chunk shifts subsequent positions
        position_offsets = torch.cumsum(audio_chunk_mask * (audio_len - 1), dim=-1)
        old_positions = (
            torch.arange(max_text_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        )
        new_positions = (
            old_positions
            + position_offsets
            - audio_chunk_mask.long().cumsum(dim=-1) * (audio_len - 1)
        )

        # 3. Scatter text embeds and labels to their new positions
        # Create batch indices for 2D indexing
        batch_indices = (
            torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand(-1, max_text_len)
        )

        # Filter valid text positions (non-padding, non-audio-chunk)
        valid_mask = text_indices_mask
        batch_idx_flat = batch_indices[valid_mask]
        old_pos_flat = old_positions[valid_mask]
        new_pos_flat = new_positions[valid_mask]

        # Place text embeddings and labels
        inputs_embeds[batch_idx_flat, new_pos_flat] = text_embeds[batch_idx_flat, old_pos_flat]
        new_labels[batch_idx_flat, new_pos_flat] = labels[batch_idx_flat, old_pos_flat]

        # 4. Place audio embeddings
        # Find where each audio chunk starts in the new sequence
        audio_positions = torch.where(audio_chunk_mask)
        for i, pos in zip(audio_positions[0], audio_positions[1]):
            new_audio_start = new_positions[i, pos]
            inputs_embeds[i, new_audio_start : new_audio_start + audio_len] = audio_embeds[i]

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
    def transcribe(
        self,
        audio: Union[np.ndarray, torch.Tensor, str],
        sampling_rate: int = 16000,
        **generate_kwargs,
    ) -> str:
        """
        Transcribe audio to text. Convenience method that handles audio loading and processing.

        Args:
            audio: Can be:
                - numpy array of audio samples
                - torch tensor of audio samples
                - string path to audio file
            sampling_rate: Audio sampling rate (default: 16000)
            **generate_kwargs: Additional arguments for generate() like max_new_tokens, temperature, etc.

        Returns:
            Transcribed text as a string
        """
        # Handle different input types
        if isinstance(audio, str):
            # Load audio file
            import librosa

            audio, _ = librosa.load(audio, sr=sampling_rate)
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Extract features using the property
        inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        # Set default generation parameters
        if "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = 448
        if "min_new_tokens" not in generate_kwargs:
            generate_kwargs["min_new_tokens"] = 10
        if "do_sample" not in generate_kwargs:
            generate_kwargs["do_sample"] = False
        if "num_beams" not in generate_kwargs:
            generate_kwargs["num_beams"] = 1

        # Generate
        generated_ids = self.generate(input_features, **generate_kwargs)

        # Decode using the property
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        # Simplified ASR prompt - just the audio token
        messages = [
            {
                "role": "user",
                "content": "<|audio_chunk|>",
            }
        ]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        audio_embeds = self._encode_audio(input_features, attention_mask=attention_mask)
        embed_layer = self.decoder.get_input_embeddings()

        prompt_ids = self.tokenizer(
            prompt_str, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(input_features.device)

        chunk_idx_tensor = torch.where(prompt_ids[0] == self.audio_chunk_id)[0]
        if len(chunk_idx_tensor) == 0:
            raise ValueError(
                "Internal error: '<|audio_chunk|>' token not found in the generation prompt."
            )
        chunk_idx = int(chunk_idx_tensor[0].item())

        prompt_embeds = embed_layer(prompt_ids[0])

        initial_embeds = torch.cat(
            [
                prompt_embeds[:chunk_idx],
                audio_embeds[0],
                prompt_embeds[chunk_idx + 1 :],
            ],
            dim=0,
        ).unsqueeze(0)

        # Set default generation parameters
        if "pad_token_id" not in generate_kwargs:
            generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if "eos_token_id" not in generate_kwargs:
            generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if "max_new_tokens" not in generate_kwargs and "max_length" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = 448  # Default max tokens to generate
        if "min_new_tokens" not in generate_kwargs:
            generate_kwargs["min_new_tokens"] = 1  # Ensure at least one token is generated

        # Add Whisper-inspired generation parameters
        if "num_beams" not in generate_kwargs:
            generate_kwargs["num_beams"] = 1
        if "suppress_tokens" not in generate_kwargs:
            from transformers.models.whisper.configuration_whisper import NON_SPEECH_TOKENS

            generate_kwargs["suppress_tokens"] = list(NON_SPEECH_TOKENS)
        if "repetition_penalty" not in generate_kwargs:
            generate_kwargs["repetition_penalty"] = 1.1
        if "no_repeat_ngram_size" not in generate_kwargs:
            generate_kwargs["no_repeat_ngram_size"] = 3

        # Create attention mask for the input embeddings
        attention_mask = torch.ones(
            initial_embeds.shape[0],
            initial_embeds.shape[1],
            dtype=torch.long,
            device=initial_embeds.device,
        )

        return self.decoder.generate(
            inputs_embeds=initial_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Optional[Callable] = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save the model and its components (including feature extractor).
        """
        # First, save the model using parent class method
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs,
        )

        # Save the feature extractor so pipeline can auto-detect it
        if is_main_process and hasattr(self, "feature_extractor"):
            self.feature_extractor.save_pretrained(save_directory)

        # Save the tokenizer
        if is_main_process and hasattr(self, "tokenizer"):
            self.tokenizer.save_pretrained(save_directory)


AutoConfig.register("asr_model", ASRModelConfig)
AutoModelForSpeechSeq2Seq.register(ASRModelConfig, ASRModel)
