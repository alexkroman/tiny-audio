"""ASR Model for Whisper-LLM integration"""

from pathlib import Path
from typing import Optional, Union, cast

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    WhisperModel,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm


class WhisperEncoder(nn.Module):
    def __init__(self, encoder_model_name="openai/whisper-small"):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(encoder_model_name)

        self.whisper.requires_grad_(False)
        self.whisper.eval()

        self.d_model = self.whisper.config.d_model

    def forward(
        self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.FloatTensor:
        with torch.no_grad():
            input_features = input_features.to(self.whisper.dtype)
            outputs = self.whisper.encoder(input_features, attention_mask=attention_mask)
            last_hidden_state: torch.FloatTensor = outputs.last_hidden_state
            return last_hidden_state


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


class AudioProjector(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int):
        super().__init__()
        self.norm = RMSNorm(audio_dim, eps=1e-6)

        self.linear_1 = nn.Linear(audio_dim, text_dim, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_dim, text_dim, bias=True)

        torch.nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.linear_1.bias)
        torch.nn.init.xavier_normal_(self.linear_2.weight)
        torch.nn.init.zeros_(self.linear_2.bias)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(audio_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        result: torch.Tensor = hidden_states * 0.01
        return result


class LLMDecoder(nn.Module):
    def __init__(self, config: ASRModelConfig):
        super().__init__()
        decoder_model_name = config.decoder_model_name
        lora_r = config.lora_r
        lora_alpha = config.lora_alpha
        lora_target_modules = config.lora_target_modules
        lora_dropout = config.lora_dropout

        self.model = AutoModelForCausalLM.from_pretrained(
            decoder_model_name,
            use_cache=False,
        )

        self.tokenizer = None

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=(
                list(lora_target_modules) if lora_target_modules else ["q_proj", "v_proj"]
            ),
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

    def forward(self, **kwargs):
        return self.model(**kwargs)


class ASRModel(PreTrainedModel):
    config_class = ASRModelConfig
    base_model_prefix = "asr"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoder", "LLMDecoder", "AudioProjector"]
    main_input_name = "input_features"  # Tell the pipeline to use input_features instead of input_ids

    def __init__(self, config: Union[ASRModelConfig, dict]) -> None:
        if isinstance(config, dict):
            config = ASRModelConfig(**config)

        super().__init__(config)

        self.encoder = WhisperEncoder(config.encoder_model_name)
        self.decoder = LLMDecoder(config)

        from transformers import WhisperFeatureExtractor

        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.encoder_model_name)

        self.decoder.tokenizer = self.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        text_dim = getattr(self.decoder.model.config, "hidden_size", 768)
        audio_dim = self.encoder.d_model
        self.audio_projector = AudioProjector(audio_dim, text_dim)

        # Initialize but don't add tokens yet - will be done in from_pretrained or after init
        self._tokens_initialized = False
        self.add_audio_special_tokens()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load model, tokenizer, and feature extractor.

        This follows HuggingFace conventions and automatically handles:
        - Loading the model weights
        - Loading the tokenizer that was saved with the model
        - Loading the feature extractor that was saved with the model
        """
        from transformers import WhisperFeatureExtractor

        # Override the model loading to not trigger embedding resizing
        kwargs["ignore_mismatched_sizes"] = True
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        model.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path
        )

        model.decoder.tokenizer = model.tokenizer

        # Just set the token IDs without resizing since they're already in the loaded model
        model.audio_start_id = model.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        model.audio_end_id = model.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        model.audio_pad_id = model.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        model.audio_chunk_id = model.tokenizer.convert_tokens_to_ids("<|audio_chunk|>")
        model._tokens_initialized = True

        # Always resize embeddings to match tokenizer vocabulary size
        model.decoder.model.resize_token_embeddings(len(model.tokenizer))

        return model

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs) -> None:
        """Save the model, tokenizer, and feature extractor.

        This properly saves all components to the same directory so they can be
        loaded together with from_pretrained().
        """
        import inspect
        import shutil

        super().save_pretrained(save_directory, **kwargs)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory)

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        modeling_file = Path(inspect.getfile(self.__class__))
        target_file = save_path / "modeling.py"

        shutil.copy2(modeling_file, target_file)

    def add_audio_special_tokens(self):
        """Add audio-specific special tokens for better audio-text alignment."""
        if self._tokens_initialized:
            return  # Already initialized during from_pretrained

        audio_tokens = [
            "<|audio_start|>",
            "<|audio_end|>",
            "<|audio_pad|>",
            "<|audio_sep|>",
            "<|audio_chunk|>",
        ]

        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": audio_tokens})

        # Only resize if tokens were added and not on meta device
        if num_added > 0:
            try:
                # Check if model is on meta device
                if hasattr(self.decoder.model, "embeddings"):
                    embed_device = self.decoder.model.embeddings.weight.device
                elif hasattr(self.decoder.model.model, "embed_tokens"):
                    embed_device = self.decoder.model.model.embed_tokens.weight.device
                else:
                    embed_device = next(self.decoder.model.parameters()).device

                if embed_device.type != "meta":
                    self.decoder.model.resize_token_embeddings(len(self.tokenizer))
            except Exception:
                # If we can't resize, it's probably because model is on meta device
                pass

        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.audio_pad_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.audio_chunk_id = self.tokenizer.convert_tokens_to_ids("<|audio_chunk|>")
        self._tokens_initialized = True

    def _encode_audio(
        self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encodes audio features and projects them to the decoder's embedding space."""
        audio_features = self.encoder(input_features, attention_mask=attention_mask)
        audio_embeds = self.audio_projector(audio_features)
        return cast(torch.Tensor, audio_embeds.to(self.decoder.model.dtype))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Inference mode - when only input_features is provided (ASR pipeline)
        if input_features is not None and input_ids is None:
            # Generate token IDs
            generated_ids = self.generate(input_features, attention_mask=audio_attention_mask, **kwargs)
            # Return a simple object that the pipeline can handle
            from transformers.modeling_outputs import CausalLMOutput
            return CausalLMOutput(logits=generated_ids.unsqueeze(-1).float())
        
        # Training mode - require both input_ids and input_features
        if not (input_ids is not None and input_features is not None):
            raise ValueError("Both input_ids and input_features are required for training")

        audio_embeds = self._encode_audio(input_features, attention_mask=audio_attention_mask)

        embed_layer = self.decoder.model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)

        audio_chunk_mask = input_ids == self.audio_chunk_id
        batch_size = input_ids.shape[0]
        final_inputs_embeds = []
        final_labels = []

        for i in range(batch_size):
            chunk_pos = audio_chunk_mask[i].nonzero(as_tuple=False)
            if len(chunk_pos) == 0:
                raise ValueError(f"'<|audio_chunk|>' token not found in batch item {i}")

            chunk_idx = chunk_pos[0, 0].item() if chunk_pos.device.type != "meta" else 0
            inputs_embeds = torch.cat(
                [text_embeds[i, :chunk_idx], audio_embeds[i], text_embeds[i, chunk_idx + 1 :]]
            )

            labels_with_audio = torch.cat(
                [
                    labels[i, :chunk_idx],
                    labels.new_full((audio_embeds.shape[1],), -100),
                    labels[i, chunk_idx + 1 :],
                ]
            )

            final_inputs_embeds.append(inputs_embeds)
            final_labels.append(labels_with_audio)

        inputs_embeds = pad_sequence(final_inputs_embeds, batch_first=True, padding_value=0.0)
        labels = pad_sequence(final_labels, batch_first=True, padding_value=-100)

        attention_mask = torch.zeros(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        for i, length in enumerate(len(emb) for emb in final_inputs_embeds):
            attention_mask[i, :length] = 1

        return self.decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> torch.LongTensor:
        """Generate text from audio input."""
        audio_embeds = self._encode_audio(input_features, attention_mask=attention_mask)

        batch_size = audio_embeds.shape[0]
        embed_layer = self.decoder.model.get_input_embeddings()

        prompt = "Please transcribe the following audio recording.\n<|audio_chunk|>"
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(input_features.device)

        chunk_idx = (prompt_ids[0] == self.audio_chunk_id).nonzero()
        if chunk_idx.shape[0] == 0:
            raise ValueError("'<|audio_chunk|>' token not found in instruction.")
        chunk_idx = chunk_idx[0].item() if chunk_idx.device.type != "meta" else 0

        prompt_embeds = embed_layer(prompt_ids)
        initial_embeds = torch.cat(
            [
                prompt_embeds[0, :chunk_idx],
                audio_embeds[0],
                prompt_embeds[0, chunk_idx + 1 :],
            ],
            dim=0,
        ).unsqueeze(0)

        if batch_size > 1:
            initial_embeds = initial_embeds.expand(batch_size, -1, -1)

        input_attention_mask = torch.ones(
            initial_embeds.shape[:2], dtype=torch.long, device=initial_embeds.device
        )

        generated: torch.LongTensor = self.decoder.model.generate(
            inputs_embeds=initial_embeds,
            attention_mask=input_attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        return generated

    def pipeline(self, task: str = "automatic-speech-recognition", **kwargs):
        from transformers import pipeline

        return pipeline(
            task,
            model=self,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            **kwargs,
        )


AutoConfig.register("asr_model", ASRModelConfig)
AutoModel.register(ASRModelConfig, ASRModel)
