"""ASR Model for Whisper-LLM integration"""

from typing import Optional, Union, cast

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as torchaudio_functional
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
    def __init__(self):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained("openai/whisper-small", dtype="auto")

        # Use HuggingFace utility to freeze parameters
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
        lora_r=32,
        lora_alpha=64,
        lora_target_modules=None,
        lora_dropout=0.05,
        **kwargs,
    ):
        self.decoder_model_name = decoder_model_name
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

        # Use torch.nn.init methods consistently
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
            dtype="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

        # Use HuggingFace's device check utility
        embeddings = self.model.get_input_embeddings()
        is_meta_device = (
            embeddings is not None
            and hasattr(embeddings, "weight")
            and embeddings.weight.device.type == "meta"
        )

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            if not is_meta_device:
                self.model.resize_token_embeddings(len(self.tokenizer))

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

    def __init__(self, config: Union[ASRModelConfig, dict]) -> None:
        if isinstance(config, dict):
            config = ASRModelConfig(**config)

        super().__init__(config)

        self.encoder = WhisperEncoder()
        self.decoder = LLMDecoder(config)
        text_dim = getattr(self.decoder.model.config, "hidden_size", 768)
        audio_dim = self.encoder.d_model
        self.audio_projector = AudioProjector(audio_dim, text_dim)
        self.add_audio_special_tokens()

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model, tokenizer, and feature extractor."""
        if not hasattr(self, "feature_extractor"):
            raise AttributeError(
                "The model does not have a `feature_extractor` attribute. "
                "Please attach it to the model instance before saving."
            )
        super().save_pretrained(save_directory, **kwargs)
        self.decoder.tokenizer.save_pretrained(save_directory)
        self.feature_extractor.save_pretrained(save_directory)

    def add_audio_special_tokens(self):
        """Add audio-specific special tokens for better audio-text alignment."""
        audio_tokens = [
            "<|audio_start|>",
            "<|audio_end|>",
            "<|audio_pad|>",
            "<|audio_sep|>",
            "<|audio_chunk|>",
        ]

        num_added = self.decoder.tokenizer.add_special_tokens(
            {"additional_special_tokens": audio_tokens}
        )

        if num_added > 0:
            # Check if model is on meta device
            embeddings = self.decoder.model.get_input_embeddings()
            is_meta_device = (
                embeddings is not None
                and hasattr(embeddings, "weight")
                and embeddings.weight.device.type == "meta"
            )

            if is_meta_device:
                self.decoder.model.resize_token_embeddings(
                    len(self.decoder.tokenizer), mean_resizing=False
                )
            else:
                self.decoder.model.resize_token_embeddings(len(self.decoder.tokenizer))

                # Initialize new embeddings with smart initialization
                with torch.no_grad():
                    embeddings = self.decoder.model.get_input_embeddings()
                    if embeddings is not None and hasattr(embeddings, "weight"):
                        existing_embeds = embeddings.weight[:-num_added]
                        mean_embedding = existing_embeds.mean(dim=0)
                        std_embedding = existing_embeds.std()

                        # Use torch.nn.init style initialization
                        new_embeds = embeddings.weight[-num_added:]
                        torch.nn.init.normal_(new_embeds, mean=0, std=std_embedding * 0.02)
                        new_embeds.data.add_(mean_embedding)

        self.audio_start_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.audio_pad_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.audio_chunk_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_chunk|>")

    def _encode_audio(
        self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encodes audio features and projects them to the decoder's embedding space."""
        audio_features = self.encoder(input_features, attention_mask=attention_mask)
        audio_embeds = self.audio_projector(audio_features)
        return cast(torch.Tensor, audio_embeds.to(self.decoder.model.dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        input_features: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        audio_embeds = self._encode_audio(input_features, attention_mask=audio_attention_mask)

        embed_layer = self.decoder.model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)

        final_inputs_embeds = []
        final_labels = []
        final_attention_masks = []

        for i in range(input_ids.shape[0]):
            chunk_idx = (input_ids[i] == self.audio_chunk_id).nonzero()
            if chunk_idx.shape[0] == 0:
                raise ValueError("'<|audio_chunk|>' token not found in input_ids.")
            chunk_idx = chunk_idx[0].item()

            combined_embeds = torch.cat(
                [
                    text_embeds[i, :chunk_idx],
                    audio_embeds[i],
                    text_embeds[i, chunk_idx + 1 :],
                ],
                dim=0,
            )
            final_inputs_embeds.append(combined_embeds)

            audio_len = audio_embeds[i].shape[0]
            label_before = labels[i, :chunk_idx]
            label_after = labels[i, chunk_idx + 1 :]
            audio_labels = torch.full((audio_len,), -100, dtype=labels.dtype, device=labels.device)

            combined_labels = torch.cat([label_before, audio_labels, label_after], dim=0)
            final_labels.append(combined_labels)

            final_attention_masks.append(
                torch.ones(
                    combined_embeds.shape[0], dtype=torch.long, device=combined_embeds.device
                )
            )

        # Use imported pad_sequence
        inputs_embeds = pad_sequence(final_inputs_embeds, batch_first=True, padding_value=0.0)
        labels = pad_sequence(final_labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(final_attention_masks, batch_first=True, padding_value=0)

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

        messages = [
            {
                "role": "user",
                "content": "Please transcribe the following audio recording.\n<|audio_chunk|>",
            }
        ]
        prompt_ids = self.decoder.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        prompt_ids = prompt_ids.to(input_features.device)

        chunk_idx = (prompt_ids[0] == self.audio_chunk_id).nonzero()
        if chunk_idx.shape[0] == 0:
            raise ValueError("'<|audio_chunk|>' token not found in instruction.")
        chunk_idx = chunk_idx[0].item()

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

    @torch.no_grad()
    def transcribe(self, audio_path: str, **kwargs) -> str:

        # Use backend="soundfile" to avoid deprecation warning
        # This is the recommended approach until TorchCodec is available
        waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")

        # Use torchaudio functional API for resampling
        if sample_rate != 16000:
            waveform = torchaudio_functional.resample(waveform, sample_rate, 16000)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )

        audio_features = inputs.input_features.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generate_kwargs = {
            "max_new_tokens": 200,
            "do_sample": False,
        }

        if self.decoder.tokenizer.pad_token_id is not None:
            generate_kwargs["pad_token_id"] = self.decoder.tokenizer.pad_token_id
        if self.decoder.tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = self.decoder.tokenizer.eos_token_id

        generate_kwargs.update(kwargs)

        generated_ids = self.generate(
            input_features=audio_features,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return cast(str, self.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True))


AutoConfig.register("asr_model", ASRModelConfig)
AutoModel.register(ASRModelConfig, ASRModel)
