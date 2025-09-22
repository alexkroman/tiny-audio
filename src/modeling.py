"""ASR Model for Whisper-LLM integration"""

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
    WhisperModel,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm


class WhisperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(
            "openai/whisper-small", dtype="auto", token=False
        )

        for param in self.whisper.parameters():
            param.requires_grad = False

        self.d_model = self.whisper.config.d_model
        self.whisper.eval()

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

        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.normal_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

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
        # Extract config values directly from ASRModelConfig
        decoder_model_name = config.decoder_model_name
        lora_r = config.lora_r
        lora_alpha = config.lora_alpha
        lora_target_modules = config.lora_target_modules
        lora_dropout = config.lora_dropout

        self.model = AutoModelForCausalLM.from_pretrained(
            decoder_model_name,
            dtype="auto",
            token=False,
            use_cache=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name, token=False)

        # Add padding token if missing to avoid HF warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(lora_target_modules)
            if lora_target_modules
            else ["q_proj", "v_proj"],
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
    INSTRUCTION_TEMPLATE = (
        "User: Please transcribe the following audio recording.\n<|audio_chunk|>\nAssistant: "
    )

    def __init__(self, config: Union[ASRModelConfig, dict]) -> None:
        # Convert dict config to ASRModelConfig if needed
        if isinstance(config, dict):
            config = ASRModelConfig(**config)

        super().__init__(config)

        # Create encoder, decoder, and projector
        self.encoder = WhisperEncoder()
        self.decoder = LLMDecoder(config)
        text_dim = getattr(self.decoder.model.config, "hidden_size", 768)
        audio_dim = self.encoder.d_model
        self.audio_projector = AudioProjector(audio_dim, text_dim)
        self.add_audio_special_tokens()

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model, tokenizer, and feature extractor."""
        super().save_pretrained(save_directory, **kwargs)

        # Save the tokenizer
        self.decoder.tokenizer.save_pretrained(save_directory)

        # Save the feature extractor
        from transformers import WhisperFeatureExtractor

        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        feature_extractor.save_pretrained(save_directory)

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

                with torch.no_grad():
                    embeddings = self.decoder.model.get_input_embeddings()
                    if embeddings is not None and hasattr(embeddings, "weight"):
                        existing_embeds = embeddings.weight[:-num_added]
                        mean_embedding = existing_embeds.mean(dim=0)
                        std_embedding = existing_embeds.std()

                        for i in range(num_added):
                            embeddings.weight[-num_added + i] = mean_embedding + torch.randn_like(
                                embeddings.weight[0]
                            ) * (std_embedding * 0.02)

        self.audio_start_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.audio_pad_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.audio_chunk_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_chunk|>")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        input_features: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        audio_features = self.encoder(input_features, attention_mask=audio_attention_mask)
        audio_embeds = self.audio_projector(audio_features)
        audio_embeds = audio_embeds.to(self.decoder.model.dtype)

        embed_layer = self.decoder.model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)

        final_inputs_embeds = []
        final_labels = []
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

        max_len = max(emb.shape[0] for emb in final_inputs_embeds)

        padded_embeds = []
        padded_labels = []
        padded_attention = []

        for emb, lab in zip(final_inputs_embeds, final_labels):
            pad_len = max_len - emb.shape[0]
            if pad_len > 0:
                pad_emb = torch.zeros((pad_len, emb.shape[-1]), dtype=emb.dtype, device=emb.device)
                emb = torch.cat([emb, pad_emb], dim=0)

                pad_lab = torch.full((pad_len,), -100, dtype=lab.dtype, device=lab.device)
                lab = torch.cat([lab, pad_lab], dim=0)

                att_mask = torch.cat(
                    [
                        torch.ones(emb.shape[0] - pad_len, dtype=torch.long, device=emb.device),
                        torch.zeros(pad_len, dtype=torch.long, device=emb.device),
                    ]
                )
            else:
                att_mask = torch.ones(emb.shape[0], dtype=torch.long, device=emb.device)

            padded_embeds.append(emb)
            padded_labels.append(lab)
            padded_attention.append(att_mask)

        inputs_embeds = torch.stack(padded_embeds)
        labels = torch.stack(padded_labels)
        attention_mask = torch.stack(padded_attention)

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
        audio_features = self.encoder(input_features, attention_mask=attention_mask)
        audio_embeds = self.audio_projector(audio_features)
        audio_embeds = audio_embeds.to(self.decoder.model.dtype)

        batch_size = audio_embeds.shape[0]
        embed_layer = self.decoder.model.get_input_embeddings()

        prompt_ids = self.decoder.tokenizer(
            self.INSTRUCTION_TEMPLATE, return_tensors="pt"
        ).input_ids
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

        generated: torch.LongTensor = self.decoder.model.generate(
            inputs_embeds=initial_embeds,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        return generated


# Register model with HuggingFace Auto classes
AutoConfig.register("asr_model", ASRModelConfig)
AutoModel.register(ASRModelConfig, ASRModel)
