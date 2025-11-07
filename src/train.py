#!/usr/bin/env python3

import logging
import re
from typing import Any, Dict, List

import hydra
import nltk
import numpy as np
import torch
import truecase
import wandb
from datasets import Audio, Dataset, interleave_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    WhisperTokenizer,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel

# Download required NLTK data for truecase
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class CollatorSwapCallback(TrainerCallback):
    """Callback to swap data collators between training and evaluation."""

    def __init__(self, train_collator, eval_collator):
        self.train_collator = train_collator
        self.eval_collator = eval_collator

    def on_train_begin(self, args, state, control, model, **kwargs):
        # Store reference to trainer to swap collators
        pass

    def on_evaluate_begin(self, args, state, control, model, **kwargs):
        # Temporarily disable augmentation for evaluation
        self.train_collator.apply_augmentation = False

    def on_evaluate_end(self, args, state, control, model, **kwargs):
        # Re-enable augmentation after evaluation
        self.train_collator.apply_augmentation = True


class DatasetLoader:
    def __init__(self, config: DictConfig):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir

    def _prepare_split(self, dataset_cfg: DictConfig, split: str) -> Dataset:
        # Get dataset path (required)
        dataset_path = dataset_cfg.get("path")
        if not dataset_path:
            raise ValueError("Dataset path is required")

        # Get optional name/config for datasets with multiple configurations
        dataset_name = dataset_cfg.get("name", None)

        ds = load_dataset(
            dataset_path,
            name=dataset_name,  # Can be None for datasets without configs
            split=split,
            streaming=True,
            cache_dir=self.cache_dir,
        )

        text_column = dataset_cfg.get("text_column", "text")
        if text_column != "text" and text_column in ds.column_names:
            ds = ds.rename_column(text_column, "text")

        audio_column = dataset_cfg.get("audio_column", "audio")
        if audio_column != "audio" and audio_column in ds.column_names:
            ds = ds.rename_column(audio_column, "audio")

        # Cast audio column to correct format
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        # Add task type to each example - store it as metadata, will be added during iteration
        # Can't use map here because it breaks with AudioDecoder objects
        task = dataset_cfg.get("task", "transcribe")  # Default to transcribe
        # Wrap the dataset to add task field during iteration
        original_iter = ds.__iter__
        def wrapped_iter():
            for item in original_iter():
                item["task"] = task
                yield item
        ds.__iter__ = wrapped_iter

        return ds

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []
        train_weights = []
        for d_cfg in self.config.datasets:
            train_splits = d_cfg.get("train_splits", [d_cfg.get("train_split", "train")])
            eval_splits = d_cfg.get("eval_splits", [d_cfg.get("eval_split", "validation")])
            sampling_weight = d_cfg.get("sampling_weight", 1.0)  # Default equal weight

            if "configs" in d_cfg:
                configs = d_cfg.configs
                for config, train_split, eval_split in zip(configs, train_splits, eval_splits):
                    split_cfg = OmegaConf.create(d_cfg)
                    split_cfg.name = config
                    train_datasets.append(self._prepare_split(split_cfg, train_split))
                    val_datasets.append(self._prepare_split(split_cfg, eval_split))
                    train_weights.append(sampling_weight)
            else:
                for train_split in train_splits:
                    train_datasets.append(self._prepare_split(d_cfg, train_split))
                    train_weights.append(sampling_weight)
                for eval_split in eval_splits:
                    val_datasets.append(self._prepare_split(d_cfg, eval_split))

        # Use sampling weights if provided and we have multiple datasets
        if len(train_datasets) > 1:
            # Normalize weights
            total_weight = sum(train_weights[:len(train_datasets)])
            probabilities = [w / total_weight for w in train_weights[:len(train_datasets)]]
            train_ds = interleave_datasets(train_datasets, probabilities=probabilities)
        else:
            train_ds = train_datasets[0]

        val_ds = interleave_datasets(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

        # train_ds = train_ds.shuffle(seed=42)

        if self.config.max_train_samples:
            train_ds = train_ds.take(self.config.max_train_samples)
        if self.config.max_eval_samples:
            val_ds = val_ds.take(self.config.max_eval_samples)

        return train_ds, val_ds


class DataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        system_prompt: str = None,
        mask_time_prob: float = 0.065,
        mask_time_length: int = 10,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        apply_augmentation: bool = True,
    ):
        super().__init__(tokenizer=tokenizer, padding=True)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.system_prompt = system_prompt

        # Check if this is a Whisper feature extractor
        self.is_whisper = feature_extractor.__class__.__name__ == 'WhisperFeatureExtractor'

        # SpecAugment parameters for data augmentation
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.apply_augmentation = apply_augmentation

        # Initialize WhisperTokenizer for text normalization (matches eval script)
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before Whisper normalization (matches eval script)."""
        # Remove <inaudible> tags
        text = re.sub(r"<inaudible>", "", text, flags=re.IGNORECASE)
        # Remove disfluencies (uh, um)
        text = re.sub(r"\b(uh|um)\b", "", text, flags=re.IGNORECASE)
        return text

    def _normalize_text(self, text: str) -> str:
        """Apply Whisper normalization (matches eval script)."""
        return self.whisper_tokenizer.normalize(self._preprocess_text(text))

    def _extract_audio(self, audio_decoder) -> Any:
        # Note: Audio() does peak normalization → [-1, 1]
        # Wav2Vec2FeatureExtractor does z-normalization → mean=0, std=1
        # No additional normalization needed here!
        audio_samples = audio_decoder.get_all_samples()
        audio_array = audio_samples.data
        return audio_array.squeeze().numpy()

    def _apply_spec_augment(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply SpecAugment using the exact Transformers implementation.
        This masks time steps (and optionally features) during training.

        Args:
            input_values: Input features [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Masked input_values
        """
        # Skip augmentation if disabled or masking is set to 0
        if not self.apply_augmentation or (self.mask_time_prob == 0 and self.mask_feature_prob == 0):
            return input_values

        batch_size, sequence_length = input_values.shape

        # Apply time masking
        if self.mask_time_prob > 0:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.mask_time_prob,
                mask_length=self.mask_time_length,
                attention_mask=attention_mask,
                min_masks=2,  # wav2vec2 default
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.bool)
            input_values[mask_time_indices] = 0.0

        return input_values

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [self._extract_audio(f["audio"]) for f in features]

        audio_features = self.feature_extractor(
            audio_arrays, sampling_rate=self.sample_rate, padding=True, return_tensors="pt"
        )

        # Handle different feature key names and augmentation
        if self.is_whisper:
            # Whisper uses 'input_features' and requires padding to 3000 frames
            if "input_features" in audio_features:
                input_features = audio_features["input_features"]
                batch_size, feature_dim, seq_len = input_features.shape

                # Whisper expects exactly 3000 frames
                expected_length = 3000

                if seq_len < expected_length:
                    # Pad with zeros to reach 3000 frames
                    padding_length = expected_length - seq_len
                    padding = torch.zeros(batch_size, feature_dim, padding_length,
                                        dtype=input_features.dtype,
                                        device=input_features.device)
                    audio_features["input_features"] = torch.cat([input_features, padding], dim=-1)

                    # Create attention mask to indicate valid vs padded frames
                    # 1 for valid frames, 0 for padded frames
                    attention_mask = torch.ones(batch_size, expected_length,
                                               dtype=torch.long,
                                               device=input_features.device)
                    attention_mask[:, seq_len:] = 0
                    audio_features["attention_mask"] = attention_mask
                elif seq_len > expected_length:
                    # Truncate if longer than 3000 frames
                    audio_features["input_features"] = input_features[:, :, :expected_length]
                    audio_features["attention_mask"] = torch.ones(batch_size, expected_length,
                                                                  dtype=torch.long,
                                                                  device=input_features.device)
                else:
                    # Exactly 3000 frames, create full attention mask
                    audio_features["attention_mask"] = torch.ones(batch_size, expected_length,
                                                                  dtype=torch.long,
                                                                  device=input_features.device)
        else:
            # Wav2Vec2/HuBERT use 'input_values' - apply our SpecAugment implementation
            if "input_values" in audio_features:
                input_values = audio_features["input_values"]
                batch_size, seq_len = input_values.shape

                # For Wav2Vec2/HuBERT, ensure proper padding and attention mask
                # The feature extractor should have already done padding, but let's ensure we have attention mask
                if "attention_mask" not in audio_features:
                    # Create attention mask based on non-zero values (assuming padding is zeros)
                    audio_features["attention_mask"] = (input_values != 0.0).long()

                # Apply SpecAugment if enabled
                if self.apply_augmentation:
                    audio_features["input_values"] = self._apply_spec_augment(
                        input_values,
                        audio_features.get("attention_mask")
                    )

        text_features = []
        for f in features:
            text = f["text"].strip() if isinstance(f["text"], str) else f["text"]

            # Apply Whisper normalization (matches eval script preprocessing)
            text = self._normalize_text(text)

            # Apply truecasing in main process (not in DataLoader workers)
            text = text.replace("<COMMA>", ",").replace("<PERIOD>", ".")
            text = truecase.get_true_case(text)

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            # Choose prompt based on task type
            task = f.get("task", "transcribe")  # Default to transcribe
            if task == "continue":
                messages.append(
                    {
                        "role": "user",
                        "content": "Continue: <audio>",
                    }
                )
            elif task == "describe":
                messages.append(
                    {
                        "role": "user",
                        "content": "Describe: <audio>",
                    }
                )
            elif task == "emotion":
                messages.append(
                    {
                        "role": "user",
                        "content": "Emotion: <audio>",
                    }
                )
            else:  # Default to transcribe
                messages.append(
                    {
                        "role": "user",
                        "content": "Transcribe: <audio>",
                    }
                )
            messages.append({"role": "assistant", "content": text})

            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=256,
                enable_thinking=False,
            )

            # Create labels - only train on the actual transcription text, not thinking tags
            labels = [-100] * len(tokens)  # Start with everything masked

            # Get special token IDs
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            think_start_id = self.tokenizer.convert_tokens_to_ids("<think>")
            think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")

            # Find where </think> ends (if present) - the actual transcription starts after it
            content_start = -1
            for i in range(len(tokens)):
                if tokens[i] == think_end_id:
                    # Skip the </think> token and any newlines after it
                    content_start = i + 1
                    # Skip newlines
                    while (
                        content_start < len(tokens)
                        and self.tokenizer.decode([tokens[content_start]]).strip() == ""
                    ):
                        content_start += 1
                    break

            # If no thinking tags found, look for assistant content directly
            if content_start == -1:
                im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
                assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
                for i in range(len(tokens) - 1):
                    if tokens[i] == im_start_id and tokens[i + 1] == assistant_id:
                        content_start = i + 3  # Skip <|im_start|>, assistant, \n
                        break

            # Find the closing <|im_end|> for the assistant message
            content_end = -1
            if content_start > 0:
                for i in range(content_start, len(tokens)):
                    if tokens[i] == im_end_id:
                        content_end = i
                        break

            # Unmask only the actual transcription text (not thinking tags)
            if content_start > 0 and content_end > 0:
                for i in range(content_start, content_end + 1):  # +1 to include <|im_end|>
                    labels[i] = tokens[i]

            text_features.append(
                {
                    "input_ids": tokens,
                    "labels": labels,
                }
            )

        batch = super().__call__(text_features)

        # Handle both Wav2Vec2 (input_values) and Whisper (input_features)
        if "input_values" in audio_features:
            batch["input_values"] = audio_features.input_values
        elif "input_features" in audio_features:
            batch["input_features"] = audio_features.input_features

        if "attention_mask" in audio_features:
            batch["audio_attention_mask"] = audio_features.attention_mask

        return batch


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("datasets.utils.file_utils").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.INFO)
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb
    if cfg.training.get("report_to") == "wandb":
        wandb.init(
            project="tiny-audio",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.training.get("run_name"),
        )

    # Get encoder and decoder dimensions
    from transformers import AutoConfig as HFAutoConfig

    encoder_config = HFAutoConfig.from_pretrained(cfg.model.encoder_model_name)
    decoder_config = HFAutoConfig.from_pretrained(
        cfg.model.decoder_model_name, trust_remote_code=True
    )

    asr_config = ASRConfig(
        text_model_id=cfg.model.decoder_model_name,
        audio_model_id=cfg.model.encoder_model_name,
        attn_implementation=cfg.training.attn_implementation,
        model_dtype=cfg.training.model_dtype,
        audio_downsample_rate=cfg.model.audio_downsample_rate,
        system_prompt=cfg.model.system_prompt,
        encoder_dim=encoder_config.hidden_size,
        llm_dim=decoder_config.hidden_size,
        projector_hidden_dim=cfg.model.get("projector_hidden_dim", 2048),
    )

    # Extract PEFT configs if present
    peft_config = None
    if "peft" in cfg and cfg.peft.get("peft_method"):
        peft_config = OmegaConf.to_container(cfg.peft, resolve=True)

    encoder_lora_config = None
    if "encoder_lora" in cfg and cfg.encoder_lora.get("r", 0) > 0:
        encoder_lora_config = OmegaConf.to_container(cfg.encoder_lora, resolve=True)

    # Load from pretrained if specified, otherwise create new model
    if cfg.model.get("pretrained_model_path"):
        print(f"Loading pretrained model from: {cfg.model.pretrained_model_path}")
        # from_pretrained will automatically load LoRA weights if they exist in the checkpoint
        # It reads encoder_lora_config.json and decoder_lora_config.json from the Hub
        model = ASRModel.from_pretrained(
            cfg.model.pretrained_model_path,
            config=asr_config,
        )
        print("✓ Loaded pretrained model (projector + LoRA weights if present)")

        # If no LoRA weights were in checkpoint but we want to add them, apply fresh LoRA
        # Check if encoder already has LoRA
        has_encoder_lora = any(
            "lora" in name.lower() for name, _ in model.encoder.named_parameters()
        )
        if encoder_lora_config and encoder_lora_config.get("r", 0) > 0 and not has_encoder_lora:
            from peft import TaskType

            print("⚠️  No encoder LoRA in checkpoint - applying fresh encoder LoRA adapters...")
            model.encoder = model._apply_lora(
                model.encoder, encoder_lora_config, TaskType.FEATURE_EXTRACTION, "encoder"
            )
            model.encoder_lora_config = encoder_lora_config

        # Check if decoder already has LoRA
        has_decoder_lora = any(
            "lora" in name.lower() for name, _ in model.decoder.named_parameters()
        )
        if peft_config and peft_config.get("peft_method") == "lora" and not has_decoder_lora:
            from peft import TaskType

            print("⚠️  No decoder LoRA in checkpoint - applying fresh decoder LoRA adapters...")
            model.decoder = model._apply_lora(
                model.decoder, peft_config, TaskType.CAUSAL_LM, "decoder"
            )
            model.peft_config = peft_config
    else:
        model = ASRModel(
            asr_config, peft_config=peft_config, encoder_lora_config=encoder_lora_config
        )

    # Disable cache during training (required for gradient checkpointing)
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if hasattr(model.generation_config, 'use_cache'):
        model.generation_config.use_cache = False

    train_dataset, val_dataset = DatasetLoader(cfg).load()

    # Create separate collators for training (with augmentation) and eval (without)
    train_collator = DataCollator(
        tokenizer=model.tokenizer,
        feature_extractor=model.feature_extractor,
        sample_rate=cfg.data.sample_rate,
        system_prompt=cfg.model.system_prompt,
        mask_time_prob=cfg.data.get("mask_time_prob", 0.05),
        mask_time_length=cfg.data.get("mask_time_length", 10),
        mask_feature_prob=cfg.data.get("mask_feature_prob", 0.0),
        mask_feature_length=cfg.data.get("mask_feature_length", 10),
        apply_augmentation=True,  # Enable augmentation for training
    )

    eval_collator = DataCollator(
        tokenizer=model.tokenizer,
        feature_extractor=model.feature_extractor,
        sample_rate=cfg.data.sample_rate,
        system_prompt=cfg.model.system_prompt,
        mask_time_prob=cfg.data.get("mask_time_prob", 0.00),
        mask_time_length=cfg.data.get("mask_time_length", 0),
        mask_feature_prob=cfg.data.get("mask_feature_prob", 0.0),
        mask_feature_length=cfg.data.get("mask_feature_length", 0),
        apply_augmentation=False,  # Disable augmentation for evaluation
    )

    callbacks = [
        CollatorSwapCallback(train_collator, eval_collator)
    ]

    # PEFT's trainer integration automatically handles LoRA checkpoint saving

    if cfg.early_stopping.patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping.patience,
                early_stopping_threshold=cfg.early_stopping.threshold,
            )
        )

    processor = model.get_processor()

    training_args = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_args, dict), "training_args must be a dict"
    training_args.pop("model_dtype", None)
    training_args.pop("attn_implementation", None)

    # Disable FLOPs computation to avoid warnings with custom model architecture
    training_args["include_num_input_tokens_seen"] = False

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=train_collator,  # Callback will toggle augmentation for eval
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    trainer.save_model()


if __name__ == "__main__":
    main()
