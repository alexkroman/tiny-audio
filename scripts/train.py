#!/usr/bin/env python3
"""Training script for ASR models using Hydra configuration.

This script handles:
- Loading and preparing datasets from multiple sources
- Creating ASR models with configurable projector types
- Training with HuggingFace Trainer and optional WandB logging
- Checkpoint saving and Hub pushing

Usage:
    poetry run python scripts/train.py +experiments=mlp
    poetry run python scripts/train.py training.learning_rate=1e-4
"""

import contextlib
import re
from dataclasses import fields
from typing import Any

import hydra
import nltk
import torch
import wandb
from datasets import (
    Audio,
    Dataset,
    interleave_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf
from punctuators.models import PunctCapSegModelONNX
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl.trainer.utils import DataCollatorForChatML

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel

TRANSCRIBE_PREFIX = "Transcribe: "  # Used in DataCollator, matches ASRConfig.user_prompt


class DatasetLoader:
    """Loads and prepares datasets for training."""

    def __init__(self, config: DictConfig):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir
        self.seed = config.training.get("seed", 42)
        self.num_proc = self.config.get("num_proc", 16)

    def _prepare_split(self, dataset_cfg: DictConfig, split: str):
        dataset_path = dataset_cfg.get("path")
        if not dataset_path:
            raise ValueError("Dataset path is required")

        ds = load_dataset(
            dataset_path,
            name=dataset_cfg.get("name"),
            split=split,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc,
            trust_remote_code=True,
        )

        # Normalize column names
        col_map = {
            "text": dataset_cfg.get("text_column", "text"),
            "audio": dataset_cfg.get("audio_column", "audio"),
        }
        for target, source in col_map.items():
            if source != target and source in ds.column_names:
                if target in ds.column_names:
                    ds = ds.remove_columns([target])
                ds = ds.rename_column(source, target)

        ds = ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        # Remove extra columns, keep only audio and text
        extra_cols = [c for c in (ds.column_names or []) if c not in {"audio", "text"}]
        if extra_cols:
            ds = ds.remove_columns(extra_cols)

        # Filter TEDLIUM ignore markers only for TEDLIUM dataset
        # Duration filtering happens in DataCollator to avoid loading all audio upfront
        if "tedlium" in dataset_path.lower():

            def filter_tedlium(text):
                return text.strip() != "ignore_time_segment_in_scoring"

            ds = ds.filter(filter_tedlium, num_proc=self.num_proc, input_columns="text")

        return ds

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []
        train_weights, val_weights = [], []

        for d_cfg in self.config.datasets:
            train_splits = d_cfg.get("train_splits", [d_cfg.get("train_split", "train")])
            eval_splits = d_cfg.get("eval_splits", [d_cfg.get("eval_split", "validation")])
            weight = d_cfg.get("sampling_weight", 1.0)

            # Handle multiple configs or direct splits
            if "configs" in d_cfg:
                configs = [
                    (OmegaConf.create({**d_cfg, "name": c}), t, e)
                    for c, t, e in zip(d_cfg.configs, train_splits, eval_splits)
                ]
                for cfg, train_split, eval_split in configs:
                    train_datasets.append(self._prepare_split(cfg, train_split))
                    train_weights.append(weight)
                    if eval_split:
                        val_datasets.append(self._prepare_split(cfg, eval_split))
                        val_weights.append(weight)
            else:
                # Add all train splits
                for train_split in train_splits:
                    train_datasets.append(self._prepare_split(d_cfg, train_split))
                    train_weights.append(weight)
                # Add all eval splits (once, not per train split)
                for eval_split in eval_splits:
                    val_datasets.append(self._prepare_split(d_cfg, eval_split))
                    val_weights.append(weight)

        # Skip samples BEFORE combining/shuffling (for stage 2 to skip stage 1 data)
        skip_samples = self.config.get("skip_train_samples", 0)
        if skip_samples:
            print(f"Skipping first {skip_samples} training samples")
            train_datasets = [ds.skip(skip_samples) for ds in train_datasets]

        train_ds = self._combine_datasets(train_datasets, train_weights, shuffle=True)
        val_ds = self._combine_datasets(val_datasets, val_weights, shuffle=False)

        if train_ds and self.config.max_train_samples:
            train_ds = train_ds.take(self.config.max_train_samples)
        if val_ds and self.config.max_eval_samples:
            val_ds = val_ds.take(self.config.max_eval_samples)

        return train_ds, val_ds

    def _combine_datasets(self, datasets: list, weights: list, shuffle: bool = True):
        if not datasets:
            return None
        # Shuffle each dataset before interleaving for better randomization
        if shuffle:
            datasets = [ds.shuffle(seed=self.seed) for ds in datasets]
        if len(datasets) == 1:
            return datasets[0]
        probs = [w / sum(weights) for w in weights]
        return interleave_datasets(
            datasets, probabilities=probs, stopping_strategy="first_exhausted"
        )


class DataCollator:
    """Collates audio and text data for training."""

    # Default conv layers for Whisper/GLM-ASR: [(pad, kernel, stride), ...]
    DEFAULT_ENCODER_CONV_LAYERS = [(1, 3, 1), (1, 3, 2)]

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        system_prompt: str = None,
        projector: Any = None,
        encoder_conv_layers: list = None,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.system_prompt = system_prompt
        self.projector = projector
        self.encoder_conv_layers = encoder_conv_layers or self.DEFAULT_ENCODER_CONV_LAYERS

        # Use trl's DataCollatorForChatML for label masking
        # max_length needs to accommodate audio tokens (1500 for 30s) + prompt + response
        self.text_collator = DataCollatorForChatML(
            tokenizer=tokenizer,
            max_length=2048,
        )

        # Punctuation and truecasing model (lazy loaded)
        self.punctuator = None

    def _compute_encoder_output_length(self, mel_length: int) -> int:
        """Compute encoder output length using conv layer formulas."""
        length = mel_length
        for padding, kernel_size, stride in self.encoder_conv_layers:
            length = (length + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        return length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Process audio
        audio_arrays = []
        valid_features = []
        min_duration = 0.5  # Skip audio shorter than 0.5 seconds
        for f in features:
            try:
                audio = f["audio"]["array"]
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                # Skip very short audio samples
                if len(audio) / self.sample_rate < min_duration:
                    continue
                audio_arrays.append(audio)
                valid_features.append(f)
            except Exception:
                continue
            finally:
                f["audio"] = None

        if not audio_arrays:
            raise ValueError("No valid audio samples in batch")

        audio_out = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Compute per-sample audio token counts (like GlmAsr)
        mel_lengths = audio_out.attention_mask.sum(dim=-1)  # Per-sample mel lengths
        audio_token_counts = []
        for mel_len in mel_lengths:
            encoder_len = self._compute_encoder_output_length(int(mel_len.item()))
            num_tokens = self.projector.get_output_length(encoder_len)
            audio_token_counts.append(num_tokens)

        # Apply punctuation and truecasing to texts in batch (model expects lowercase)
        if self.punctuator is None:
            self.punctuator = PunctCapSegModelONNX.from_pretrained("pcs_en")

        def strip_html(text: str) -> str:
            """Remove HTML tags from text."""
            return re.sub(r"<[^>]+>", "", text)

        raw_texts = [strip_html(f.get("text") or "").strip().lower() for f in valid_features]
        results = self.punctuator.infer(raw_texts)
        processed_texts = [" ".join(r) if r else t for t, r in zip(raw_texts, results)]

        # Build messages for each sample with per-sample audio token counts
        text_features = []
        for text, num_audio_tokens in zip(processed_texts, audio_token_counts):
            audio_placeholder = "<audio>" * num_audio_tokens
            user_content = TRANSCRIBE_PREFIX + audio_placeholder

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": text})

            text_features.append({"messages": messages})

        # Let trl handle tokenization, label masking, and padding
        batch = self.text_collator(text_features)
        batch["input_features"] = audio_out.input_features
        batch["audio_attention_mask"] = audio_out.attention_mask

        return batch


class ASRTrainer(Trainer):
    """Trainer subclass for ASR models."""

    pass


class PushToHubCallback(TrainerCallback):
    """Pushes model to Hub on every save."""

    def on_save(self, args, state, control, **kwargs):
        if not (args.push_to_hub and args.hub_model_id):
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        with contextlib.suppress(Exception):
            model.push_to_hub(
                repo_id=args.hub_model_id,
                commit_message=f"Training in progress - step {state.global_step}",
                private=args.hub_private_repo,
            )

        return control


def get_valid_training_args(config: dict) -> dict:
    """Filter config to only valid TrainingArguments fields."""
    valid_fields = {f.name for f in fields(TrainingArguments)}
    return {k: v for k, v in config.items() if k in valid_fields}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    nltk.download("punkt_tab", quiet=True)

    # Initialize wandb
    if cfg.training.get("report_to") == "wandb":
        wandb.init(
            project=cfg.training.get("wandb_project", "tiny-audio"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Create model config from hydra config
    # Merge model config with training config for model-specific params
    model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_config_dict, dict), "model config must be a dict"
    # Add training params that affect model behavior
    training_model_params = [
        "label_smoothing",
        "projector_dropout",
        "use_specaugment",
        "num_time_masks",
        "time_mask_length",
        "num_freq_masks",
        "freq_mask_length",
        "attn_implementation",
        # LoRA params (Stage 2 fine-tuning)
        "use_lora",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "freeze_projector",
    ]
    for param in training_model_params:
        if cfg.training.get(param) is not None:
            model_config_dict[param] = cfg.training[param]
    asr_config = ASRConfig(**model_config_dict)

    # Load or create model
    if cfg.model.get("pretrained_model_path"):
        model = ASRModel.from_pretrained(cfg.model.pretrained_model_path, config=asr_config)
    else:
        model = ASRModel(asr_config)

    model.config.use_cache = False

    # Disable Qwen3 thinking mode by patching the chat template
    # This is a workaround for TRL's DataCollatorForChatML not passing enable_thinking=False
    # See: https://github.com/huggingface/trl/issues/3387
    if model.tokenizer.chat_template and "enable_thinking" in model.tokenizer.chat_template:
        # Replace the conditional check with a hardcoded False
        model.tokenizer.chat_template = model.tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "true",  # Always disable thinking
        )

    # Load datasets
    train_dataset, val_dataset = DatasetLoader(cfg).load()

    # Create data collator
    data_collator = DataCollator(
        tokenizer=model.tokenizer,
        feature_extractor=model.feature_extractor,
        sample_rate=cfg.data.sample_rate,
        system_prompt=cfg.model.system_prompt,
        projector=model.projector,
        encoder_conv_layers=model.config.encoder_conv_layers,
    )

    # Setup callbacks
    callbacks = []
    if cfg.early_stopping.patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping.patience,
                early_stopping_threshold=cfg.early_stopping.threshold,
            )
        )
    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        callbacks.append(PushToHubCallback())

    # Configure torch.compile if specified
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_config, dict)
    if compile_config := training_config.pop("torch_compile_config", None):
        torch._dynamo.config.cache_size_limit = compile_config.get("cache_size_limit", 64)
        torch._dynamo.config.capture_scalar_outputs = compile_config.get(
            "capture_scalar_outputs", True
        )
        torch._inductor.config.compile_threads = compile_config.get("compile_threads", 4)

    # Create trainer with only valid TrainingArguments
    valid_args = get_valid_training_args(training_config)
    trainer = ASRTrainer(
        model=model,
        args=TrainingArguments(**valid_args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    trainer.save_model()

    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        trainer.push_to_hub(commit_message="Training complete - final model")


if __name__ == "__main__":
    main()
