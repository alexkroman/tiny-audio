#!/usr/bin/env python3
"""Training script for ASR models using Hydra configuration.

This script handles:
- Loading and preparing datasets from multiple sources
- Creating ASR models with configurable projector types
- Training with HuggingFace Trainer and optional WandB logging
- Checkpoint saving and Hub pushing

Usage:
    poetry run python scripts/train.py +experiments=transcription
    poetry run python scripts/train.py training.learning_rate=1e-4
"""

import contextlib
import os
import random
from dataclasses import fields
from typing import Any

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

import hydra
import torch
import wandb
from datasets import (
    Audio,
    Dataset,
    concatenate_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl.experimental.utils import DataCollatorForChatML

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel

# Prompts for ASR training (transcription task)
# Target: lowercased verbatim transcription of speech
TRANSCRIBE_PROMPTS = [
    # Direct/simple
    "Transcribe this.",
    "What is being said?",
    "What did they say?",
    "What words are spoken?",
    "Write out the speech.",
    # Slightly varied
    "Transcribe the audio.",
    "What do you hear them saying?",
    "Type out what's said.",
    "What's the transcript?",
    "Transcribe the spoken words.",
    # Question form
    "What is the speaker saying?",
    "What was spoken here?",
    "What are the words?",
    "What did you hear?",
    "What's being said?",
]

# Prompts for SIFT training (speaker description task)
# All prompts elicit comprehensive output: emotion + gender + quoted speech + voice quality
# Target format: "Sounds like a neutral woman saying '...' in a calm, even voice."
DESCRIBE_PROMPTS = [
    # Direct requests for comprehensive description
    "Describe what you hear.",
    "Describe the speaker and what they say.",
    "Describe the voice and speech.",
    "Characterize this audio.",
    "What do you hear?",
    # Slightly more specific but still comprehensive
    "Describe the speaker's voice, emotion, and words.",
    "Who is speaking and how do they sound?",
    "Summarize the speaker and their message.",
    "Describe the vocal qualities and content.",
    "What kind of voice is this and what are they saying?",
    # Informal/conversational
    "Tell me about this audio.",
    "What's going on in this clip?",
    "Break down this speech for me.",
    "Give me the details on this speaker.",
    "What can you tell me about this voice?",
]

# Prompts for audio/music captioning task (non-speech sounds)
# Target: descriptions like "The recording features a ballad with sustained strings..."
CAPTION_PROMPTS = [
    # Direct requests
    "Describe the audio.",
    "Describe what you hear.",
    "What sounds are in this recording?",
    "Describe this sound.",
    "What do you hear?",
    # Scene-focused
    "Describe the audio scene.",
    "What's happening in this audio?",
    "Describe the sounds.",
    "Caption this audio.",
    "What does this sound like?",
    # Slightly more specific
    "Describe the sound environment.",
    "What audio events do you hear?",
    "Summarize this audio.",
    "What's in this recording?",
    "Describe the audio content.",
]


class DatasetLoader:
    """Loads and prepares datasets for training."""

    def __init__(self, config: DictConfig, multitask_enabled: bool = False):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir
        self.seed = config.training.get("seed", 42)
        self.num_proc = self.config.get("num_proc", 16)
        self.multitask_enabled = multitask_enabled

    def _prepare_split(self, dataset_cfg: DictConfig, split: str):
        dataset_path = dataset_cfg.get("path")
        if not dataset_path:
            raise ValueError("Dataset path is required")

        use_streaming = dataset_cfg.get("streaming", False)
        ds = load_dataset(
            dataset_path,
            name=dataset_cfg.get("name"),
            split=split,
            cache_dir=self.cache_dir,
            num_proc=None if use_streaming else self.num_proc,
            trust_remote_code=True,
            streaming=use_streaming,
        )

        # Normalize column names for audio and text
        col_map = {
            "text": dataset_cfg.get("text_column", "text"),
            "audio": dataset_cfg.get("audio_column", "audio"),
        }
        for target, source in col_map.items():
            if source != target and source in ds.column_names:
                if target in ds.column_names:
                    ds = ds.remove_columns([target])
                ds = ds.rename_column(source, target)

        # Remove extra columns BEFORE casting to avoid schema mismatch errors
        # (some datasets have complex column types that can't be cast)
        if self.multitask_enabled:
            keep_cols = {"audio", "text", "duration", "sift_response", "caption"}
            extra_cols = [c for c in (ds.column_names or []) if c not in keep_cols]
        else:
            keep_cols = {"audio", "text", "duration"}
            extra_cols = [c for c in (ds.column_names or []) if c not in keep_cols]

        if extra_cols:
            ds = ds.remove_columns(extra_cols)

        # Cast audio column after removing problematic columns
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        # For multitask, add task column to identify sample type
        if self.multitask_enabled:
            task = dataset_cfg.get("task", "transcribe")  # Default to transcribe
            if use_streaming:
                # For streaming datasets, use map to add task column
                ds = ds.map(lambda x: {**x, "task": task})
            else:
                ds = ds.add_column("task", [task] * len(ds))

        # Filter TEDLIUM ignore markers only for TEDLIUM dataset
        # Duration filtering happens in DataCollator to avoid loading all audio upfront
        if "tedlium" in dataset_path.lower():

            def filter_tedlium(text):
                return text.strip() != "ignore_time_segment_in_scoring"

            if use_streaming:
                ds = ds.filter(filter_tedlium, input_columns="text")
            else:
                ds = ds.filter(filter_tedlium, num_proc=self.num_proc, input_columns="text")

        return ds, use_streaming

    def _resample_to_target(self, ds: Dataset, target: int) -> Dataset:
        """Upsample or downsample dataset to target size."""
        current = len(ds)
        if current == target:
            return ds
        if current > target:
            # Downsample
            return ds.select(range(target))
        # Upsample by repeating
        repeats = (target // current) + 1
        indices = list(range(current)) * repeats
        return ds.select(indices[:target])

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []
        streaming_datasets = []

        for d_cfg in tqdm(self.config.datasets, desc="Loading datasets"):
            train_splits = d_cfg.get("train_splits", [d_cfg.get("train_split", "train")])
            eval_splits = d_cfg.get("eval_splits", [d_cfg.get("eval_split", "validation")])
            target_samples = d_cfg.get("target_samples")

            for train_split in train_splits:
                ds, is_streaming = self._prepare_split(d_cfg, train_split)
                if is_streaming:
                    # For streaming datasets, take target samples and collect
                    if target_samples:
                        ds = ds.take(target_samples)
                    streaming_datasets.append(ds)
                else:
                    if target_samples:
                        ds = self._resample_to_target(ds, target_samples)
                    train_datasets.append(ds)

            for eval_split in eval_splits:
                ds, is_streaming = self._prepare_split(d_cfg, eval_split)
                if not is_streaming:
                    val_datasets.append(ds)

        # Convert streaming datasets to regular datasets
        for streaming_ds in streaming_datasets:
            samples = list(tqdm(streaming_ds, desc="Loading streaming dataset"))
            if samples:
                train_datasets.append(Dataset.from_list(samples))

        # Concatenate and shuffle
        train_ds = (
            concatenate_datasets(train_datasets).shuffle(seed=self.seed) if train_datasets else None
        )
        val_ds = concatenate_datasets(val_datasets) if val_datasets else None

        if val_ds and self.config.get("max_eval_samples"):
            n_samples = min(len(val_ds), self.config.max_eval_samples)
            val_ds = val_ds.select(range(n_samples))

        return train_ds, val_ds


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
        for f in features:
            try:
                audio = f["audio"]["array"]
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
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
            padding="longest",  # Pad to longest in batch, not fixed 30s
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

        # Lowercase all training texts
        processed_texts = [(f.get("text") or "").strip().lower() for f in valid_features]

        # Build messages for each sample with per-sample audio token counts
        text_features = []
        for text, num_audio_tokens in zip(processed_texts, audio_token_counts):
            audio_placeholder = "<audio>" * num_audio_tokens
            prompt = random.choice(TRANSCRIBE_PROMPTS)
            user_content = audio_placeholder + " " + prompt

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


# SIFT configuration for multi-task training
SIFT_SYSTEM_MESSAGE = ""  # No system message


class MultiTaskDataCollator(DataCollator):
    """Collates audio and text data for multi-task training."""

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        projector: Any = None,
        encoder_conv_layers: list = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            sample_rate=sample_rate,
            system_prompt=SIFT_SYSTEM_MESSAGE,
            projector=projector,
            encoder_conv_layers=encoder_conv_layers,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Process audio
        audio_arrays = []
        valid_features = []
        for f in features:
            try:
                audio = f["audio"]["array"]
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
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
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Compute per-sample audio token counts
        mel_lengths = audio_out.attention_mask.sum(dim=-1)
        audio_token_counts = []
        for mel_len in mel_lengths:
            encoder_len = self._compute_encoder_output_length(int(mel_len.item()))
            num_tokens = self.projector.get_output_length(encoder_len)
            audio_token_counts.append(num_tokens)

        # Build messages - differentiate between ASR, SIFT, and QA tasks using task column
        text_features = []
        for f, num_audio_tokens in zip(valid_features, audio_token_counts):
            audio_placeholder = "<audio>" * num_audio_tokens
            task = f.get("task", "transcribe")

            if task == "caption":
                # Audio/music captioning task
                response = (f.get("caption") or f.get("text") or "").strip()
                task_prompt = random.choice(CAPTION_PROMPTS)
                # Multitask prompt strategy: 50% no instruction, 50% with instruction
                if random.random() < 0.5:
                    user_content = audio_placeholder
                else:
                    user_content = f"{task_prompt}\n\n{audio_placeholder}"
            elif task == "sift":
                # SIFT task: describe audio
                response = (f.get("sift_response") or f.get("text") or "").strip()
                task_prompt = random.choice(DESCRIBE_PROMPTS)
                # Multitask prompt strategy: 50% no instruction, 50% with instruction
                if random.random() < 0.5:
                    user_content = audio_placeholder
                else:
                    user_content = f"{task_prompt}\n\n{audio_placeholder}"
            else:
                # ASR task: transcription
                response = (f.get("text") or "").strip().lower()
                task_prompt = random.choice(TRANSCRIBE_PROMPTS)
                # Multitask prompt strategy: 50% no instruction, 50% with instruction
                if random.random() < 0.5:
                    user_content = audio_placeholder
                else:
                    user_content = f"{task_prompt}\n\n{audio_placeholder}"

            # Build messages (skip system if empty)
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": response})

            text_features.append({"messages": messages})

        # Let trl handle tokenization, label masking, and padding
        batch = self.text_collator(text_features)
        batch["input_features"] = audio_out.input_features
        batch["audio_attention_mask"] = audio_out.attention_mask

        return batch


class ASRTrainer(Trainer):
    """Trainer subclass for ASR models."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with proper label shifting for causal LM.

        HuggingFace Trainer's label_smoother checks MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        to decide whether to shift labels. Since ASRModel isn't in that mapping,
        it incorrectly uses shift_labels=False, causing misaligned predictions.
        This override forces shift_labels=True for correct causal LM behavior.
        """
        # Pop labels if using label smoothing (matches Trainer behavior)
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            # Force shift_labels=True since ASRModel is a causal LM
            loss = self.label_smoother(outputs, labels, shift_labels=True)
            # Scale loss for gradient accumulation (Trainer expects this)
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


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
    # Check HF_TOKEN is set if pushing to hub
    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        import os

        if not os.environ.get("HF_TOKEN"):
            raise ValueError(
                "HF_TOKEN environment variable is required when push_to_hub is enabled. "
                "Set it with: export HF_TOKEN=your_token"
            )

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

    # Store hub_model_id in config so save_pretrained() can set base_model_name_or_path correctly
    if hub_model_id := cfg.training.get("hub_model_id"):
        model.config.pretrained_model_path = hub_model_id

    # Disable Qwen3 thinking mode by patching the chat template
    # This is a workaround for TRL's DataCollatorForChatML not passing enable_thinking=False
    # See: https://github.com/huggingface/trl/issues/3387
    if model.tokenizer.chat_template and "enable_thinking" in model.tokenizer.chat_template:
        # Replace the conditional check with a hardcoded False
        model.tokenizer.chat_template = model.tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "true",  # Always disable thinking
        )

    # Check if multi-task training is enabled
    multitask_enabled = cfg.get("multitask", {}).get("enabled", False)

    # Load datasets
    train_dataset, val_dataset = DatasetLoader(cfg, multitask_enabled=multitask_enabled).load()

    # Create data collator (multi-task or standard)
    if multitask_enabled:
        data_collator = MultiTaskDataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=cfg.data.sample_rate,
            projector=model.projector,
            encoder_conv_layers=model.config.encoder_conv_layers,
        )
    else:
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
        # Use model's push_to_hub which properly sets base_model_name_or_path in adapter_config.json
        trainer.model.push_to_hub(
            cfg.training.hub_model_id,
            commit_message="Training complete - final model",
            private=cfg.training.get("hub_private_repo", False),
        )


if __name__ == "__main__":
    main()
