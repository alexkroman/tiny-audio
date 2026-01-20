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
    interleave_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl.experimental.utils import DataCollatorForChatML

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel

# Transcription prompts (randomly selected during training)
# Audio tokens come BEFORE the prompt for proper causal attention
TRANSCRIBE_PROMPTS = [
    "Repeat the above",
    "Transcribe speech to text",
    "Transcribe audio to text",
]

# Task prompts; config can override per-task
TASK_PROMPTS = {
    "transcription": TRANSCRIBE_PROMPTS,
    "speaker_id": ["Who is speaking?", "Identify the speaker"],
}

# Task weights for multi-task training
DEFAULT_TASK_WEIGHTS = {
    "transcription": 1.0,
    "speaker_id": 1.0,
}


class DatasetLoader:
    """Loads and prepares datasets for training."""

    # Task-specific column mappings (target -> possible source column names)
    TASK_COLUMNS = {
        "speaker": ["speaker_column", "speaker_id", "speaker"],
    }

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

        ds = load_dataset(
            dataset_path,
            name=dataset_cfg.get("name"),
            split=split,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc,
            trust_remote_code=True,
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

        ds = ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        # Handle multi-task columns
        if self.multitask_enabled:
            supported_tasks = list(dataset_cfg.get("tasks", ["transcription"]))
            # Map task columns from config-specified names to standard names
            task_col_map = {}
            for target, source_options in self.TASK_COLUMNS.items():
                # Check config for explicit column name
                for opt in source_options:
                    source = dataset_cfg.get(opt)
                    if source and source in ds.column_names:
                        task_col_map[target] = source
                        break
                    if opt in ds.column_names:
                        task_col_map[target] = opt
                        break

            # Rename task columns to standard names
            for target, source in task_col_map.items():
                if source != target and source in ds.column_names:
                    if target in ds.column_names:
                        ds = ds.remove_columns([target])
                    ds = ds.rename_column(source, target)

            # Add supported_tasks column
            ds = ds.map(
                lambda _: {"_supported_tasks": supported_tasks},
                num_proc=self.num_proc,
            )

            # Keep audio, text, duration, task columns, and _supported_tasks
            keep_cols = {"audio", "text", "duration", "_supported_tasks"}
            keep_cols.update(self.TASK_COLUMNS.keys())
            extra_cols = [c for c in (ds.column_names or []) if c not in keep_cols]
        else:
            # Remove extra columns, keep only audio, text, and duration (for group_by_length)
            extra_cols = [
                c for c in (ds.column_names or []) if c not in {"audio", "text", "duration"}
            ]

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
            # Audio BEFORE prompt for proper causal attention (matches Auden)
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


# SIFT system message for multi-task training
# Describes the model's capabilities without requiring specific tag formats
SIFT_SYSTEM_MESSAGE = (
    "You are a powerful virtual human who is capable of perceiving speech inputs "
    "and generating precise natural responses. "
    "You can understand both what is being said and how it is being said, "
    "including the speaker's emotion, age, and gender."
)

# SIFT instruction for sit_ssp mode (matches Azeros)
SIFT_INSTRUCTION = "Describe all information you can hear."


class MultiTaskDataCollator(DataCollator):
    """Collates audio and text data for multi-task training.

    Extends DataCollator with task selection and weighted sampling.
    Supports SIFT training using preprocessed sift_response column.

    SIFT modes (1/4 each when mixed):
    - transcription: Standard transcription with prompts (like mlp.yaml)
    - sift_s: Semantic only (just transcription, no system message, no prompt)
    - sift_ssp: Preprocessed sift_response with system message, no instruction
    - sit_ssp: Preprocessed sift_response with system message + instruction
    """

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        task_config: dict,
        system_prompt: str = None,
        projector: Any = None,
        encoder_conv_layers: list = None,
        sift_mode: str = None,  # None, "transcription", "sift_s", "sift_ssp", "sit_ssp", or "mixed"
    ):
        super().__init__(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            sample_rate=sample_rate,
            system_prompt=system_prompt,
            projector=projector,
            encoder_conv_layers=encoder_conv_layers,
        )
        self.task_config = task_config
        self.sift_mode = sift_mode

        # Use SIFT system message for paralinguistic tasks
        if sift_mode in ("sift_ssp", "sit_ssp", "mixed"):
            self.system_prompt = SIFT_SYSTEM_MESSAGE

        # Extract task weights from config
        self.task_weights = {
            name: cfg.get("weight", DEFAULT_TASK_WEIGHTS.get(name, 1.0))
            for name, cfg in task_config.items()
        }
        # Extract task prompts from config (with fallback to defaults)
        self.task_prompts = {
            name: list(cfg.get("prompts", TASK_PROMPTS.get(name, [])))
            for name, cfg in task_config.items()
        }

    def _select_sift_mode(self) -> str:
        """Select SIFT mode for this sample (for mixed mode)."""
        if self.sift_mode == "mixed":
            # 1/4 each: transcription, sift_s, sift_ssp, sit_ssp
            return random.choice(["transcription", "sift_s", "sift_ssp", "sit_ssp"])
        return self.sift_mode

    def _select_task(self, sample: dict) -> tuple[str, str, str, bool]:
        """Select a task for this sample based on SIFT mode.

        Returns (task_name, prompt, response_text, use_system_prompt).

        Requires preprocessed dataset with 'sift_response' column.
        """
        mode = self._select_sift_mode()

        if mode == "transcription":
            # Standard transcription with prompts (like mlp.yaml)
            text = (sample.get("text") or "").strip().lower()
            return ("transcription", random.choice(TRANSCRIBE_PROMPTS), text, False)
        if mode == "sift_s":
            # Semantic only - no system message, no instruction, just transcript
            text = (sample.get("text") or "").strip()
            return ("transcription", "", text, False)
        if mode == "sift_ssp":
            # Preprocessed paralinguistic response, no instruction
            response = (sample.get("sift_response") or "").strip()
            if not response:
                # Fallback to transcript if no sift_response
                response = (sample.get("text") or "").strip()
            return ("sift", "", response, True)
        if mode == "sit_ssp":
            # Preprocessed paralinguistic response with instruction
            response = (sample.get("sift_response") or "").strip()
            if not response:
                response = (sample.get("text") or "").strip()
            return ("sift", SIFT_INSTRUCTION, response, True)

        # Fallback
        text = (sample.get("text") or "").strip().lower()
        return ("transcription", random.choice(TRANSCRIBE_PROMPTS), text, False)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Process audio (same as parent)
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

        # Build messages for each sample with task selection
        text_features = []
        task_ids = []  # Track which task was selected for each sample
        for f, num_audio_tokens in zip(valid_features, audio_token_counts):
            task_name, prompt, response, use_system_prompt = self._select_task(f)
            task_ids.append(task_name)

            audio_placeholder = "<audio>" * num_audio_tokens
            # Only add space before prompt if prompt is non-empty
            user_content = audio_placeholder + (" " + prompt if prompt else "")

            messages = []
            if use_system_prompt and self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": response})

            text_features.append({"messages": messages})

        # Let trl handle tokenization, label masking, and padding
        batch = self.text_collator(text_features)
        batch["input_features"] = audio_out.input_features
        batch["audio_attention_mask"] = audio_out.attention_mask
        batch["task_ids"] = task_ids  # Pass task IDs for weighted loss

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
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class MultiTaskASRTrainer(ASRTrainer):
    """Trainer subclass for multi-task ASR training with weighted loss.

    Applies per-task weights to compute a weighted average loss,
    matching Auden's multi-task training approach.
    """

    def __init__(self, task_weights: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.task_weights = task_weights or DEFAULT_TASK_WEIGHTS
        # Track per-task losses for logging
        self._task_loss_accum = {}
        self._task_count_accum = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted loss across tasks.

        Each sample's loss is weighted by its task weight.
        Per-task losses are tracked for WandB logging.
        """
        # Extract task_ids before model forward (not a model input)
        task_ids = inputs.pop("task_ids", None)

        # Pop labels if using label smoothing (matches Trainer behavior)
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            # Force shift_labels=True since ASRModel is a causal LM
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = outputs.loss

        # Apply task weights if we have task_ids
        if task_ids is not None and loss is not None:
            # Get weights for each sample in the batch
            weights = torch.tensor(
                [self.task_weights.get(t, 1.0) for t in task_ids],
                device=loss.device,
                dtype=loss.dtype,
            )
            # Weight the loss (assuming batch reduction)
            weighted_loss = loss * weights.mean()

            # Track per-task losses for logging
            for task_name in set(task_ids):
                task_mask = [1 if t == task_name else 0 for t in task_ids]
                task_count = sum(task_mask)
                if task_count > 0:
                    if task_name not in self._task_loss_accum:
                        self._task_loss_accum[task_name] = 0.0
                        self._task_count_accum[task_name] = 0
                    # Approximate per-task loss contribution
                    self._task_loss_accum[task_name] += loss.item() * task_count
                    self._task_count_accum[task_name] += task_count

            loss = weighted_loss

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time: float | None = None):
        """Override log to add per-task loss metrics."""
        # Add per-task losses to logs
        for task_name, total_loss in self._task_loss_accum.items():
            count = self._task_count_accum.get(task_name, 1)
            if count > 0:
                logs[f"train_loss_{task_name}"] = total_loss / count

        # Reset accumulators after logging
        self._task_loss_accum = {}
        self._task_count_accum = {}

        super().log(logs, start_time)


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
        task_config = dict(cfg.multitask.get("tasks", {}))
        sift_mode = cfg.multitask.get("sift_mode")
        data_collator = MultiTaskDataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=cfg.data.sample_rate,
            task_config=task_config,
            system_prompt=cfg.model.system_prompt,
            projector=model.projector,
            encoder_conv_layers=model.config.encoder_conv_layers,
            sift_mode=sift_mode,
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
    if multitask_enabled:
        # Extract task weights for the trainer
        task_weights = {
            name: task_cfg.get("weight", DEFAULT_TASK_WEIGHTS.get(name, 1.0))
            for name, task_cfg in task_config.items()
        }
        trainer = MultiTaskASRTrainer(
            task_weights=task_weights,
            model=model,
            args=TrainingArguments(**valid_args),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
    else:
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
