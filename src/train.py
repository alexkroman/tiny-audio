#!/usr/bin/env python3

import os
import re
from dataclasses import fields
from typing import Any

import hydra
import nltk
import torch
import truecase
from datasets import (
    Audio,
    Dataset,
    Features,
    IterableDataset,
    Value,
    interleave_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    WhisperTokenizer,
)

import wandb
from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel

# Shared task prompts (matches ASRModel.TASK_PROMPTS)
TASK_PROMPTS = {
    "transcribe": "Transcribe: <audio>",
    "continue": "Continue: <audio>",
    "describe": "Describe: <audio>",
    "emotion": "Emotion: <audio>",
}


class DatasetLoader:
    """Loads and prepares streaming datasets for training."""

    def __init__(self, config: DictConfig):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir
        self.seed = config.training.get("seed", 42)
        self.max_audio_duration = self.config.get("max_audio_duration_seconds", 30.0)

    def _prepare_split(self, dataset_cfg: DictConfig, split: str) -> IterableDataset:
        dataset_path = dataset_cfg.get("path")
        if not dataset_path:
            raise ValueError("Dataset path is required")

        ds = load_dataset(
            dataset_path,
            name=dataset_cfg.get("name"),
            split=split,
            streaming=True,
            cache_dir=self.cache_dir,
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

        task = dataset_cfg.get("task", "transcribe")
        features = Features(
            {
                "audio": Audio(sampling_rate=self.sample_rate),
                "text": Value("string"),
                "task": Value("string"),
            }
        )

        return IterableDataset.from_generator(
            self._add_task_generator,
            gen_kwargs={"dataset": ds, "task": task, "max_audio_duration": self.max_audio_duration},
            features=features,
        )

    @staticmethod
    def _add_task_generator(dataset, task: str, max_audio_duration: float = 30.0):
        """Generator that adds task field and filters invalid samples."""
        for example in dataset:
            # Skip invalid audio
            audio = example.get("audio")
            if audio is None:
                continue

            try:
                # Soundfile backend returns dict with 'array' and 'sampling_rate'
                if not isinstance(audio, dict) or "array" not in audio:
                    continue

                arr = audio["array"]
                sample_rate = audio.get("sampling_rate", 16000)
                num_samples = len(arr) if hasattr(arr, "__len__") else arr.shape[-1]

                # Skip audio that's too long (causes OOM)
                duration = num_samples / sample_rate
                if duration > max_audio_duration:
                    continue
            except Exception:
                continue

            # Skip TEDLIUM ignore markers
            text = example.get("text", "")
            if isinstance(text, str) and text.strip() == "ignore_time_segment_in_scoring":
                continue

            example["task"] = task
            yield example

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

        train_ds = self._combine_datasets(train_datasets, train_weights, shuffle=True)
        val_ds = self._combine_datasets(val_datasets, val_weights, shuffle=False)

        if train_ds:
            if self.config.max_train_samples:
                train_ds = train_ds.take(self.config.max_train_samples)
        if val_ds and self.config.max_eval_samples:
            val_ds = val_ds.take(self.config.max_eval_samples)

        return train_ds, val_ds

    def _combine_datasets(self, datasets: list, weights: list, shuffle: bool = True):
        if not datasets:
            return None
        # Shuffle each dataset before interleaving for better randomization
        if shuffle:
            datasets = [ds.shuffle(seed=self.seed, buffer_size=1000) for ds in datasets]
        if len(datasets) == 1:
            return datasets[0]
        probs = [w / sum(weights) for w in weights]
        return interleave_datasets(datasets, probabilities=probs)


class DataCollator(DataCollatorForSeq2Seq):
    """Collates audio and text data for training."""

    # Text preprocessing patterns
    TEXT_REPLACEMENTS = {
        r"<PERIOD>": ".",
        r"<COMMA>": ",",
        r"<QUESTIONMARK>": "?",
        r"<EXCLAMATIONPOINT>": "!",
        r"<inaudible>": "",
        r"\b(uh|um|ah)\b": "",
    }

    def __init__(
        self, tokenizer: Any, feature_extractor: Any, sample_rate: int, system_prompt: str = None
    ):
        super().__init__(tokenizer=tokenizer, padding=True)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.system_prompt = system_prompt
        self.is_whisper = feature_extractor.__class__.__name__ == "WhisperFeatureExtractor"

        # Text normalizer
        self.text_normalizer = (
            tokenizer
            if hasattr(tokenizer, "normalize")
            else WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        )

        # Cache special token IDs
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._assistant_id = tokenizer.convert_tokens_to_ids("assistant")
        self._think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    def _normalize_text(self, text: str) -> str:
        """Preprocess and normalize text."""
        if not isinstance(text, str):
            return ""
        for pattern, repl in self.TEXT_REPLACEMENTS.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)  # Strip HTML-like tags
        return self.text_normalizer.normalize(text)

    def _find_assistant_content_range(self, tokens: list[int]) -> tuple[int, int]:
        """Find the start and end indices of assistant content for label masking."""
        # Try to find </think> tag first (for thinking models)
        if self._think_end_id in tokens:
            start = tokens.index(self._think_end_id) + 1
        else:
            # Find <|im_start|>assistant pattern
            start = -1
            for i in range(len(tokens) - 1):
                if tokens[i] == self._im_start_id and tokens[i + 1] == self._assistant_id:
                    start = i + 2
                    break

        if start < 0:
            return -1, -1

        # Skip whitespace tokens
        while start < len(tokens) and self.tokenizer.decode([tokens[start]]).strip() == "":
            start += 1

        # Find <|im_end|> after content start
        try:
            end = tokens.index(self._im_end_id, start)
        except ValueError:
            return -1, -1

        return start, end

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Process audio - extract numpy arrays from soundfile dict format
        audio_arrays = []
        valid_features = []
        for f in features:
            try:
                audio_obj = f["audio"]
                # Soundfile backend returns dict with 'array' and 'sampling_rate'
                audio = audio_obj["array"]
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                elif audio.ndim == 0:
                    audio = audio.reshape(1)
                audio_arrays.append(audio)
                valid_features.append(f)
            except Exception:
                # Skip corrupted audio files silently
                continue
            finally:
                f["audio"] = None

        if not audio_arrays:
            raise ValueError("No valid audio samples in batch - all samples were corrupted")

        features = valid_features

        audio_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding="max_length" if self.is_whisper else True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Process text
        text_features = []
        for f in features:
            text = self._normalize_text(f.get("text", ""))
            text = truecase.get_true_case(text)

            task = f.get("task", "transcribe")
            instruction = TASK_PROMPTS.get(task, TASK_PROMPTS["transcribe"])

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": instruction})
            messages.append({"role": "assistant", "content": text})

            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=512,
                enable_thinking=False,
            )

            # Ensure <|im_end|> is present - truncation may have cut it off
            if tokens[-1] != self._im_end_id:
                tokens = tokens[:-1] + [self._im_end_id]  # Replace last token with im_end

            # Create labels - mask everything except assistant content
            labels = [-100] * len(tokens)
            start, end = self._find_assistant_content_range(tokens)
            if start > 0 and end > 0:
                labels[start : end + 1] = tokens[start : end + 1]

            text_features.append({"input_ids": tokens, "labels": labels})

        batch = super().__call__(text_features)

        # Add audio to batch
        if "input_values" in audio_features:
            batch["input_values"] = audio_features.input_values
        elif "input_features" in audio_features:
            batch["input_features"] = audio_features.input_features
        if "attention_mask" in audio_features:
            batch["audio_attention_mask"] = audio_features.attention_mask

        return batch


class PushToHubCallback(TrainerCallback):
    """Pushes model to Hub on every save."""

    def on_save(self, args, state, control, **kwargs):
        if not (args.push_to_hub and args.hub_model_id):
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        print(f"\n📤 Pushing checkpoint (step {state.global_step}) to Hub...")
        try:
            model.push_to_hub(
                repo_id=args.hub_model_id,
                commit_message=f"Training in progress - step {state.global_step}",
                private=args.hub_private_repo,
            )
            print(f"✅ Successfully pushed to {args.hub_model_id}")
        except Exception as e:
            print(f"⚠️  Failed to push to hub: {e}")

        return control


def get_valid_training_args(config: dict) -> dict:
    """Filter config to only valid TrainingArguments fields."""
    valid_fields = {f.name for f in fields(TrainingArguments)}
    return {k: v for k, v in config.items() if k in valid_fields}


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    nltk.download("punkt_tab", quiet=True)

    if cfg.get("verbose"):
        print(OmegaConf.to_yaml(cfg))

    # Initialize wandb
    if cfg.training.get("report_to") == "wandb":
        wandb.init(
            project="tiny-audio",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Create model config (uses defaults from ASRConfig)
    asr_config = ASRConfig()

    # Load or create model
    if cfg.model.get("pretrained_model_path"):
        print(f"Loading pretrained model from: {cfg.model.pretrained_model_path}")
        model = ASRModel.from_pretrained(cfg.model.pretrained_model_path, config=asr_config)
    else:
        model = ASRModel(asr_config)

    model.config.use_cache = False

    # Load datasets
    train_dataset, val_dataset = DatasetLoader(cfg).load()

    # Create data collator
    data_collator = DataCollator(
        tokenizer=model.tokenizer,
        feature_extractor=model.feature_extractor,
        sample_rate=cfg.data.sample_rate,
        system_prompt=cfg.model.system_prompt,
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
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**get_valid_training_args(training_config)),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    trainer.save_model()

    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        print(f"Pushing final model to Hub: {cfg.training.hub_model_id}")
        trainer.push_to_hub(commit_message="Training complete - final model")


if __name__ == "__main__":
    main()
