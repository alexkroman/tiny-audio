#!/usr/bin/env python3

import logging
import re
from typing import Any, Dict, List

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


class DatasetLoader:
    def __init__(self, config: DictConfig):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir
        self.seed = config.training.get("seed", 42)  # Get seed from training config

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

        # Normalize column names
        for target_col, source_col in [
            ("text", dataset_cfg.get("text_column", "text")),
            ("audio", dataset_cfg.get("audio_column", "audio")),
        ]:
            if source_col != target_col and source_col in ds.column_names:
                if target_col in ds.column_names:
                    ds = ds.remove_columns([target_col])
                ds = ds.rename_column(source_col, target_col)

        # Cast audio column to correct format
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        # Get task for this dataset
        task = dataset_cfg.get("task", "transcribe")

        # Get original features before any modifications
        original_features = ds.info.features if hasattr(ds, "info") and ds.info else None

        # Remove extra columns that might cause feature conflicts
        # Keep only: audio, text
        columns_to_keep = {"audio", "text"}
        current_columns = ds.column_names if hasattr(ds, "column_names") and ds.column_names else []
        columns_to_remove = [col for col in current_columns if col not in columns_to_keep]

        if columns_to_remove:
            ds = ds.remove_columns(columns_to_remove)

        # Add task using a simple wrapper generator
        def add_task_gen(dataset, task_val):
            for example in dataset:
                # Skip samples with invalid/empty audio
                try:
                    audio_decoder = example.get("audio")
                    if audio_decoder is None:
                        continue
                    # Test if audio can be decoded
                    _ = audio_decoder.get_all_samples()
                except (RuntimeError, Exception):
                    # Skip samples that fail to decode
                    continue

                # Skip TEDLIUM segments marked to be ignored
                text = example.get("text", "")
                if isinstance(text, str):
                    if text.strip() == "ignore_time_segment_in_scoring":
                        continue

                example["task"] = task_val
                yield example

        # Build new features dict from original, adding task
        if original_features:
            new_features = {k: v for k, v in original_features.items() if k in columns_to_keep}
            new_features["task"] = Value("string")
            new_features = Features(new_features)
        else:
            # Fallback if no features available
            new_features = Features(
                {
                    "audio": Audio(sampling_rate=self.sample_rate),
                    "text": Value("string"),
                    "task": Value("string"),
                }
            )

        # Create new dataset with explicit features
        return IterableDataset.from_generator(
            add_task_gen,
            gen_kwargs={"dataset": ds, "task_val": task},
            features=new_features,
        )

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []
        train_weights, val_weights = [], []

        for d_cfg in self.config.datasets:
            train_splits = d_cfg.get("train_splits", [d_cfg.get("train_split", "train")])
            eval_splits = d_cfg.get("eval_splits", [d_cfg.get("eval_split", "validation")])
            weight = d_cfg.get("sampling_weight", 1.0)

            # Process configs if present, otherwise use direct splits
            configs_to_process = []
            if "configs" in d_cfg:
                for config, train, eval in zip(d_cfg.configs, train_splits, eval_splits):
                    cfg = OmegaConf.create(d_cfg)
                    cfg.name = config
                    configs_to_process.append((cfg, train, eval))
            else:
                configs_to_process = [
                    (d_cfg, train, eval) for train in train_splits for eval in eval_splits
                ]

            for cfg, train_split, eval_split in configs_to_process:
                train_datasets.append(self._prepare_split(cfg, train_split))
                train_weights.append(weight)
                if eval_split:
                    val_datasets.append(self._prepare_split(cfg, eval_split))
                    val_weights.append(weight)

        # Helper to combine datasets with weights
        def combine_datasets(datasets, weights):
            if not datasets:
                return None
            if len(datasets) == 1:
                return datasets[0]
            # Normalize weights and interleave
            probs = [w / sum(weights) for w in weights]
            return interleave_datasets(datasets, probabilities=probs)

        train_ds = combine_datasets(train_datasets, train_weights)
        val_ds = combine_datasets(val_datasets, val_weights)

        # Shuffle and limit datasets
        if train_ds:
            train_ds = train_ds.shuffle(seed=self.seed, buffer_size=1000)
            if self.config.max_train_samples:
                train_ds = train_ds.take(self.config.max_train_samples)
        if val_ds and self.config.max_eval_samples:
            val_ds = val_ds.take(self.config.max_eval_samples)

        return train_ds, val_ds


class DataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        system_prompt: str = None,
    ):
        super().__init__(tokenizer=tokenizer, padding=True)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.system_prompt = system_prompt

        # Check if this is a Whisper feature extractor
        self.is_whisper = feature_extractor.__class__.__name__ == "WhisperFeatureExtractor"

        # Use tokenizer's normalize method if available, otherwise use WhisperTokenizer for normalization
        # The Whisper normalizer is a standard text preprocessing utility
        if hasattr(tokenizer, "normalize"):
            self.text_normalizer = tokenizer
        else:
            # Fallback to whisper-tiny tokenizer for its normalize() method only
            self.text_normalizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before Whisper normalization (matches eval script)."""
        if not isinstance(text, str) or text is None:
            return ""

        # Apply all text replacements
        replacements = {
            r"<PERIOD>": ".",
            r"<COMMA>": ",",
            r"<QUESTIONMARK>": "?",
            r"<EXCLAMATIONPOINT>": "!",
            r"<inaudible>": "",
            r"\b(uh|um|ah)\b": "",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Strip any remaining HTML-like tags
        text = re.sub(r'<[^>]+>', '', text)

        return text

    def _normalize_text(self, text: str) -> str:
        """Apply Whisper normalization (matches eval script)."""
        return self.text_normalizer.normalize(self._preprocess_text(text))

    def _extract_audio(self, audio_decoder) -> Any:
        """Extract and normalize audio to mono."""
        audio_array = audio_decoder.get_all_samples().data.numpy().squeeze()

        # Convert to mono if multi-channel
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0)
        elif audio_array.ndim == 0:
            audio_array = audio_array.reshape(1)

        return audio_array

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [self._extract_audio(f["audio"]) for f in features]

        # Extract audio features with attention mask
        # For Whisper: padding="max_length" pads to 3000 frames (30 seconds)
        # For Wav2Vec2: padding=True pads to longest in batch
        padding_strategy = "max_length" if self.is_whisper else True

        audio_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding=padding_strategy,
            return_tensors="pt",
            return_attention_mask=True,  # Required for both Whisper and Wav2Vec2
        )

        text_features = []
        for f in features:
            text = f["text"].strip() if isinstance(f["text"], str) else f["text"]

            # Apply Whisper normalization (matches eval script preprocessing)
            text = self._normalize_text(text)

            # Apply truecasing to restore proper capitalization
            text = text.replace("<COMMA>", ",").replace("<PERIOD>", ".")
            text = truecase.get_true_case(text)

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            # Choose prompt based on task type
            task = f.get("task", "transcribe")  # Get task field added by add_column

            # Use default single prompt per task
            if task == "continue":
                instruction = "Continue: <audio>"
            elif task == "describe":
                instruction = "Describe: <audio>"
            elif task == "emotion":
                instruction = "Emotion: <audio>"
            else:  # Default to transcribe
                instruction = "Transcribe: <audio>"

            messages.append({"role": "user", "content": instruction})
            messages.append({"role": "assistant", "content": text})

            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=1024,
                enable_thinking=False,
            )

            # Create labels - only train on the actual transcription text
            labels = [-100] * len(tokens)

            # Helper to skip whitespace tokens
            def skip_whitespace(idx, token_list):
                while (
                    idx < len(token_list)
                    and self.tokenizer.decode([int(token_list[idx])]).strip() == ""
                ):
                    idx += 1
                return idx

            # Find content boundaries
            special_tokens = {
                "im_end": self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                "think_end": self.tokenizer.convert_tokens_to_ids("</think>"),
                "im_start": self.tokenizer.convert_tokens_to_ids("<|im_start|>"),
                "assistant": self.tokenizer.convert_tokens_to_ids("assistant"),
            }

            # Find content start
            content_start = -1
            # Check for thinking tag end
            if special_tokens["think_end"] in tokens:
                idx = tokens.index(special_tokens["think_end"])
                content_start = skip_whitespace(idx + 1, tokens)
            else:
                # Look for assistant marker
                for i in range(len(tokens) - 2):
                    if (
                        tokens[i] == special_tokens["im_start"]
                        and tokens[i + 1] == special_tokens["assistant"]
                    ):
                        content_start = skip_whitespace(i + 2, tokens)
                        break

            # Find content end (im_end token)
            content_end = -1
            if content_start > 0 and special_tokens["im_end"] in tokens[content_start:]:
                content_end = tokens[content_start:].index(special_tokens["im_end"]) + content_start

            # Unmask the actual transcription including EOS token
            if content_start > 0 and content_end > 0:
                labels[content_start : content_end + 1] = tokens[content_start : content_end + 1]

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


class PushToHubCallback(TrainerCallback):
    """Custom callback to push model to hub root directory on every save."""

    def on_save(self, args, state, control, **kwargs):
        """Called after a checkpoint is saved."""
        if args.push_to_hub and args.hub_model_id:
            # Get the model from kwargs
            model = kwargs.get("model")
            if model is not None:
                print(f"\n📤 Pushing checkpoint (step {state.global_step}) to Hub root...")
                try:
                    # model.push_to_hub() will call model.save_pretrained() internally
                    # which triggers your custom save logic (encoder, decoder, projector split)
                    commit_message = f"Training in progress - step {state.global_step}"
                    model.push_to_hub(
                        repo_id=args.hub_model_id,
                        commit_message=commit_message,
                        private=args.hub_private_repo,
                    )
                    print(f"✅ Successfully pushed to {args.hub_model_id}")
                except Exception as e:
                    print(f"⚠️  Failed to push to hub: {e}")
        return control


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    # Use HuggingFace's logging utilities
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()  # Reduces transformer trainer logs

    # Suppress HTTP and dataset loading logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Download NLTK data for truecasing
    nltk.download("punkt_tab")

    # Log configuration if needed
    if cfg.get("verbose", False):
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
        system_prompt=cfg.model.system_prompt,
        encoder_dim=encoder_config.hidden_size,
        llm_dim=decoder_config.hidden_size,
        projector_hidden_dim=cfg.model.get("projector_hidden_dim"),
        use_specaugment=cfg.training.get("use_specaugment", False),
        label_smoothing=cfg.training.get("label_smoothing", 0.1),
    )

    # Load from pretrained if specified, otherwise create new model
    if cfg.model.get("pretrained_model_path"):
        print(f"Loading pretrained model from: {cfg.model.pretrained_model_path}")
        model = ASRModel.from_pretrained(
            cfg.model.pretrained_model_path,
            config=asr_config,
        )
    else:
        model = ASRModel(asr_config)

    # Disable cache during training (required for gradient checkpointing)
    model.config.use_cache = False

    train_dataset, val_dataset = DatasetLoader(cfg).load()

    # Create data collator for both training and evaluation
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

    # Add custom push to hub callback if configured
    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        callbacks.append(PushToHubCallback())

    # Convert config to dict and remove non-TrainingArguments fields
    training_args = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_args, dict), "training_args must be a dict"

    # Apply torch.compile config if present
    if compile_config := training_args.pop("torch_compile_config", None):
        # Configure torch compilation settings
        torch._dynamo.config.cache_size_limit = compile_config.get("cache_size_limit", 64)
        torch._dynamo.config.capture_scalar_outputs = compile_config.get(
            "capture_scalar_outputs", True
        )
        torch._dynamo.config.allow_unspec_int_on_nn_module = compile_config.get(
            "allow_unspec_int_on_nn_module", True
        )
        torch._inductor.config.compile_threads = compile_config.get("compile_threads", 4)

    # Remove non-TrainingArguments fields (model/projector configs)
    non_training_args = [
        "torch_compile_dynamic",
        "torch_compile_backend",
        "torch_compile_mode",
        "torch_compile_fullgraph",
        "model_dtype",
        "attn_implementation",
        "use_specaugment",
        "label_smoothing",
    ]

    # Also remove any projector-specific configs
    projector_keys = [k for k in training_args.keys() if k.startswith("projector_")]
    for key in non_training_args + projector_keys:
        training_args.pop(key, None)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Check for checkpoint resumption
    resume_from_checkpoint = cfg.training.get("resume_from_checkpoint", None)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()

    # Push final model to hub if configured
    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        print(f"Pushing final model to Hub: {cfg.training.hub_model_id}")
        trainer.push_to_hub(commit_message="Training complete - final model")


if __name__ == "__main__":
    main()
