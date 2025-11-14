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

        text_column = dataset_cfg.get("text_column", "text")
        if text_column != "text" and text_column in ds.column_names:
            # If "text" already exists in the dataset, remove it first
            if "text" in ds.column_names:
                ds = ds.remove_columns(["text"])
            ds = ds.rename_column(text_column, "text")

        audio_column = dataset_cfg.get("audio_column", "audio")
        if audio_column != "audio" and audio_column in ds.column_names:
            # If "audio" already exists in the dataset, remove it first
            if "audio" in ds.column_names:
                ds = ds.remove_columns(["audio"])
            ds = ds.rename_column(audio_column, "audio")

        # Cast audio column to correct format
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        # Get task for this dataset
        task = dataset_cfg.get("task", "transcribe")

        # Get original features before any modifications
        original_features = ds.info.features if hasattr(ds, 'info') and ds.info else None

        # Remove extra columns that might cause feature conflicts
        # Keep only: audio, text
        columns_to_keep = {"audio", "text"}
        current_columns = ds.column_names if hasattr(ds, 'column_names') and ds.column_names else []
        columns_to_remove = [col for col in current_columns if col not in columns_to_keep]

        if columns_to_remove:
            ds = ds.remove_columns(columns_to_remove)

        # Add task using a simple wrapper generator
        def add_task_gen(dataset, task_val):
            for example in dataset:
                example["task"] = task_val
                yield example

        # Build new features dict from original, adding task
        if original_features:
            new_features = {k: v for k, v in original_features.items() if k in columns_to_keep}
            new_features["task"] = Value("string")
            new_features = Features(new_features)
        else:
            # Fallback if no features available
            new_features = Features({
                "audio": Audio(sampling_rate=self.sample_rate),
                "text": Value("string"),
                "task": Value("string"),
            })

        # Create new dataset with explicit features
        ds_with_task = IterableDataset.from_generator(
            add_task_gen,
            gen_kwargs={"dataset": ds, "task_val": task},
            features=new_features,
        )

        return ds_with_task

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []
        train_weights = []
        val_weights = []

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
                    val_weights.append(sampling_weight)
            else:
                for train_split in train_splits:
                    train_datasets.append(self._prepare_split(d_cfg, train_split))
                    train_weights.append(sampling_weight)
                for eval_split in eval_splits:
                    val_datasets.append(self._prepare_split(d_cfg, eval_split))
                    val_weights.append(sampling_weight)

        # Use sampling weights if provided and we have multiple datasets
        if len(train_datasets) > 1:
            # Normalize weights
            total_weight = sum(train_weights[: len(train_datasets)])
            probabilities = [w / total_weight for w in train_weights[: len(train_datasets)]]
            train_ds = interleave_datasets(train_datasets, probabilities=probabilities)
        else:
            train_ds = train_datasets[0]

        if not val_datasets:
            val_ds = None
        elif len(val_datasets) > 1:
            # Use same sampling weights as training for validation
            total_weight = sum(val_weights[: len(val_datasets)])
            probabilities = [w / total_weight for w in val_weights[: len(val_datasets)]]
            val_ds = interleave_datasets(val_datasets, probabilities=probabilities)
        else:
            val_ds = val_datasets[0]

        # Shuffle training data for better generalization
        # Buffer size of 1000 balances memory usage and randomness
        train_ds = train_ds.shuffle(seed=self.seed, buffer_size=1000)

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
        # Handle None or empty text
        if text is None or not isinstance(text, str):
            return ""

        # Replace GigaSpeech punctuation tags with actual punctuation
        text = re.sub(r"<PERIOD>", ".", text)
        text = re.sub(r"<COMMA>", ",", text)
        text = re.sub(r"<QUESTIONMARK>", "?", text)
        text = re.sub(r"<EXCLAMATIONPOINT>", "!", text)

        # Remove <inaudible> tags
        text = re.sub(r"<inaudible>", "", text, flags=re.IGNORECASE)
        # Remove disfluencies (uh, um)
        return re.sub(r"\b(uh|um)\b", "", text, flags=re.IGNORECASE)

    def _normalize_text(self, text: str) -> str:
        """Apply Whisper normalization (matches eval script)."""
        return self.text_normalizer.normalize(self._preprocess_text(text))

    def _extract_audio(self, audio_decoder) -> Any:
        # Note: Audio() does peak normalization → [-1, 1]
        # Wav2Vec2FeatureExtractor does z-normalization → mean=0, std=1
        # No additional normalization needed here!
        audio_samples = audio_decoder.get_all_samples()
        audio_array = audio_samples.data.numpy()

        # Ensure mono audio: squeeze all dimensions, then handle multi-channel
        audio_array = audio_array.squeeze()

        # If still multi-dimensional (stereo/multi-channel), convert to mono by averaging
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0)

        # Ensure 1D array
        if audio_array.ndim == 0:
            # Single sample - expand to 1D
            audio_array = audio_array.reshape(1)

        return audio_array

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [self._extract_audio(f["audio"]) for f in features]

        # Extract audio features with attention mask
        # SpecAugment is applied automatically by the encoder model during training
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
                max_length=256,
                enable_thinking=False,
            )

            # Create labels - only train on the actual transcription text, not thinking tags
            labels = [-100] * len(tokens)  # Start with everything masked

            # Get special token IDs
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
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
                        and self.tokenizer.decode([int(tokens[content_start])]).strip() == ""
                    ):
                        content_start += 1
                    break

            # If no thinking tags found, look for assistant content directly
            if content_start == -1:
                im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
                assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
                for i in range(len(tokens) - 2):  # -2 to safely check i+1 and i+2
                    if tokens[i] == im_start_id and tokens[i + 1] == assistant_id:
                        # Start after <|im_start|> and assistant
                        # Now, find the actual content start by skipping newlines
                        content_start = i + 2
                        while (
                            content_start < len(tokens)
                            and self.tokenizer.decode([int(tokens[content_start])]).strip() == ""
                        ):
                            content_start += 1
                        break

            # Find the closing <|im_end|> for the assistant message
            content_end = -1
            if content_start > 0:
                for i in range(content_start, len(tokens)):
                    if tokens[i] == im_end_id:
                        content_end = i
                        break

            # Unmask the actual transcription text AND the EOS token
            if content_start > 0 and content_end > 0:
                # Include the <|im_end|> token in the loss so model learns to stop
                for i in range(content_start, content_end + 1):  # +1 to include EOS
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
        audio_downsample_rate=cfg.model.audio_downsample_rate,
        system_prompt=cfg.model.system_prompt,
        encoder_dim=encoder_config.hidden_size,
        llm_dim=decoder_config.hidden_size,
        projector_dropout=cfg.model.get("projector_dropout", 0.05),
        projector_hidden_dim=cfg.model.get("projector_hidden_dim"),
        label_smoothing=cfg.model.get("label_smoothing", 0.0),
        mask_time_prob=cfg.data.get("mask_time_prob", 0.05),
        mask_time_length=cfg.data.get("mask_time_length", 10),
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

        # Apply fresh LoRA if needed (when loading base model without LoRA)
        if (
            encoder_lora_config
            and encoder_lora_config.get("r", 0) > 0
            and not any("lora" in n.lower() for n, _ in model.encoder.named_parameters())
        ):
            from peft import TaskType

            model.encoder = model._apply_lora(
                model.encoder, encoder_lora_config, TaskType.FEATURE_EXTRACTION, "encoder"
            )
            model.encoder_lora_config = encoder_lora_config

        if (
            peft_config
            and peft_config.get("peft_method") == "lora"
            and not any("lora" in n.lower() for n, _ in model.decoder.named_parameters())
        ):
            from peft import TaskType

            model.decoder = model._apply_lora(
                model.decoder, peft_config, TaskType.CAUSAL_LM, "decoder"
            )
            model.peft_config = peft_config
    else:
        model = ASRModel(
            asr_config, peft_config=peft_config, encoder_lora_config=encoder_lora_config
        )

    # Disable cache during training (required for gradient checkpointing)
    model.config.use_cache = False

    train_dataset, val_dataset = DatasetLoader(cfg).load()

    # Create data collator for both training and evaluation
    # SpecAugment is now handled automatically by the model (enabled in train(), disabled in eval())
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

    # processor = model.get_processor()

    # Convert config to dict and remove non-TrainingArguments fields
    training_args = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_args, dict), "training_args must be a dict"

    # Apply torch.compile config before creating TrainingArguments
    if training_args.get("torch_compile_config"):
        compile_config = training_args.pop("torch_compile_config")
        # Configure torch._dynamo settings
        cache_limit = compile_config.get("cache_size_limit", 64)
        torch._dynamo.config.cache_size_limit = cache_limit
        torch._dynamo.config.capture_scalar_outputs = compile_config.get(
            "capture_scalar_outputs", True
        )
        torch._dynamo.config.allow_unspec_int_on_nn_module = compile_config.get(
            "allow_unspec_int_on_nn_module", True
        )

        # Enable parallel compilation for faster initial compile
        compile_threads = compile_config.get("compile_threads", 4)
        torch._inductor.config.compile_threads = compile_threads

    # Handle torch.compile settings (TrainingArguments doesn't support all options)
    training_args.get("torch_compile", False)
    training_args.pop("torch_compile_dynamic", False)
    training_args.pop("torch_compile_backend", "inductor")
    training_args.pop("torch_compile_mode", None)
    training_args.pop("torch_compile_fullgraph", False)

    # Remove other custom fields that aren't TrainingArguments parameters
    for key in ["model_dtype", "attn_implementation"]:
        training_args.pop(key, None)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model()

    # Push final model to hub if configured
    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        print(f"Pushing final model to Hub: {cfg.training.hub_model_id}")
        trainer.push_to_hub(commit_message="Training complete - final model")


if __name__ == "__main__":
    main()
