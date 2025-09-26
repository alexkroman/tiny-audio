#!/usr/bin/env python3
"""
🎙️ Simplified ASR Training Script
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from datasets import Audio, Dataset, interleave_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from modeling import ASRModel, ASRModelConfig

# --- Text Cleaning Utilities ---

PUNCT_MAP = {"COMMA": ",", "PERIOD": ".", "QUESTIONMARK": "?", "EXCLAMATIONPOINT": "!"}
TAG_REGEX = re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>")
PUNCT_REGEX = re.compile(r"\s*<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>")


def clean_gigaspeech_text(text: str) -> Optional[str]:
    """Cleans transcription text from the GigaSpeech dataset."""
    if TAG_REGEX.search(text):
        cleaned = TAG_REGEX.sub("", text).strip()
        if not cleaned or len(cleaned) < 3:
            return None

    text = PUNCT_REGEX.sub(lambda m: PUNCT_MAP[m.group(1)], text)
    text = TAG_REGEX.sub(" ", text)
    return " ".join(text.split()).strip() or None


# --- Data Loading ---


def _load_and_prepare_dataset(
    cfg: DictConfig, split: str, cache_dir: str, sample_rate: int
) -> Dataset:
    """Loads and prepares a single dataset split based on its configuration."""
    # Handle different dataset structures
    if cfg.get("path"):
        # For datasets with explicit path (e.g., GigaSpeech)
        ds = load_dataset(
            cfg.path,
            name=cfg.get("name"),
            split=split,
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    else:
        # For datasets identified by name (e.g., librispeech_asr)
        # Use the first config if multiple configs are provided
        config_name = cfg.configs[0] if hasattr(cfg, "configs") and cfg.configs else None
        ds = load_dataset(
            cfg.name,
            name=config_name,
            split=split,
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    # Handle text column cleaning
    text_column = cfg.get("text_column", "text")

    if cfg.get("cleaner") == "gigaspeech":
        ds = ds.filter(lambda x: clean_gigaspeech_text(x.get(text_column, "")) is not None)
        ds = ds.map(lambda x: {text_column: clean_gigaspeech_text(x[text_column])})

    # Handle audio column (could be "audio" or "wav" or other)
    audio_column = cfg.get("audio_column", "audio")

    # Ensure we have a 'text' column
    if "text" not in ds.column_names and text_column in ds.column_names:
        ds = ds.rename_column(text_column, "text")

    # Ensure we have an 'audio' column (normalize different audio column names)
    if "audio" not in ds.column_names and audio_column in ds.column_names:
        ds = ds.rename_column(audio_column, "audio")

    return ds.cast_column("audio", Audio(sampling_rate=sample_rate))


def load_all_datasets(config: DictConfig) -> Tuple[Dataset, Dataset]:
    """Loads and interleaves all specified training and validation datasets."""
    train_datasets, val_datasets = [], []

    for dataset_cfg in config.data.datasets:
        # Handle both singular and plural forms, take first split if multiple
        train_split = (
            dataset_cfg.train_splits[0]
            if hasattr(dataset_cfg, "train_splits")
            else dataset_cfg.train_split
        )
        eval_split = (
            dataset_cfg.eval_splits[0]
            if hasattr(dataset_cfg, "eval_splits")
            else dataset_cfg.eval_split
        )

        train_datasets.append(
            _load_and_prepare_dataset(
                dataset_cfg,
                train_split,
                config.data.dataset_cache_dir,
                config.data.sample_rate,
            )
        )
        val_datasets.append(
            _load_and_prepare_dataset(
                dataset_cfg,
                eval_split,
                config.data.dataset_cache_dir,
                config.data.sample_rate,
            )
        )

    train_dataset = (
        interleave_datasets(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    )
    val_dataset = interleave_datasets(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    if config.data.max_train_samples:
        train_dataset = train_dataset.take(config.data.max_train_samples)
    if config.data.max_eval_samples:
        val_dataset = val_dataset.take(config.data.max_eval_samples)

    return train_dataset, val_dataset


# --- Data Collator ---


class DataCollator:
    def __init__(self, tokenizer: Any, feature_extractor: Any, config: DictConfig):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = config.data.sample_rate
        self.max_audio_seconds = config.data.max_audio_seconds
        self.audio_chunk_token_id = tokenizer.convert_tokens_to_ids("<|audio_chunk|>")
        if self.audio_chunk_token_id == tokenizer.unk_token_id:
            raise ValueError("'<|audio_chunk|>' token not found in tokenizer vocabulary!")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Filter out invalid samples (all data should be normalized by _load_and_prepare_dataset)
        valid_features = []
        for f in features:
            # Both "text" and "audio" columns should be normalized by dataset loading
            text = f.get("text", "").strip()
            audio_data = f.get("audio")

            # Skip if no audio data or text
            if not audio_data or not text:
                continue

            # Skip if audio is too long
            audio_array = audio_data.get("array") if isinstance(audio_data, dict) else audio_data
            if audio_array is None or len(audio_array) == 0:
                continue

            duration = len(audio_array) / self.sample_rate
            if duration <= self.max_audio_seconds:
                valid_features.append(f)

        if not valid_features:
            return {}  # Trainer handles empty batches

        # 2. Extract audio features
        audio_arrays = [f["audio"]["array"] for f in valid_features]
        # SeamlessM4TFeatureExtractor processes audio arrays
        audio_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,  # Use automatic padding
        )

        # 3. Tokenize text with prompt format: <|audio_chunk|> {text}
        texts = [f"<|audio_chunk|> {f['text']}" for f in valid_features]
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # 4. Create labels: mask tokens up to and including <|audio_chunk|>
        labels = batch["input_ids"].clone()
        for i in range(labels.shape[0]):
            # Find the first occurrence of the audio chunk token
            chunk_token_indices = (labels[i] == self.audio_chunk_token_id).nonzero(as_tuple=True)[0]
            if len(chunk_token_indices) > 0:
                mask_end_pos = chunk_token_indices[0] + 1
                labels[i, :mask_end_pos] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding tokens
        batch["labels"] = labels

        # 5. Add audio features to batch
        batch["input_features"] = audio_features.input_features
        return batch


# --- Evaluation & Logging ---


def evaluate_wer(model, tokenizer, feature_extractor, eval_samples: List[Dict]) -> float:
    """Calculates Word Error Rate (WER) on a list of evaluation samples."""
    import evaluate

    device = model.device
    predictions, references = [], []

    with torch.no_grad():
        for sample in eval_samples:
            inputs = feature_extractor(
                sample["audio"]["array"],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,  # Use automatic padding for evaluation
            )
            generated_ids = model.generate(
                input_features=inputs.input_features.to(device),
                max_new_tokens=100,
            )
            predictions.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
            references.append(sample.get("text", ""))

    wer_metric = evaluate.load("wer")
    return wer_metric.compute(predictions=predictions, references=references)


class PredictionLoggingCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer,
        feature_extractor,
        num_samples=10,
        log_every_n_steps=500,
    ):
        self.eval_samples = list(eval_dataset.take(num_samples))
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples

    def on_step_end(self, args: TrainingArguments, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_every_n_steps == 0:
            print(f"\n📊 Step {state.global_step}: Generating predictions for logging...")
            wer = evaluate_wer(model, self.tokenizer, self.feature_extractor, self.eval_samples)

            with SummaryWriter(log_dir=args.logging_dir) as writer:
                writer.add_scalar("eval/wer_on_samples", wer, state.global_step)

            print(f"📈 WER on {self.num_samples} samples: {wer:.2%}\n")
            model.train()


# --- Main Training Function ---


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function driven by Hydra configuration."""
    print("--- Configuration ---\n" + OmegaConf.to_yaml(cfg))

    # 1. Initialize Model, Tokenizer, and Feature Extractor
    asr_config = ASRModelConfig(
        decoder_model_name=cfg.model.decoder_model_name,
        encoder_model_name="facebook/w2v-bert-2.0",
        lora_r=cfg.model.lora_r if cfg.model.use_lora else 0,
        lora_alpha=cfg.model.lora_alpha if cfg.model.use_lora else 0,
        lora_dropout=cfg.model.lora_dropout if cfg.model.use_lora else 0.0,
        lora_target_modules=list(cfg.model.lora_target_modules) if cfg.model.use_lora else [],
    )
    model = ASRModel(asr_config)

    # 2. Load Datasets
    train_dataset, val_dataset = load_all_datasets(cfg)

    # 3. Configure Training Arguments
    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    training_args = TrainingArguments(**training_args_dict)

    # 4. Set up Callbacks
    callbacks = [
        PredictionLoggingCallback(
            eval_dataset=val_dataset,
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            log_every_n_steps=cfg.log_predictions_every_n_steps,
        )
    ]

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            config=cfg,
        ),
        callbacks=callbacks,
    )

    # 6. Start Training
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
