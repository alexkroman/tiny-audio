#!/usr/bin/env python3
"""
🎙️ Simplified ASR Training Script
Refactored for clarity, modularity, and easier maintenance.
"""

import re
from typing import Any, Dict, List, Optional

import hydra
import torch
import numpy as np
from datasets import Audio, Dataset, interleave_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainerCallback, TrainingArguments

from modeling import ASRModel, ASRModelConfig

# --- Constants and Utilities ---

# Pre-compile regex for performance and clarity
TAG_REGEX = re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>")
PUNCT_REGEX = re.compile(r"\s*<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>")
PUNCT_MAP = {"COMMA": ",", "PERIOD": ".", "QUESTIONMARK": "?", "EXCLAMATIONPOINT": "!"}


def clean_gigaspeech_text(text: str) -> Optional[str]:
    """
    Cleans and normalizes transcription text from the GigaSpeech dataset.
    Removes special tags and normalizes punctuation.
    """
    if TAG_REGEX.search(text):
        cleaned = TAG_REGEX.sub("", text).strip()
        # Return None for empty or very short transcriptions after cleaning
        if not cleaned or len(cleaned) < 3:
            return None
    text = PUNCT_REGEX.sub(lambda m: PUNCT_MAP.get(m.group(1), ""), text)
    text = TAG_REGEX.sub(" ", text)
    # Remove extra whitespace
    return " ".join(text.split()).strip() or None


# --- Data Handling ---


class DatasetLoader:
    """Encapsulates all logic for loading and preparing datasets."""

    def __init__(self, config: DictConfig):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir

    def _prepare_split(self, dataset_cfg: DictConfig, split: str) -> Dataset:
        """Loads, cleans, and standardizes a single dataset split."""
        ds = load_dataset(
            dataset_cfg.get("path", dataset_cfg.name),
            name=dataset_cfg.get("name"),
            split=split,
            streaming=True,
            cache_dir=self.cache_dir,
        )

        # Apply text cleaning if specified
        text_column = dataset_cfg.get("text_column", "text")
        if dataset_cfg.get("cleaner") == "gigaspeech":
            ds = ds.map(lambda x: {text_column: clean_gigaspeech_text(x[text_column])})
            ds = ds.filter(lambda x: x[text_column] is not None)

        # Standardize column names to 'audio' and 'text'
        if text_column != "text" and text_column in ds.column_names:
            ds = ds.rename_column(text_column, "text")

        audio_column = dataset_cfg.get("audio_column", "audio")
        if audio_column != "audio" and audio_column in ds.column_names:
            ds = ds.rename_column(audio_column, "audio")

        # Cast audio column to the correct sample rate
        return ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

    def load(self) -> tuple[Dataset, Dataset]:
        """Loads and interleaves all configured training and validation datasets."""
        train_datasets, val_datasets = [], []
        for d_cfg in self.config.datasets:
            # Handle multiple train/eval splits per dataset
            train_splits = d_cfg.get("train_splits", [d_cfg.get("train_split", "train")])
            eval_splits = d_cfg.get("eval_splits", [d_cfg.get("eval_split", "validation")])

            # For datasets with multiple configs/splits, zip them together
            if "configs" in d_cfg:
                configs = d_cfg.configs
                for config, train_split, eval_split in zip(configs, train_splits, eval_splits):
                    # Create a modified config for this specific split
                    split_cfg = OmegaConf.create(d_cfg)
                    split_cfg.name = config
                    train_datasets.append(self._prepare_split(split_cfg, train_split))
                    val_datasets.append(self._prepare_split(split_cfg, eval_split))
            else:
                # Single config dataset - use all splits
                for train_split in train_splits:
                    train_datasets.append(self._prepare_split(d_cfg, train_split))
                for eval_split in eval_splits:
                    val_datasets.append(self._prepare_split(d_cfg, eval_split))

        # Interleave datasets if more than one is provided
        train_ds = (
            interleave_datasets(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        )
        val_ds = interleave_datasets(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

        # Apply sample limits if specified
        if self.config.max_train_samples:
            train_ds = train_ds.take(self.config.max_train_samples)
        if self.config.max_eval_samples:
            val_ds = val_ds.take(self.config.max_eval_samples)

        return train_ds, val_ds


class DataCollator:
    """Prepares batches of audio and text data for model training."""

    def __init__(self, tokenizer: Any, feature_extractor: Any, config: DictConfig):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = config.data.sample_rate
        self.max_audio_seconds = config.data.max_audio_seconds
        self.max_audio_samples = int(self.max_audio_seconds * self.sample_rate)
        self.audio_chunk_token_id = tokenizer.convert_tokens_to_ids("<|audio_chunk|>")

        if self.audio_chunk_token_id == tokenizer.unk_token_id:
            raise ValueError("'<|audio_chunk|>' token is missing from the tokenizer's vocabulary!")


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Filter out invalid samples
        valid_features = [f for f in features if f.get("audio") and f.get("text")]
        if not valid_features:
            return {}

        # 2. Manually truncate AND pad each audio array to the global max length
        audio_arrays = []
        for f in valid_features:
            array = f["audio"]["array"]
            # Truncate if longer
            if len(array) > self.max_audio_samples:
                array = array[:self.max_audio_samples]
            # Pad with zeros if shorter
            if len(array) < self.max_audio_samples:
                padded_array = np.pad(
                    array, (0, self.max_audio_samples - len(array)),
                    mode='constant', constant_values=0
                )
                audio_arrays.append(padded_array)
            else:
                audio_arrays.append(array)

        # 3. Process the uniformly-sized arrays with the feature extractor
        audio_features = self.feature_extractor(
            audio_arrays, sampling_rate=self.sample_rate, return_tensors="pt"
        )

        # 4. Format as a simple text sequence for the whole batch
        # We add the EOS token to teach the model when a transcription is complete.
        prompts = [f"<|audio_chunk|>{f['text']}{self.tokenizer.eos_token}" for f in valid_features]

        # 5. Tokenize the batch
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        batch = {}
        batch["input_ids"] = tokenized["input_ids"]
        batch["attention_mask"] = tokenized["attention_mask"]
        batch["input_features"] = audio_features.input_features

        # 6. Create labels: mask out the prompt part (<|audio_chunk|>)
        labels = batch["input_ids"].clone()

        # Tokenize the prompt-only part to find its length
        prompt_only_tokenized = self.tokenizer(
            ["<|audio_chunk|>"], add_special_tokens=False, return_tensors="pt"
        )
        prompt_len = prompt_only_tokenized.input_ids.shape[1]

        # Mask the prompt tokens by setting their labels to -100
        labels[:, :prompt_len] = -100

        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch


# --- Evaluation and Logging ---

class PredictionLoggingCallback(TrainerCallback):

 def __init__(
    self,
    eval_dataset: Dataset,
    tokenizer: Any,
    feature_extractor: Any,
    sample_rate: int,
    max_audio_seconds: float,
    num_samples: int = 10,
    log_every_n_steps: int = 500,
):
    import evaluate

    self.eval_samples = list(eval_dataset.take(num_samples))
    self.tokenizer = tokenizer
    self.feature_extractor = feature_extractor
    self.sample_rate = sample_rate
    self.max_audio_samples = int(max_audio_seconds * sample_rate)
    self.log_every_n_steps = log_every_n_steps
    self.wer_metric = evaluate.load("wer")

    def on_step_end(self, args: TrainingArguments, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_every_n_steps == 0:
            model.eval()
            device = next(model.parameters()).device
            predictions, references = [], []

            # Prepare the inference prompt
            inference_prompt = "<|audio_chunk|>"
            prompt_ids = self.tokenizer(inference_prompt, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                for sample in self.eval_samples:
                    array = sample["audio"]["array"]
                    # Manually truncate AND pad the single audio array
                    if len(array) > self.max_audio_samples:
                        array = array[:self.max_audio_samples]
                    if len(array) < self.max_audio_samples:
                        array = np.pad(
                            array, (0, self.max_audio_samples - len(array)),
                            mode='constant', constant_values=0
                        )

                    inputs = self.feature_extractor(
                        array,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt",
                    )

                    generated_ids = model.generate(
                        input_features=inputs.input_features.to(device),
                        decoder_input_ids=prompt_ids,
                        max_new_tokens=100,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    predictions.append(
                        self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    )
                    references.append(sample.get("text", ""))

            wer = self.wer_metric.compute(predictions=predictions, references=references)

            print(f"Prediction: '{predictions[0]}'")
            print(f"Reference:  '{references[0]}'")
            print(f"📈 WER on {len(self.eval_samples)} samples: {wer:.2%}\n")
            model.train()  # Ensure model is back in training mode


# --- Main Training ---


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to configure and run the ASR model training."""
    print("--- 🚀 Initializing ASR Training ---")
    print(OmegaConf.to_yaml(cfg))

    # 1. Initialize Model from Configuration
    lora_config = {}
    if cfg.model.use_lora:
        lora_config = {
            "lora_r": cfg.model.lora_r,
            "lora_alpha": cfg.model.lora_alpha,
            "lora_dropout": cfg.model.lora_dropout,
            "lora_target_modules": list(cfg.model.lora_target_modules),
        }

    asr_config = ASRModelConfig(
        decoder_model_name=cfg.model.decoder_model_name,
        encoder_model_name="facebook/w2v-bert-2.0",
        **lora_config,
    )
    model = ASRModel(asr_config)
    print("✅ Model, Tokenizer, and Feature Extractor loaded.")

    # 2. Load and Prepare Datasets
    data_loader = DatasetLoader(cfg)
    train_dataset, val_dataset = data_loader.load()
    print("✅ Datasets configured and ready.")

    # 3. Configure Training Arguments
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=model.tokenizer,
        data_collator=DataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            config=cfg,
        ),
        callbacks=[
            PredictionLoggingCallback(
                eval_dataset=val_dataset,
                tokenizer=model.tokenizer,
                feature_extractor=model.feature_extractor,
                sample_rate=cfg.data.sample_rate,
                max_audio_seconds=cfg.data.max_audio_seconds,
                log_every_n_steps=cfg.log_predictions_every_n_steps,
            )
        ],
    )

    # 5. Start Training
    print("--- 🏋️ Starting Training ---")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    print("--- 🎉 Training Complete ---")
    trainer.save_model()
    print(f"💾 Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
