#!/usr/bin/env python3
"""
🎙️ ASR Training
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    WhisperFeatureExtractor,
)

from modeling import (
    ASRModel,
    ASRModelConfig,
)


def create_asr_model(config: DictConfig) -> ASRModel:
    asr_config = ASRModelConfig(
        decoder_model_name=config.model.decoder_model_name,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_target_modules=list(config.model.lora_target_modules),
        lora_dropout=config.model.lora_dropout,
    )

    return ASRModel(asr_config)


def evaluate_samples(model, tokenizer, feature_extractor, eval_samples, device=None):
    import evaluate

    if device is None:
        device = model.device

    predictions = []
    references = []

    with torch.no_grad():
        for sample in eval_samples:
            audio_array = sample["audio"]["array"]
            inputs = feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )

            input_features = inputs.input_features.to(device)
            attention_mask = (
                inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None
            )

            generated_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
            )

            prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            reference = sample.get("text", sample.get("sentence", ""))

            predictions.append(prediction)
            references.append(reference)

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)

    return predictions, references, wer


def clean_gigaspeech_text(text: str) -> Optional[str]:
    garbage_tags = ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"]
    for tag in garbage_tags:
        if tag in text:
            cleaned = text.replace(tag, "").strip()
            if not cleaned or len(cleaned) < 3:
                return None

    # Replace punctuation tags with proper spacing
    text = text.replace(" <COMMA>", ",")
    text = text.replace(" <PERIOD>", ".")
    text = text.replace(" <QUESTIONMARK>", "?")
    text = text.replace(" <EXCLAMATIONPOINT>", "!")

    # Handle any remaining tags without leading space (edge cases)
    text = text.replace("<COMMA>", ",")
    text = text.replace("<PERIOD>", ".")
    text = text.replace("<QUESTIONMARK>", "?")
    text = text.replace("<EXCLAMATIONPOINT>", "!")

    for tag in garbage_tags:
        text = text.replace(tag, " ")

    text = " ".join(text.split())
    text = text.strip()

    return text if text else None


class DataCollator:
    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: WhisperFeatureExtractor,
        config: DictConfig,
        model: ASRModel,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = config.data.sample_rate
        self.max_audio_seconds = config.data.max_audio_seconds
        self.model = model

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, Union[torch.Tensor, Optional[torch.Tensor], np.ndarray]]:

        valid_features = []
        for f in features:
            try:
                audio_array = f["audio"]["array"]
                audio_len_sec = len(audio_array) / self.sample_rate

                text = f.get("text") or f.get("sentence") or ""

                if "<" in text and ">" in text:
                    cleaned_text = clean_gigaspeech_text(text)
                    if cleaned_text is None:
                        continue
                    text = cleaned_text

                text = text.strip()
                if not text:
                    continue

                if audio_len_sec <= self.max_audio_seconds:
                    f["text"] = text
                    valid_features.append(f)
            except Exception:
                continue

        if not valid_features:
            return {
                "input_ids": torch.zeros((0, 0), dtype=torch.long),
                "labels": torch.zeros((0, 0), dtype=torch.long),
                "attention_mask": torch.zeros((0, 0), dtype=torch.long),
                "input_features": torch.zeros((0, 80, 3000)),
                "audio_attention_mask": torch.zeros((0, 1500)),
            }

        audio_arrays = [f["audio"]["array"] for f in valid_features]
        audio_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            max_length=480000,
        )

        # Apply chat template
        texts = []
        for f in valid_features:
            messages = [
                {
                    "role": "user",
                    "content": "Please transcribe the following audio recording.\n<|audio_chunk|>",
                },
                {"role": "assistant", "content": f["text"]},
            ]
            texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

        batch = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        # Mask the prompt part of the labels
        prompt_messages = [
            {
                "role": "user",
                "content": "Please transcribe the following audio recording.\n<|audio_chunk|>",
            }
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_length = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))

        labels[:, :prompt_length] = -100

        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
            "input_features": audio_features.input_features,
            "audio_attention_mask": (
                audio_features.attention_mask if hasattr(audio_features, "attention_mask") else None
            ),
        }


def _load_single_dataset(
    dataset_info: DictConfig, split: str, cache_dir: str, sample_rate: int
) -> Dataset:
    """Load a single dataset split based on its configuration."""
    import os

    from datasets import Audio

    dataset_name = dataset_info.name
    token = os.environ.get("HF_TOKEN")

    if dataset_name == "librispeech_asr":
        ds = load_dataset(
            "librispeech_asr",
            dataset_info.configs[0],
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        )
    elif dataset_name == "gigaspeech":
        ds = load_dataset(
            "speechcolab/gigaspeech",
            dataset_info.subset,
            split=split,
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
        )

        def filter_gigaspeech(example):
            text = example.get("text", "")
            cleaned = clean_gigaspeech_text(text)
            if cleaned:
                example["text"] = cleaned
                return True
            return False

        ds = ds.filter(filter_gigaspeech)
    elif dataset_name == "common_voice":
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            dataset_info.language,
            split=split,
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
        )
        ds = ds.rename_column("sentence", "text")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return ds.cast_column("audio", Audio(sampling_rate=sample_rate))


def load_datasets(config: DictConfig) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets in streaming mode."""
    from datasets import interleave_datasets

    train_datasets = []
    val_datasets = []

    for dataset_info in config.data.datasets:
        train_split = dataset_info.train_splits[0]
        eval_split = dataset_info.eval_splits[0]

        train_ds = _load_single_dataset(
            dataset_info, train_split, config.data.dataset_cache_dir, config.data.sample_rate
        )
        val_ds = _load_single_dataset(
            dataset_info, eval_split, config.data.dataset_cache_dir, config.data.sample_rate
        )

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    if len(train_datasets) > 1:
        train_dataset = interleave_datasets(train_datasets, stopping_strategy="first_exhausted")
        val_dataset = interleave_datasets(val_datasets, stopping_strategy="first_exhausted")
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]

    if config.data.max_train_samples:
        train_dataset = train_dataset.take(config.data.max_train_samples)

    if config.data.max_eval_samples:
        val_dataset = val_dataset.take(config.data.max_eval_samples)

    return train_dataset, val_dataset


class ModelingFileCopyCallback(TrainerCallback):
    """Copy modeling.py to the output directory so it gets pushed to Hub."""

    def on_save(self, args, state, control, **kwargs):
        modeling_src = Path(__file__).parent / "modeling.py"
        if modeling_src.exists():
            modeling_dst = Path(args.output_dir) / "modeling.py"
            shutil.copy2(modeling_src, modeling_dst)
            print(f"✅ Copied modeling.py to {modeling_dst} for Hub upload")


class PredictionLoggingCallback(TrainerCallback):
    def __init__(
        self, eval_dataset, tokenizer, feature_extractor, cfg, log_predictions_every_n_steps=500
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.cfg = cfg
        self.log_predictions_every_n_steps = log_predictions_every_n_steps
        self.writer = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_predictions_every_n_steps == 0:
            self._log_predictions(args, state, model)

    def _log_predictions(self, args, state, model):
        if model is None:
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logging_dir)

        model.eval()

        num_samples = getattr(self.cfg, "log_predictions_samples", 10)
        eval_samples = list(self.eval_dataset.take(num_samples))

        print(f"\n📊 Generating predictions for logging (step {state.global_step})...")

        predictions, references, wer = evaluate_samples(
            model, self.tokenizer, self.feature_extractor, eval_samples, device=model.device
        )

        self.writer.add_scalar("eval/wer", wer, state.global_step)

        predictions_text = []
        for i, (ref, pred) in enumerate(zip(references[:5], predictions[:5])):
            predictions_text.append(
                f"**Sample {i + 1}**\n\nTruth: {ref}\n\nPrediction: {pred}\n\n---"
            )

        full_text = "\n".join(predictions_text)
        self.writer.add_text("eval/predictions", full_text, state.global_step)

        print(f"📈 WER at step {state.global_step}: {wer:.2%}")
        print("\nSample predictions:")
        for _i, (ref, pred) in enumerate(zip(references[:3], predictions[:3])):
            print(f"  Truth:      {ref[:80]}...")
            print(f"  Prediction: {pred[:80]}...")
            print()

        model.train()


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration."""
    import os

    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")

    model = create_asr_model(cfg)
    tokenizer = model.decoder.tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    model.feature_extractor = feature_extractor
    train_dataset, val_dataset = load_datasets(cfg)

    if cfg.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {cfg.resume_from_checkpoint}")

    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_args_dict, dict), "Training args must be a dict"

    # Convert output_dir to absolute path for clearer logging
    output_dir = training_args_dict.get("output_dir", "./outputs")
    training_args_dict["output_dir"] = os.path.abspath(output_dir)

    # Also convert logging_dir to absolute path if present
    if "logging_dir" in training_args_dict:
        training_args_dict["logging_dir"] = os.path.abspath(training_args_dict["logging_dir"])

    if training_args_dict.get("push_to_hub", False) and not training_args_dict.get("hub_token"):
        hub_token = os.environ.get("HF_TOKEN")
        if hub_token:
            training_args_dict["hub_token"] = hub_token

    training_args = TrainingArguments(**training_args_dict)

    callbacks = [ModelingFileCopyCallback()]
    if val_dataset and "tensorboard" in training_args.report_to:
        callbacks.append(
            PredictionLoggingCallback(
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                cfg=cfg,
                log_predictions_every_n_steps=cfg.get("log_predictions_every_n_steps", 500),
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            config=cfg,
            model=model,
        ),
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model()

    if training_args.push_to_hub:
        print("\n🚀 Pushing final model to Hub...")
        trainer.push_to_hub()

if __name__ == "__main__":
    main()
