#!/usr/bin/env python3

import logging
from typing import Any, Dict, List

import hydra
import torch
import wandb
from datasets import Audio, Dataset, interleave_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments

from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel


class DatasetLoader:
    def __init__(self, config: DictConfig):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir

    def _prepare_split(self, dataset_cfg: DictConfig, split: str) -> Dataset:
        ds = load_dataset(
            dataset_cfg.get("path", dataset_cfg.name),
            name=dataset_cfg.get("name"),
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

        return ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []
        for d_cfg in self.config.datasets:
            train_splits = d_cfg.get("train_splits", [d_cfg.get("train_split", "train")])
            eval_splits = d_cfg.get("eval_splits", [d_cfg.get("eval_split", "validation")])

            if "configs" in d_cfg:
                configs = d_cfg.configs
                for config, train_split, eval_split in zip(configs, train_splits, eval_splits):
                    split_cfg = OmegaConf.create(d_cfg)
                    split_cfg.name = config
                    train_datasets.append(self._prepare_split(split_cfg, train_split))
                    val_datasets.append(self._prepare_split(split_cfg, eval_split))
            else:
                for train_split in train_splits:
                    train_datasets.append(self._prepare_split(d_cfg, train_split))
                for eval_split in eval_splits:
                    val_datasets.append(self._prepare_split(d_cfg, eval_split))

        train_ds = (
            interleave_datasets(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        )
        val_ds = interleave_datasets(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

        # train_ds = train_ds.shuffle(seed=42, buffer_size=1000)

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
        max_audio_seconds: float,
        system_prompt: str = None,
    ):
        super().__init__(tokenizer=tokenizer, padding=True)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_seconds * sample_rate)
        self.system_prompt = system_prompt

    def _extract_audio(self, audio_decoder) -> Any:
        # Note: Audio() does peak normalization → [-1, 1]
        # Wav2Vec2FeatureExtractor does z-normalization → mean=0, std=1
        # No additional normalization needed here!
        audio_samples = audio_decoder.get_all_samples()
        audio_array = audio_samples.data[: self.max_audio_samples]
        return audio_array.squeeze().numpy()

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [self._extract_audio(f["audio"]) for f in features]

        audio_features = self.feature_extractor(
            audio_arrays, sampling_rate=self.sample_rate, padding=True, return_tensors="pt"
        )

        text_features = []
        for f in features:
            text = f["text"].strip() if isinstance(f["text"], str) else f["text"]

            # Build messages for prefix (everything before assistant response)
            prefix_messages = []
            if self.system_prompt:
                prefix_messages.append({"role": "system", "content": self.system_prompt})
            prefix_messages.append(
                {
                    "role": "user",
                    "content": "Repeat the following text, without any explanation: <|audio_start|><|audio_end|>",
                }
            )

            # Tokenize prefix with add_generation_prompt=True to get position where assistant content starts
            prefix_tokens = self.tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=True,
                add_generation_prompt=True,  # Adds assistant start marker
                truncation=False,
                enable_thinking=False,
            )

            # Build full messages including assistant response
            full_messages = prefix_messages.copy()
            full_messages.append({"role": "assistant", "content": text})

            # Tokenize full conversation
            tokens = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=256,
                enable_thinking=False,
            )

            # FAST label masking: Mask everything before assistant response starts
            # Only train on the assistant's actual response (from len(prefix_tokens) onwards)
            labels = [-100] * len(tokens)
            for i in range(len(prefix_tokens), len(tokens)):
                labels[i] = tokens[i]

            text_features.append(
                {
                    "input_ids": tokens,
                    "labels": labels,
                }
            )

        batch = super().__call__(text_features)

        batch["input_values"] = audio_features.input_values
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
    decoder_config = HFAutoConfig.from_pretrained(cfg.model.decoder_model_name, trust_remote_code=True)

    asr_config = ASRConfig(
        text_model_id=cfg.model.decoder_model_name,
        audio_model_id=cfg.model.encoder_model_name,
        attn_implementation=cfg.training.attn_implementation,
        model_dtype=cfg.training.model_dtype,
        audio_downsample_rate=cfg.model.audio_downsample_rate,
        system_prompt=cfg.model.system_prompt,
        encoder_dim=encoder_config.hidden_size,
        llm_dim=decoder_config.hidden_size,
    )
    model = ASRModel(asr_config)

    train_dataset, val_dataset = DatasetLoader(cfg).load()

    data_collator = DataCollator(
        tokenizer=model.tokenizer,
        feature_extractor=model.feature_extractor,
        sample_rate=cfg.data.sample_rate,
        max_audio_seconds=cfg.data.max_audio_seconds,
        system_prompt=cfg.model.system_prompt,
    )

    callbacks = []

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

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    trainer.save_model()


if __name__ == "__main__":
    main()
