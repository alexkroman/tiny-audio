#!/usr/bin/env python3
"""Training script for ASR models using Hydra configuration."""

import contextlib
import os
import random
from dataclasses import fields
from typing import Any

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

import hydra
import numpy as np
import torch
import wandb
from datasets import (
    Audio,
    Dataset,
    Value,
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

from tiny_audio.asr_config import (
    DEFAULT_ENCODER_CONV_LAYERS,
    ASRConfig,
    compute_encoder_output_length,
)
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.augmentation import NoiseAugmentation, RIRAugmentation, SpeedPerturbation

TRANSCRIBE_PROMPTS = ["Transcribe the speech to text"]
DESCRIBE_PROMPTS = ["Describe all the information you can hear"]


class DatasetLoader:
    """Loads and prepares datasets for training."""

    def __init__(self, config: DictConfig, multitask_enabled: bool = False):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir
        self.seed = config.training.get("seed", 42)
        self.num_proc = self.config.get("num_proc", 16)
        # `duration` is only consumed by Trainer's group_by_length sampler; skip
        # any duration prep when group_by_length is off. Avoids a slow column
        # cast (and potentially a full-decode map) on every train run.
        self.needs_duration = bool(config.training.get("group_by_length", False))
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

        col_map = {
            "text": dataset_cfg.get("text_column", "text"),
            "audio": dataset_cfg.get("audio_column", "audio"),
        }
        if duration_source := dataset_cfg.get("duration_column"):
            col_map["duration"] = duration_source

        for target, source in col_map.items():
            if source != target and source in ds.column_names:
                if target in ds.column_names:
                    ds = ds.remove_columns([target])
                ds = ds.rename_column(source, target)

        ds = ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        # For multitask, keep sift_response and add task column
        if self.multitask_enabled:
            task = dataset_cfg.get("task", "transcribe")  # Default to transcribe
            # Add task column to identify ASR vs SIFT samples
            ds = ds.add_column("task", [task] * len(ds))
            keep_cols = {"audio", "text", "sift_response", "task"}
        else:
            keep_cols = {"audio", "text"}
        if self.needs_duration:
            keep_cols = keep_cols | {"duration"}
        extra_cols = [c for c in (ds.column_names or []) if c not in keep_cols]

        if extra_cols:
            ds = ds.remove_columns(extra_cols)

        # Filter TEDLIUM ignore markers only for TEDLIUM dataset
        # Duration filtering happens in DataCollator to avoid loading all audio upfront
        if "tedlium" in dataset_path.lower():

            def filter_tedlium(text):
                return text.strip() != "ignore_time_segment_in_scoring"

            ds = ds.filter(filter_tedlium, num_proc=self.num_proc, input_columns="text")

        return ds

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

    def _ensure_duration(self, ds: Dataset) -> Dataset:
        # `group_by_length` requires a `duration` column. Compute from audio
        # length when no source field provides one. Run AFTER resampling so we
        # don't decode samples that get trimmed. Cast to float32 only when the
        # source ships a different dtype (sources mix float32/float64/null and
        # concatenate_datasets rejects the mismatch); skipping a no-op cast
        # avoids a multi-minute column rewrite on large streams.
        if "duration" not in ds.column_names:
            sr = self.sample_rate

            def _add_duration(batch):
                return {"duration": [a["array"].shape[0] / sr for a in batch["audio"]]}

            ds = ds.map(_add_duration, batched=True, num_proc=self.num_proc)
        if ds.features["duration"].dtype != "float32":
            ds = ds.cast_column("duration", Value("float32"))
        return ds

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []

        for d_cfg in tqdm(self.config.datasets, desc="Loading datasets"):
            train_splits = d_cfg.get("train_splits", [d_cfg.get("train_split", "train")])
            eval_splits = d_cfg.get("eval_splits", [d_cfg.get("eval_split", "validation")])
            target_samples = d_cfg.get("target_samples")

            for train_split in train_splits:
                ds = self._prepare_split(d_cfg, train_split)
                if target_samples:
                    ds = self._resample_to_target(ds, target_samples)
                if self.needs_duration:
                    ds = self._ensure_duration(ds)
                train_datasets.append(ds)

            for eval_split in eval_splits:
                ds = self._prepare_split(d_cfg, eval_split)
                if self.needs_duration:
                    ds = self._ensure_duration(ds)
                val_datasets.append(ds)

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
        self.encoder_conv_layers = encoder_conv_layers or DEFAULT_ENCODER_CONV_LAYERS
        # Whisper's encoder requires a fixed 3000 mel frames; other encoders
        # (GLM-ASR) accept variable-length input, so only pad to longest.
        self._audio_padding = (
            "max_length"
            if type(feature_extractor).__name__ == "WhisperFeatureExtractor"
            else "longest"
        )
        self.text_collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=2048)

    def _extract_audio_arrays(self, features):
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
                # Drop samples that would poison the gradient: empty audio,
                # NaN/Inf samples (encoding glitches in community datasets),
                # or empty text labels (would yield 0/0 loss under label
                # smoothing). One bad sample is enough to NaN the optimizer
                # state and every subsequent step.
                if audio.size == 0:
                    continue
                if not np.isfinite(audio).all():
                    continue
                if not (f.get("text") or "").strip():
                    continue
                audio_arrays.append(audio)
                valid_features.append(f)
            except Exception:
                continue
            finally:
                f["audio"] = None
        if not audio_arrays:
            raise ValueError("No valid audio samples in batch")
        return audio_arrays, valid_features

    def _build_sample(self, feature: dict, num_audio_tokens: int) -> dict:
        """Build a single chat sample. Subclasses can override for task-specific prompts."""
        text = (feature.get("text") or "").strip().lower()
        return self._make_messages(num_audio_tokens, random.choice(TRANSCRIBE_PROMPTS), text)

    def _make_messages(self, num_audio_tokens: int, prompt: str, response: str) -> dict:
        user_content = ("<audio>" * num_audio_tokens) + " " + prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": response})
        return {"messages": messages}

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_arrays, valid_features = self._extract_audio_arrays(features)

        audio_out = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding=self._audio_padding,
            return_attention_mask=True,
            return_tensors="pt",
        )

        mel_lengths = audio_out.attention_mask.sum(dim=-1)
        encoder_lengths = compute_encoder_output_length(mel_lengths, self.encoder_conv_layers)
        token_counts_tensor = self.projector.get_output_length(encoder_lengths).to(torch.long)
        audio_token_counts = token_counts_tensor.tolist()

        text_features = [
            self._build_sample(f, n) for f, n in zip(valid_features, audio_token_counts)
        ]

        batch = self.text_collator(text_features)
        batch["input_features"] = audio_out.input_features
        batch["audio_attention_mask"] = audio_out.attention_mask
        batch["audio_token_counts"] = token_counts_tensor
        return batch


class MultiTaskDataCollator(DataCollator):
    """Collates audio and text data for multi-task ASR + SIFT training."""

    def __init__(self, *args, **kwargs):
        kwargs["system_prompt"] = ""
        super().__init__(*args, **kwargs)

    def _build_sample(self, feature: dict, num_audio_tokens: int) -> dict:
        if feature.get("task") == "sift":
            response = (feature.get("sift_response") or feature.get("text") or "").strip()
            prompt = random.choice(DESCRIBE_PROMPTS)
        else:
            response = (feature.get("text") or "").strip().lower()
            prompt = random.choice(TRANSCRIBE_PROMPTS)
        return self._make_messages(num_audio_tokens, prompt, response)


class ASRTrainer(Trainer):
    """Trainer subclass for ASR models."""

    def __init__(
        self,
        *args,
        decoder_learning_rate: float | None = None,
        decoder_weight_decay: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.decoder_learning_rate = decoder_learning_rate
        self.decoder_weight_decay = decoder_weight_decay

    def create_optimizer(self):
        """Optimizer with separate LR / weight decay for the language model.

        Mirrors HF Trainer.create_optimizer's decay/no-decay split, but adds a
        second axis: parameters under `language_model.` get `decoder_learning_rate`
        and `decoder_weight_decay` (when set), while the projector keeps
        `args.learning_rate` and `args.weight_decay`.
        """
        decoder_overrides = (
            self.decoder_learning_rate is not None or self.decoder_weight_decay is not None
        )
        if self.optimizer is not None or not decoder_overrides:
            return super().create_optimizer()

        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        opt_model = self.model
        decay_parameters = set(get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS))
        decay_parameters = {n for n in decay_parameters if "bias" not in n}

        groups: dict[tuple[bool, bool], list] = {
            (True, True): [],  # decoder, decay
            (True, False): [],  # decoder, no decay
            (False, True): [],  # other, decay
            (False, False): [],  # other, no decay
        }
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            is_decoder = name.startswith("language_model.")
            decay = name in decay_parameters
            groups[(is_decoder, decay)].append(param)

        base_wd = self.args.weight_decay
        base_lr = self.args.learning_rate
        dec_lr = self.decoder_learning_rate if self.decoder_learning_rate is not None else base_lr
        dec_wd = self.decoder_weight_decay if self.decoder_weight_decay is not None else base_wd
        optimizer_grouped_parameters = [
            {"params": groups[(False, True)], "weight_decay": base_wd, "lr": base_lr},
            {"params": groups[(False, False)], "weight_decay": 0.0, "lr": base_lr},
            {"params": groups[(True, True)], "weight_decay": dec_wd, "lr": dec_lr},
            {"params": groups[(True, False)], "weight_decay": 0.0, "lr": dec_lr},
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with proper label shifting for causal LM.

        HuggingFace Trainer's label_smoother checks MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        to decide whether to shift labels. Since ASRModel isn't in that mapping,
        it incorrectly uses shift_labels=False, causing misaligned predictions.
        This override forces shift_labels=True for correct causal LM behavior.
        """
        labels = (
            inputs.pop("labels") if self.label_smoother is not None and "labels" in inputs else None
        )

        outputs = model(**inputs)

        if labels is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
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


TRAINING_MODEL_PARAMS = [
    "use_specaugment",
    "num_time_masks",
    "time_mask_length",
    "num_freq_masks",
    "freq_mask_length",
    "attn_implementation",
    "use_lora",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "freeze_projector",
    "freeze_language_model",
]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    push_to_hub = cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id")
    if push_to_hub and not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN environment variable is required when push_to_hub is enabled. "
            "Set it with: export HF_TOKEN=your_token"
        )

    if cfg.training.get("report_to") == "wandb":
        wandb.init(
            project=cfg.training.get("wandb_project", "tiny-audio"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_config_dict, dict), "model config must be a dict"
    for param in TRAINING_MODEL_PARAMS:
        if cfg.training.get(param) is not None:
            model_config_dict[param] = cfg.training[param]
    asr_config = ASRConfig(**model_config_dict)

    if cfg.model.get("pretrained_model_path"):
        model = ASRModel.from_pretrained(cfg.model.pretrained_model_path, config=asr_config)
    else:
        model = ASRModel(asr_config)

    model.config.use_cache = False

    if hub_model_id := cfg.training.get("hub_model_id"):
        model.config.pretrained_model_path = hub_model_id

    # Workaround: TRL's DataCollatorForChatML doesn't pass enable_thinking=False to Qwen3.
    # See https://github.com/huggingface/trl/issues/3387
    if model.tokenizer.chat_template and "enable_thinking" in model.tokenizer.chat_template:
        model.tokenizer.chat_template = model.tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "true",
        )

    multitask_enabled = cfg.get("multitask", {}).get("enabled", False)

    train_dataset, val_dataset = DatasetLoader(cfg, multitask_enabled=multitask_enabled).load()

    augmentations: list = []

    # Speed perturbation runs FIRST so reverb / noise apply to the perturbed
    # signal — Kaldi convention also followed by NeMo / ESPnet / K2 / Icefall.
    speed_cfg = cfg.training.get("speed_perturbation") or {}
    if speed_cfg.get("enabled"):
        augmentations.append(
            SpeedPerturbation(
                sample_rate=cfg.data.sample_rate,
                rates=tuple(speed_cfg.get("rates", [0.9, 1.0, 1.1])),
                prob=speed_cfg.get("prob", 1.0),
            )
        )

    rir_aug: RIRAugmentation | None = None
    rir_cfg = cfg.training.get("rir_augmentation") or {}
    if rir_cfg.get("enabled"):
        rir_aug = RIRAugmentation(
            sample_rate=cfg.data.sample_rate,
            prob=rir_cfg.get("prob", 0.5),
            pool_size=rir_cfg.get("pool_size", 2048),
            corpus_path=rir_cfg.get("corpus_path"),
            room_x_range=tuple(rir_cfg.get("room_x_range", [3.0, 10.0])),
            room_y_range=tuple(rir_cfg.get("room_y_range", [3.0, 10.0])),
            room_z_range=tuple(rir_cfg.get("room_z_range", [2.4, 4.0])),
            t60_range=tuple(rir_cfg.get("t60_range", [0.1, 1.0])),
            seed=rir_cfg.get("seed"),
        )
        augmentations.append(rir_aug)

    noise_cfg = cfg.training.get("noise_augmentation") or {}
    if noise_cfg.get("enabled"):
        # Share the RIR pool with NoiseAugmentation when reverb_noise is set
        # so noise gets the same room response as the clean signal.
        noise_rir = rir_aug if (rir_aug is not None and noise_cfg.get("reverb_noise")) else None
        augmentations.append(
            NoiseAugmentation(
                sample_rate=cfg.data.sample_rate,
                prob=noise_cfg.get("prob", 0.5),
                min_snr_db=noise_cfg.get("min_snr_db", 0.0),
                max_snr_db=noise_cfg.get("max_snr_db", 25.0),
                corpus_path=noise_cfg.get("corpus_path"),
                babble_weight=noise_cfg.get("babble_weight", 0.0),
                rir_augmentation=noise_rir,
            )
        )

    if augmentations:

        def _apply_aug(batch):
            audios = batch.get("audio") or []
            for a in audios:
                if a and "array" in a:
                    arr = a["array"]
                    for aug in augmentations:
                        arr = aug(arr)
                    a["array"] = arr
            return batch

        train_dataset = train_dataset.with_transform(_apply_aug)

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

    callbacks = []
    if cfg.early_stopping.patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping.patience,
                early_stopping_threshold=cfg.early_stopping.threshold,
            )
        )
    if push_to_hub:
        callbacks.append(PushToHubCallback())

    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_config, dict)
    decoder_learning_rate = training_config.pop("decoder_learning_rate", None)
    decoder_weight_decay = training_config.pop("decoder_weight_decay", None)
    if compile_config := training_config.pop("torch_compile_config", None):
        torch._dynamo.config.cache_size_limit = compile_config.get("cache_size_limit", 64)
        torch._dynamo.config.capture_scalar_outputs = compile_config.get(
            "capture_scalar_outputs", True
        )
        torch._inductor.config.compile_threads = compile_config.get("compile_threads", 4)

    trainer = ASRTrainer(
        model=model,
        args=TrainingArguments(**get_valid_training_args(training_config)),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=model.tokenizer,
        callbacks=callbacks,
        decoder_learning_rate=decoder_learning_rate,
        decoder_weight_decay=decoder_weight_decay,
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    trainer.save_model()

    if push_to_hub:
        trainer.model.push_to_hub(
            cfg.training.hub_model_id,
            commit_message="Training complete - final model",
            private=cfg.training.get("hub_private_repo", False),
        )


if __name__ == "__main__":
    main()
