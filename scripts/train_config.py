"""Hydra structured-config schema for scripts/train.py.

The YAML files under configs/ still hold every value; these dataclasses only
declare which keys exist and what type each one has. `configs/config.yaml`
lists `base_config` first in its defaults, so every data/training/experiment
file is merged onto this schema. A misspelled key (`training.lerning_rate=`)
or a wrongly typed value (`training.seed=abc`) now fails at compose time
instead of being silently ignored by `get_valid_training_args` or
`ASRConfig(**kwargs)` hours into a pod launch.

Two conventions to keep in mind when adding a field:

- `ModelConfig` fields all default to None, meaning "unset": train.py drops
  None-valued keys before building `ASRConfig`, so the defaults stay owned by
  `tiny_audio/asr_config.py` instead of being duplicated here. Every field
  name must be an `ASRConfig.__init__` parameter (tests/test_train_config.py).
- `TrainingConfig` fields that mirror a HF `TrainingArguments` argument either
  carry its real default or None ("let TrainingArguments decide"); train.py
  drops None-valued keys before constructing it. The `TRAINING_MODEL_PARAMS`
  fields are None so they only override the `model:` block when set.

A key that is not declared here can still be added for a one-off run with
Hydra's append syntax, e.g. `+training.warmup_ratio=0.05`.
"""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class ExcludeWhereConfig:
    """Row filter on a source-metadata column (see DatasetLoader._prepare_split)."""

    column: str | None = None
    values: list[Any] | None = None
    above: float | None = None
    below: float | None = None


@dataclass
class DatasetConfig:
    """One entry of `data.datasets`."""

    path: str | None = None
    name: str | None = None
    audio_column: str = "audio"
    text_column: str = "text"
    train_splits: list[str] = field(default_factory=lambda: ["train"])
    eval_splits: list[str] = field(default_factory=lambda: ["validation"])
    task: str | None = None
    # "mono" or "cased"; validated in DatasetLoader._prepare_split.
    text_case: str | None = None
    text_punct: bool | None = None
    target_samples: int | None = None
    exclude_where: ExcludeWhereConfig | None = None


@dataclass
class DataConfig:
    datasets: list[DatasetConfig] = field(default_factory=list[DatasetConfig])
    sample_rate: int = 16000
    dataset_cache_dir: str | None = None
    num_proc: int = 16
    epoch_expansion: int = 1
    max_eval_samples: int | None = None
    max_eval_samples_per_dataset: int | None = None


@dataclass
class ModelConfig:
    """Keyword arguments for `tiny_audio.asr_config.ASRConfig`; None = its default."""

    audio_model_id: str | None = None
    text_model_id: str | None = None
    attn_implementation: str | None = None
    model_dtype: str | None = None
    transcribe_prompt: str | None = None
    encoder_dim: int | None = None
    llm_dim: int | None = None
    # list of (padding, kernel_size, stride) triples
    encoder_conv_layers: list[list[int]] | None = None
    audio_sample_rate: int | None = None
    audio_features_time_major: bool | None = None
    encoder_attention_mask: bool | None = None
    audio_token: str | None = None
    projector_dtype: str | None = None
    encoder_dtype: str | None = None
    projector_pool_stride: int | None = None
    projector_hidden_dim: int | None = None
    projector_type: str | None = None
    label_smoothing: float | None = None
    use_lora: bool | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    # "all-linear" or an explicit list of module names
    lora_target_modules: Any = None
    lora_rank_pattern: dict[str, int] | None = None
    lora_alpha_pattern: dict[str, int] | None = None
    inference_lead_in_seconds: float | None = None
    freeze_projector: bool | None = None
    freeze_language_model: bool | None = None
    freeze_text_embed_tokens: bool | None = None
    freeze_audio_encoder: bool | None = None
    encoder_trainable_top_layers: int | None = None
    encoder_trainable_post_projections: bool | None = None
    apply_spec_augment: bool | None = None
    mask_time_prob: float | None = None
    mask_time_length: int | None = None
    mask_time_min_masks: int | None = None
    max_new_tokens: int | None = None
    use_cache: bool | None = None
    no_repeat_ngram_size: int | None = None


@dataclass
class TrainingConfig:
    # ---- read by scripts/train.py itself, not by TrainingArguments --------
    use_liger: bool = True
    allow_unfused_ce: bool = False
    wandb_project: str = "tiny-audio"
    # Per-group optimizer knobs consumed by ASRTrainer (None = share the
    # global learning_rate / weight_decay).
    decoder_learning_rate: float | None = None
    projector_weight_decay: float | None = None
    encoder_learning_rate: float | None = None
    encoder_weight_decay: float | None = None

    # ---- TRAINING_MODEL_PARAMS: override the `model:` block when set -------
    attn_implementation: str | None = None
    use_lora: bool | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_target_modules: Any = None
    lora_rank_pattern: dict[str, int] | None = None
    lora_alpha_pattern: dict[str, int] | None = None
    freeze_projector: bool | None = None
    freeze_language_model: bool | None = None
    freeze_text_embed_tokens: bool | None = None
    freeze_audio_encoder: bool | None = None
    encoder_trainable_top_layers: int | None = None
    encoder_trainable_post_projections: bool | None = None

    # ---- HF TrainingArguments ----------------------------------------------
    output_dir: str | None = None
    seed: int = 42
    num_train_epochs: float = 3.0
    max_steps: int = -1
    optim: str | None = None
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_epsilon: float | None = None
    max_grad_norm: float | None = None
    warmup_steps: int | None = None
    lr_scheduler_type: str | None = None
    # Set to null in an experiment to drop kwargs inherited from production.
    lr_scheduler_kwargs: dict[str, Any] | None = None
    label_smoothing_factor: float | None = None
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int | None = None
    eval_accumulation_steps: int | None = None
    auto_find_batch_size: bool | None = None
    gradient_checkpointing: bool = False
    logging_steps: int | None = None
    logging_first_step: bool | None = None
    logging_nan_inf_filter: bool | None = None
    report_to: str | None = None
    eval_strategy: str | None = None
    eval_steps: int | None = None
    save_strategy: str | None = None
    save_steps: int | None = None
    save_total_limit: int | None = None
    load_best_model_at_end: bool | None = None
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_strategy: str | None = None
    hub_token: str | None = None
    hub_private_repo: bool = False
    fp16: bool = False
    bf16: bool = False
    tf32: bool | None = None
    torch_compile: bool | None = None
    use_cache: bool | None = None
    dataloader_num_workers: int | None = None
    dataloader_prefetch_factor: int | None = None
    dataloader_pin_memory: bool | None = None
    dataloader_persistent_workers: bool | None = None
    dataloader_drop_last: bool | None = None
    disable_tqdm: bool | None = None
    log_level: str | None = None
    ignore_data_skip: bool | None = None
    # A checkpoint path, or true to resume from the latest in output_dir.
    resume_from_checkpoint: Any = None
    remove_unused_columns: bool | None = None
    label_names: list[str] | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def register_configs() -> None:
    ConfigStore.instance().store(name="base_config", node=Config)


register_configs()
