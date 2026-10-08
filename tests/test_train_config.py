"""Tests for the Hydra structured-config schema in scripts/train_config.py."""

import inspect
from dataclasses import fields
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from transformers import TrainingArguments

from scripts.train import TRAINING_MODEL_PARAMS
from scripts.train_config import ModelConfig, TrainingConfig
from tiny_audio.asr_config import ASRConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = sorted(p.stem for p in (CONFIGS / "experiments").glob("*.yaml"))

# Keys TrainingConfig declares for scripts/train.py itself rather than for
# TrainingArguments (TRAINING_MODEL_PARAMS go to ASRConfig, the rest are popped
# or read directly in main()).
TRAIN_SCRIPT_KEYS = {
    "use_liger",
    "allow_unfused_ce",
    "wandb_project",
    "decoder_learning_rate",
    "projector_weight_decay",
    "encoder_learning_rate",
    "encoder_weight_decay",
    "use_cache",
    *TRAINING_MODEL_PARAMS,
}


def _compose(overrides=()):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def test_model_fields_are_asrconfig_params() -> None:
    params = set(inspect.signature(ASRConfig.__init__).parameters)
    missing = {f.name for f in fields(ModelConfig)} - params
    assert not missing, f"ModelConfig fields not accepted by ASRConfig: {missing}"


def test_model_fields_default_to_unset() -> None:
    # ASRConfig owns the model defaults; None lets train.py fall through to them.
    assert all(f.default is None for f in fields(ModelConfig))


def test_training_model_params_default_to_unset() -> None:
    # A non-None default here would silently override the `model:` block.
    defaults = {f.name: f.default for f in fields(TrainingConfig)}
    for param in TRAINING_MODEL_PARAMS:
        assert param in defaults, f"{param} missing from TrainingConfig"
        assert defaults[param] is None, param


def test_training_fields_are_known() -> None:
    valid = {f.name for f in fields(TrainingArguments)} | TRAIN_SCRIPT_KEYS
    unknown = {f.name for f in fields(TrainingConfig)} - valid
    assert not unknown, f"TrainingConfig fields TrainingArguments would drop: {unknown}"


def test_default_config_composes() -> None:
    cfg = _compose()
    assert OmegaConf.get_type(cfg.training) is TrainingConfig
    assert cfg.data.datasets, "default data group should list datasets"
    assert isinstance(cfg.training.learning_rate, float)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_composes(experiment: str) -> None:
    cfg = _compose([f"+experiments={experiment}"])
    # Typed nodes: every dataset entry validated against DatasetConfig.
    for entry in cfg.data.datasets:
        assert entry.path


def test_unknown_key_rejected() -> None:
    with pytest.raises(ConfigCompositionException):
        _compose(["training.lerning_rate=1e-4"])


def test_wrong_type_rejected() -> None:
    with pytest.raises(ConfigCompositionException):
        _compose(["training.seed=not-a-number"])


def test_append_syntax_adds_undeclared_key() -> None:
    cfg = _compose(["+training.warmup_ratio=0.05"])
    assert cfg.training.warmup_ratio == 0.05
