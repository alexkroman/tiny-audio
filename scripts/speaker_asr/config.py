"""Hydra config access outside `@hydra.main`, and the pool signature.

Same contract as scripts/turn_aware: `build-pool` and training resolve the
same configs/speaker_asr tree with the same overrides, each pool dir records
the settings that built it in pool_config.json, build-pool skips a split that
matches and training refuses one that does not.
"""

from __future__ import annotations

from dataclasses import fields

from omegaconf import DictConfig, OmegaConf

from scripts.speaker_asr.context import ContextConfig
from scripts.speaker_asr.data import WindowConfig
from scripts.speaker_asr.junction import JunctionConfig
from scripts.utils import get_project_root

CONFIG_DIR = get_project_root() / "configs" / "speaker_asr"

# Settings that change WHERE things are read from, not what the pool holds.
_NOT_IN_SIGNATURE = ("transcript_cache", "hub_repo")


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose configs/speaker_asr/config.yaml with Hydra-style overrides."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides or []))


def window_config(cfg: DictConfig) -> WindowConfig:
    """The `pool:` section's window knobs as a WindowConfig, field by field."""
    values = {}
    for f in fields(WindowConfig):
        value = cfg.pool[f.name]
        values[f.name] = tuple(value) if OmegaConf.is_list(value) else value
    return WindowConfig(**values)


def context_config(cfg: DictConfig) -> ContextConfig:
    """The `context:` section as a ContextConfig, field by field."""
    values = {}
    for f in fields(ContextConfig):
        value = cfg.context[f.name]
        values[f.name] = tuple(value) if OmegaConf.is_list(value) else value
    return ContextConfig(**values)


def junction_config(cfg: DictConfig) -> JunctionConfig:
    """The `junction:` section as a JunctionConfig, field by field."""
    values = {}
    for f in fields(JunctionConfig):
        value = cfg.junction[f.name]
        values[f.name] = tuple(value) if OmegaConf.is_list(value) else value
    return JunctionConfig(**values)


def n_speaker_tokens(cfg: DictConfig) -> int:
    """Speaker tokens a run registers: per window, or per recording with context."""
    if cfg.context.enabled:
        return max(cfg.pool.max_speakers, cfg.context.max_speakers_total)
    return cfg.pool.max_speakers


def pool_signature(cfg: DictConfig, max_samples: int | None = None) -> dict:
    """Everything that determines a pool's contents, as plain JSON-able data."""
    pool = OmegaConf.to_container(cfg.pool, resolve=True)
    for key in _NOT_IN_SIGNATURE:
        pool.pop(key, None)
    data = OmegaConf.to_container(cfg.data, resolve=True)
    source = {k: data[k] for k in ("dataset_id", "data_files", "columns")}
    return {"source": source, "pool": pool, "max_samples": max_samples}
