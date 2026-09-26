"""Hydra config access outside `@hydra.main`, and the pool signature.

`build-pool`, `ta runpod` and training all resolve the same
configs/turn_aware tree with the same overrides, so a pool and the run that
trains on it cannot disagree. Each pool dir records the settings that built
it in pool_config.json: build-pool skips a split whose manifest matches the
current settings, and training refuses one that does not.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scripts.turn_aware.data import PoolConfig
from scripts.utils import get_project_root

CONFIG_DIR = get_project_root() / "configs" / "turn_aware"
SIGNATURE_FILE = "pool_config.json"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose configs/turn_aware/config.yaml with Hydra-style overrides."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides or []))


def pool_config(cfg: DictConfig) -> PoolConfig:
    """The `pool:` section as a PoolConfig, field by field from the dataclass.

    Generic on purpose: a knob added to PoolConfig and the YAML is picked up
    here without a third edit (and a knob missing from the YAML fails loudly).
    Lists become tuples and `hold_copies` a sorted tuple of pairs, so the
    result stays frozen and hashable.
    """
    values = {}
    for f in fields(PoolConfig):
        value = cfg.pool[f.name]
        if f.name == "hold_copies":
            value = tuple(sorted((str(k), int(v)) for k, v in value.items()))
        elif OmegaConf.is_list(value):
            value = tuple(value)
        values[f.name] = value
    return PoolConfig(**values)


# Settings that change WHERE things are read from, not what the pool holds.
_NOT_IN_SIGNATURE = ("transcript_cache",)


def pool_signature(cfg: DictConfig, max_samples: int | None = None) -> dict:
    """Everything that determines a pool's contents, as plain JSON-able data."""
    pool = OmegaConf.to_container(cfg.pool, resolve=True)
    for key in _NOT_IN_SIGNATURE:
        pool.pop(key, None)
    return {"dataset_id": cfg.data.dataset_id, "pool": pool, "max_samples": max_samples}


def read_signature(pool_dir: Path, split: str) -> dict | None:
    """The settings `{split}.parquet` was built with (None if unrecorded)."""
    path = Path(pool_dir) / SIGNATURE_FILE
    return json.loads(path.read_text()).get(split) if path.exists() else None


def write_signature(pool_dir: Path, split: str, signature: dict) -> None:
    """Record one split's settings; other splits' entries are kept.

    Per split, not per dir: rebuilding train under new settings must not make
    an older validation manifest look current.
    """
    path = Path(pool_dir) / SIGNATURE_FILE
    entries = json.loads(path.read_text()) if path.exists() else {}
    entries[split] = signature
    path.write_text(json.dumps(entries, indent=2, sort_keys=True))


def pool_is_current(pool_dir: Path, split: str, signature: dict) -> bool:
    """True when `{split}.parquet` exists and was built with exactly `signature`."""
    exists = (Path(pool_dir) / f"{split}.parquet").exists()
    return exists and read_signature(pool_dir, split) == signature


def signature_diff(built: dict, wanted: dict, prefix: str = "") -> list[str]:
    """Human-readable `key: built -> wanted` lines for mismatched leaves."""
    out = []
    for key in sorted(set(built) | set(wanted)):
        a, b = built.get(key), wanted.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out += signature_diff(a, b, f"{prefix}{key}.")
        elif a != b:
            out.append(f"{prefix}{key}: {a!r} -> {b!r}")
    return out
