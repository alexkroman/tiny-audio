"""Hydra config access outside `@hydra.main`, and the pool signature.

`build-pool`, `ta runpod` and training all resolve the same
configs/turn_aware tree with the same overrides, so a pool and the run that
trains on it cannot disagree. Each pool dir records the settings that built
it in pool_config.json: build-pool skips a split whose manifest matches the
current settings, and training refuses one that does not.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scripts.turn_aware.data import PoolConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "turn_aware"
SIGNATURE_FILE = "pool_config.json"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose configs/turn_aware/config.yaml with Hydra-style overrides."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides or []))


def pool_config(cfg: DictConfig) -> PoolConfig:
    """The `pool:` section as a PoolConfig (lists -> tuples, hold_copies -> pairs)."""
    p = cfg.pool
    return PoolConfig(
        fire_tail_s=tuple(p.fire_tail_s),
        hold_tail_s=tuple(p.hold_tail_s),
        long_tail_s=tuple(p.long_tail_s),
        long_tail_prob=p.long_tail_prob,
        nosil_prob=p.nosil_prob,
        lead_sil_s=tuple(p.lead_sil_s),
        lead_sil_prob=p.lead_sil_prob,
        silence_only_frac=p.silence_only_frac,
        silence_only_s=tuple(p.silence_only_s),
        ctx_prob=p.ctx_prob,
        max_audio_s=p.max_audio_s,
        hold_copies=tuple(sorted((str(k), int(v)) for k, v in p.hold_copies.items())),
        strip_hold_punct=p.strip_hold_punct,
    )


def pool_signature(cfg: DictConfig, max_samples: int | None = None) -> dict:
    """Everything that determines a pool's contents, as plain JSON-able data."""
    return {
        "dataset_id": cfg.data.dataset_id,
        "pool": OmegaConf.to_container(cfg.pool, resolve=True),
        "max_samples": max_samples,
    }


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
