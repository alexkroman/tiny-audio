"""Trained-projector weight conversion + atomic cache markers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

# Bump this constant when the cache layout changes incompatibly.
MLX_FORMAT_VERSION = 1


def _t(t) -> mx.array:
    """Convert a torch.Tensor (or array-like) to an mx.array at fp16."""
    import torch

    if hasattr(t, "detach"):
        arr = t.detach().cpu().to(torch.float16).numpy()
    else:
        arr = np.asarray(t)
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
    return mx.array(arr)


def convert_projector_weights(pt_state: dict) -> dict[str, mx.array]:
    """Strip the 'projector.' prefix from a trained tiny-audio state_dict.

    The trained checkpoint at e.g. mazesmazes/tiny-audio-embedded contains only
    the projector weights — see asr_modeling.py:367-369:

        def state_dict(self):
            return {f"projector.{k}": v for k, v in self.projector.state_dict().items()}

    For our MLX projector module, we want the keys WITHOUT the prefix
    (e.g. "linear_1.weight" rather than "projector.linear_1.weight").
    """
    out: dict[str, mx.array] = {}
    for k, v in pt_state.items():
        if not k.startswith("projector."):
            continue
        out[k[len("projector.") :]] = _t(v)
    if not out:
        raise ValueError("No projector.* keys found in state_dict")
    return out


def safe_repo_id(repo_id: str) -> str:
    """Make an HF repo id safe to use as a directory name."""
    return repo_id.replace("/", "__")


def default_cache_root() -> Path:
    """Default location for converted MLX cache (~/.cache/tiny-audio/mlx)."""
    return Path(os.environ.get("HOME", "/tmp")) / ".cache" / "tiny-audio" / "mlx"


def _marker_path(cache_dir: Path) -> Path:
    return cache_dir / ".mlx_converted"


def mark_cache_complete(cache_dir: Path, version: int = MLX_FORMAT_VERSION) -> None:
    """Atomically write the completion marker. Tempfile + rename so partial
    writes can never look complete."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = _marker_path(cache_dir)
    fd, tmp = tempfile.mkstemp(prefix=".mlx_converted.", dir=str(cache_dir))
    try:
        try:
            os.write(fd, json.dumps({"version": version}).encode())
        finally:
            os.close(fd)
        Path(tmp).replace(marker)
    except Exception:
        tmp_path = Path(tmp)
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def is_cache_valid(cache_dir: Path, expected_version: int = MLX_FORMAT_VERSION) -> bool:
    """Return True iff cache_dir contains a complete-marker for the expected version."""
    marker = _marker_path(cache_dir)
    if not marker.exists():
        return False
    try:
        with marker.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("version") == expected_version
