#!/usr/bin/env python3
"""Shared utilities for scripts."""

from pathlib import Path


def _extract_model_from_dir(dir_name: str) -> str:
    """Extract model name from directory name.

    Format: {timestamp_date}_{timestamp_time}_{model}[_{endpoint}]_{dataset}.
    `scripts.eval.cli._dir_segment` strips `_` out of each field as the run is
    written, so field 2 is the whole model label. Directories written before
    that sanitizer landed can still carry `_` inside the model name and will
    parse short here.
    """
    parts = dir_name.split("_")
    if len(parts) < 3:
        return dir_name
    return parts[2]


def find_model_dirs(
    outputs_dir: Path,
    model_pattern: str,
    exclude: list[str] | None = None,
    latest: bool = False,
) -> list[Path]:
    """Find output directories matching a model pattern.

    Args:
        outputs_dir: Base directory containing evaluation outputs.
        model_pattern: Pattern to match model name exactly (not substring).
            Empty matches every model, which is what `extract-entities`
            documents its empty default to mean.
        exclude: List of patterns to exclude from matching.
        latest: If True, keep only the most recent run of each distinct
            evaluation (same model, endpoint and dataset).

    Returns:
        Sorted list of matching directory paths.
    """
    exclude = [ex for ex in (exclude or []) if ex]
    dirs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        model_name = _extract_model_from_dir(d.name)
        if (not model_pattern or model_name.lower() == model_pattern.lower()) and not any(
            ex.lower() in d.name.lower() for ex in exclude
        ):
            dirs.append(d)

    if latest:
        # Key on everything after the timestamp -- model, optional endpoint
        # suffix and dataset -- so "latest" means the newest run of the same
        # evaluation. Keying on the trailing field alone used the `_mcq` /
        # `_diarization` suffix as the dataset and ignored the model entirely,
        # so an empty `model_pattern` collapsed every model into one entry and
        # only one MCQ dataset could ever reach the comparison table.
        latest_by_run: dict[str, Path] = {}
        for d in sorted(dirs, reverse=True):
            parts = d.name.split("_", 2)
            latest_by_run.setdefault(parts[2] if len(parts) >= 3 else d.name, d)
        dirs = list(latest_by_run.values())

    return sorted(dirs)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent
