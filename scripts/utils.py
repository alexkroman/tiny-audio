#!/usr/bin/env python3
"""Shared utilities for scripts."""

import re
from pathlib import Path


def parse_results_file(results_path: Path) -> list[dict]:
    """Parse a results.txt file and return list of samples.

    Args:
        results_path: Path to a results.txt file from evaluation.

    Returns:
        List of dicts with keys: sample_num, ground_truth, prediction, wer,
        word_count, ground_truth_raw, prediction_raw. The `_raw` values are the
        un-normalized transcripts; they are `None` for runs written before
        those lines existed, so formatting metrics must skip such samples
        rather than silently score normalized text.
    """
    samples = []
    content = results_path.read_text()
    blocks = content.split("-" * 80)

    for block in blocks:
        sample_match = re.search(r"Sample (\d+) - WER: ([\d.]+)%", block)
        gt_match = re.search(r"^Ground Truth: (.+?)$", block, re.MULTILINE)
        pred_match = re.search(r"^Prediction:[ \t]*(.+?)$", block, re.MULTILINE)
        gt_raw_match = re.search(r"^Ground Truth Raw: (.+?)$", block, re.MULTILINE)
        pred_raw_match = re.search(r"^Prediction Raw:[ \t]*(.+?)$", block, re.MULTILINE)

        if sample_match and gt_match and pred_match:
            wer = float(sample_match.group(2))
            ground_truth = gt_match.group(1).strip()
            prediction = pred_match.group(1).strip()
            word_count = len(ground_truth.split())

            samples.append(
                {
                    "sample_num": int(sample_match.group(1)),
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "wer": wer,
                    "word_count": word_count,
                    "ground_truth_raw": (gt_raw_match.group(1).strip() if gt_raw_match else None),
                    "prediction_raw": (pred_raw_match.group(1).strip() if pred_raw_match else None),
                }
            )

    return samples


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


def load_data_config(name: str = "multiasr") -> list[dict]:
    """Return the `datasets` list from `configs/data/<name>.yaml`.

    Loaded through OmegaConf, the same parser Hydra uses for the training
    configs, so the debug probes read the file exactly as `ta train` does.
    Interpolations elsewhere in the file are left unresolved.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(get_project_root() / "configs" / "data" / f"{name}.yaml")
    return OmegaConf.to_container(cfg["datasets"])
