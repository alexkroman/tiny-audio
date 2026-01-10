#!/usr/bin/env python3
"""Shared utilities for scripts."""

import re
from pathlib import Path


def parse_results_file(results_path: Path) -> list[dict]:
    """Parse a results.txt file and return list of samples.

    Args:
        results_path: Path to a results.txt file from evaluation.

    Returns:
        List of dicts with keys: sample_num, ground_truth, prediction, wer, word_count
    """
    samples = []
    content = results_path.read_text()
    blocks = content.split("-" * 80)

    for block in blocks:
        sample_match = re.search(r"Sample (\d+) - WER: ([\d.]+)%", block)
        gt_match = re.search(r"Ground Truth: (.+?)(?:\n|$)", block)
        pred_match = re.search(r"Prediction:\s*(.+?)(?:\n|$)", block)

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
                }
            )

    return samples


def find_model_dirs(
    outputs_dir: Path,
    model_pattern: str,
    exclude: list[str] | None = None,
    latest: bool = False,
) -> list[Path]:
    """Find output directories matching a model pattern.

    Args:
        outputs_dir: Base directory containing evaluation outputs.
        model_pattern: Pattern to match in directory names.
        exclude: List of patterns to exclude from matching.
        latest: If True, only return the most recent run per dataset.

    Returns:
        Sorted list of matching directory paths.
    """
    exclude = [ex for ex in (exclude or []) if ex]  # Filter out empty strings
    dirs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        if model_pattern.lower() in d.name.lower() and not any(
            ex.lower() in d.name.lower() for ex in exclude
        ):
            dirs.append(d)

    if latest:
        # Group by dataset and keep only most recent (dirs are sorted by timestamp in name)
        latest_by_dataset: dict[str, Path] = {}
        for d in sorted(dirs, reverse=True):  # Sort descending by name (timestamp first)
            # Extract dataset from dir name: <timestamp>_<model>_<dataset>
            parts = d.name.split("_")
            if len(parts) >= 3:
                dataset = parts[-1]  # Last part is dataset
                if dataset not in latest_by_dataset:
                    latest_by_dataset[dataset] = d
        dirs = list(latest_by_dataset.values())

    return sorted(dirs)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent
