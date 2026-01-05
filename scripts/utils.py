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
    outputs_dir: Path, model_pattern: str, exclude: list[str] | None = None
) -> list[Path]:
    """Find output directories matching a model pattern.

    Args:
        outputs_dir: Base directory containing evaluation outputs.
        model_pattern: Pattern to match in directory names.
        exclude: List of patterns to exclude from matching.

    Returns:
        Sorted list of matching directory paths.
    """
    exclude = exclude or []
    dirs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        if model_pattern.lower() in d.name.lower() and not any(
            ex.lower() in d.name.lower() for ex in exclude
        ):
            dirs.append(d)
    return sorted(dirs)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent
