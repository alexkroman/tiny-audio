#!/usr/bin/env python3
"""Calculate WER broken down by reference utterance length (word count)."""

import re
from collections import defaultdict
from pathlib import Path

import typer

app = typer.Typer()


def parse_results_file(results_path: Path) -> list[dict]:
    """Parse a results.txt file and return list of samples."""
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


def analyze_by_word_count(samples: list[dict], max_words: int = 10) -> dict:
    """Group samples by word count and calculate stats."""
    by_length = defaultdict(list)

    for s in samples:
        wc = s["word_count"]
        if wc <= max_words:
            by_length[wc].append(s)
        else:
            by_length[max_words + 1].append(s)  # Group all longer ones

    stats = {}
    for wc in sorted(by_length.keys()):
        samples_at_length = by_length[wc]
        n = len(samples_at_length)
        avg_wer = sum(s["wer"] for s in samples_at_length) / n if n > 0 else 0
        perfect = sum(1 for s in samples_at_length if s["wer"] == 0)
        failures = sum(1 for s in samples_at_length if s["wer"] == 100)

        stats[wc] = {
            "count": n,
            "avg_wer": avg_wer,
            "perfect": perfect,
            "perfect_pct": perfect / n * 100 if n > 0 else 0,
            "failures": failures,
            "failures_pct": failures / n * 100 if n > 0 else 0,
        }

    return stats


def find_model_dirs(outputs_dir: Path, model_pattern: str) -> list[Path]:
    """Find output directories matching a model pattern."""
    dirs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        if model_pattern.lower() in d.name.lower():
            dirs.append(d)
    return sorted(dirs)


@app.command()
def main(
    model: str = typer.Argument(
        ..., help="Model name pattern to match (e.g., 'tiny-audio', 'Universal-Streaming')"
    ),
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
    max_words: int = typer.Option(
        10, help="Max word count to show individually (longer grouped as N+)"
    ),
    exclude: list[str] = typer.Option([], help="Patterns to exclude from matching"),
):
    """Calculate WER broken down by reference utterance length."""
    if not outputs_dir.exists():
        typer.echo(f"Error: {outputs_dir} does not exist")
        raise typer.Exit(1)

    # Find matching directories
    model_dirs = find_model_dirs(outputs_dir, model)

    # Apply exclusions
    if exclude:
        model_dirs = [
            d for d in model_dirs if not any(ex.lower() in d.name.lower() for ex in exclude)
        ]

    if not model_dirs:
        typer.echo(f"No output directories found matching '{model}'")
        raise typer.Exit(1)

    typer.echo(f"Found {len(model_dirs)} directories matching '{model}':")
    for d in model_dirs:
        typer.echo(f"  {d.name}")

    # Collect all samples
    all_samples = []
    for dir_path in model_dirs:
        results_file = dir_path / "results.txt"
        if results_file.exists():
            samples = parse_results_file(results_file)
            for s in samples:
                s["dataset"] = dir_path.name.split("_")[1]
            all_samples.extend(samples)

    if not all_samples:
        typer.echo("No samples found in results files")
        raise typer.Exit(1)

    typer.echo(f"\nTotal samples: {len(all_samples)}")

    # Calculate stats by word count
    stats = analyze_by_word_count(all_samples, max_words)

    # Print results
    typer.echo(f"\nWER by Utterance Length: {model}")
    typer.echo("=" * 70)
    typer.echo(f"{'Words':<8} {'Count':<8} {'WER':<10} {'Perfect':<12} {'Failures':<12}")
    typer.echo("-" * 70)

    for wc in sorted(stats.keys()):
        s = stats[wc]
        label = f"{wc}" if wc <= max_words else f"{max_words}+"
        typer.echo(
            f"{label:<8} {s['count']:<8} {s['avg_wer']:<10.1f} "
            f"{s['perfect']:<4} ({s['perfect_pct']:>5.1f}%) "
            f"{s['failures']:<4} ({s['failures_pct']:>5.1f}%)"
        )

    # Overall stats
    total_wer = sum(s["wer"] for s in all_samples) / len(all_samples)
    total_perfect = sum(1 for s in all_samples if s["wer"] == 0)
    total_failures = sum(1 for s in all_samples if s["wer"] == 100)

    typer.echo("-" * 70)
    typer.echo(
        f"{'Total':<8} {len(all_samples):<8} {total_wer:<10.1f} "
        f"{total_perfect:<4} ({total_perfect / len(all_samples) * 100:>5.1f}%) "
        f"{total_failures:<4} ({total_failures / len(all_samples) * 100:>5.1f}%)"
    )


if __name__ == "__main__":
    app()
