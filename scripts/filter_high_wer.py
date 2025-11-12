#!/usr/bin/env python3
"""
Filter eval results to show only samples with WER above a threshold.

Usage:
    python scripts/filter_high_wer.py <input_file> <wer_threshold> [--output <output_file>]

Examples:
    # Show samples with WER > 20%
    python scripts/filter_high_wer.py outputs/eval_loquacious_model/results.txt 20

    # Save filtered results to a file
    python scripts/filter_high_wer.py outputs/eval_loquacious_model/results.txt 20 --output high_wer_samples.txt

    # Show only high error samples (WER > 50%)
    python scripts/filter_high_wer.py outputs/eval_loquacious_model/results.txt 50
"""

import argparse
import re
import sys
from pathlib import Path


def parse_results_file(file_path: Path) -> tuple[dict, list[dict]]:
    """
    Parse the eval results file.

    Returns:
        tuple: (metadata dict, list of sample dicts)
    """
    from pathlib import Path

    with Path(file_path).open() as f:
        content = f.read()

    # Extract metadata (first section)
    metadata = {}
    lines = content.split("\n")
    for line in lines[:10]:  # Metadata is in first few lines
        if line.startswith("Model:"):
            metadata["model"] = line.split("Model:", 1)[1].strip()
        elif line.startswith("Dataset:"):
            metadata["dataset"] = line.split("Dataset:", 1)[1].strip()
        elif line.startswith("Samples:"):
            metadata["samples"] = line.split("Samples:", 1)[1].strip()
        elif line.startswith("WER:"):
            metadata["overall_wer"] = line.split("WER:", 1)[1].strip()
        elif line.startswith("Avg Response Time:"):
            metadata["avg_time"] = line.split("Avg Response Time:", 1)[1].strip()

    # Split into sample blocks
    # Pattern: "Sample N - WER: X.XX%, Time: X.XXs"
    sample_pattern = re.compile(
        r"Sample (\d+) - WER: ([\d.]+)%, Time: ([\d.]+)s\n"
        r"Ground Truth: (.*?)\n"
        r"Prediction:\s+(.*?)\n",
        re.MULTILINE,
    )

    samples = []
    for match in sample_pattern.finditer(content):
        sample_num = int(match.group(1))
        wer = float(match.group(2))
        time = float(match.group(3))
        ground_truth = match.group(4).strip()
        prediction = match.group(5).strip()

        samples.append(
            {
                "sample_num": sample_num,
                "wer": wer,
                "time": time,
                "ground_truth": ground_truth,
                "prediction": prediction,
            }
        )

    return metadata, samples


def filter_samples_by_wer(samples: list[dict], threshold: float) -> list[dict]:
    """Filter samples to only those with WER above threshold."""
    return [s for s in samples if s["wer"] > threshold]


def format_output(
    metadata: dict, filtered_samples: list[dict], threshold: float, total_samples: int
) -> str:
    """Format the filtered results for output."""
    output_lines = []

    # Header with metadata
    output_lines.append("=" * 80)
    output_lines.append(f"Filtered Results: WER > {threshold}%")
    output_lines.append("=" * 80)
    output_lines.append(f"Model: {metadata.get('model', 'Unknown')}")
    output_lines.append(f"Dataset: {metadata.get('dataset', 'Unknown')}")
    output_lines.append(f"Overall WER: {metadata.get('overall_wer', 'Unknown')}")
    output_lines.append(f"Total Samples: {total_samples}")
    output_lines.append(
        f"Filtered Samples: {len(filtered_samples)} ({len(filtered_samples) / total_samples * 100:.1f}%)"
    )
    output_lines.append("=" * 80)
    output_lines.append("")

    # Sort by WER descending (worst first)
    sorted_samples = sorted(filtered_samples, key=lambda x: x["wer"], reverse=True)

    # Output each filtered sample
    for sample in sorted_samples:
        output_lines.append(
            f"Sample {sample['sample_num']} - WER: {sample['wer']:.2f}%, Time: {sample['time']:.2f}s"
        )
        output_lines.append(f"Ground Truth: {sample['ground_truth']}")
        output_lines.append(f"Prediction:   {sample['prediction']}")
        output_lines.append("-" * 80)
        output_lines.append("")

    # Summary statistics
    if filtered_samples:
        avg_wer = sum(s["wer"] for s in filtered_samples) / len(filtered_samples)
        max_wer = max(s["wer"] for s in filtered_samples)
        min_wer = min(s["wer"] for s in filtered_samples)
        avg_time = sum(s["time"] for s in filtered_samples) / len(filtered_samples)

        output_lines.append("=" * 80)
        output_lines.append("Summary Statistics (for filtered samples)")
        output_lines.append("=" * 80)
        output_lines.append(f"Count: {len(filtered_samples)}")
        output_lines.append(f"Average WER: {avg_wer:.2f}%")
        output_lines.append(f"Max WER: {max_wer:.2f}%")
        output_lines.append(f"Min WER: {min_wer:.2f}%")
        output_lines.append(f"Avg Time: {avg_time:.2f}s")
        output_lines.append("=" * 80)

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Filter eval results to show only samples with WER above threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show samples with WER > 20%
  %(prog)s outputs/eval_loquacious_model/results.txt 20

  # Save filtered results to a file
  %(prog)s outputs/eval_loquacious_model/results.txt 20 --output high_wer.txt

  # Show only high error samples (WER > 50%)
  %(prog)s outputs/eval_loquacious_model/results.txt 50
        """,
    )
    parser.add_argument("input_file", type=Path, help="Path to the eval results.txt file")
    parser.add_argument(
        "wer_threshold", type=float, help="WER threshold percentage (e.g., 20 for 20%%)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (default: print to stdout)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Parse the results file
    try:
        metadata, samples = parse_results_file(args.input_file)
    except Exception as e:
        print(f"Error parsing results file: {e}", file=sys.stderr)
        sys.exit(1)

    if not samples:
        print("Error: No samples found in results file", file=sys.stderr)
        sys.exit(1)

    # Filter samples
    filtered_samples = filter_samples_by_wer(samples, args.wer_threshold)

    # Format output
    output = format_output(metadata, filtered_samples, args.wer_threshold, len(samples))

    # Write or print
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            f.write(output)
        print(f"Filtered results saved to: {args.output}")
        print(f"Found {len(filtered_samples)} samples with WER > {args.wer_threshold}%")
    else:
        print(output)


if __name__ == "__main__":
    main()
