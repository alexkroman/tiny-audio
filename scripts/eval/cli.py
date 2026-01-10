"""CLI for ASR, diarization, and alignment evaluation."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scripts.eval.audio import TextNormalizer
from scripts.eval.datasets import (
    ALIGNMENT_DATASETS,
    DATASET_REGISTRY,
    DIARIZATION_DATASETS,
    load_eval_dataset,
)
from scripts.eval.evaluators import (
    AssemblyAIAlignmentEvaluator,
    AssemblyAIDiarizationEvaluator,
    AssemblyAIEvaluator,
    AssemblyAIStreamingEvaluator,
    DeepgramAlignmentEvaluator,
    DeepgramDiarizationEvaluator,
    DeepgramEvaluator,
    DiarizationEvaluator,
    EndpointEvaluator,
    EvalResult,
    LocalEvaluator,
    LocalStreamingEvaluator,
    TimestampAlignmentEvaluator,
)

console = Console()


def build_model_id(
    provider: str, model_variant: str | None = None, api_key: str | None = None
) -> str:
    """Build a model identifier including provider, variant, and API key prefix.

    Examples:
        - assemblyai_slam1_abc
        - deepgram_nova3_xyz
        - assemblyai_abc (no variant)
    """
    parts = [provider]
    if model_variant:
        # Normalize variant name (remove underscores, hyphens for cleaner names)
        clean_variant = model_variant.replace("_", "").replace("-", "")
        parts.append(clean_variant)
    if api_key:
        parts.append(api_key[:3])
    return "_".join(parts)


def save_results(
    model_name: str,
    dataset_name: str,
    results: list[EvalResult],
    metrics: dict,
    output_dir: str = "outputs",
    base_url: str | None = None,
) -> Path:
    """Save evaluation results and metrics to a timestamped directory."""
    normalizer = TextNormalizer()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")

    # Extract short identifier from base_url (e.g., "sandbox013" from the URL)
    url_suffix = ""
    if base_url:
        # Extract hostname and create a short identifier
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        host = parsed.netloc or parsed.path
        # Extract meaningful part (e.g., "sandbox013" from "api.sandbox013.assemblyai-labs.com")
        parts = host.split(".")
        for part in parts:
            if "sandbox" in part.lower():
                url_suffix = f"_{part}"
                break
        if not url_suffix and host:
            # Fallback: use first part of hostname
            url_suffix = f"_{parts[0]}"

    result_dir = Path(output_dir) / f"{timestamp}_{safe_model_name}{url_suffix}_{dataset_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = result_dir / "results.txt"
    with results_file.open("w") as f:
        for i, r in enumerate(results, 1):
            norm_pred = normalizer.normalize(r.prediction)
            norm_ref = normalizer.normalize(r.reference)
            f.write(f"Sample {i} - WER: {r.wer:.2f}%\n")
            f.write(f"Ground Truth: {norm_ref}\n")
            f.write(f"Prediction: {norm_pred}\n")
            f.write("-" * 80 + "\n")

    # Save summary metrics
    metrics_file = result_dir / "metrics.txt"
    with metrics_file.open("w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        if base_url:
            f.write(f"Base URL: {base_url}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

    console.print(f"\nResults saved to: [bold]{result_dir}[/bold]")
    return result_dir


def save_diarization_results(
    model_name: str,
    dataset_name: str,
    results,
    metrics: dict,
    output_dir: str = "outputs",
) -> Path:
    """Save diarization evaluation results and metrics."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    result_dir = Path(output_dir) / f"{timestamp}_{safe_model_name}_{dataset_name}_diarization"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = result_dir / "results.txt"
    with results_file.open("w") as f:
        for i, r in enumerate(results, 1):
            f.write(f"Sample {i}\n")
            f.write(f"  DER: {r.der:.2f}%\n")
            f.write(
                f"  Components: conf={r.confusion:.2f}%, miss={r.missed:.2f}%, fa={r.false_alarm:.2f}%\n"
            )
            f.write(f"  Speakers: ref={r.num_speakers_ref}, hyp={r.num_speakers_hyp}\n")
            f.write(f"  Time: {r.time:.2f}s\n")
            f.write("-" * 80 + "\n")

    # Save summary metrics
    metrics_file = result_dir / "metrics.txt"
    with metrics_file.open("w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

    console.print(f"\nResults saved to: [bold]{result_dir}[/bold]")
    return result_dir


def save_alignment_results(
    model_name: str,
    dataset_name: str,
    results,
    metrics: dict,
    output_dir: str = "outputs",
) -> Path:
    """Save alignment evaluation results and metrics."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    result_dir = Path(output_dir) / f"{timestamp}_{safe_model_name}_{dataset_name}_alignment"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = result_dir / "results.txt"
    with results_file.open("w") as f:
        for i, r in enumerate(results, 1):
            f.write(f"Sample {i}\n")
            f.write(f"  Aligned: {r.num_aligned_words}/{r.num_ref_words} words\n")
            if r.num_aligned_words > 0:
                mae_start = sum(abs(p - ref) for p, ref in zip(r.pred_starts, r.ref_starts)) / len(
                    r.pred_starts
                )
                mae_end = sum(abs(p - ref) for p, ref in zip(r.pred_ends, r.ref_ends)) / len(
                    r.pred_ends
                )
                f.write(f"  MAE (start): {mae_start * 1000:.1f}ms\n")
                f.write(f"  MAE (end): {mae_end * 1000:.1f}ms\n")
            f.write(f"  Time: {r.time:.2f}s\n")
            f.write(f"  Reference: {r.reference_text[:100]}...\n")
            f.write(f"  Prediction: {r.predicted_text[:100]}...\n")
            f.write("-" * 80 + "\n")

    # Save summary metrics
    metrics_file = result_dir / "metrics.txt"
    with metrics_file.open("w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

    console.print(f"\nResults saved to: [bold]{result_dir}[/bold]")
    return result_dir


def print_asr_metrics(dataset_name: str, metrics: dict):
    """Print ASR metrics using rich table."""
    table = Table(title=f"Results: {dataset_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("WER", f"{metrics['wer']:.2f}%")
    table.add_row("Samples", str(metrics["num_samples"]))
    table.add_row("Avg Time", f"{metrics['avg_time']:.2f}s")

    if "avg_ttfb" in metrics:
        table.add_row("Avg TTFB", f"{metrics['avg_ttfb'] * 1000:.0f}ms")
        table.add_row(
            "TTFB Range", f"{metrics['min_ttfb'] * 1000:.0f}ms - {metrics['max_ttfb'] * 1000:.0f}ms"
        )
    if "avg_processing" in metrics:
        table.add_row("Avg Processing", f"{metrics['avg_processing'] * 1000:.0f}ms")

    console.print(table)


def print_diarization_metrics(dataset_name: str, metrics: dict):
    """Print diarization metrics using rich table."""
    table = Table(title=f"Results: {dataset_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("DER", f"{metrics['der']:.2f}%")
    table.add_row("Confusion", f"{metrics['confusion']:.2f}%")
    table.add_row("Missed", f"{metrics['missed']:.2f}%")
    table.add_row("False Alarm", f"{metrics['false_alarm']:.2f}%")
    table.add_row("Samples", str(metrics["num_samples"]))
    table.add_row("Avg Time", f"{metrics['avg_time']:.2f}s")

    console.print(table)


def print_alignment_metrics(dataset_name: str, metrics: dict):
    """Print alignment metrics using rich table."""
    table = Table(title=f"Results: {dataset_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("MAE", f"{metrics['mae'] * 1000:.1f}ms")
    table.add_row("Alignment Error", f"{metrics.get('alignment_error', 0) * 100:.1f}%")
    table.add_row(
        "Words Aligned",
        f"{metrics.get('total_aligned_words', 0)}/{metrics.get('total_ref_words', 0)}",
    )
    table.add_row("Samples", str(metrics["num_samples"]))
    table.add_row("Avg Time", f"{metrics['avg_time']:.2f}s")

    console.print(table)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate ASR models on standard datasets")
    parser.add_argument("model", help="Model path/ID, 'assemblyai', or 'deepgram'")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["loquacious"],
        choices=["all", "all-full"] + list(DATASET_REGISTRY.keys()),
        help="Datasets to evaluate on (default: loquacious, 'all' for ASR only, 'all-full' includes diarization/alignment)",
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--endpoint", action="store_true", help="Use HF Inference Endpoint")
    parser.add_argument(
        "--assemblyai-model",
        default="slam_1",
        choices=["best", "universal", "slam_1", "nano"],
        help="AssemblyAI model (default: slam_1)",
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming evaluation (for local or AAI)"
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for diarization models")
    parser.add_argument(
        "--num-speakers", type=int, default=None, help="Number of speakers (for diarization)"
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None, help="Min speakers (for diarization)"
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None, help="Max speakers (for diarization)"
    )
    parser.add_argument(
        "--config", default=None, help="Dataset config override (e.g., 'en' for CommonVoice)"
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory for results")
    parser.add_argument(
        "--user-prompt", type=str, default=None, help="Custom user prompt for the model"
    )
    parser.add_argument(
        "--base-url", type=str, default=None, help="Custom API base URL (for AssemblyAI sandbox)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for API evaluations (default: 1)",
    )
    args = parser.parse_args()

    # Expand "all" to ASR datasets only (exclude diarization and alignment)
    if "all" in args.datasets:
        args.datasets = [
            k
            for k in DATASET_REGISTRY
            if k not in DIARIZATION_DATASETS and k not in ALIGNMENT_DATASETS
        ]
    # Expand "all-full" to include diarization and alignment datasets too
    elif "all-full" in args.datasets:
        args.datasets = list(DATASET_REGISTRY.keys())

    for dataset_name in args.datasets:
        console.print(f"\n[bold blue]Evaluating on: {dataset_name}[/bold blue]")

        cfg = DATASET_REGISTRY[dataset_name]
        split = cfg.default_split if args.split == "test" else args.split

        # Handle diarization datasets
        if dataset_name in DIARIZATION_DATASETS:
            dataset = load_eval_dataset(dataset_name, split, args.config, decode_audio=False)
            model_id = args.model  # Default, override for API providers

            if args.model == "assemblyai":
                api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
                if not api_key:
                    console.print(
                        "[red]Error: ASSEMBLYAI_API_KEY environment variable not set[/red]"
                    )
                    sys.exit(1)
                model_id = build_model_id("assemblyai", args.assemblyai_model, api_key)
                evaluator = AssemblyAIDiarizationEvaluator(
                    api_key=api_key,
                    model=args.assemblyai_model,
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                )
            elif args.model == "deepgram":
                api_key = os.environ.get("DEEPGRAM_API_KEY", "")
                if not api_key:
                    console.print("[red]Error: DEEPGRAM_API_KEY environment variable not set[/red]")
                    sys.exit(1)
                model_id = build_model_id("deepgram", None, api_key)
                evaluator = DeepgramDiarizationEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                )
            else:
                evaluator = DiarizationEvaluator(
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                    hf_token=args.hf_token,
                    num_speakers=args.num_speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )

            results = evaluator.evaluate(dataset, args.max_samples)
            metrics = evaluator.compute_metrics()
            save_diarization_results(model_id, dataset_name, results, metrics, args.output_dir)
            print_diarization_metrics(dataset_name, metrics)
            continue

        # Handle alignment datasets
        if dataset_name in ALIGNMENT_DATASETS:
            dataset = load_eval_dataset(dataset_name, split, args.config)
            model_id = args.model  # Default, override for API providers

            if args.model == "assemblyai":
                api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
                if not api_key:
                    console.print(
                        "[red]Error: ASSEMBLYAI_API_KEY environment variable not set[/red]"
                    )
                    sys.exit(1)
                model_id = build_model_id("assemblyai", args.assemblyai_model, api_key)
                evaluator = AssemblyAIAlignmentEvaluator(
                    api_key=api_key,
                    model=args.assemblyai_model,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    words_field=cfg.words_field,
                )
            elif args.model == "deepgram":
                api_key = os.environ.get("DEEPGRAM_API_KEY", "")
                if not api_key:
                    console.print("[red]Error: DEEPGRAM_API_KEY environment variable not set[/red]")
                    sys.exit(1)
                model_id = build_model_id("deepgram", None, api_key)
                evaluator = DeepgramAlignmentEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    words_field=cfg.words_field,
                )
            else:
                evaluator = TimestampAlignmentEvaluator(
                    model_path=args.model,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    words_field=cfg.words_field,
                    user_prompt=args.user_prompt,
                )

            results = evaluator.evaluate(dataset, args.max_samples)
            metrics = evaluator.compute_metrics()
            save_alignment_results(model_id, dataset_name, results, metrics, args.output_dir)
            print_alignment_metrics(dataset_name, metrics)
            continue

        # ASR evaluation
        dataset = load_eval_dataset(dataset_name, split, args.config)
        model_id = args.model  # Default to args.model, override for API providers

        if args.model == "assemblyai":
            api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
            if not api_key:
                console.print("[red]Error: ASSEMBLYAI_API_KEY environment variable not set[/red]")
                sys.exit(1)
            model_id = build_model_id("assemblyai", args.assemblyai_model, api_key)

            if args.streaming:
                evaluator = AssemblyAIStreamingEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                )
            else:
                evaluator = AssemblyAIEvaluator(
                    api_key=api_key,
                    model=args.assemblyai_model,
                    base_url=args.base_url,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    num_workers=args.num_workers,
                )
        elif args.model == "deepgram":
            api_key = os.environ.get("DEEPGRAM_API_KEY", "")
            if not api_key:
                console.print("[red]Error: DEEPGRAM_API_KEY environment variable not set[/red]")
                sys.exit(1)
            model_id = build_model_id("deepgram", None, api_key)
            evaluator = DeepgramEvaluator(
                api_key=api_key,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
                num_workers=args.num_workers,
            )
        elif args.endpoint:
            evaluator = EndpointEvaluator(
                endpoint_url=args.model,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
            )
        elif args.streaming:
            evaluator = LocalStreamingEvaluator(
                model_path=args.model,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
                user_prompt=args.user_prompt,
            )
        else:
            evaluator = LocalEvaluator(
                model_path=args.model,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
                user_prompt=args.user_prompt,
            )

        results = evaluator.evaluate(dataset, args.max_samples)
        metrics = evaluator.compute_metrics()
        save_results(model_id, dataset_name, results, metrics, args.output_dir, args.base_url)
        print_asr_metrics(dataset_name, metrics)


if __name__ == "__main__":
    main()
