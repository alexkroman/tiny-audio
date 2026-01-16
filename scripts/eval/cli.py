"""CLI for ASR, diarization, and alignment evaluation."""

import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
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
    LocalDiarizationEvaluator,
    LocalEvaluator,
    LocalStreamingEvaluator,
    TimestampAlignmentEvaluator,
)

app = typer.Typer(help="Evaluate ASR models on standard datasets")
console = Console()


class AssemblyAIModel(str, Enum):
    """AssemblyAI model options."""

    best = "best"
    universal = "universal"
    slam_1 = "slam_1"
    nano = "nano"


# Valid dataset choices
VALID_DATASETS = ["all", "all-full"] + list(DATASET_REGISTRY.keys())


def get_model_name(model_path: str) -> str:
    """Extract model name from a HuggingFace model path.

    Examples:
        - mazesmazes/tiny-audio -> tiny-audio
        - /path/to/checkpoint -> checkpoint
    """
    return model_path.rstrip("/").split("/")[-1]


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


def validate_datasets(datasets: list[str]) -> list[str]:
    """Validate and expand dataset names."""
    for ds in datasets:
        if ds not in VALID_DATASETS:
            console.print(f"[red]Error: Invalid dataset '{ds}'[/red]")
            console.print(f"Valid choices: {', '.join(VALID_DATASETS)}")
            raise typer.Exit(1)

    # Expand "all" to ASR datasets only (exclude diarization and alignment)
    if "all" in datasets:
        return [
            k
            for k in DATASET_REGISTRY
            if k not in DIARIZATION_DATASETS and k not in ALIGNMENT_DATASETS
        ]
    # Expand "all-full" to include diarization and alignment datasets too
    if "all-full" in datasets:
        return list(DATASET_REGISTRY.keys())

    return datasets


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model path/ID, 'assemblyai', or 'deepgram'"),
    ] = None,
    datasets: Annotated[
        Optional[list[str]],
        typer.Option(
            "--datasets",
            "-d",
            help="Datasets to evaluate on ('all' for ASR only, 'all-full' includes diarization/alignment)",
        ),
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    max_samples: Annotated[
        Optional[int], typer.Option("--max-samples", "-n", help="Maximum samples to evaluate")
    ] = None,
    endpoint: Annotated[
        bool, typer.Option("--endpoint", "-e", help="Use HF Inference Endpoint")
    ] = False,
    assemblyai_model: Annotated[
        AssemblyAIModel, typer.Option("--assemblyai-model", help="AssemblyAI model")
    ] = AssemblyAIModel.slam_1,
    streaming: Annotated[
        bool, typer.Option("--streaming", "-s", help="Use streaming evaluation (for local or AAI)")
    ] = False,
    hf_token: Annotated[
        Optional[str], typer.Option("--hf-token", help="HuggingFace token for diarization models")
    ] = None,
    num_speakers: Annotated[
        Optional[int], typer.Option("--num-speakers", help="Number of speakers (for diarization)")
    ] = None,
    min_speakers: Annotated[
        Optional[int], typer.Option("--min-speakers", help="Min speakers (for diarization)")
    ] = None,
    max_speakers: Annotated[
        Optional[int], typer.Option("--max-speakers", help="Max speakers (for diarization)")
    ] = None,
    config: Annotated[
        Optional[str],
        typer.Option("--config", "-c", help="Dataset config override (e.g., 'en' for CommonVoice)"),
    ] = None,
    output_dir: Annotated[
        str, typer.Option("--output-dir", "-o", help="Output directory for results")
    ] = "outputs",
    user_prompt: Annotated[
        Optional[str], typer.Option("--user-prompt", help="Custom user prompt for the model")
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", help="Custom API base URL (for AssemblyAI sandbox)"),
    ] = None,
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", "-w", help="Number of parallel workers for API evaluations"),
    ] = 1,
):
    """Evaluate ASR models on standard datasets."""
    # If a subcommand was invoked, skip
    if ctx.invoked_subcommand is not None:
        return

    # Require model when running directly
    if model is None:
        console.print("[red]Error: --model / -m is required[/red]")
        console.print("Example: ta eval -m assemblyai -d loquacious")
        raise typer.Exit(1)

    # Default to loquacious if no datasets specified
    if datasets is None:
        datasets = ["loquacious"]

    # Validate and expand datasets
    datasets = validate_datasets(datasets)

    for dataset_name in datasets:
        console.print(f"\n[bold blue]Evaluating on: {dataset_name}[/bold blue]")

        cfg = DATASET_REGISTRY[dataset_name]
        actual_split = cfg.default_split if split == "test" else split

        # Handle diarization datasets
        if dataset_name in DIARIZATION_DATASETS:
            dataset = load_eval_dataset(dataset_name, actual_split, config, decode_audio=False)
            if model == "assemblyai":
                api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
                if not api_key:
                    console.print(
                        "[red]Error: ASSEMBLYAI_API_KEY environment variable not set[/red]"
                    )
                    raise typer.Exit(1)
                model_id = assemblyai_model.value.replace("_", "-")
                evaluator = AssemblyAIDiarizationEvaluator(
                    api_key=api_key,
                    model=assemblyai_model.value,
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                )
            elif model == "deepgram":
                api_key = os.environ.get("DEEPGRAM_API_KEY", "")
                if not api_key:
                    console.print("[red]Error: DEEPGRAM_API_KEY environment variable not set[/red]")
                    raise typer.Exit(1)
                model_id = "nova-3"
                evaluator = DeepgramDiarizationEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                )
            elif model == "local":
                # Local diarization using TEN-VAD + ERes2NetV2 + spectral clustering
                model_id = "local"
                evaluator = LocalDiarizationEvaluator(
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers or 2,
                    max_speakers=max_speakers or 3,
                )
            elif model == "pyannote":
                # Pyannote diarization (requires HF token with model access)
                model_id = "pyannote"
                evaluator = DiarizationEvaluator(
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                    hf_token=hf_token,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
            else:
                # Default to pyannote for other model paths
                model_id = get_model_name(model)
                evaluator = DiarizationEvaluator(
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                    hf_token=hf_token,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )

            results = evaluator.evaluate(dataset, max_samples)
            metrics = evaluator.compute_metrics()
            save_diarization_results(model_id, dataset_name, results, metrics, output_dir)
            print_diarization_metrics(dataset_name, metrics)
            continue

        # Handle alignment datasets
        if dataset_name in ALIGNMENT_DATASETS:
            dataset = load_eval_dataset(dataset_name, actual_split, config)

            if model == "assemblyai":
                api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
                if not api_key:
                    console.print(
                        "[red]Error: ASSEMBLYAI_API_KEY environment variable not set[/red]"
                    )
                    raise typer.Exit(1)
                model_id = assemblyai_model.value.replace("_", "-")
                evaluator = AssemblyAIAlignmentEvaluator(
                    api_key=api_key,
                    model=assemblyai_model.value,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    words_field=cfg.words_field,
                )
            elif model == "deepgram":
                api_key = os.environ.get("DEEPGRAM_API_KEY", "")
                if not api_key:
                    console.print("[red]Error: DEEPGRAM_API_KEY environment variable not set[/red]")
                    raise typer.Exit(1)
                model_id = "nova-3"
                evaluator = DeepgramAlignmentEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    words_field=cfg.words_field,
                )
            else:
                model_id = get_model_name(model)
                evaluator = TimestampAlignmentEvaluator(
                    model_path=model,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    words_field=cfg.words_field,
                    user_prompt=user_prompt,
                )

            results = evaluator.evaluate(dataset, max_samples)
            metrics = evaluator.compute_metrics()
            save_alignment_results(model_id, dataset_name, results, metrics, output_dir)
            print_alignment_metrics(dataset_name, metrics)
            continue

        # ASR evaluation
        dataset = load_eval_dataset(dataset_name, actual_split, config)

        if model == "assemblyai":
            api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
            if not api_key:
                console.print("[red]Error: ASSEMBLYAI_API_KEY environment variable not set[/red]")
                raise typer.Exit(1)

            if streaming:
                model_id = "universal-streaming"
                evaluator = AssemblyAIStreamingEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    num_workers=num_workers,
                )
            else:
                model_id = assemblyai_model.value.replace("_", "-")
                evaluator = AssemblyAIEvaluator(
                    api_key=api_key,
                    model=assemblyai_model.value,
                    base_url=base_url,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    num_workers=num_workers,
                )
        elif model == "deepgram":
            api_key = os.environ.get("DEEPGRAM_API_KEY", "")
            if not api_key:
                console.print("[red]Error: DEEPGRAM_API_KEY environment variable not set[/red]")
                raise typer.Exit(1)
            model_id = "nova-3"
            evaluator = DeepgramEvaluator(
                api_key=api_key,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
                num_workers=num_workers,
            )
        elif endpoint:
            model_id = get_model_name(model)
            evaluator = EndpointEvaluator(
                endpoint_url=model,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
            )
        elif streaming:
            model_id = get_model_name(model)
            evaluator = LocalStreamingEvaluator(
                model_path=model,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
                user_prompt=user_prompt,
            )
        else:
            model_id = get_model_name(model)
            evaluator = LocalEvaluator(
                model_path=model,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
                user_prompt=user_prompt,
            )

        results = evaluator.evaluate(dataset, max_samples)
        metrics = evaluator.compute_metrics()
        save_results(model_id, dataset_name, results, metrics, output_dir, base_url)
        print_asr_metrics(dataset_name, metrics)


if __name__ == "__main__":
    app()
