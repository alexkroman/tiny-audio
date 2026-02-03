"""CLI for ASR, diarization, and alignment evaluation."""

import os

# Use soundfile for audio decoding instead of torchaudio (avoids compatibility issues)
os.environ.setdefault("HF_DATASETS_AUDIO_DECODER", "soundfile")
from datetime import datetime, timezone
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
    MCQ_DATASETS,
    load_eval_dataset,
)
from scripts.eval.evaluators import (
    ALL_TASKS as AIRBENCH_ALL_TASKS,
)
from scripts.eval.evaluators import (
    AIRBenchEvaluator,
    AIRBenchResult,
    AssemblyAIAlignmentEvaluator,
    AssemblyAIDiarizationEvaluator,
    AssemblyAIEvaluator,
    AssemblyAIMMAUEvaluator,
    AssemblyAIStreamingEvaluator,
    DeepgramAlignmentEvaluator,
    DeepgramDiarizationEvaluator,
    DeepgramEvaluator,
    ElevenLabsAlignmentEvaluator,
    ElevenLabsDiarizationEvaluator,
    ElevenLabsEvaluator,
    EndpointEvaluator,
    EvalResult,
    LocalDiarizationEvaluator,
    LocalEvaluator,
    LocalStreamingEvaluator,
    MCQResult,
    MMAUEvaluator,
    TimestampAlignmentEvaluator,
    print_airbench_metrics,
)

app = typer.Typer(help="Evaluate ASR models on standard datasets")
console = Console()


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
    table.add_row("Samples", str(metrics["num_samples"]))
    table.add_row("Avg Time", f"{metrics['avg_time']:.2f}s")

    console.print(table)


def save_mcq_results(
    model_name: str,
    dataset_name: str,
    results: list[MCQResult],
    metrics: dict,
    output_dir: str = "outputs",
) -> Path:
    """Save MCQ evaluation results and metrics."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    result_dir = Path(output_dir) / f"{timestamp}_{safe_model_name}_{dataset_name}_mcq"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = result_dir / "results.txt"
    with results_file.open("w") as f:
        for i, r in enumerate(results, 1):
            status = "✓" if r.correct else "✗"
            f.write(f"Sample {i} [{status}]\n")
            f.write(f"  Category: {r.category}\n")
            f.write(f"  Question: {r.question}\n")
            f.write(f"  Choices: {r.choices}\n")
            f.write(f"  Prediction: {r.prediction}\n")
            f.write(f"  Matched: {r.matched_choice}\n")
            f.write(f"  Reference: {r.reference}\n")
            f.write(f"  Time: {r.time:.2f}s\n")
            f.write("-" * 80 + "\n")

    # Save summary metrics
    metrics_file = result_dir / "metrics.txt"
    with metrics_file.open("w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Correct: {metrics['correct']}/{metrics['total']}\n")
        f.write(f"Avg Time: {metrics['avg_time']:.2f}s\n")
        f.write(f"Num Samples: {metrics['num_samples']}\n")
        f.write("-" * 40 + "\n")
        f.write("Per-Category Accuracy:\n")
        for cat, acc in sorted(metrics.get("category_accuracy", {}).items()):
            f.write(f"  {cat}: {acc:.2f}%\n")

    console.print(f"\nResults saved to: [bold]{result_dir}[/bold]")
    return result_dir


def print_mcq_metrics(dataset_name: str, metrics: dict):
    """Print MCQ metrics using rich table."""
    table = Table(title=f"Results: {dataset_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Accuracy", f"{metrics['accuracy']:.2f}%")
    table.add_row("Correct", f"{metrics['correct']}/{metrics['total']}")
    table.add_row("Samples", str(metrics["num_samples"]))
    table.add_row("Avg Time", f"{metrics['avg_time']:.2f}s")

    console.print(table)

    # Print per-category breakdown if available
    if "category_accuracy" in metrics and metrics["category_accuracy"]:
        cat_table = Table(title="Per-Category Accuracy")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Accuracy", style="green")
        for cat, acc in sorted(metrics["category_accuracy"].items()):
            cat_table.add_row(cat, f"{acc:.2f}%")
        console.print(cat_table)


def validate_datasets(datasets: list[str]) -> list[str]:
    """Validate and expand dataset names."""
    for ds in datasets:
        if ds not in VALID_DATASETS:
            console.print(f"[red]Error: Invalid dataset '{ds}'[/red]")
            console.print(f"Valid choices: {', '.join(VALID_DATASETS)}")
            raise typer.Exit(1)

    # Expand "all" to ASR datasets only (exclude diarization, alignment, MCQ)
    if "all" in datasets:
        return [
            k
            for k in DATASET_REGISTRY
            if k not in DIARIZATION_DATASETS
            and k not in ALIGNMENT_DATASETS
            and k not in MCQ_DATASETS
        ]
    # Expand "all-full" to include diarization, alignment, and MCQ datasets too
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
    streaming: Annotated[
        bool, typer.Option("--streaming", "-s", help="Use streaming evaluation (for local or AAI)")
    ] = False,
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
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show word-by-word alignment details"),
    ] = False,
    temperature: Annotated[
        Optional[float],
        typer.Option("--temperature", "-t", help="Temperature for AssemblyAI SLAM-1 (e.g., 0.5)"),
    ] = None,
    keyterms: Annotated[
        Optional[str],
        typer.Option("--keyterms", "-k", help="Comma-separated key terms for AssemblyAI SLAM-1"),
    ] = None,
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
                model_id = "slam-1"
                evaluator = AssemblyAIDiarizationEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                    num_workers=num_workers,
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
                    num_workers=num_workers,
                )
            elif model == "elevenlabs":
                api_key = os.environ.get("ELEVENLABS_API_KEY", "")
                if not api_key:
                    console.print(
                        "[red]Error: ELEVENLABS_API_KEY environment variable not set[/red]"
                    )
                    raise typer.Exit(1)
                model_id = "scribe-v2"
                evaluator = ElevenLabsDiarizationEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                    num_workers=num_workers,
                )
            else:
                # Local diarization using TEN-VAD + ECAPA-TDNN + spectral clustering
                model_id = "local" if model == "local" else get_model_name(model)
                evaluator = LocalDiarizationEvaluator(
                    audio_field=cfg.audio_field,
                    speakers_field=cfg.speakers_field,
                    timestamps_start_field=cfg.timestamps_start_field,
                    timestamps_end_field=cfg.timestamps_end_field,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers or 2,
                    max_speakers=max_speakers or 10,
                    num_workers=num_workers,
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
                model_id = "slam-1"
                evaluator = AssemblyAIAlignmentEvaluator(
                    api_key=api_key,
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
            elif model == "elevenlabs":
                api_key = os.environ.get("ELEVENLABS_API_KEY", "")
                if not api_key:
                    console.print(
                        "[red]Error: ELEVENLABS_API_KEY environment variable not set[/red]"
                    )
                    raise typer.Exit(1)
                model_id = "scribe-v2"
                evaluator = ElevenLabsAlignmentEvaluator(
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

        # Handle MCQ datasets (audio understanding benchmarks)
        if dataset_name in MCQ_DATASETS:
            from datasets import load_dataset as hf_load_dataset

            dataset = hf_load_dataset(cfg.path, split=actual_split, streaming=True)

            if model == "assemblyai":
                api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
                if not api_key:
                    console.print(
                        "[red]Error: ASSEMBLYAI_API_KEY environment variable not set[/red]"
                    )
                    raise typer.Exit(1)
                model_id = "slam-1"
                evaluator = AssemblyAIMMAUEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    question_field=cfg.question_field,
                    answer_field=cfg.answer_field,
                    choices_field=cfg.choices_field,
                    category_field=cfg.category_field,
                    num_workers=num_workers,
                )
            else:
                model_id = get_model_name(model)
                evaluator = MMAUEvaluator(
                    model_path=model,
                    audio_field=cfg.audio_field,
                    question_field=cfg.question_field,
                    answer_field=cfg.answer_field,
                    choices_field=cfg.choices_field,
                    category_field=cfg.category_field,
                    user_prompt=user_prompt,
                    num_workers=num_workers,
                )

            results = evaluator.evaluate(dataset, max_samples)
            metrics = evaluator.compute_metrics()
            save_mcq_results(model_id, dataset_name, results, metrics, output_dir)
            print_mcq_metrics(dataset_name, metrics)
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
                model_id = "slam-1"
                if temperature is not None:
                    console.print(f"[cyan]Using temperature: {temperature}[/cyan]")
                if user_prompt is not None:
                    console.print(f"[cyan]Using prompt: {user_prompt}[/cyan]")
                keyterms_list = None
                if keyterms is not None:
                    keyterms_list = [k.strip() for k in keyterms.split(",")]
                    console.print(f"[cyan]Using keyterms: {keyterms_list}[/cyan]")
                evaluator = AssemblyAIEvaluator(
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    prompt=user_prompt,
                    keyterms_prompt=keyterms_list,
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
        elif model == "elevenlabs":
            api_key = os.environ.get("ELEVENLABS_API_KEY", "")
            if not api_key:
                console.print("[red]Error: ELEVENLABS_API_KEY environment variable not set[/red]")
                raise typer.Exit(1)
            model_id = "scribe-v2"
            evaluator = ElevenLabsEvaluator(
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


def save_airbench_results(
    model_name: str,
    task_name: str | None,
    results: list[AIRBenchResult],
    metrics: dict,
    output_dir: str = "outputs",
) -> Path:
    """Save AIR-Bench evaluation results and metrics."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    task_suffix = f"_{task_name}" if task_name else "_speech"
    result_dir = Path(output_dir) / f"{timestamp}_{safe_model_name}_airbench{task_suffix}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = result_dir / "results.txt"
    with results_file.open("w") as f:
        for i, r in enumerate(results, 1):
            status = "✓" if r.correct else "✗"
            f.write(f"Sample {i} [{status}]\n")
            f.write(f"  Task: {r.task_name}_{r.dataset_name}\n")
            f.write(f"  Question: {r.question}\n")
            f.write(f"  Prediction: {r.prediction}\n")
            f.write(f"  Extracted: {r.extracted_answer} | Reference: {r.reference}\n")
            f.write(f"  Time: {r.time:.2f}s\n")
            f.write("-" * 80 + "\n")

    # Save summary metrics
    metrics_file = result_dir / "metrics.txt"
    with metrics_file.open("w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Task Filter: {task_name or 'speech (all)'}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Correct: {metrics['correct']}/{metrics['total']}\n")
        f.write(f"Avg Time: {metrics['avg_time']:.2f}s\n")
        f.write(f"Num Samples: {metrics['num_samples']}\n")

        if metrics.get("category_accuracy"):
            f.write("-" * 40 + "\n")
            f.write("Per-Category Accuracy:\n")
            for cat, acc in sorted(metrics["category_accuracy"].items()):
                f.write(f"  {cat}: {acc:.2f}%\n")

        if metrics.get("task_accuracy"):
            f.write("-" * 40 + "\n")
            f.write("Per-Task Accuracy:\n")
            for task_id, task_metrics in sorted(metrics["task_accuracy"].items()):
                f.write(
                    f"  {task_id}: {task_metrics['accuracy']:.2f}% "
                    f"({task_metrics['correct']}/{task_metrics['total']})\n"
                )

    console.print(f"\nResults saved to: [bold]{result_dir}[/bold]")
    return result_dir


@app.command()
def airbench(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="HuggingFace model path/ID"),
    ],
    task_name: Annotated[
        Optional[str],
        typer.Option(
            "--task",
            "-t",
            help="Filter to specific task (e.g., 'Speaker_Gender_Recognition'). If not specified, runs all speech tasks.",
        ),
    ] = None,
    max_samples: Annotated[
        Optional[int],
        typer.Option(
            "--max-samples",
            "-n",
            help="Max samples (per task if running all tasks, total if single task)",
        ),
    ] = None,
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Output directory for results"),
    ] = "outputs",
    list_tasks: Annotated[
        bool,
        typer.Option("--list-tasks", "-l", help="List available tasks and exit"),
    ] = False,
):
    """Evaluate models on AIR-Bench Foundation benchmark.

    AIR-Bench is a benchmark for Large Audio-Language Models with 19 tasks
    covering speech, sound, and music understanding.

    By default, runs only speech tasks. Use --task to run a specific task.
    When running multiple tasks (no -t flag), -n limits samples PER TASK.
    When running a single task, -n limits total samples.

    Examples:
        # Run all speech tasks
        ta eval airbench -m mazesmazes/tiny-audio-omni

        # Run 50 samples per task (450 total for 9 speech tasks)
        ta eval airbench -m mazesmazes/tiny-audio-omni -n 50

        # Run specific task (100 total samples)
        ta eval airbench -m mazesmazes/tiny-audio-omni -t Speaker_Gender_Recognition -n 100

        # List available tasks
        ta eval airbench -m dummy --list-tasks
    """
    if list_tasks:
        console.print("\n[bold]Available AIR-Bench Foundation Tasks:[/bold]\n")

        console.print("[cyan]Speech Tasks:[/cyan]")
        from scripts.eval.evaluators.airbench import SPEECH_TASKS

        for task in sorted(SPEECH_TASKS):
            console.print(f"  • {task}")

        console.print("\n[cyan]Sound Tasks:[/cyan]")
        from scripts.eval.evaluators.airbench import SOUND_TASKS

        for task in sorted(SOUND_TASKS):
            console.print(f"  • {task}")

        console.print("\n[cyan]Music Tasks:[/cyan]")
        from scripts.eval.evaluators.airbench import MUSIC_TASKS

        for task in sorted(MUSIC_TASKS):
            console.print(f"  • {task}")

        return

    # Validate task_name if provided
    if task_name and task_name not in AIRBENCH_ALL_TASKS:
        console.print(f"[red]Error: Invalid task '{task_name}'[/red]")
        console.print(f"Valid tasks: {', '.join(sorted(AIRBENCH_ALL_TASKS))}")
        raise typer.Exit(1)

    console.print("\n[bold blue]AIR-Bench Foundation Evaluation[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Task: {task_name or 'all speech tasks'}")
    if max_samples:
        console.print(f"Max samples: {max_samples}")

    model_id = get_model_name(model)
    evaluator = AIRBenchEvaluator(
        model_path=model,
        task_name=task_name,
    )

    results = evaluator.evaluate(max_samples=max_samples)
    metrics = evaluator.compute_metrics()

    save_airbench_results(model_id, task_name, results, metrics, output_dir)
    print_airbench_metrics(metrics)


if __name__ == "__main__":
    app()
