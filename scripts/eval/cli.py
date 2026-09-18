"""CLI for ASR evaluation."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from scripts.eval.datasets import (
    DATASET_REGISTRY,
    load_eval_dataset,
)
from scripts.eval.evaluators import (
    AppleSpeechEvaluator,
    AssemblyAIEvaluator,
    AssemblyAIModel,
    AssemblyAIStreamingEvaluator,
    DeepgramEvaluator,
    ElevenLabsEvaluator,
    EndpointEvaluator,
    EvalResult,
    LocalEvaluator,
    LocalStreamingEvaluator,
    SwiftSDKEvaluator,
)

app = typer.Typer(help="Evaluate ASR models on standard datasets")
console = Console()


# Valid dataset choices
VALID_DATASETS = ["all"] + list(DATASET_REGISTRY.keys())


def get_model_name(model_path: str) -> str:
    """Extract model name from a HuggingFace model path.

    Examples:
        - mazesmazes/tiny-audio -> tiny-audio
        - /path/to/checkpoint -> checkpoint
    """
    return model_path.rstrip("/").split("/")[-1]


def _require_api_key(env_var: str) -> str:
    """Read an API key from the environment or exit with an error."""
    api_key = os.environ.get(env_var, "")
    if not api_key:
        console.print(f"[red]Error: {env_var} environment variable not set[/red]")
        raise typer.Exit(1)
    return api_key


def _one_line(text: str) -> str:
    """Collapse whitespace so a raw transcript stays on a single results.txt line."""
    return " ".join(text.split())


def save_results(
    model_name: str,
    dataset_name: str,
    results: list[EvalResult],
    metrics: dict,
    output_dir: str = "outputs",
    base_url: str | None = None,
) -> Path:
    """Save evaluation results and metrics to a timestamped directory."""
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
            norm_pred = r.norm_prediction
            norm_ref = r.norm_reference
            f.write(f"Sample {i} - WER: {r.wer:.2f}%\n")
            f.write(f"Ground Truth: {norm_ref}\n")
            f.write(f"Prediction: {norm_pred}\n")
            # Raw (un-normalized) pair, written after the normalized one so the
            # existing "Ground Truth: "/"Prediction: " parsers keep matching the
            # WER-bearing text. Formatting metrics (ITN, casing, punctuation)
            # need these: EnglishTextNormalizer both performs ITN
            # ("twenty five dollars" -> "$25") and destroys it
            # ("3:00 p.m." -> "3 0 p m"), so the normalized pair carries no
            # formatting signal. Newlines are flattened to keep one line each.
            f.write(f"Ground Truth Raw: {_one_line(r.reference)}\n")
            f.write(f"Prediction Raw: {_one_line(r.prediction)}\n")
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


def validate_datasets(datasets: list[str]) -> list[str]:
    """Validate and expand dataset names."""
    for ds in datasets:
        if ds not in VALID_DATASETS:
            console.print(f"[red]Error: Invalid dataset '{ds}'[/red]")
            console.print(f"Valid choices: {', '.join(VALID_DATASETS)}")
            raise typer.Exit(1)

    # Expand "all" to the ASR datasets (expresso is TTS-style, opt in by name)
    if "all" in datasets:
        return [k for k in DATASET_REGISTRY if k != "expresso"]

    return datasets


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model path/ID, 'assemblyai', 'deepgram', 'elevenlabs', or 'apple-speech'",
        ),
    ] = None,
    datasets: Annotated[
        Optional[list[str]],
        typer.Option(
            "--datasets",
            "-d",
            help="Datasets to evaluate on ('all' for every ASR dataset)",
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
    ] = AssemblyAIModel.universal_3_pro,
    streaming: Annotated[
        bool, typer.Option("--streaming", "-s", help="Use streaming evaluation (for local or AAI)")
    ] = False,
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
    locale: Annotated[
        str,
        typer.Option(
            "--locale",
            help="Locale for apple-speech (e.g. en-US, es-ES, fr-FR)",
        ),
    ] = "en-US",
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", "-w", help="Number of parallel workers for API evaluations"),
    ] = 1,
    model_name: Annotated[
        Optional[str],
        typer.Option(
            "--model-name",
            help="Override the auto-derived model label used in output dir names "
            "and downstream `ta analysis` matching (otherwise derived from --model).",
        ),
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

        # ASR evaluation
        dataset = load_eval_dataset(dataset_name, actual_split, config)

        if model == "assemblyai":
            api_key = _require_api_key("ASSEMBLYAI_API_KEY")

            if streaming:
                model_id = "universal-streaming"
                evaluator = AssemblyAIStreamingEvaluator(
                    api_key=api_key,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    num_workers=num_workers,
                )
            else:
                model_id = assemblyai_model.value
                evaluator = AssemblyAIEvaluator(
                    api_key=api_key,
                    model=assemblyai_model.value,
                    base_url=base_url,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                    num_workers=num_workers,
                )
        elif model == "deepgram":
            api_key = _require_api_key("DEEPGRAM_API_KEY")
            model_id = "nova-3"
            evaluator = DeepgramEvaluator(
                api_key=api_key,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
                num_workers=num_workers,
            )
        elif model == "elevenlabs":
            api_key = _require_api_key("ELEVENLABS_API_KEY")
            model_id = "scribe-v2"
            evaluator = ElevenLabsEvaluator(
                api_key=api_key,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
                num_workers=num_workers,
            )
        elif model == "apple-speech":
            model_id = "apple-speech"
            evaluator = AppleSpeechEvaluator(
                locale=locale,
                audio_field=cfg.audio_field,
                text_field=cfg.text_field,
            )
        elif model == "swift" or model.startswith("swift://"):
            suffix = model[len("swift://") :] if model.startswith("swift://") else ""
            # Path-like suffix → load that local bundle (TINY_AUDIO_LOCAL_MODEL_DIR
            # path: the Swift binary actually honors this).
            # Empty suffix → load the Swift SDK's pinned default bundle.
            # Anything else (a repo id) is rejected — the Swift binary ignores
            # `repo_id`, which previously produced output dirs labeled with a
            # model that was never actually evaluated.
            if suffix.startswith(("/", "~", "./", "../")):
                model_dir = Path(suffix).expanduser().resolve()
                if not model_dir.is_dir():
                    raise typer.BadParameter(
                        f"swift:// path does not resolve to a directory: {model_dir}"
                    )
                model_id = f"swift-local-{model_dir.name}"
                evaluator = SwiftSDKEvaluator(
                    model_dir=model_dir,
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                )
            elif suffix == "":
                model_id = "swift-default-bundle"
                evaluator = SwiftSDKEvaluator(
                    audio_field=cfg.audio_field,
                    text_field=cfg.text_field,
                )
            else:
                raise typer.BadParameter(
                    f"swift://{suffix!r} is not supported — the Swift binary loads "
                    "the SDK-pinned bundle and ignores arbitrary repo ids, so this "
                    "form would silently evaluate the default bundle while labeling "
                    "outputs with your repo id (see asr_pipeline.py docstring).\n\n"
                    "To evaluate that model via the Swift SDK, build a local bundle "
                    "from it first:\n"
                    "  cd ~/Code/ios/tiny-audio-swift\n"
                    f"  poetry run python -m scripts.bundle.cli build-bundle --projector {suffix}\n"
                    "  cd -\n"
                    "  ta eval -m swift://~/Code/ios/tiny-audio-swift/swift/Sources/TinyAudio/Resources/Model -d ...\n\n"
                    "Or evaluate the HF checkpoint directly (PyTorch path, no Swift):\n"
                    f"  ta eval -m {suffix} -d ..."
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
        save_results(model_name or model_id, dataset_name, results, metrics, output_dir, base_url)
        print_asr_metrics(dataset_name, metrics)


if __name__ == "__main__":
    app()
