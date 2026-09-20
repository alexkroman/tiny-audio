"""CLI for ASR evaluation."""

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

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
    Evaluator,
    LocalEvaluator,
    LocalStreamingEvaluator,
    SwiftSDKEvaluator,
)

app = typer.Typer(add_completion=False)
console = Console()


# `--datasets` choices, built from the registry so Click validates them and the
# help text lists them. "all" expands to every ASR dataset (expresso is
# TTS-style, opt in by name).
Dataset = StrEnum("Dataset", {name: name for name in ("all", *DATASET_REGISTRY)})
ALL_DATASETS = [name for name in DATASET_REGISTRY if name != "expresso"]


def get_model_name(model_path: str) -> str:
    """Extract model name from a HuggingFace model path.

    Examples:
        - mazesmazes/tiny-audio -> tiny-audio
        - /path/to/checkpoint -> checkpoint
    """
    return model_path.rstrip("/").split("/")[-1]


def _require_api_key(api_key: str | None, option: str, env_var: str) -> str:
    """Return a provider API key or fail like any other missing option."""
    if not api_key:
        raise typer.BadParameter(f"set {env_var} or pass {option}", param_hint=option)
    return api_key


def _one_line(text: str) -> str:
    """Collapse whitespace so a raw transcript stays on a single results.txt line."""
    return " ".join(text.split())


_DIR_SEGMENT_RE = re.compile(r"[^A-Za-z0-9.-]+")


def _dir_segment(value: str) -> str:
    """Sanitize one `_`-delimited field of a run directory name.

    Run directories are `{date}_{time}_{model}[_{endpoint}]_{dataset}`, and
    every consumer splits them on `_` (`scripts.utils._extract_model_from_dir`
    takes field 2, `scripts.analysis.extract_dataset_name` takes the last).
    An underscore inside a field silently shifts all the others: with
    `--model-name granite_qwen` the model parses as "granite", so
    `ta analysis compare granite_qwen` can never find the run it just wrote.
    """
    return _DIR_SEGMENT_RE.sub("-", value).strip("-") or "unknown"


def save_results(
    model_name: str,
    dataset_name: str,
    results: list[EvalResult],
    metrics: dict,
    output_dir: str = "outputs",
    base_url: str | None = None,
) -> Path:
    """Save evaluation results and metrics to a timestamped directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_model_name = _dir_segment(model_name)

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
                url_suffix = f"_{_dir_segment(part)}"
                break
        if not url_suffix and host:
            # Fallback: use first part of hostname
            url_suffix = f"_{_dir_segment(parts[0])}"

    result_dir = (
        Path(output_dir) / f"{timestamp}_{safe_model_name}{url_suffix}_{_dir_segment(dataset_name)}"
    )
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


def expand_datasets(datasets: list[str]) -> list[str]:
    """Expand the "all" choice; Click has already validated every name."""
    if "all" in datasets:
        return ALL_DATASETS
    return datasets


def _build_evaluator(
    *,
    model: str,
    endpoint: bool,
    streaming: bool,
    assemblyai_model: AssemblyAIModel,
    assemblyai_api_key: str | None,
    deepgram_api_key: str | None,
    elevenlabs_api_key: str | None,
    base_url: str | None,
    locale: str,
    num_workers: int,
    user_prompt: str | None,
) -> tuple[str, Evaluator]:
    """Construct the evaluator once for a whole sweep. Returns (model_id, evaluator).

    Takes no dataset argument, deliberately. This used to run inside the
    per-dataset loop because `audio_field` / `text_field` are constructor
    state and vary by corpus (`wav` for Loquacious, `sentence` for
    Earnings22, `transcript` for SPGISpeech) -- which welded a few bytes of
    per-dataset config to a very expensive per-model setup. `ta eval -d all`
    therefore paid that setup 12 times: 12 full `from_pretrained` loads for a
    local model, 12 `swift build` + subprocess spawns + warmups for
    `swift://`, 12 SFSpeechRecognizer authorizations for `apple-speech`.

    The column names now ride on `Evaluator.evaluate(...)` instead, so one
    instance serves every dataset. Anything an evaluator accumulates across
    `transcribe` calls must be cleared in `_reset_run_state`.
    """
    if model == "assemblyai":
        api_key = _require_api_key(assemblyai_api_key, "--assemblyai-api-key", "ASSEMBLYAI_API_KEY")

        if streaming:
            model_id = "universal-streaming"
            evaluator = AssemblyAIStreamingEvaluator(
                api_key=api_key,
                num_workers=num_workers,
            )
        else:
            model_id = assemblyai_model.value
            evaluator = AssemblyAIEvaluator(
                api_key=api_key,
                model=assemblyai_model.value,
                base_url=base_url,
                num_workers=num_workers,
            )
    elif model == "deepgram":
        api_key = _require_api_key(deepgram_api_key, "--deepgram-api-key", "DEEPGRAM_API_KEY")
        model_id = "nova-3"
        evaluator = DeepgramEvaluator(
            api_key=api_key,
            num_workers=num_workers,
        )
    elif model == "elevenlabs":
        api_key = _require_api_key(elevenlabs_api_key, "--elevenlabs-api-key", "ELEVENLABS_API_KEY")
        model_id = "scribe-v2"
        evaluator = ElevenLabsEvaluator(
            api_key=api_key,
            num_workers=num_workers,
        )
    elif model == "apple-speech":
        model_id = "apple-speech"
        evaluator = AppleSpeechEvaluator(
            locale=locale,
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
            )
        elif suffix == "":
            model_id = "swift-default-bundle"
            evaluator = SwiftSDKEvaluator()
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
        )
    elif streaming:
        model_id = get_model_name(model)
        evaluator = LocalStreamingEvaluator(
            model_path=model,
            user_prompt=user_prompt,
        )
    else:
        model_id = get_model_name(model)
        evaluator = LocalEvaluator(
            model_path=model,
            user_prompt=user_prompt,
        )

    return model_id, evaluator


@app.command()
def main(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model path/ID, 'assemblyai', 'deepgram', 'elevenlabs', or 'apple-speech'",
        ),
    ],
    datasets: Annotated[
        list[Dataset],
        typer.Option(
            "--datasets",
            "-d",
            help="Datasets to evaluate on ('all' for every ASR dataset)",
        ),
    ] = (Dataset.loquacious,),
    split: Annotated[
        str | None, typer.Option("--split", help="Dataset split (default: the dataset's own)")
    ] = None,
    max_samples: Annotated[
        int | None,
        typer.Option("--max-samples", "-n", help="Maximum samples to evaluate per dataset"),
    ] = None,
    endpoint: Annotated[
        bool, typer.Option("--endpoint", help="Treat --model as an HF Inference Endpoint URL")
    ] = False,
    assemblyai_model: Annotated[
        AssemblyAIModel, typer.Option("--assemblyai-model", help="AssemblyAI model")
    ] = AssemblyAIModel.universal_3_pro,
    streaming: Annotated[
        bool,
        typer.Option("--streaming", "-s", help="Use streaming evaluation (local or AssemblyAI)"),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Dataset config override (e.g., 'en' for CommonVoice)"),
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory to write eval results to")
    ] = Path("outputs"),
    user_prompt: Annotated[
        str | None, typer.Option("--user-prompt", help="Custom user prompt for the model")
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Custom API base URL (for AssemblyAI sandbox)"),
    ] = None,
    assemblyai_api_key: Annotated[
        str | None,
        typer.Option(
            "--assemblyai-api-key", envvar="ASSEMBLYAI_API_KEY", help="AssemblyAI API key"
        ),
    ] = None,
    deepgram_api_key: Annotated[
        str | None,
        typer.Option("--deepgram-api-key", envvar="DEEPGRAM_API_KEY", help="Deepgram API key"),
    ] = None,
    elevenlabs_api_key: Annotated[
        str | None,
        typer.Option(
            "--elevenlabs-api-key", envvar="ELEVENLABS_API_KEY", help="ElevenLabs API key"
        ),
    ] = None,
    locale: Annotated[
        str,
        typer.Option("--locale", help="Locale for apple-speech (e.g. en-US, es-ES, fr-FR)"),
    ] = "en-US",
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", "-w", help="Number of parallel workers for API evaluations"),
    ] = 1,
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model-name",
            help="Override the auto-derived model label used in output dir names "
            "and downstream `ta analysis` matching (otherwise derived from --model).",
        ),
    ] = None,
):
    """Evaluate ASR models on standard datasets."""
    # Built once, before the loop: model load / Swift build / API client setup
    # is per-model, not per-dataset. See _build_evaluator.
    model_id, evaluator = _build_evaluator(
        model=model,
        endpoint=endpoint,
        streaming=streaming,
        assemblyai_model=assemblyai_model,
        assemblyai_api_key=assemblyai_api_key,
        deepgram_api_key=deepgram_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
        base_url=base_url,
        locale=locale,
        num_workers=num_workers,
        user_prompt=user_prompt,
    )

    for dataset_name in expand_datasets([d.value for d in datasets]):
        console.print(f"\n[bold blue]Evaluating on: {dataset_name}[/bold blue]")

        cfg = DATASET_REGISTRY[dataset_name]
        dataset = load_eval_dataset(dataset_name, split or cfg.default_split, config)

        results = evaluator.evaluate(
            dataset,
            max_samples,
            audio_field=cfg.audio_field,
            text_field=cfg.text_field,
        )
        metrics = evaluator.compute_metrics()
        save_results(
            model_name or model_id, dataset_name, results, metrics, str(output_dir), base_url
        )
        print_asr_metrics(dataset_name, metrics)


if __name__ == "__main__":
    app()
