#!/usr/bin/env python3
"""Local runner to test the HuggingFace inference endpoint handler."""

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from scripts.utils import get_project_root

app = typer.Typer(help="Test HuggingFace inference endpoint handler locally")


def find_latest_model(base_dir: str = "outputs") -> str | None:
    """Find the most recent saved model in outputs directory."""
    outputs_path = Path(base_dir)
    if not outputs_path.exists():
        return None
    model_files = sorted(
        outputs_path.glob("**/model.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(model_files[0].parent) if model_files else None


def find_test_audio() -> str | None:
    """Find a test audio file in the project."""
    base_dir = get_project_root()

    # Gradio ships sample audio; glob the python version out of the path so
    # this keeps working across interpreter upgrades (was hardcoded to 3.11).
    for pattern in (
        ".venv/lib/python3.*/site-packages/gradio/test_data/test_audio.wav",
        ".venv/lib/python3.*/site-packages/gradio/media_assets/audio/cantina.wav",
    ):
        match = next(iter(sorted(base_dir.glob(pattern))), None)
        if match:
            return str(match)

    for test_path in ("demo/sample.wav", "tests/test_audio.wav"):
        full_path = base_dir / test_path
        if full_path.exists():
            return str(full_path)

    for ext in ("*.wav", "*.mp3", "*.flac"):
        audio_files = list(base_dir.glob(f"**/{ext}"))
        if audio_files:
            return str(audio_files[0])

    return None


@app.command()
def test(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model path/ID to load"),
    ] = "mazesmazes/tiny-audio",
    audio: Annotated[
        Path | None,
        typer.Option(
            "--audio",
            "-a",
            exists=True,
            dir_okay=False,
            help="Audio file to transcribe (default: auto-detect a test clip)",
        ),
    ] = None,
    max_new_tokens: Annotated[
        int, typer.Option("--max-new-tokens", help="Maximum number of tokens to generate")
    ] = 200,
    num_beams: Annotated[
        int, typer.Option("--num-beams", help="Number of beams for beam search (1 for greedy)")
    ] = 1,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Temperature for sampling")
    ] = 1.0,
    do_sample: Annotated[
        bool, typer.Option("--do-sample", help="Use sampling instead of greedy/beam search")
    ] = False,
    batch_test: Annotated[
        bool, typer.Option("--batch-test", help="Test batch processing with multiple audio files")
    ] = False,
):
    """Test the inference endpoint handler locally."""
    # Imported here, not at module scope: it pulls in transformers + torch
    # (~2.8s), which every other `ta dev` command would otherwise pay because
    # scripts/dev.py imports this module to register the command.
    try:
        from tiny_audio.handler import EndpointHandler
    except ImportError as e:
        typer.echo(f"Failed to import handler: {e}", err=True)
        typer.echo("   Make sure tiny_audio package is installed", err=True)
        raise typer.Exit(1) from None

    model_path = model

    typer.echo("=" * 80)
    typer.echo("HuggingFace Inference Endpoint Handler - Local Test Runner")
    typer.echo("=" * 80)

    typer.echo(f"\nLoading model from: {model_path}")
    typer.echo("   This may take a moment on first load...")

    start_time = time.time()
    try:
        handler = EndpointHandler(path=model_path)
    except Exception as e:
        typer.echo(f"Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from None
    typer.echo(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")

    audio_path = str(audio) if audio else None
    if audio_path is None:
        typer.echo("\nNo audio file specified, searching for test audio...")
        audio_path = find_test_audio()
        if audio_path:
            typer.echo(f"   Found test audio: {audio_path}")
        else:
            raise typer.BadParameter(
                "no test audio found; pass one explicitly", param_hint="--audio"
            )

    typer.echo(f"\nUsing audio file: {audio_path}")
    typer.echo("\nPreparing inference request...")

    params: dict = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
    }
    if do_sample:
        params["temperature"] = temperature

    if not batch_test:
        data = {"inputs": audio_path, "parameters": params}
        typer.echo(f"   Parameters: max_new_tokens={max_new_tokens}, num_beams={num_beams}")
        typer.echo(f"              temperature={temperature}, do_sample={do_sample}")

        typer.echo("\nRunning transcription...")
        start_time = time.time()
        try:
            result = handler(data)
        except Exception as e:
            typer.echo(f"Inference failed: {e}")
            import traceback

            traceback.print_exc()
        else:
            typer.echo(f"Inference completed in {time.time() - start_time:.2f} seconds")
            typer.echo("\nTranscription Result:")
            typer.echo("-" * 40)
            typer.echo(result.get("text", json.dumps(result, indent=2)))
            typer.echo("-" * 40)

    if batch_test:
        typer.echo("\nTesting batch processing...")
        batch_size = 3
        batch_params = {**params, "batch_size": batch_size}
        data = {"inputs": [audio_path] * batch_size, "parameters": batch_params}
        typer.echo(f"   Batch size: {batch_size}")

        start_time = time.time()
        try:
            result = handler(data)
        except Exception as e:
            typer.echo(f"Batch inference failed: {e}")
            import traceback

            traceback.print_exc()
        else:
            inference_time = time.time() - start_time
            typer.echo(f"Batch inference completed in {inference_time:.2f} seconds")
            typer.echo(f"   Average time per sample: {inference_time / batch_size:.2f} seconds")
            typer.echo("\nBatch Results:")
            typer.echo("-" * 40)
            if "texts" in result:
                for i, text in enumerate(result["texts"], 1):
                    typer.echo(f"Sample {i}: {text}")
            else:
                typer.echo(json.dumps(result, indent=2))
            typer.echo("-" * 40)

    typer.echo("\nTest completed!")
    typer.echo("=" * 80)


if __name__ == "__main__":
    app()
