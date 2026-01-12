#!/usr/bin/env python3
"""
Local runner script to test the HuggingFace inference endpoint handler.
This simulates the inference endpoint environment locally for debugging.
"""

import json
import sys
import time
from pathlib import Path

import typer

# Try importing required modules
try:
    from tiny_audio.handler import EndpointHandler
except ImportError as e:
    print(f"Failed to import handler: {e}")
    print("   Make sure tiny_audio package is installed")
    sys.exit(1)

app = typer.Typer(help="Test HuggingFace inference endpoint handler locally")


def find_latest_model(base_dir: str = "outputs") -> str | None:
    """Find the most recent saved model in outputs directory."""
    outputs_path = Path(base_dir)
    if not outputs_path.exists():
        return None

    # Find all model.safetensors files
    model_files = list(outputs_path.glob("**/model.safetensors"))

    if not model_files:
        return None

    # Sort by modification time and get the most recent
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Return the parent directory of the model file
    return str(model_files[0].parent)


def find_test_audio() -> str | None:
    """Find a test audio file in the project."""
    # Look for test audio files in common locations
    test_paths = [
        ".venv/lib/python3.11/site-packages/gradio/test_data/test_audio.wav",
        ".venv/lib/python3.11/site-packages/gradio/media_assets/audio/cantina.wav",
        "demo/sample.wav",
        "tests/test_audio.wav",
    ]

    base_dir = Path(__file__).parent.parent.parent

    for test_path in test_paths:
        full_path = base_dir / test_path
        if full_path.exists():
            return str(full_path)

    # Try to find any audio file
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        audio_files = list(base_dir.glob(f"**/{ext}"))
        if audio_files:
            return str(audio_files[0])

    return None


@app.command()
def test(
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to model directory or HuggingFace model ID (default: mazesmazes/tiny-audio)",
    ),
    audio: Path | None = typer.Option(
        None,
        "--audio",
        "-a",
        help="Path to audio file for transcription (default: auto-detect test audio)",
    ),
    max_new_tokens: int = typer.Option(
        200,
        "--max-new-tokens",
        help="Maximum number of tokens to generate",
    ),
    num_beams: int = typer.Option(
        1,
        "--num-beams",
        help="Number of beams for beam search (1 for greedy)",
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        help="Temperature for sampling",
    ),
    do_sample: bool = typer.Option(
        False,
        "--do-sample",
        help="Use sampling instead of greedy/beam search",
    ),
    batch_test: bool = typer.Option(
        False,
        "--batch-test",
        help="Test batch processing with multiple audio files",
    ),
):
    """Test the EndpointHandler with various configurations."""
    # Determine model path
    model_path = model
    if model_path is None:
        typer.echo("No model specified, using default HuggingFace Hub model...")
        model_path = "mazesmazes/tiny-audio"
        typer.echo(f"   Using model: {model_path}")

    typer.echo("=" * 80)
    typer.echo("HuggingFace Inference Endpoint Handler - Local Test Runner")
    typer.echo("=" * 80)

    # Step 1: Initialize the handler
    typer.echo(f"\nLoading model from: {model_path}")
    typer.echo("   This may take a moment on first load...")

    start_time = time.time()
    try:
        handler = EndpointHandler(path=model_path)
        load_time = time.time() - start_time
        typer.echo(f"Model loaded successfully in {load_time:.2f} seconds")
    except Exception as e:
        typer.echo(f"Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from None

    # Step 2: Find or use provided audio file
    audio_path = str(audio) if audio else None
    if audio_path is None:
        typer.echo("\nNo audio file specified, searching for test audio...")
        audio_path = find_test_audio()
        if audio_path:
            typer.echo(f"   Found test audio: {audio_path}")
        else:
            typer.echo("No test audio found. Please provide an audio file path.")
            typer.echo("   Usage: run-handler --audio path/to/audio.wav")
            raise typer.Exit(1)

    if not Path(audio_path).exists():
        typer.echo(f"Audio file not found: {audio_path}")
        raise typer.Exit(1)

    typer.echo(f"\nUsing audio file: {audio_path}")

    # Step 3: Prepare the request data
    typer.echo("\nPreparing inference request...")

    # Basic single file test
    if not batch_test:
        params = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
        }
        # Only add temperature when sampling is enabled
        if do_sample:
            params["temperature"] = temperature

        data = {
            "inputs": audio_path,
            "parameters": params,
        }

        typer.echo(f"   Parameters: max_new_tokens={max_new_tokens}, num_beams={num_beams}")
        typer.echo(f"              temperature={temperature}, do_sample={do_sample}")

        # Step 4: Run inference
        typer.echo("\nRunning transcription...")
        start_time = time.time()
        try:
            result = handler(data)
            inference_time = time.time() - start_time
            typer.echo(f"Inference completed in {inference_time:.2f} seconds")

            # Step 5: Display results
            typer.echo("\nTranscription Result:")
            typer.echo("-" * 40)
            if "text" in result:
                typer.echo(result["text"])
            else:
                typer.echo(json.dumps(result, indent=2))
            typer.echo("-" * 40)

        except Exception as e:
            typer.echo(f"Inference failed: {e}")
            import traceback

            traceback.print_exc()

    # Batch test (if requested)
    if batch_test:
        typer.echo("\nTesting batch processing...")

        # Create a batch of the same audio file
        batch_size = 3
        params = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "batch_size": batch_size,
            "do_sample": do_sample,
        }
        # Only add temperature when sampling is enabled
        if do_sample:
            params["temperature"] = temperature

        data = {
            "inputs": [audio_path] * batch_size,
            "parameters": params,
        }

        typer.echo(f"   Batch size: {batch_size}")

        start_time = time.time()
        try:
            result = handler(data)
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

        except Exception as e:
            typer.echo(f"Batch inference failed: {e}")
            import traceback

            traceback.print_exc()

    typer.echo("\nTest completed!")
    typer.echo("=" * 80)


def main():
    """Entry point for pyproject.toml scripts."""
    app()


if __name__ == "__main__":
    app()
