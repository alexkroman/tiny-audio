#!/usr/bin/env python3
"""
Local runner script to test the HuggingFace inference endpoint handler.
This simulates the inference endpoint environment locally for debugging.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Try importing required modules
try:
    from tiny_audio.handler import EndpointHandler

    print("Successfully imported EndpointHandler from tiny_audio.handler")
except ImportError as e:
    print(f"Failed to import handler: {e}")
    print("   Make sure tiny_audio package is installed")
    sys.exit(1)


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


def test_handler(
    model_path: str,
    audio_path: str | None = None,
    max_new_tokens: int = 200,
    num_beams: int = 1,
    temperature: float = 1.0,
    do_sample: bool = False,
    batch_test: bool = False,
):
    """Test the EndpointHandler with various configurations."""

    print("=" * 80)
    print("HuggingFace Inference Endpoint Handler - Local Test Runner")
    print("=" * 80)

    # Step 1: Initialize the handler
    print(f"\nLoading model from: {model_path}")
    print("   This may take a moment on first load...")

    start_time = time.time()
    try:
        handler = EndpointHandler(path=model_path)
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Step 2: Find or use provided audio file
    if audio_path is None:
        print("\nNo audio file specified, searching for test audio...")
        audio_path = find_test_audio()
        if audio_path:
            print(f"   Found test audio: {audio_path}")
        else:
            print("No test audio found. Please provide an audio file path.")
            print("   Usage: python run_handler_local.py --audio path/to/audio.wav")
            return

    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return

    print(f"\nUsing audio file: {audio_path}")

    # Step 3: Prepare the request data
    print("\nPreparing inference request...")

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

        print(f"   Parameters: max_new_tokens={max_new_tokens}, num_beams={num_beams}")
        print(f"              temperature={temperature}, do_sample={do_sample}")

        # Step 4: Run inference
        print("\nRunning transcription...")
        start_time = time.time()
        try:
            result = handler(data)
            inference_time = time.time() - start_time
            print(f"Inference completed in {inference_time:.2f} seconds")

            # Step 5: Display results
            print("\nTranscription Result:")
            print("-" * 40)
            if "text" in result:
                print(result["text"])
            else:
                print(json.dumps(result, indent=2))
            print("-" * 40)

        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback

            traceback.print_exc()

    # Batch test (if requested)
    if batch_test:
        print("\nTesting batch processing...")

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

        print(f"   Batch size: {batch_size}")

        start_time = time.time()
        try:
            result = handler(data)
            inference_time = time.time() - start_time
            print(f"Batch inference completed in {inference_time:.2f} seconds")
            print(f"   Average time per sample: {inference_time / batch_size:.2f} seconds")

            print("\nBatch Results:")
            print("-" * 40)
            if "texts" in result:
                for i, text in enumerate(result["texts"], 1):
                    print(f"Sample {i}: {text}")
            else:
                print(json.dumps(result, indent=2))
            print("-" * 40)

        except Exception as e:
            print(f"Batch inference failed: {e}")
            import traceback

            traceback.print_exc()

    print("\nTest completed!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test HuggingFace inference endpoint handler locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use the latest saved model
  python scripts/deploy/handler_local.py

  # Use a specific model
  python scripts/deploy/handler_local.py --model outputs/2025-09-24/10-39-34/outputs/mac_model

  # Use a HuggingFace Hub model
  python scripts/deploy/handler_local.py --model mazesmazes/tiny-audio

  # Specify custom audio file
  python scripts/deploy/handler_local.py --audio path/to/audio.wav

  # Test with beam search
  python scripts/deploy/handler_local.py --num-beams 5 --max-new-tokens 256

  # Test batch processing
  python scripts/deploy/handler_local.py --batch-test
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Path to model directory or HuggingFace model ID (default: latest saved model)",
    )

    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        default=None,
        help="Path to audio file for transcription (default: auto-detect test audio)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 200)",
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search (default: 1 for greedy)",
    )

    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling (default: 1.0)"
    )

    parser.add_argument(
        "--do-sample", action="store_true", help="Use sampling instead of greedy/beam search"
    )

    parser.add_argument(
        "--batch-test", action="store_true", help="Test batch processing with multiple audio files"
    )

    args = parser.parse_args()

    # Determine model path
    model_path = args.model
    if model_path is None:
        print("No model specified, using default HuggingFace Hub model...")
        model_path = "mazesmazes/tiny-audio"
        print(f"   Using model: {model_path}")

    # Run the test
    test_handler(
        model_path=model_path,
        audio_path=args.audio,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
        do_sample=args.do_sample,
        batch_test=args.batch_test,
    )


if __name__ == "__main__":
    main()
