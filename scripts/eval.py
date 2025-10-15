#!/usr/bin/env python3
"""
Evaluate ASR models on the LoquaciousSet dataset using HuggingFace Inference API.

Supports:
- HuggingFace Hub models (e.g., mazesmazes/tiny-audio)
- HuggingFace Inference Endpoints (via endpoint URL)
- AssemblyAI API for comparison

Uses HuggingFace's evaluate library to compute WER and log predictions.
"""

import argparse
import io
import os
import tempfile
import time
from pathlib import Path

import evaluate
import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import InferenceClient


def audio_to_wav_bytes(audio_array, sample_rate):
    """Convert audio array to WAV bytes in memory."""
    if not isinstance(audio_array, torch.Tensor):
        audio_tensor = torch.from_numpy(audio_array)
    else:
        audio_tensor = audio_array

    # Ensure 2D shape (channels, samples)
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Write to BytesIO buffer
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
    return buffer.getvalue()


def wav_bytes_to_audio(wav_bytes):
    """Convert WAV bytes to audio array."""
    audio_tensor, sample_rate = torchaudio.load(io.BytesIO(wav_bytes), format="wav")
    return audio_tensor.squeeze().numpy(), sample_rate


def prepare_wav_bytes(wav_data):
    """Convert various WAV data formats to bytes for API calls."""
    if isinstance(wav_data, dict) and "bytes" in wav_data:
        # Already in bytes format
        return wav_data["bytes"]
    if isinstance(wav_data, dict):
        # Dict with array and sampling_rate
        return audio_to_wav_bytes(wav_data["array"], wav_data["sampling_rate"])
    # Audio object format
    return audio_to_wav_bytes(wav_data.array, wav_data.sampling_rate)


def evaluate_huggingface(dataset, model_or_endpoint):
    """Evaluate using HuggingFace InferenceClient (works for both Hub models and endpoints)."""

    # Create client - use base_url for endpoints instead of model parameter
    if model_or_endpoint.startswith("http"):
        print(f"Using HuggingFace Inference Endpoint: {model_or_endpoint}")
        client = InferenceClient(base_url=model_or_endpoint)
    else:
        print(f"Using HuggingFace Hub model: {model_or_endpoint}")
        client = InferenceClient(model=model_or_endpoint)

    predictions = []
    references = []

    # Create a temporary directory for WAV files
    temp_dir = tempfile.mkdtemp()

    try:
        for i, sample in enumerate(dataset):
            # Get WAV bytes for API
            wav_bytes = prepare_wav_bytes(sample["wav"])

            # Write to temporary file with .wav extension
            # InferenceClient needs a file path to detect content type properly
            temp_path = Path(temp_dir) / f"temp_{i}.wav"
            temp_path.write_bytes(wav_bytes)

            try:
                result = client.automatic_speech_recognition(str(temp_path))

                # Parse result - InferenceClient returns different formats
                if isinstance(result, dict):
                    prediction = result.get("text", result.get("transcription", ""))
                elif isinstance(result, str):
                    prediction = result
                elif hasattr(result, "text"):
                    # Handle AutomaticSpeechRecognitionOutput objects
                    prediction = result.text
                else:
                    prediction = str(result)
            except Exception as e:
                import traceback

                print(f"Error processing sample {i + 1}:")
                print(f"  Exception type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print("  Full traceback:")
                traceback.print_exc()
                prediction = ""
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

            predictions.append(prediction)
            references.append(sample["text"])

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} samples")
    finally:
        # Clean up temporary directory
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    return predictions, references


def evaluate_assemblyai(dataset, api_key, model="best"):
    """Evaluate using AssemblyAI API."""
    import assemblyai as aai

    aai.settings.api_key = api_key

    # Map string to SpeechModel enum
    model_map = {
        "best": aai.types.SpeechModel.best,
        "universal": aai.types.SpeechModel.universal,
        "slam_1": aai.types.SpeechModel.slam_1,
        "nano": aai.types.SpeechModel.nano,
    }

    if model not in model_map:
        raise ValueError(f"Invalid model '{model}'. Choose from: {list(model_map.keys())}")

    config = aai.TranscriptionConfig(speech_model=model_map[model])
    transcriber = aai.Transcriber(config=config)

    print(f"Using AssemblyAI model: {model}")

    predictions = []
    references = []

    for i, sample in enumerate(dataset):
        # Get WAV bytes for API
        wav_bytes = prepare_wav_bytes(sample["wav"])

        try:
            transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
            prediction = transcript.text if transcript.text else ""
        except Exception as e:
            print(f"Error processing sample {i + 1}: {e}")
            prediction = ""

        predictions.append(prediction)
        references.append(sample["text"])

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} samples")

        # Rate limiting
        time.sleep(0.5)

    return predictions, references


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR models on LoquaciousSet dataset")
    parser.add_argument(
        "model",
        type=str,
        help="Model ID (e.g., mazesmazes/tiny-audio) or endpoint URL",
    )
    parser.add_argument(
        "--assemblyai",
        action="store_true",
        help="Use AssemblyAI API instead of HuggingFace",
    )
    parser.add_argument(
        "--assemblyai-model",
        type=str,
        default="best",
        choices=["best", "universal", "slam_1", "nano"],
        help="AssemblyAI model to use (default: best)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ASSEMBLYAI_API_KEY"),
        help="AssemblyAI API key (required if --assemblyai)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="large",
        help="LoquaciousSet config name (default: large)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: outputs/eval_{model_name})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    args = parser.parse_args()

    # Validate AssemblyAI requirements
    if args.assemblyai and not args.api_key:
        raise ValueError("AssemblyAI API key required. Set --api-key or ASSEMBLYAI_API_KEY env var")

    # Set default output dir
    if args.output_dir is None:
        if args.assemblyai:
            args.output_dir = Path(f"outputs/eval_assemblyai_{args.assemblyai_model}")
        else:
            # Sanitize model name for directory
            model_name = args.model.replace("/", "_").replace(":", "_")
            if args.model.startswith("http"):
                model_name = "endpoint_" + model_name.split("/")[-1]
            args.output_dir = Path(f"outputs/eval_{model_name}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load WER metric
    wer_metric = evaluate.load("wer")

    # Load dataset in streaming mode
    print(
        f"Loading speechbrain/LoquaciousSet dataset (config: {args.config}, split: {args.split})..."
    )
    dataset = load_dataset(
        "speechbrain/LoquaciousSet", args.config, split=args.split, streaming=True
    )

    if args.max_samples:
        dataset = dataset.take(args.max_samples)

    # Run inference
    print("Running inference...")

    if args.assemblyai:
        predictions, references = evaluate_assemblyai(dataset, args.api_key, args.assemblyai_model)
        model_name = f"AssemblyAI ({args.assemblyai_model})"
    else:
        predictions, references = evaluate_huggingface(dataset, args.model)
        model_name = args.model

    # Normalize text before computing WER
    from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase

    normalizer = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    normalized_predictions = [normalizer(p) for p in predictions]
    normalized_references = [normalizer(r) for r in references]

    # Compute WER
    wer = wer_metric.compute(predictions=normalized_predictions, references=normalized_references)

    # Save results
    num_samples = len(predictions)
    wer_percent = wer * 100
    results_file = args.output_dir / "results.txt"

    with results_file.open("w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(
            f"Dataset: speechbrain/LoquaciousSet (config: {args.config}, split: {args.split})\n"
        )
        f.write(f"Samples: {num_samples}\n")
        f.write(f"WER: {wer_percent:.2f}%\n\n")
        f.write("=" * 80 + "\n")
        f.write("Predictions vs Ground Truth\n")
        f.write("=" * 80 + "\n\n")

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            f.write(f"Sample {i + 1}\n")
            f.write(f"Ground Truth: {ref}\n")
            f.write(f"Prediction:   {pred}\n")
            f.write("-" * 80 + "\n\n")

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: speechbrain/LoquaciousSet (config: {args.config}, split: {args.split})")
    print(f"Samples: {num_samples}")
    print(f"WER: {wer_percent:.2f}%")
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
