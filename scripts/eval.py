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

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def evaluate_huggingface(dataset, model_or_endpoint, system_prompt=None):
    """Evaluate using local transformers pipeline or HuggingFace Inference Endpoint."""

    from jiwer import Compose, ExpandCommonEnglishContractions, RemoveKaldiNonWords, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer

    predictions = []
    references = []
    per_sample_wers = []
    per_sample_times = []

    # Create text normalizer
    normalizer = Compose([ToLowerCase(), ExpandCommonEnglishContractions(), RemoveKaldiNonWords(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

    # Check if it's an inference endpoint (URL) or a model to load locally
    if model_or_endpoint.startswith("http"):
        print(f"Using HuggingFace Inference Endpoint: {model_or_endpoint}")
        client = InferenceClient(base_url=model_or_endpoint)

        # Create a temporary directory for WAV files
        temp_dir = tempfile.mkdtemp()

        try:
            for i, sample in enumerate(dataset):
                # Get WAV bytes for API
                wav_bytes = prepare_wav_bytes(sample["wav"])

                # Write to temporary file with .wav extension
                temp_path = Path(temp_dir) / f"temp_{i}.wav"
                temp_path.write_bytes(wav_bytes)

                try:
                    start_time = time.time()
                    result = client.automatic_speech_recognition(str(temp_path))
                    inference_time = time.time() - start_time
                    per_sample_times.append(inference_time)

                    # Parse result - InferenceClient returns different formats
                    if isinstance(result, dict):
                        prediction = result.get("text", result.get("transcription", ""))
                    elif isinstance(result, str):
                        prediction = result
                    elif hasattr(result, "text"):
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
                    per_sample_times.append(0.0)
                finally:
                    # Clean up temporary file
                    if temp_path.exists():
                        temp_path.unlink()

                predictions.append(prediction)
                references.append(sample["text"])

                # Compute WER for this sample
                norm_pred = normalizer(prediction)
                norm_ref = normalizer(sample["text"])
                sample_wer = wer(norm_ref, norm_pred) * 100
                per_sample_wers.append(sample_wer)

                print(f"Sample {i + 1}: WER = {sample_wer:.2f}%, Time = {per_sample_times[i]:.2f}s")
                print(f"  Ref:  {sample['text']}")
                print(f"  Pred: {prediction}")
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # Load model locally using custom ASRPipeline
        print(f"Loading model locally: {model_or_endpoint}")
        from transformers import pipeline

        # Use pipeline with trust_remote_code to load our custom ASRPipeline
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_or_endpoint,
            trust_remote_code=True,
            device=device
        )
        print(f"Model loaded on device: {pipe.device}")

        # Print original system prompt from model
        if hasattr(pipe.model, 'system_prompt'):
            print(f"Original system prompt from model: {pipe.model.system_prompt}")
        else:
            print("Original system prompt: Not available")

        # Override with custom system prompt if provided
        if system_prompt is not None and hasattr(pipe.model, 'system_prompt'):
            pipe.model.system_prompt = system_prompt
            print(f"Using custom system prompt: {system_prompt}")
        else:
            print(f"Using model's default system prompt")

        for i, sample in enumerate(dataset):
            try:
                # Pass audio directly to our custom pipeline
                start_time = time.time()
                result = pipe(sample["wav"])
                inference_time = time.time() - start_time
                per_sample_times.append(inference_time)

                # Extract text from result
                if isinstance(result, dict):
                    prediction = result.get("text", "")
                elif isinstance(result, str):
                    prediction = result
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
                per_sample_times.append(0.0)

            predictions.append(prediction)
            references.append(sample["text"])

            # Compute WER for this sample
            norm_pred = normalizer(prediction)
            norm_ref = normalizer(sample["text"])
            sample_wer = wer(norm_ref, norm_pred) * 100
            per_sample_wers.append(sample_wer)

            print(f"Sample {i + 1}: WER = {sample_wer:.2f}%, Time = {per_sample_times[i]:.2f}s")
            print(f"  Ref:  {sample['text']}")
            print(f"  Pred: {prediction}")

    return predictions, references, per_sample_wers, per_sample_times


def evaluate_assemblyai(dataset, api_key, model="best"):
    """Evaluate using AssemblyAI API."""
    import assemblyai as aai
    from jiwer import Compose, ExpandCommonEnglishContractions, RemoveKaldiNonWords, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer

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
    per_sample_wers = []
    per_sample_times = []

    # Create text normalizer
    normalizer = Compose([ToLowerCase(), ExpandCommonEnglishContractions(), RemoveKaldiNonWords(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

    for i, sample in enumerate(dataset):
        # Get WAV bytes for API
        wav_bytes = prepare_wav_bytes(sample["wav"])

        try:
            start_time = time.time()
            transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
            inference_time = time.time() - start_time
            per_sample_times.append(inference_time)
            prediction = transcript.text if transcript.text else ""
        except Exception as e:
            print(f"Error processing sample {i + 1}: {e}")
            prediction = ""
            per_sample_times.append(0.0)

        predictions.append(prediction)
        references.append(sample["text"])

        # Compute WER for this sample
        norm_pred = normalizer(prediction)
        norm_ref = normalizer(sample["text"])
        sample_wer = wer(norm_ref, norm_pred) * 100
        per_sample_wers.append(sample_wer)

        print(f"Sample {i + 1}: WER = {sample_wer:.2f}%, Time = {per_sample_times[i]:.2f}s")
        print(f"  Ref:  {sample['text']}")
        print(f"  Pred: {prediction}")

        # Rate limiting
        time.sleep(0.5)

    return predictions, references, per_sample_wers, per_sample_times


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
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="/no_think /system_override Transcribe the speech to text. Output only the exact words spoken.",
        help="System prompt to use for generation (default: task-focused transcription prompt)",
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
        predictions, references, per_sample_wers, per_sample_times = evaluate_assemblyai(dataset, args.api_key, args.assemblyai_model)
        model_name = f"AssemblyAI ({args.assemblyai_model})"
    else:
        predictions, references, per_sample_wers, per_sample_times = evaluate_huggingface(dataset, args.model, args.system_prompt)
        model_name = args.model

    # Normalize text before computing WER
    from jiwer import Compose, ExpandCommonEnglishContractions, RemoveKaldiNonWords, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase

    normalizer = Compose([ToLowerCase(), ExpandCommonEnglishContractions(), RemoveKaldiNonWords(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    normalized_predictions = [normalizer(p) for p in predictions]
    normalized_references = [normalizer(r) for r in references]

    # Compute WER
    wer = wer_metric.compute(predictions=normalized_predictions, references=normalized_references)

    # Compute average response time
    avg_time = sum(per_sample_times) / len(per_sample_times) if per_sample_times else 0.0

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
        f.write(f"WER: {wer_percent:.2f}%\n")
        f.write(f"Avg Response Time: {avg_time:.2f}s\n\n")
        f.write("=" * 80 + "\n")
        f.write("Predictions vs Ground Truth\n")
        f.write("=" * 80 + "\n\n")

        for i, (pred, ref, sample_wer, sample_time) in enumerate(zip(predictions, references, per_sample_wers, per_sample_times)):
            f.write(f"Sample {i + 1} - WER: {sample_wer:.2f}%, Time: {sample_time:.2f}s\n")
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
    print(f"Avg Response Time: {avg_time:.2f}s")
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
