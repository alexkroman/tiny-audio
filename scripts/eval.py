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
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import InferenceClient
from torchcodec.decoders import AudioDecoder


def audio_to_wav_bytes(audio_array, sample_rate):
    """Convert audio array to WAV bytes using torchcodec."""
    # Convert to numpy if needed
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.numpy()

    # Ensure 1D shape
    if audio_array.ndim > 1:
        audio_array = audio_array.squeeze()

    # Write to temp file and use torchcodec to encode as WAV
    # torchcodec is primarily a decoder, so we'll use a simple WAV writer
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        # Convert float32 to int16
        audio_int16 = (audio_array * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


def wav_bytes_to_audio(wav_bytes):
    """Convert WAV bytes to audio array using torchcodec."""
    # Write bytes to temp file for torchcodec
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        temp_path = f.name

    try:
        decoder = AudioDecoder(temp_path)
        audio_samples = decoder.get_all_samples()
        audio_array = audio_samples.data.squeeze().numpy()
        sample_rate = decoder.metadata.sample_rate
        return audio_array, sample_rate
    finally:
        Path(temp_path).unlink()


def prepare_wav_bytes(wav_data):
    """Convert various WAV data formats to bytes for API calls."""
    # Handle AudioDecoder objects (datasets library lazy loading)
    if (
        hasattr(wav_data, "__class__")
        and "AudioDecoder" in str(type(wav_data))
        and hasattr(wav_data, "get_all_samples")
    ):
        samples = wav_data.get_all_samples()
        # samples should have .data and metadata with sample_rate
        audio_array = samples.data.squeeze().numpy()
        sample_rate = wav_data.metadata.sample_rate
        return audio_to_wav_bytes(audio_array, sample_rate)

    if isinstance(wav_data, dict):
        if "bytes" in wav_data:
            # Already in bytes format
            return wav_data["bytes"]
        if "array" in wav_data and "sampling_rate" in wav_data:
            # Dict with array and sampling_rate (earnings22 format)
            return audio_to_wav_bytes(wav_data["array"], wav_data["sampling_rate"])

    # Audio object format (LoquaciousSet)
    if hasattr(wav_data, "array") and hasattr(wav_data, "sampling_rate"):
        return audio_to_wav_bytes(wav_data.array, wav_data.sampling_rate)

    raise ValueError(
        f"Unsupported audio format: {type(wav_data)}, available attributes: {dir(wav_data) if hasattr(wav_data, '__dir__') else 'N/A'}"
    )


def evaluate_huggingface(
    dataset,
    model_or_endpoint,
    system_prompt=None,
    user_prompt=None,
    audio_field="wav",
    text_field="text",
):
    """Evaluate using local transformers pipeline or HuggingFace Inference Endpoint.

    Args:
        dataset: Dataset to evaluate on
        model_or_endpoint: Model path or endpoint URL
        system_prompt: Optional system prompt override
        user_prompt: Optional user prompt override
        audio_field: Name of the audio field in the dataset (default: "wav")
        text_field: Name of the text field in the dataset (default: "text")
    """

    import re

    from jiwer import wer
    from transformers import WhisperTokenizer

    # Custom preprocessing to remove <inaudible> tags and disfluencies before Whisper normalization
    def preprocess_text(text: str) -> str:
        # Remove <inaudible> tags
        text = re.sub(r"<inaudible>", "", text, flags=re.IGNORECASE)
        # Remove disfluencies (uh, um) - these are in Whisper's ignore patterns already
        # but we keep this for compatibility with non-Whisper datasets
        return re.sub(r"\b(uh|um)\b", "", text, flags=re.IGNORECASE)

    predictions = []
    references = []
    per_sample_wers = []
    per_sample_times = []

    # Use Whisper's English text normalizer (includes number normalization, contractions, etc.)
    # Load once at the start to avoid repeated downloads
    whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    def normalize_text(text: str) -> str:
        return whisper_tokenizer.normalize(preprocess_text(text))

    # Check if it's an inference endpoint (URL) or a model to load locally
    if model_or_endpoint.startswith("http"):
        print(f"Using HuggingFace Inference Endpoint: {model_or_endpoint}")
        client = InferenceClient(base_url=model_or_endpoint)

        # Create a temporary directory for WAV files
        temp_dir = tempfile.mkdtemp()

        try:
            sample_count = 0
            for i, sample in enumerate(dataset):
                sample_count += 1

                # Get WAV bytes for API
                wav_bytes = prepare_wav_bytes(sample[audio_field])

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

                    print(f"Error processing sample {sample_count}:")
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
                references.append(sample[text_field])

                # Compute WER for this sample using Whisper normalization
                norm_pred = normalize_text(prediction)
                norm_ref = normalize_text(sample[text_field])
                sample_wer = wer(norm_ref, norm_pred) * 100
                per_sample_wers.append(sample_wer)

                print(f"Sample {sample_count}: WER = {sample_wer:.2f}%, Time = {per_sample_times[-1]:.2f}s")
                print(f"  Ref:  {sample[text_field]}")
                print(f"  Pred: {prediction}")

                # Print cumulative WER every 100 samples
                if sample_count % 100 == 0:
                    # Compute corpus-level WER (same as final metric)
                    normalized_preds = [normalize_text(p) for p in predictions]
                    normalized_refs = [normalize_text(r) for r in references]
                    corpus_wer = wer(normalized_refs, normalized_preds) * 100
                    avg_time_so_far = sum(per_sample_times) / len(per_sample_times)
                    print(f"\n{'=' * 80}")
                    print(f"CHECKPOINT @ {sample_count} samples:")
                    print(f"  Corpus WER: {corpus_wer:.2f}%")
                    print(f"  Avg Time/Sample: {avg_time_so_far:.2f}s")
                    print(f"{'=' * 80}\n")
        finally:
            # Clean up temporary directory
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # Load model locally using custom ASRPipeline
        from src.asr_modeling import ASRModel
        from src.asr_pipeline import ASRPipeline

        # Determine device
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load model directly to device
        model = ASRModel.from_pretrained(
            model_or_endpoint,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=None,
            device=device,
        )

        # Create pipeline using local ASRPipeline class directly
        pipe = ASRPipeline(
            model=model,
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            device=device,
        )

        sample_count = 0
        for i, sample in enumerate(dataset):
            sample_count += 1

            try:
                # Pass audio directly to our custom pipeline
                start_time = time.time()
                # Pass user_prompt to pipeline if provided
                if user_prompt is not None:
                    result = pipe(sample[audio_field], user_prompt=user_prompt)
                else:
                    result = pipe(sample[audio_field])
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

                print(f"Error processing sample {sample_count}:")
                print(f"  Exception type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print("  Full traceback:")
                traceback.print_exc()
                prediction = ""
                per_sample_times.append(0.0)

            predictions.append(prediction)
            references.append(sample[text_field])

            # Compute WER for this sample using Whisper normalization
            norm_pred = normalize_text(prediction)
            norm_ref = normalize_text(sample[text_field])
            sample_wer = wer(norm_ref, norm_pred) * 100
            per_sample_wers.append(sample_wer)

            print(f"Sample {sample_count}: WER = {sample_wer:.2f}%, Time = {per_sample_times[-1]:.2f}s")
            print(f"  Ref:  {sample[text_field]}")
            print(f"  Pred: {prediction}")

            # Print cumulative WER every 100 samples
            if sample_count % 100 == 0:
                # Compute corpus-level WER (same as final metric)
                normalized_preds = [normalize_text(p) for p in predictions]
                normalized_refs = [normalize_text(r) for r in references]
                corpus_wer = wer(normalized_refs, normalized_preds) * 100
                avg_time_so_far = sum(per_sample_times) / len(per_sample_times)
                print(f"\n{'=' * 80}")
                print(f"CHECKPOINT @ {sample_count} samples:")
                print(f"  Corpus WER: {corpus_wer:.2f}%")
                print(f"  Avg Time/Sample: {avg_time_so_far:.2f}s")
                print(f"{'=' * 80}\n")

    return predictions, references, per_sample_wers, per_sample_times


def evaluate_assemblyai(dataset, api_key, model="best", audio_field="wav", text_field="text"):
    """Evaluate using AssemblyAI API.

    Args:
        dataset: Dataset to evaluate on
        api_key: AssemblyAI API key
        model: AssemblyAI model to use
        audio_field: Name of the audio field in the dataset (default: "wav")
        text_field: Name of the text field in the dataset (default: "text")
    """
    import re

    import assemblyai as aai
    from jiwer import wer
    from transformers import WhisperTokenizer

    # Custom preprocessing to remove <inaudible> tags and disfluencies before Whisper normalization
    def preprocess_text(text: str) -> str:
        # Remove <inaudible> tags
        text = re.sub(r"<inaudible>", "", text, flags=re.IGNORECASE)
        # Remove disfluencies (uh, um)
        return re.sub(r"\b(uh|um)\b", "", text, flags=re.IGNORECASE)

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

    # Use Whisper's English text normalizer
    whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    def normalize_text(text: str) -> str:
        return whisper_tokenizer.normalize(preprocess_text(text))

    sample_count = 0
    for i, sample in enumerate(dataset):
        sample_count += 1

        # Get WAV bytes for API
        wav_bytes = prepare_wav_bytes(sample[audio_field])

        try:
            start_time = time.time()
            transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
            inference_time = time.time() - start_time
            per_sample_times.append(inference_time)
            prediction = transcript.text if transcript.text else ""
        except Exception as e:
            print(f"Error processing sample {sample_count}: {e}")
            prediction = ""
            per_sample_times.append(0.0)

        predictions.append(prediction)
        references.append(sample[text_field])

        # Compute WER for this sample using Whisper normalization
        norm_pred = normalize_text(prediction)
        norm_ref = normalize_text(sample[text_field])
        sample_wer = wer(norm_ref, norm_pred) * 100
        per_sample_wers.append(sample_wer)

        print(f"Sample {sample_count}: WER = {sample_wer:.2f}%, Time = {per_sample_times[-1]:.2f}s")
        print(f"  Ref:  {sample[text_field]}")
        print(f"  Pred: {prediction}")

        # Print cumulative WER every 100 samples
        if sample_count % 100 == 0:
            # Compute corpus-level WER (same as final metric)
            normalized_preds = [normalize_text(p) for p in predictions]
            normalized_refs = [normalize_text(r) for r in references]
            corpus_wer = wer(normalized_refs, normalized_preds) * 100
            avg_time_so_far = sum(per_sample_times) / len(per_sample_times)
            print(f"\n{'=' * 80}")
            print(f"CHECKPOINT @ {sample_count} samples:")
            print(f"  Corpus WER: {corpus_wer:.2f}%")
            print(f"  Avg Time/Sample: {avg_time_so_far:.2f}s")
            print(f"{'=' * 80}\n")

        # Rate limiting
        time.sleep(0.5)

    return predictions, references, per_sample_wers, per_sample_times


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR models on audio datasets")
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="Model ID (e.g., mazesmazes/tiny-audio) or endpoint URL (not required when using --assemblyai)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="loquacious",
        choices=["loquacious", "earnings22", "ami", "gigaspeech"],
        help="Dataset to evaluate on (default: loquacious)",
    )
    parser.add_argument(
        "--assemblyai",
        action="store_true",
        help="Use AssemblyAI API instead of HuggingFace",
    )
    parser.add_argument(
        "--assemblyai-model",
        type=str,
        default="slam_1",
        choices=["best", "universal", "slam_1", "nano"],
        help="AssemblyAI model to use (default: slam_1)",
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
        default="medium",
        help="Dataset config name (default: medium for loquacious, chunked for earnings22)",
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
        default="/no_think /system_override",
        help="System prompt to use for generation (default: task-focused transcription prompt)",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=None,
        help="User prompt to override the default 'Repeat the following text, without any explanation: <audio>'. Must include <audio> token.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling (default: 42)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.assemblyai:
        if not args.api_key:
            raise ValueError(
                "AssemblyAI API key required. Set --api-key or ASSEMBLYAI_API_KEY env var"
            )
    else:
        if not args.model:
            raise ValueError("Model argument is required when not using --assemblyai")

    # Set default output dir
    if args.output_dir is None:
        if args.assemblyai:
            args.output_dir = Path(
                f"outputs/eval_{args.dataset}_assemblyai_{args.assemblyai_model}"
            )
        else:
            # Sanitize model name for directory
            model_name = args.model.replace("/", "_").replace(":", "_")
            if args.model.startswith("http"):
                model_name = "endpoint_" + model_name.split("/")[-1]
            args.output_dir = Path(f"outputs/eval_{args.dataset}_{model_name}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load WER metric
    wer_metric = evaluate.load("wer")

    # Load dataset based on selection
    if args.dataset == "loquacious":
        dataset_name = "speechbrain/LoquaciousSet"
        dataset_config = args.config if args.config != "medium" else "medium"
        audio_field = "wav"
        text_field = "text"
        print(f"Loading {dataset_name} dataset (config: {dataset_config}, split: {args.split})...")
        dataset = load_dataset(dataset_name, dataset_config, split=args.split, streaming=True)
    elif args.dataset == "earnings22":
        dataset_name = "sanchit-gandhi/earnings22_robust_split"
        # Use default config for earnings22_robust_split
        dataset_config = args.config if args.config != "medium" else "default"
        audio_field = "audio"
        text_field = "sentence"  # earnings22_robust_split uses "sentence" field
        print(f"Loading {dataset_name} dataset (config: {dataset_config}, split: {args.split})...")
        dataset = load_dataset(dataset_name, dataset_config, split=args.split, streaming=True)
    elif args.dataset == "ami":
        dataset_name = "TakalaWang/AMI_ASR"
        dataset_config = None  # No config needed for AMI
        audio_field = "audio"
        text_field = "text"
        print(f"Loading {dataset_name} dataset (split: {args.split})...")
        dataset = load_dataset(dataset_name, split=args.split, streaming=True)
    elif args.dataset == "gigaspeech":
        dataset_name = "fixie-ai/gigaspeech"
        # Use xl-empty-audio-removed config by default, or user-specified config
        dataset_config = args.config if args.config != "medium" else "dev"
        audio_field = "audio"
        text_field = "text"
        print(f"Loading {dataset_name} dataset (config: {dataset_config}, split: {args.split})...")
        dataset = load_dataset(dataset_name, dataset_config, split="dev", streaming=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.max_samples:
        dataset = dataset.take(args.max_samples)

    # Run inference
    print("Running inference...")

    if args.assemblyai:
        predictions, references, per_sample_wers, per_sample_times = evaluate_assemblyai(
            dataset, args.api_key, args.assemblyai_model, audio_field, text_field
        )
        model_name = f"AssemblyAI ({args.assemblyai_model})"
    else:
        predictions, references, per_sample_wers, per_sample_times = evaluate_huggingface(
            dataset, args.model, args.system_prompt, args.user_prompt, audio_field, text_field
        )
        model_name = args.model

    # Normalize text before computing WER using Whisper's normalizer
    import re

    from transformers import WhisperTokenizer

    # Custom preprocessing to remove <inaudible> tags and disfluencies
    def preprocess_text(text: str) -> str:
        text = re.sub(r"<inaudible>", "", text, flags=re.IGNORECASE)
        return re.sub(r"\b(uh|um)\b", "", text, flags=re.IGNORECASE)

    whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
    normalized_predictions = [whisper_tokenizer.normalize(preprocess_text(p)) for p in predictions]
    normalized_references = [whisper_tokenizer.normalize(preprocess_text(r)) for r in references]

    # Compute WER
    wer = wer_metric.compute(predictions=normalized_predictions, references=normalized_references)

    # Compute average response time
    avg_time = sum(per_sample_times) / len(per_sample_times) if per_sample_times else 0.0

    # Save results
    num_samples = len(predictions)
    wer_percent = wer * 100
    results_file = args.output_dir / "results.txt"

    # Build dataset description
    if dataset_config:
        dataset_desc = f"{dataset_name} (config: {dataset_config}, split: {args.split})"
    else:
        dataset_desc = f"{dataset_name} (split: {args.split})"

    with results_file.open("w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_desc}\n")
        f.write(f"Samples: {num_samples}\n")
        f.write(f"WER: {wer_percent:.2f}%\n")
        f.write(f"Avg Response Time: {avg_time:.2f}s\n\n")
        f.write("=" * 80 + "\n")
        f.write("Predictions vs Ground Truth\n")
        f.write("=" * 80 + "\n\n")

        for i, (pred, ref, sample_wer, sample_time) in enumerate(
            zip(predictions, references, per_sample_wers, per_sample_times)
        ):
            f.write(f"Sample {i + 1} - WER: {sample_wer:.2f}%, Time: {sample_time:.2f}s\n")
            f.write(f"Ground Truth: {ref}\n")
            f.write(f"Prediction:   {pred}\n")
            f.write("-" * 80 + "\n\n")

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_desc}")
    print(f"Samples: {num_samples}")
    print(f"WER: {wer_percent:.2f}%")
    print(f"Avg Response Time: {avg_time:.2f}s")
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
