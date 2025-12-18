#!/usr/bin/env python3
"""
Evaluate ASR models on audio datasets.

Supports:
- Local models (e.g., mazesmazes/tiny-audio)
- HuggingFace Inference Endpoints (via endpoint URL)
- AssemblyAI API for comparison
"""

import io
import os
import re
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import load_dataset
from jiwer import wer
from transformers import WhisperTokenizer

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Audio Utilities
# =============================================================================


def audio_to_wav_bytes(audio_array: np.ndarray | torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes."""
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.numpy()
    if audio_array.ndim > 1:
        audio_array = audio_array.squeeze()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


def prepare_wav_bytes(wav_data) -> bytes:
    """Convert various audio formats to WAV bytes."""
    # Dict with array (most common HF datasets format)
    if isinstance(wav_data, dict):
        if "array" in wav_data and "sampling_rate" in wav_data:
            return audio_to_wav_bytes(wav_data["array"], wav_data["sampling_rate"])
        if "bytes" in wav_data:
            return wav_data["bytes"]
        if "path" in wav_data and wav_data["path"]:
            # Load from file path
            import soundfile as sf

            audio_array, sample_rate = sf.read(wav_data["path"])
            return audio_to_wav_bytes(audio_array, sample_rate)

    # Audio object with array/sampling_rate attributes
    if hasattr(wav_data, "array") and hasattr(wav_data, "sampling_rate"):
        return audio_to_wav_bytes(wav_data.array, wav_data.sampling_rate)

    # AudioDecoder - try to get path and load with soundfile
    if hasattr(wav_data, "path") and wav_data.path:
        import soundfile as sf

        audio_array, sample_rate = sf.read(wav_data.path)
        return audio_to_wav_bytes(audio_array, sample_rate)

    raise ValueError(f"Unsupported audio format: {type(wav_data)}")


# =============================================================================
# Text Normalization
# =============================================================================


class TextNormalizer:
    """Whisper-based text normalizer with ASR-specific preprocessing."""

    def __init__(self):
        self._tokenizer = None

    @property
    def tokenizer(self) -> WhisperTokenizer:
        if self._tokenizer is None:
            self._tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        return self._tokenizer

    def preprocess(self, text: str) -> str:
        """Remove inaudible tags and disfluencies."""
        text = re.sub(r"<inaudible>", "", text, flags=re.IGNORECASE)
        return re.sub(r"\b(uh|um)\b", "", text, flags=re.IGNORECASE)

    def normalize(self, text: str) -> str:
        """Full normalization pipeline."""
        return self.tokenizer.normalize(self.preprocess(text))


# =============================================================================
# Dataset Configuration
# =============================================================================


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    path: str
    audio_field: str
    text_field: str
    config: str | None = None
    default_split: str = "test"
    weight: float = 1.0


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "loquacious": DatasetConfig(
        name="loquacious",
        path="speechbrain/LoquaciousSet",
        config="medium",
        audio_field="wav",
        text_field="text",
    ),
    "earnings22": DatasetConfig(
        name="earnings22",
        path="sanchit-gandhi/earnings22_robust_split",
        config="default",
        audio_field="audio",
        text_field="sentence",
    ),
    "ami": DatasetConfig(
        name="ami",
        path="TakalaWang/AMI_ASR",
        config=None,
        audio_field="audio",
        text_field="text",
    ),
    "gigaspeech": DatasetConfig(
        name="gigaspeech",
        path="fixie-ai/gigaspeech",
        config="dev",
        audio_field="audio",
        text_field="text",
        default_split="dev",
    ),
    "tedlium": DatasetConfig(
        name="tedlium",
        path="sanchit-gandhi/tedlium-data",
        config="default",
        audio_field="audio",
        text_field="text",
    ),
    "commonvoice": DatasetConfig(
        name="commonvoice",
        path="fixie-ai/common_voice_17_0",
        config="en",
        audio_field="audio",
        text_field="sentence",
    ),
    "peoples": DatasetConfig(
        name="peoples",
        path="fixie-ai/peoples_speech",
        config="clean",
        audio_field="audio",
        text_field="text",
    ),
    "librispeech": DatasetConfig(
        name="librispeech",
        path="openslr/librispeech_asr",
        config="clean",
        audio_field="audio",
        text_field="text",
    ),
    "librispeech-other": DatasetConfig(
        name="librispeech-other",
        path="openslr/librispeech_asr",
        config="other",
        audio_field="audio",
        text_field="text",
    ),
}

# Combined dataset proportions
COMBINED_WEIGHTS = {
    "loquacious": 0.50,
    "gigaspeech": 0.10,
    "earnings22": 0.10,
    "ami": 0.10,
    "tedlium": 0.10,
}


def load_single_dataset(name: str, split: str, config_override: str | None = None):
    """Load a single dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    cfg = DATASET_REGISTRY[name]
    config = config_override if config_override else cfg.config

    print(f"Loading {cfg.path} (config: {config}, split: {split})...")
    if config:
        return load_dataset(cfg.path, config, split=split, streaming=True)
    return load_dataset(cfg.path, split=split, streaming=True)


def load_combined_dataset(max_samples: int | None, seed: int) -> tuple[Iterator, int]:
    """Load proportionally sampled combined dataset."""
    import random

    random.seed(seed)

    total_samples = max_samples or 1000
    samples_per_dataset = {
        name: max(1, int(total_samples * weight)) for name, weight in COMBINED_WEIGHTS.items()
    }
    print(f"Loading combined dataset: {samples_per_dataset}")

    all_samples = []
    for name, num_samples in samples_per_dataset.items():
        cfg = DATASET_REGISTRY[name]
        validation_split = "validation" if name != "loquacious" else "dev"

        if cfg.config:
            ds = load_dataset(cfg.path, cfg.config, split=validation_split, streaming=True)
        else:
            ds = load_dataset(cfg.path, split=validation_split, streaming=True)

        ds = ds.shuffle(seed=seed, buffer_size=num_samples * 10)

        count = 0
        for sample in ds:
            # Skip TEDLIUM ignore markers
            text = sample.get(cfg.text_field, "")
            if isinstance(text, str) and text.strip() == "ignore_time_segment_in_scoring":
                continue

            all_samples.append(
                {
                    "audio": sample[cfg.audio_field],
                    "text": sample[cfg.text_field],
                    "source": name,
                }
            )
            count += 1
            if count >= num_samples:
                break

    random.shuffle(all_samples)
    print(f"Combined dataset ready: {len(all_samples)} samples")
    return iter(all_samples), len(all_samples)


# =============================================================================
# Evaluation Core
# =============================================================================


@dataclass
class EvalResult:
    """Result of a single sample evaluation."""

    prediction: str
    reference: str
    wer: float
    time: float


class Evaluator:
    """Base evaluator with common evaluation loop logic."""

    def __init__(self, audio_field: str = "audio", text_field: str = "text"):
        self.audio_field = audio_field
        self.text_field = text_field
        self.normalizer = TextNormalizer()
        self.results: list[EvalResult] = []

    def transcribe(self, audio) -> tuple[str, float]:
        """Transcribe audio and return (text, inference_time). Override in subclass."""
        raise NotImplementedError

    def evaluate(self, dataset, max_samples: int | None = None) -> list[EvalResult]:
        """Run evaluation loop on dataset."""
        self.results = []
        processed = 0

        for sample in dataset:
            reference = sample[self.text_field]

            # Skip TEDLIUM ignore markers
            if isinstance(reference, str) and reference.strip() == "ignore_time_segment_in_scoring":
                continue

            processed += 1
            if max_samples and processed > max_samples:
                break

            try:
                prediction, inference_time = self.transcribe(sample[self.audio_field])
            except Exception as e:
                print(f"Error on sample {processed}: {e}")
                prediction, inference_time = "", 0.0
            norm_pred = self.normalizer.normalize(prediction)
            norm_ref = self.normalizer.normalize(reference)
            sample_wer = wer(norm_ref, norm_pred) * 100 if norm_ref else 0.0

            result = EvalResult(prediction, reference, sample_wer, inference_time)
            self.results.append(result)

            print(f"Sample {processed}: WER={sample_wer:.1f}%, Time={inference_time:.2f}s")
            print(f"  Ref:  {reference}")
            print(f"  Pred: {prediction}")

            # Checkpoint every 100 samples
            if processed % 100 == 0:
                self._print_checkpoint(processed)

        return self.results

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        preds = [self.normalizer.normalize(r.prediction) for r in self.results]
        refs = [self.normalizer.normalize(r.reference) for r in self.results]
        corpus_wer = wer(refs, preds) * 100
        avg_time = sum(r.time for r in self.results) / len(self.results)

        print(f"\n{'=' * 60}")
        print(f"CHECKPOINT @ {sample_count}: WER={corpus_wer:.2f}%, Avg Time={avg_time:.2f}s")
        print(f"{'=' * 60}\n")

    def compute_metrics(self) -> dict:
        """Compute final metrics."""
        if not self.results:
            return {"wer": 0.0, "avg_time": 0.0, "num_samples": 0}

        preds = [self.normalizer.normalize(r.prediction) for r in self.results]
        refs = [self.normalizer.normalize(r.reference) for r in self.results]

        return {
            "wer": wer(refs, preds) * 100,
            "avg_time": sum(r.time for r in self.results) / len(self.results),
            "num_samples": len(self.results),
        }


class LocalEvaluator(Evaluator):
    """Evaluator for local models."""

    def __init__(self, model_path: str, user_prompt: str | None = None, **kwargs):
        super().__init__(**kwargs)
        from transformers import pipeline

        # Load using pipeline with trust_remote_code to use Hub's custom pipeline class
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            trust_remote_code=True,
        )
        self.user_prompt = user_prompt

    def transcribe(self, audio) -> tuple[str, float]:
        # Convert to pipeline-compatible format
        if isinstance(audio, dict) and "array" in audio and "raw" not in audio:
            # Standard HF datasets format: "array" -> "raw"
            audio = {"raw": audio["array"], "sampling_rate": audio["sampling_rate"]}
        elif not isinstance(audio, (str, dict)) or (isinstance(audio, dict) and "raw" not in audio):
            # For other formats (AudioDecoder, bytes, etc.), convert to WAV file
            wav_bytes = prepare_wav_bytes(audio)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_bytes)
                audio = temp_file.name

        start = time.time()
        if self.user_prompt:
            result = self.pipe(audio, user_prompt=self.user_prompt)
        else:
            result = self.pipe(audio)
        elapsed = time.time() - start

        if isinstance(result, dict):
            return result.get("text", ""), elapsed
        return str(result), elapsed


class EndpointEvaluator(Evaluator):
    """Evaluator for HuggingFace Inference Endpoints."""

    def __init__(self, endpoint_url: str, **kwargs):
        super().__init__(**kwargs)
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(base_url=endpoint_url)
        self.temp_dir = tempfile.mkdtemp()

    def transcribe(self, audio) -> tuple[str, float]:
        wav_bytes = prepare_wav_bytes(audio)
        temp_path = Path(self.temp_dir) / f"temp_{time.time_ns()}.wav"
        temp_path.write_bytes(wav_bytes)

        try:
            start = time.time()
            result = self.client.automatic_speech_recognition(str(temp_path))
            elapsed = time.time() - start

            if isinstance(result, dict):
                text = result.get("text", result.get("transcription", ""))
            elif hasattr(result, "text"):
                text = result.text
            else:
                text = str(result)
            return text, elapsed
        finally:
            temp_path.unlink(missing_ok=True)

    def __del__(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


class AssemblyAIEvaluator(Evaluator):
    """Evaluator for AssemblyAI API."""

    MODEL_MAP = {"best": "best", "universal": "universal", "slam_1": "slam_1", "nano": "nano"}

    def __init__(self, api_key: str, model: str = "slam_1", **kwargs):
        super().__init__(**kwargs)
        import assemblyai as aai

        aai.settings.api_key = api_key

        if model not in self.MODEL_MAP:
            raise ValueError(f"Invalid model '{model}'. Choose from: {list(self.MODEL_MAP.keys())}")

        model_enum = getattr(aai.types.SpeechModel, model)
        config = aai.TranscriptionConfig(speech_model=model_enum)
        self.transcriber = aai.Transcriber(config=config)

    def transcribe(self, audio) -> tuple[str, float]:
        wav_bytes = prepare_wav_bytes(audio)

        start = time.time()
        transcript = self.transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start

        time.sleep(0.5)  # Rate limiting
        return transcript.text or "", elapsed


# =============================================================================
# Results Output
# =============================================================================


def save_results(
    output_path: Path,
    model_name: str,
    dataset_desc: str,
    results: list[EvalResult],
    metrics: dict,
):
    """Save evaluation results to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_desc}\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write(f"WER: {metrics['wer']:.2f}%\n")
        f.write(f"Avg Response Time: {metrics['avg_time']:.2f}s\n\n")
        f.write("=" * 80 + "\n")
        f.write("Predictions vs Ground Truth\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"Sample {i} - WER: {r.wer:.2f}%, Time: {r.time:.2f}s\n")
            f.write(f"Ground Truth: {r.reference}\n")
            f.write(f"Prediction:   {r.prediction}\n")
            f.write("-" * 80 + "\n\n")


def print_summary(model_name: str, dataset_desc: str, metrics: dict, output_path: Path):
    """Print final evaluation summary."""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_desc}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"WER: {metrics['wer']:.2f}%")
    print(f"Avg Time: {metrics['avg_time']:.2f}s")
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================


def run_all_datasets(args, model_name: str):
    """Run evaluation on all datasets sequentially, saving results to separate folders."""
    safe_name = model_name.replace("/", "_").replace(":", "_")
    if args.model and args.model.startswith("http"):
        safe_name = "endpoint_" + safe_name.split("/")[-1]

    all_results: dict[str, dict] = {}
    output_dirs: list[Path] = []

    print(f"\n{'=' * 60}")
    print("Running evaluation on ALL datasets")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}\n")

    for dataset_name, cfg in DATASET_REGISTRY.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'=' * 60}\n")

        try:
            # Load dataset
            split = args.split if args.split != "test" else cfg.default_split
            dataset = load_single_dataset(dataset_name, split, args.config)
            audio_field, text_field = cfg.audio_field, cfg.text_field
            config_name = args.config or cfg.config
            dataset_desc = f"{cfg.path} (config: {config_name}, split: {split})"

            if args.max_samples:
                dataset = dataset.take(args.max_samples)

            # Create evaluator
            if args.assemblyai:
                evaluator = AssemblyAIEvaluator(
                    args.api_key,
                    args.assemblyai_model,
                    audio_field=audio_field,
                    text_field=text_field,
                )
            elif args.model.startswith("http"):
                evaluator = EndpointEvaluator(
                    args.model,
                    audio_field=audio_field,
                    text_field=text_field,
                )
            else:
                evaluator = LocalEvaluator(
                    args.model,
                    user_prompt=args.user_prompt,
                    audio_field=audio_field,
                    text_field=text_field,
                )

            # Run evaluation
            evaluator.evaluate(dataset, args.max_samples)
            metrics = evaluator.compute_metrics()

            # Save results using same path format as single dataset runs
            output_dir = Path(f"outputs/eval_{dataset_name}_{safe_name}")
            output_path = output_dir / "results.txt"
            save_results(output_path, model_name, dataset_desc, evaluator.results, metrics)
            output_dirs.append(output_dir)

            all_results[dataset_name] = metrics
            print(f"\n{dataset_name}: WER={metrics['wer']:.2f}%, Samples={metrics['num_samples']}")

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            all_results[dataset_name] = {"wer": -1, "error": str(e)}

    # Print summary of all datasets
    print(f"\n{'=' * 60}")
    print("SUMMARY - All Datasets")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")
    for name, metrics in all_results.items():
        if "error" in metrics:
            print(f"  {name}: ERROR - {metrics['error']}")
        else:
            print(f"  {name}: WER={metrics['wer']:.2f}% ({metrics['num_samples']} samples)")
    print(f"{'=' * 60}")
    print("Results saved to:")
    for output_dir in output_dirs:
        print(f"  {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ASR models")
    parser.add_argument("model", nargs="?", help="Model path or endpoint URL")
    parser.add_argument(
        "--dataset",
        default="loquacious",
        choices=list(DATASET_REGISTRY.keys()) + ["combined", "all"],
        help="Dataset to evaluate on ('all' runs all datasets sequentially)",
    )
    parser.add_argument("--assemblyai", action="store_true", help="Use AssemblyAI API")
    parser.add_argument(
        "--assemblyai-model",
        default="slam_1",
        choices=["best", "universal", "slam_1", "nano"],
    )
    parser.add_argument("--api-key", default=os.environ.get("ASSEMBLYAI_API_KEY"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default=None, help="Dataset config override")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--user-prompt", default=None, help="User prompt (must include <audio>)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validation
    if args.assemblyai:
        if not args.api_key:
            raise ValueError("AssemblyAI API key required")
        model_name = f"AssemblyAI ({args.assemblyai_model})"
    else:
        if not args.model:
            raise ValueError("Model argument required")
        model_name = args.model

    # Handle --dataset all: run all datasets sequentially
    if args.dataset == "all":
        run_all_datasets(args, model_name)
        return

    # Output directory
    if args.output_dir is None:
        safe_name = model_name.replace("/", "_").replace(":", "_")
        if args.model and args.model.startswith("http"):
            safe_name = "endpoint_" + safe_name.split("/")[-1]
        args.output_dir = Path(f"outputs/eval_{args.dataset}_{safe_name}")

    # Load dataset
    if args.dataset == "combined":
        dataset, total = load_combined_dataset(args.max_samples, args.seed)
        audio_field, text_field = "audio", "text"
        dataset_desc = f"combined (proportional, {total} samples)"
        args.max_samples = None  # Already limited
    else:
        cfg = DATASET_REGISTRY[args.dataset]
        # Use dataset's default_split if user didn't override
        split = args.split if args.split != "test" else cfg.default_split
        dataset = load_single_dataset(args.dataset, split, args.config)
        audio_field, text_field = cfg.audio_field, cfg.text_field
        config_name = args.config or cfg.config
        dataset_desc = f"{cfg.path} (config: {config_name}, split: {split})"

        if args.max_samples:
            dataset = dataset.take(args.max_samples)

    # Create evaluator
    if args.assemblyai:
        evaluator = AssemblyAIEvaluator(
            args.api_key,
            args.assemblyai_model,
            audio_field=audio_field,
            text_field=text_field,
        )
    elif args.model.startswith("http"):
        evaluator = EndpointEvaluator(
            args.model,
            audio_field=audio_field,
            text_field=text_field,
        )
    else:
        evaluator = LocalEvaluator(
            args.model,
            user_prompt=args.user_prompt,
            audio_field=audio_field,
            text_field=text_field,
        )

    # Run evaluation
    print("Running inference...")
    evaluator.evaluate(dataset, args.max_samples)
    metrics = evaluator.compute_metrics()

    # Save and print results
    output_path = args.output_dir / "results.txt"
    save_results(output_path, model_name, dataset_desc, evaluator.results, metrics)
    print_summary(model_name, dataset_desc, metrics, output_path)


if __name__ == "__main__":
    main()
