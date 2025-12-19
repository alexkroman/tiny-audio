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
from datasets import Audio, load_dataset
from jiwer import wer
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
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


# =============================================================================
# Diarization Dataset Configuration
# =============================================================================


@dataclass
class DiarizationDatasetConfig:
    """Configuration for a diarization dataset."""

    name: str
    path: str
    audio_field: str
    speakers_field: str
    timestamps_start_field: str
    timestamps_end_field: str
    config: str | None = None
    default_split: str = "test"


DIARIZATION_DATASET_REGISTRY: dict[str, DiarizationDatasetConfig] = {
    "callhome": DiarizationDatasetConfig(
        name="callhome",
        path="talkbank/callhome",
        config="eng",
        audio_field="audio",
        speakers_field="speakers",
        timestamps_start_field="timestamps_start",
        timestamps_end_field="timestamps_end",
        default_split="data",
    ),
}


def load_single_dataset(name: str, split: str, config_override: str | None = None):
    """Load a single dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    cfg = DATASET_REGISTRY[name]
    config = config_override if config_override else cfg.config

    print(f"Loading {cfg.path} (config: {config}, split: {split})...")
    if config:
        ds = load_dataset(cfg.path, config, split=split, streaming=True)
    else:
        ds = load_dataset(cfg.path, split=split, streaming=True)

    # Cast audio column to ensure proper decoding with streaming
    return ds.cast_column(cfg.audio_field, Audio(sampling_rate=16000))


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

        # Cast audio column to ensure proper decoding with streaming
        ds = ds.cast_column(cfg.audio_field, Audio(sampling_rate=16000))
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
# Diarization Evaluation
# =============================================================================


@dataclass
class DiarizationResult:
    """Result of a single diarization evaluation."""

    der: float  # Sample DER percentage
    confusion: float  # Sample confusion percentage
    missed: float  # Sample missed percentage
    false_alarm: float  # Sample false alarm percentage
    time: float
    num_speakers_ref: int
    num_speakers_hyp: int
    # Raw values for corpus-level calculation
    total: float = 0.0  # Total reference duration
    confusion_raw: float = 0.0  # Confusion duration
    missed_raw: float = 0.0  # Missed detection duration
    false_alarm_raw: float = 0.0  # False alarm duration


class DiarizationEvaluator:
    """Evaluator for speaker diarization using pyannote DER metric."""

    def __init__(
        self,
        audio_field: str = "audio",
        speakers_field: str = "speakers",
        timestamps_start_field: str = "timestamps_start",
        timestamps_end_field: str = "timestamps_end",
        hf_token: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        self.audio_field = audio_field
        self.speakers_field = speakers_field
        self.timestamps_start_field = timestamps_start_field
        self.timestamps_end_field = timestamps_end_field
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.results: list[DiarizationResult] = []
        self.metric = DiarizationErrorRate()

    def _build_reference_annotation(self, sample: dict) -> Annotation:
        """Build pyannote Annotation from dataset sample."""
        annotation = Annotation()
        speakers = sample[self.speakers_field]
        starts = sample[self.timestamps_start_field]
        ends = sample[self.timestamps_end_field]

        for speaker, start, end in zip(speakers, starts, ends):
            annotation[Segment(start, end)] = speaker

        return annotation

    def _build_hypothesis_annotation(self, segments: list[dict]) -> Annotation:
        """Build pyannote Annotation from diarization output."""
        annotation = Annotation()
        for seg in segments:
            annotation[Segment(seg["start"], seg["end"])] = seg["speaker"]
        return annotation

    def diarize(self, audio) -> tuple[list[dict], float]:
        """Run diarization on audio and return (segments, inference_time)."""
        import io

        import librosa

        from src.asr_pipeline import SpeakerDiarizer

        # Prepare audio array - handle both decoded and raw bytes formats
        if isinstance(audio, dict):
            if "array" in audio:
                # Already decoded
                audio_array = audio["array"]
                sample_rate = audio.get("sampling_rate", 16000)
            elif "bytes" in audio:
                # Raw bytes - decode with librosa (avoids torchcodec CPU issues)
                audio_array, sample_rate = librosa.load(io.BytesIO(audio["bytes"]), sr=16000)
            else:
                raise ValueError(f"Unsupported audio dict format: {audio.keys()}")
        else:
            raise ValueError(f"Unsupported audio format: {type(audio)}")

        # Ensure float32 dtype (pyannote requires consistent dtype)
        if hasattr(audio_array, "astype"):
            audio_array = audio_array.astype(np.float32)

        start = time.time()
        segments = SpeakerDiarizer.diarize(
            audio_array,
            sample_rate=sample_rate,
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            hf_token=self.hf_token,
        )
        elapsed = time.time() - start

        return segments, elapsed

    def evaluate(self, dataset, max_samples: int | None = None) -> list[DiarizationResult]:
        """Run diarization evaluation loop on dataset."""
        self.results = []
        processed = 0

        for sample in dataset:
            processed += 1
            if max_samples and processed > max_samples:
                break

            try:
                # Get reference annotation
                reference = self._build_reference_annotation(sample)

                # Build UEM (evaluation region) from reference extent
                uem = Timeline([reference.get_timeline().extent()])

                # Run diarization
                segments, inference_time = self.diarize(sample[self.audio_field])
                hypothesis = self._build_hypothesis_annotation(segments)

                # Compute DER with detailed components
                details = self.metric(reference, hypothesis, uem=uem, detailed=True)

                # Extract raw values for corpus calculation
                total = details["total"]
                confusion_raw = details["confusion"]
                missed_raw = details["missed detection"]
                false_alarm_raw = details["false alarm"]

                # Compute per-sample percentages
                if total > 0:
                    der = (confusion_raw + missed_raw + false_alarm_raw) / total
                    confusion = confusion_raw / total
                    missed = missed_raw / total
                    false_alarm = false_alarm_raw / total
                else:
                    der = confusion = missed = false_alarm = 0.0

                result = DiarizationResult(
                    der=der * 100,
                    confusion=confusion * 100,
                    missed=missed * 100,
                    false_alarm=false_alarm * 100,
                    time=inference_time,
                    num_speakers_ref=len(set(sample[self.speakers_field])),
                    num_speakers_hyp=len(set(seg["speaker"] for seg in segments)),
                    total=total,
                    confusion_raw=confusion_raw,
                    missed_raw=missed_raw,
                    false_alarm_raw=false_alarm_raw,
                )
                self.results.append(result)

                print(
                    f"Sample {processed}: DER={result.der:.1f}% "
                    f"(conf={result.confusion:.1f}%, miss={result.missed:.1f}%, fa={result.false_alarm:.1f}%) "
                    f"Time={inference_time:.2f}s "
                    f"Speakers: ref={result.num_speakers_ref}, hyp={result.num_speakers_hyp}"
                )

            except Exception as e:
                print(f"Error on sample {processed}: {e}")
                # Skip failed samples - don't add to results to avoid polluting corpus metrics
                continue

            # Checkpoint every 50 samples
            if processed % 50 == 0:
                self._print_checkpoint(processed)

        return self.results

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        metrics = self.compute_metrics()
        print(f"\n{'=' * 60}")
        print(
            f"CHECKPOINT @ {sample_count}: DER={metrics['der']:.2f}% "
            f"(conf={metrics['confusion']:.2f}%, miss={metrics['missed']:.2f}%, fa={metrics['false_alarm']:.2f}%)"
        )
        print(f"{'=' * 60}\n")

    def compute_metrics(self) -> dict:
        """Compute final corpus-level metrics."""
        if not self.results:
            return {
                "der": 0.0,
                "confusion": 0.0,
                "missed": 0.0,
                "false_alarm": 0.0,
                "avg_time": 0.0,
                "num_samples": 0,
            }

        # Corpus-level: sum raw values across all samples
        total_duration = sum(r.total for r in self.results)
        total_confusion = sum(r.confusion_raw for r in self.results)
        total_missed = sum(r.missed_raw for r in self.results)
        total_false_alarm = sum(r.false_alarm_raw for r in self.results)

        if total_duration > 0:
            corpus_der = (total_confusion + total_missed + total_false_alarm) / total_duration * 100
            corpus_confusion = total_confusion / total_duration * 100
            corpus_missed = total_missed / total_duration * 100
            corpus_false_alarm = total_false_alarm / total_duration * 100
        else:
            corpus_der = corpus_confusion = corpus_missed = corpus_false_alarm = 0.0

        return {
            "der": corpus_der,
            "confusion": corpus_confusion,
            "missed": corpus_missed,
            "false_alarm": corpus_false_alarm,
            "avg_time": sum(r.time for r in self.results) / len(self.results),
            "num_samples": len(self.results),
        }


class AssemblyAIDiarizationEvaluator(DiarizationEvaluator):
    """Evaluator for AssemblyAI speaker diarization."""

    MODEL_MAP = {"best": "best", "universal": "universal", "slam_1": "slam_1", "nano": "nano"}

    def __init__(self, api_key: str, model: str = "slam_1", **kwargs):
        # Remove hf_token since we don't use pyannote
        kwargs.pop("hf_token", None)
        super().__init__(**kwargs)
        import assemblyai as aai

        aai.settings.api_key = api_key

        if model not in self.MODEL_MAP:
            raise ValueError(f"Invalid model '{model}'. Choose from: {list(self.MODEL_MAP.keys())}")

        model_enum = getattr(aai.types.SpeechModel, model)
        config = aai.TranscriptionConfig(
            speech_model=model_enum,
            speaker_labels=True,
        )
        self.transcriber = aai.Transcriber(config=config)

    def diarize(self, audio) -> tuple[list[dict], float]:
        """Run AssemblyAI diarization on audio."""
        wav_bytes = prepare_wav_bytes(audio)

        start = time.time()
        transcript = self.transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start

        # Convert utterances to segment format
        segments = []
        if transcript.utterances:
            for utt in transcript.utterances:
                segments.append({
                    "speaker": utt.speaker,
                    "start": utt.start / 1000.0,  # ms -> seconds
                    "end": utt.end / 1000.0,
                })

        time.sleep(0.5)  # Rate limiting
        return segments, elapsed


def load_diarization_dataset(name: str, split: str, config_override: str | None = None):
    """Load a diarization dataset by name."""
    if name not in DIARIZATION_DATASET_REGISTRY:
        raise ValueError(
            f"Unknown diarization dataset: {name}. "
            f"Available: {list(DIARIZATION_DATASET_REGISTRY.keys())}"
        )

    cfg = DIARIZATION_DATASET_REGISTRY[name]
    config = config_override if config_override else cfg.config

    print(f"Loading {cfg.path} (config: {config}, split: {split})...")
    if config:
        ds = load_dataset(cfg.path, config, split=split, streaming=True)
    else:
        ds = load_dataset(cfg.path, split=split, streaming=True)

    # Use decode=False to avoid torchcodec CPU issues - we'll decode manually with librosa
    return ds.cast_column(cfg.audio_field, Audio(decode=False))


def save_diarization_results(
    output_path: Path,
    dataset_desc: str,
    results: list[DiarizationResult],
    metrics: dict,
):
    """Save diarization evaluation results to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write(f"Diarization Evaluation\n")
        f.write(f"Dataset: {dataset_desc}\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write(f"DER: {metrics['der']:.2f}%\n")
        f.write(f"  - Confusion: {metrics['confusion']:.2f}%\n")
        f.write(f"  - Missed Detection: {metrics['missed']:.2f}%\n")
        f.write(f"  - False Alarm: {metrics['false_alarm']:.2f}%\n")
        f.write(f"Avg Response Time: {metrics['avg_time']:.2f}s\n\n")
        f.write("=" * 80 + "\n")
        f.write("Per-Sample Results\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(results, 1):
            f.write(
                f"Sample {i} - DER: {r.der:.2f}% "
                f"(conf={r.confusion:.2f}%, miss={r.missed:.2f}%, fa={r.false_alarm:.2f}%), "
                f"Time: {r.time:.2f}s, "
                f"Speakers: ref={r.num_speakers_ref}, hyp={r.num_speakers_hyp}\n"
            )


def print_diarization_summary(dataset_desc: str, metrics: dict, output_path: Path):
    """Print final diarization evaluation summary."""
    print("\n" + "=" * 60)
    print("Diarization Evaluation Results")
    print("=" * 60)
    print(f"Dataset: {dataset_desc}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"DER: {metrics['der']:.2f}%")
    print(f"  - Confusion: {metrics['confusion']:.2f}%")
    print(f"  - Missed Detection: {metrics['missed']:.2f}%")
    print(f"  - False Alarm: {metrics['false_alarm']:.2f}%")
    print(f"Avg Time: {metrics['avg_time']:.2f}s")
    print(f"\nResults saved to: {output_path}")


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


def run_diarization_eval(args):
    """Run diarization evaluation."""
    # Set default dataset for diarization if not specified
    if args.dataset is None:
        args.dataset = "callhome"

    # Validate diarization dataset choice
    if args.dataset not in DIARIZATION_DATASET_REGISTRY:
        raise ValueError(
            f"Unknown diarization dataset: {args.dataset}. "
            f"Available: {list(DIARIZATION_DATASET_REGISTRY.keys())}"
        )

    cfg = DIARIZATION_DATASET_REGISTRY[args.dataset]
    split = args.split if args.split != "test" else cfg.default_split

    # Output directory
    model_suffix = f"_assemblyai_{args.assemblyai_model}" if args.assemblyai else "_pyannote"
    if args.output_dir is None:
        args.output_dir = Path(f"outputs/diarization_eval_{args.dataset}{model_suffix}")

    # Load dataset
    dataset = load_diarization_dataset(args.dataset, split, args.config)
    config_name = args.config or cfg.config
    dataset_desc = f"{cfg.path} (config: {config_name}, split: {split})"

    if args.max_samples:
        dataset = dataset.take(args.max_samples)

    # Create evaluator
    if args.assemblyai:
        if not args.api_key:
            raise ValueError("AssemblyAI API key required (--api-key or ASSEMBLYAI_API_KEY env var)")
        evaluator = AssemblyAIDiarizationEvaluator(
            api_key=args.api_key,
            model=args.assemblyai_model,
            audio_field=cfg.audio_field,
            speakers_field=cfg.speakers_field,
            timestamps_start_field=cfg.timestamps_start_field,
            timestamps_end_field=cfg.timestamps_end_field,
        )
        model_name = f"AssemblyAI ({args.assemblyai_model})"
    else:
        evaluator = DiarizationEvaluator(
            audio_field=cfg.audio_field,
            speakers_field=cfg.speakers_field,
            timestamps_start_field=cfg.timestamps_start_field,
            timestamps_end_field=cfg.timestamps_end_field,
            hf_token=os.environ.get("HF_TOKEN"),
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        model_name = "pyannote/speaker-diarization-3.1"

    # Run evaluation
    print(f"Running diarization evaluation with {model_name}...")
    evaluator.evaluate(dataset, args.max_samples)
    metrics = evaluator.compute_metrics()

    # Save and print results
    output_path = args.output_dir / "results.txt"
    save_diarization_results(output_path, dataset_desc, evaluator.results, metrics)
    print_diarization_summary(dataset_desc, metrics, output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ASR models and speaker diarization")
    parser.add_argument("model", nargs="?", help="Model path or endpoint URL (not needed for diarization)")
    parser.add_argument(
        "--task",
        default="asr",
        choices=["asr", "diarization"],
        help="Task to evaluate: 'asr' for speech recognition, 'diarization' for speaker diarization",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset to evaluate on (default: loquacious for ASR, callhome for diarization)",
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
    # Diarization-specific args
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers (diarization)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Min number of speakers (diarization)")
    parser.add_argument("--max-speakers", type=int, default=None, help="Max number of speakers (diarization)")
    args = parser.parse_args()

    # Handle diarization task
    if args.task == "diarization":
        run_diarization_eval(args)
        return

    # Set default dataset for ASR if not specified
    if args.dataset is None:
        args.dataset = "loquacious"

    # Validate ASR dataset choice
    valid_asr_datasets = list(DATASET_REGISTRY.keys()) + ["combined", "all"]
    if args.dataset not in valid_asr_datasets:
        raise ValueError(f"Unknown ASR dataset: {args.dataset}. Available: {valid_asr_datasets}")

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
