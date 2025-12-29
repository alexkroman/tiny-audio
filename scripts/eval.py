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
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import evaluate
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from transformers import WhisperTokenizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# Load WER metric once at module level
wer_metric = evaluate.load("wer")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# AssemblyAI model options
ASSEMBLYAI_MODELS = {"best", "universal", "slam_1", "nano"}


def setup_assemblyai(api_key: str, model: str, speaker_labels: bool = False):
    """Initialize AssemblyAI transcriber with given model."""
    import assemblyai as aai
    aai.settings.api_key = api_key
    if model not in ASSEMBLYAI_MODELS:
        raise ValueError(f"Invalid model '{model}'. Choose from: {ASSEMBLYAI_MODELS}")
    config = aai.TranscriptionConfig(speech_model=getattr(aai.types.SpeechModel, model), speaker_labels=speaker_labels)
    return aai.Transcriber(config=config)


# =============================================================================
# Audio Utilities
# =============================================================================


def audio_to_wav_bytes(audio_array: np.ndarray | torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes using soundfile."""
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.numpy()
    if audio_array.ndim > 1:
        audio_array = audio_array.squeeze()

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
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
            audio_array, sample_rate = sf.read(wav_data["path"])
            return audio_to_wav_bytes(audio_array, sample_rate)

    # Audio object with array/sampling_rate attributes
    if hasattr(wav_data, "array") and hasattr(wav_data, "sampling_rate"):
        return audio_to_wav_bytes(wav_data.array, wav_data.sampling_rate)

    # AudioDecoder - try to get path and load with soundfile
    if hasattr(wav_data, "path") and wav_data.path:
        audio_array, sample_rate = sf.read(wav_data.path)
        return audio_to_wav_bytes(audio_array, sample_rate)

    raise ValueError(f"Unsupported audio format: {type(wav_data)}")


# =============================================================================
# Text Normalization
# =============================================================================


class TextNormalizer:
    """Whisper-based text normalizer for ASR evaluation.

    Uses EnglishTextNormalizer which handles:
    - Lowercase and punctuation removal
    - Number normalization ("three" <-> "3")
    - British to American spelling ("colour" -> "color")
    - Disfluency removal ("uh", "um", "hmm")
    - Tag removal ("<inaudible>", etc.)
    """

    def __init__(self):
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        self._normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)

    def normalize(self, text: str) -> str:
        """Normalize text for WER calculation."""
        return self._normalizer(text)


# =============================================================================
# Dataset Configuration (Unified)
# =============================================================================


@dataclass
class DatasetConfig:
    """Unified configuration for all dataset types."""

    name: str
    path: str
    audio_field: str
    text_field: str = "text"
    config: str | None = None
    default_split: str = "test"
    weight: float = 1.0
    # Diarization-specific
    speakers_field: str | None = None
    timestamps_start_field: str | None = None
    timestamps_end_field: str | None = None
    # Alignment-specific
    words_field: str | None = None


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    # ASR datasets
    "loquacious": DatasetConfig(
        name="loquacious", path="speechbrain/LoquaciousSet", config="small",
        audio_field="wav", text_field="text",
    ),
    "earnings22": DatasetConfig(
        name="earnings22", path="sanchit-gandhi/earnings22_robust_split", config="default",
        audio_field="audio", text_field="sentence",
    ),
    "ami": DatasetConfig(
        name="ami", path="TakalaWang/AMI_ASR", audio_field="audio", text_field="text",
    ),
    "gigaspeech": DatasetConfig(
        name="gigaspeech", path="fixie-ai/gigaspeech", config="dev",
        audio_field="audio", text_field="text", default_split="dev",
    ),
    "tedlium": DatasetConfig(
        name="tedlium", path="sanchit-gandhi/tedlium-data", config="default",
        audio_field="audio", text_field="text",
    ),
    "commonvoice": DatasetConfig(
        name="commonvoice", path="fixie-ai/common_voice_17_0", config="en",
        audio_field="audio", text_field="sentence",
    ),
    "peoples": DatasetConfig(
        name="peoples", path="fixie-ai/peoples_speech", config="clean",
        audio_field="audio", text_field="text",
    ),
    "librispeech": DatasetConfig(
        name="librispeech", path="openslr/librispeech_asr", config="clean",
        audio_field="audio", text_field="text",
    ),
    "librispeech-other": DatasetConfig(
        name="librispeech-other", path="openslr/librispeech_asr", config="other",
        audio_field="audio", text_field="text",
    ),
    # Diarization datasets
    "callhome": DatasetConfig(
        name="callhome", path="talkbank/callhome", config="eng",
        audio_field="audio", text_field="text", default_split="data",
        speakers_field="speakers", timestamps_start_field="timestamps_start",
        timestamps_end_field="timestamps_end",
    ),
    # Alignment datasets
    "librispeech-alignments": DatasetConfig(
        name="librispeech-alignments", path="gilkeyio/librispeech-alignments",
        audio_field="audio", text_field="transcript", default_split="dev_clean",
        words_field="words",
    ),
}

COMBINED_WEIGHTS = {"loquacious": 0.50, "gigaspeech": 0.10, "earnings22": 0.10, "ami": 0.10, "tedlium": 0.10}
DIARIZATION_DATASETS = {"callhome"}
ALIGNMENT_DATASETS = {"librispeech-alignments"}


def load_eval_dataset(name: str, split: str, config_override: str | None = None, decode_audio: bool = True):
    """Load any dataset by name with unified interface."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    cfg = DATASET_REGISTRY[name]
    config = config_override or cfg.config

    print(f"Loading {cfg.path} (config: {config}, split: {split})...")
    ds = load_dataset(cfg.path, config, split=split, streaming=True) if config else load_dataset(cfg.path, split=split, streaming=True)
    audio_opts = Audio(sampling_rate=16000) if decode_audio else Audio(decode=False)
    return ds.cast_column(cfg.audio_field, audio_opts)


def load_combined_dataset(max_samples: int | None, seed: int) -> tuple[Iterator, int]:
    """Load proportionally sampled combined dataset."""
    import random
    random.seed(seed)

    total_samples = max_samples or 1000
    samples_per_dataset = {name: max(1, int(total_samples * weight)) for name, weight in COMBINED_WEIGHTS.items()}
    print(f"Loading combined dataset: {samples_per_dataset}")

    all_samples = []
    for name, num_samples in samples_per_dataset.items():
        cfg = DATASET_REGISTRY[name]
        validation_split = "dev" if name == "loquacious" else "validation"
        ds = load_eval_dataset(name, validation_split).shuffle(seed=seed, buffer_size=num_samples * 10)

        count = 0
        for sample in ds:
            text = sample.get(cfg.text_field, "")
            if isinstance(text, str) and text.strip() == "ignore_time_segment_in_scoring":
                continue
            all_samples.append({"audio": sample[cfg.audio_field], "text": sample[cfg.text_field], "source": name})
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
            sample_wer = (
                wer_metric.compute(predictions=[norm_pred], references=[norm_ref]) * 100
                if norm_ref
                else 0.0
            )

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
        corpus_wer = wer_metric.compute(predictions=preds, references=refs) * 100
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
            "wer": wer_metric.compute(predictions=preds, references=refs) * 100,
            "avg_time": sum(r.time for r in self.results) / len(self.results),
            "num_samples": len(self.results),
        }


class LocalEvaluator(Evaluator):
    """Evaluator for local models."""

    def __init__(self, model_path: str, user_prompt: str | None = None, **kwargs):
        super().__init__(**kwargs)
        from src.asr_modeling import ASRModel
        from src.asr_pipeline import ASRPipeline

        # Load model and use our custom pipeline
        model = ASRModel.from_pretrained(model_path)
        self.pipe = ASRPipeline(model=model)
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

    def __init__(self, api_key: str, model: str = "slam_1", **kwargs):
        super().__init__(**kwargs)
        self.transcriber = setup_assemblyai(api_key, model)

    def transcribe(self, audio) -> tuple[str, float]:
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()
        transcript = self.transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start
        time.sleep(0.5)
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
                    num_speakers_hyp=len({seg["speaker"] for seg in segments}),
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

    def __init__(self, api_key: str, model: str = "slam_1", **kwargs):
        kwargs.pop("hf_token", None)
        super().__init__(**kwargs)
        self.transcriber = setup_assemblyai(api_key, model, speaker_labels=True)

    def diarize(self, audio) -> tuple[list[dict], float]:
        """Run AssemblyAI diarization on audio."""
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()
        transcript = self.transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start
        segments = [{"speaker": u.speaker, "start": u.start / 1000.0, "end": u.end / 1000.0} for u in (transcript.utterances or [])]
        time.sleep(0.5)
        return segments, elapsed


def save_diarization_results(
    output_path: Path,
    dataset_desc: str,
    results: list[DiarizationResult],
    metrics: dict,
):
    """Save diarization evaluation results to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write("Diarization Evaluation\n")
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
# Timestamp Alignment Evaluation
# =============================================================================


@dataclass
class AlignmentResult:
    """Result of a single timestamp alignment evaluation."""

    pred_starts: list[float]  # Predicted start times for aligned words
    pred_ends: list[float]  # Predicted end times for aligned words
    ref_starts: list[float]  # Reference start times for aligned words
    ref_ends: list[float]  # Reference end times for aligned words
    num_aligned_words: int  # Number of words successfully aligned
    num_ref_words: int  # Number of reference words
    num_pred_words: int  # Number of predicted words
    time: float  # Inference time
    reference_text: str
    predicted_text: str


def align_words_to_reference(
    pred_words: list[dict],
    ref_words: list[dict],
    normalizer: TextNormalizer,
) -> list[tuple[dict, dict]]:
    """Align predicted words to reference words using normalized text matching.

    Uses a simple greedy alignment: for each reference word, find the best
    matching predicted word that hasn't been used yet.

    Returns:
        List of (pred_word, ref_word) tuples for successfully aligned words.
    """
    aligned_pairs = []

    # Normalize reference words
    ref_normalized = []
    for rw in ref_words:
        word = rw.get("word", "")
        # Skip <unk> tokens in reference
        if word == "<unk>":
            continue
        norm = normalizer.normalize(word).strip()
        if norm:
            ref_normalized.append((norm, rw))

    # Normalize predicted words
    pred_normalized = []
    for pw in pred_words:
        word = pw.get("word", "")
        norm = normalizer.normalize(word).strip()
        if norm:
            pred_normalized.append((norm, pw))

    # Greedy alignment: for each ref word, find matching pred word
    used_pred_indices = set()

    for ref_norm, ref_word in ref_normalized:
        best_match_idx = None
        best_time_diff = float("inf")

        for i, (pred_norm, pred_word) in enumerate(pred_normalized):
            if i in used_pred_indices:
                continue

            # Check if normalized words match
            if pred_norm == ref_norm:
                # Prefer matches closer in time
                time_diff = abs(pred_word.get("start", 0) - ref_word.get("start", 0))
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_match_idx = i

        if best_match_idx is not None:
            used_pred_indices.add(best_match_idx)
            aligned_pairs.append((pred_normalized[best_match_idx][1], ref_word))

    return aligned_pairs


class BaseAlignmentEvaluator:
    """Base class for timestamp alignment evaluators."""

    def __init__(
        self,
        audio_field: str = "audio",
        text_field: str = "transcript",
        words_field: str = "words",
    ):
        self.audio_field = audio_field
        self.text_field = text_field
        self.words_field = words_field
        self.normalizer = TextNormalizer()
        self.results: list[AlignmentResult] = []

    def transcribe_with_timestamps(self, audio) -> tuple[str, list[dict], float]:
        """Transcribe audio and return (text, word_timestamps, inference_time)."""
        raise NotImplementedError

    def evaluate(self, dataset, max_samples: int | None = None) -> list[AlignmentResult]:
        """Run timestamp alignment evaluation loop on dataset."""
        self.results = []
        processed = 0

        for sample in dataset:
            processed += 1
            if max_samples and processed > max_samples:
                break

            ref_text = sample[self.text_field]
            ref_words = sample[self.words_field]

            try:
                pred_text, pred_words, inference_time = self.transcribe_with_timestamps(
                    sample[self.audio_field]
                )
            except Exception as e:
                print(f"Error on sample {processed}: {e}")
                continue

            # Align predicted words to reference words
            aligned_pairs = align_words_to_reference(pred_words, ref_words, self.normalizer)

            if not aligned_pairs:
                print(f"Sample {processed}: No words aligned (ref={len(ref_words)}, pred={len(pred_words)})")
                result = AlignmentResult(
                    pred_starts=[],
                    pred_ends=[],
                    ref_starts=[],
                    ref_ends=[],
                    num_aligned_words=0,
                    num_ref_words=len(ref_words),
                    num_pred_words=len(pred_words),
                    time=inference_time,
                    reference_text=ref_text,
                    predicted_text=pred_text,
                )
                self.results.append(result)
                continue

            # Extract timestamps from aligned pairs
            pred_starts = []
            pred_ends = []
            ref_starts = []
            ref_ends = []

            for pred_word, ref_word in aligned_pairs:
                pred_starts.append(pred_word.get("start", 0))
                pred_ends.append(pred_word.get("end", 0))
                ref_starts.append(ref_word.get("start", 0))
                ref_ends.append(ref_word.get("end", 0))

            result = AlignmentResult(
                pred_starts=pred_starts,
                pred_ends=pred_ends,
                ref_starts=ref_starts,
                ref_ends=ref_ends,
                num_aligned_words=len(aligned_pairs),
                num_ref_words=len(ref_words),
                num_pred_words=len(pred_words),
                time=inference_time,
                reference_text=ref_text,
                predicted_text=pred_text,
            )
            self.results.append(result)

            # Compute sample-level MAE for logging
            mae_start = sum(abs(p - r) for p, r in zip(pred_starts, ref_starts)) / len(pred_starts)
            mae_end = sum(abs(p - r) for p, r in zip(pred_ends, ref_ends)) / len(pred_ends)
            mae_combined = (mae_start + mae_end) / 2

            print(
                f"Sample {processed}: MAE={mae_combined*1000:.1f}ms, "
                f"Aligned={len(aligned_pairs)}/{len(ref_words)} words, "
                f"Time={inference_time:.2f}s"
            )

            # Checkpoint every 50 samples
            if processed % 50 == 0:
                self._print_checkpoint(processed)

        return self.results

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        metrics = self.compute_metrics()
        print(f"\n{'=' * 60}")
        print(
            f"CHECKPOINT @ {sample_count}: MAE={metrics['mae']*1000:.1f}ms, "
            f"Alignment Error={metrics['alignment_error']*100:.1f}%"
        )
        print(f"{'=' * 60}\n")

    def compute_metrics(self) -> dict:
        """Compute final corpus-level metrics using evaluate library."""
        if not self.results:
            return {
                "mae": 0.0,
                "alignment_error": 1.0,
                "mae_adjusted": float("inf"),
                "avg_time": 0.0,
                "num_samples": 0,
            }

        # Collect all predictions and references across samples
        all_pred_times = []
        all_ref_times = []
        total_aligned = 0
        total_ref = 0

        for r in self.results:
            # Combine start and end times for MAE calculation
            all_pred_times.extend(r.pred_starts)
            all_pred_times.extend(r.pred_ends)
            all_ref_times.extend(r.ref_starts)
            all_ref_times.extend(r.ref_ends)
            total_aligned += r.num_aligned_words
            total_ref += r.num_ref_words

        if not all_pred_times:
            return {
                "mae": float("nan"),
                "alignment_error": 1.0,
                "mae_adjusted": float("inf"),
                "avg_time": sum(r.time for r in self.results) / len(self.results),
                "num_samples": len(self.results),
            }

        # Direct MAE calculation
        mae = sum(abs(p - r) for p, r in zip(all_pred_times, all_ref_times)) / len(all_pred_times)

        # Direct alignment rate calculation
        alignment_rate = total_aligned / total_ref if total_ref > 0 else 0.0
        alignment_error = 1.0 - alignment_rate

        return {
            "mae": mae,
            "alignment_error": alignment_error,
            "alignment_rate": alignment_rate,
            "mae_adjusted": mae / alignment_rate if alignment_rate > 0 else float("inf"),
            "total_aligned_words": total_aligned,
            "total_ref_words": total_ref,
            "avg_time": sum(r.time for r in self.results) / len(self.results),
            "num_samples": len(self.results),
        }


class TimestampAlignmentEvaluator(BaseAlignmentEvaluator):
    """Evaluator for word-level timestamp alignment accuracy using local models."""

    def __init__(
        self,
        model_path: str,
        audio_field: str = "audio",
        text_field: str = "transcript",
        words_field: str = "words",
        user_prompt: str | None = None,
    ):
        super().__init__(audio_field, text_field, words_field)
        self.user_prompt = user_prompt

        # Load model and pipeline
        from src.asr_modeling import ASRModel
        from src.asr_pipeline import ASRPipeline

        model = ASRModel.from_pretrained(model_path)
        self.pipe = ASRPipeline(model=model)

    def transcribe_with_timestamps(self, audio) -> tuple[str, list[dict], float]:
        """Transcribe audio and return (text, word_timestamps, inference_time)."""
        # Convert to pipeline-compatible format
        if isinstance(audio, dict) and "array" in audio and "raw" not in audio:
            audio_input = {"raw": audio["array"], "sampling_rate": audio["sampling_rate"]}
        else:
            audio_input = audio

        start = time.time()
        if self.user_prompt:
            result = self.pipe(audio_input, return_timestamps=True, user_prompt=self.user_prompt)
        else:
            result = self.pipe(audio_input, return_timestamps=True)
        elapsed = time.time() - start

        text = result.get("text", "")
        words = result.get("words", [])

        return text, words, elapsed


class AssemblyAIAlignmentEvaluator(BaseAlignmentEvaluator):
    """Evaluator for word-level timestamp alignment accuracy using AssemblyAI."""

    def __init__(self, api_key: str, model: str = "slam_1", audio_field: str = "audio",
                 text_field: str = "transcript", words_field: str = "words"):
        super().__init__(audio_field, text_field, words_field)
        self.transcriber = setup_assemblyai(api_key, model)

    def transcribe_with_timestamps(self, audio) -> tuple[str, list[dict], float]:
        """Transcribe audio and return (text, word_timestamps, inference_time)."""
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()
        transcript = self.transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start
        words = [{"word": w.text, "start": w.start / 1000.0, "end": w.end / 1000.0} for w in (transcript.words or [])]
        time.sleep(0.5)
        return transcript.text or "", words, elapsed


def save_alignment_results(
    output_path: Path,
    model_name: str,
    dataset_desc: str,
    results: list[AlignmentResult],
    metrics: dict,
):
    """Save timestamp alignment evaluation results to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write("Timestamp Alignment Evaluation\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_desc}\n")
        f.write(f"Samples: {metrics['num_samples']}\n\n")
        f.write(f"MAE: {metrics['mae']*1000:.1f}ms @ {metrics['alignment_rate']*100:.1f}% alignment\n")
        f.write(f"MAE (adjusted): {metrics['mae_adjusted']*1000:.1f}ms\n")
        f.write(f"Alignment Error: {metrics['alignment_error']*100:.1f}%\n\n")
        f.write("=" * 80 + "\n")
        f.write("Per-Sample Results\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(results, 1):
            # Compute sample-level MAE from raw timestamps
            if r.pred_starts:
                mae = sum(
                    abs(p - ref)
                    for p, ref in zip(r.pred_starts + r.pred_ends, r.ref_starts + r.ref_ends)
                ) / (2 * len(r.pred_starts))
                f.write(
                    f"Sample {i} - MAE: {mae*1000:.1f}ms, "
                    f"Aligned: {r.num_aligned_words}/{r.num_ref_words}, Time: {r.time:.2f}s\n"
                )
            else:
                f.write(
                    f"Sample {i} - No alignment, "
                    f"Ref words: {r.num_ref_words}, Time: {r.time:.2f}s\n"
                )
            f.write(f"  Ref:  {r.reference_text}\n")
            f.write(f"  Pred: {r.predicted_text}\n\n")


def print_alignment_summary(model_name: str, dataset_desc: str, metrics: dict, output_path: Path):
    """Print final timestamp alignment evaluation summary."""
    print("\n" + "=" * 60)
    print("Timestamp Alignment Evaluation Results")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_desc}")
    print(f"Samples: {metrics['num_samples']}")
    print()
    print(f"MAE: {metrics['mae']*1000:.1f}ms @ {metrics['alignment_rate']*100:.1f}% alignment")
    print(f"MAE (adjusted): {metrics['mae_adjusted']*1000:.1f}ms")
    print(f"Alignment Error: {metrics['alignment_error']*100:.1f}%")
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

    # Skip diarization and alignment datasets (they require special evaluation modes)
    asr_datasets = {k: v for k, v in DATASET_REGISTRY.items() if k not in DIARIZATION_DATASETS and k not in ALIGNMENT_DATASETS}

    for dataset_name, cfg in asr_datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'=' * 60}\n")

        try:
            # Load dataset
            split = args.split if args.split != "test" else cfg.default_split
            dataset = load_eval_dataset(dataset_name, split, args.config)
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
    if args.dataset is None:
        args.dataset = "callhome"

    if args.dataset not in DIARIZATION_DATASETS:
        raise ValueError(f"Unknown diarization dataset: {args.dataset}. Available: {DIARIZATION_DATASETS}")

    cfg = DATASET_REGISTRY[args.dataset]
    split = args.split if args.split != "test" else cfg.default_split

    model_suffix = f"_assemblyai_{args.assemblyai_model}" if args.assemblyai else "_pyannote"
    if args.output_dir is None:
        args.output_dir = Path(f"outputs/diarization_eval_{args.dataset}{model_suffix}")

    dataset = load_eval_dataset(args.dataset, split, args.config, decode_audio=False)
    dataset_desc = f"{cfg.path} (config: {args.config or cfg.config}, split: {split})"

    if args.max_samples:
        dataset = dataset.take(args.max_samples)

    if args.assemblyai:
        if not args.api_key:
            raise ValueError("AssemblyAI API key required (--api-key or ASSEMBLYAI_API_KEY env var)")
        evaluator = AssemblyAIDiarizationEvaluator(
            api_key=args.api_key, model=args.assemblyai_model, audio_field=cfg.audio_field,
            speakers_field=cfg.speakers_field, timestamps_start_field=cfg.timestamps_start_field,
            timestamps_end_field=cfg.timestamps_end_field,
        )
        model_name = f"AssemblyAI ({args.assemblyai_model})"
    else:
        evaluator = DiarizationEvaluator(
            audio_field=cfg.audio_field, speakers_field=cfg.speakers_field,
            timestamps_start_field=cfg.timestamps_start_field, timestamps_end_field=cfg.timestamps_end_field,
            hf_token=os.environ.get("HF_TOKEN"), num_speakers=args.num_speakers,
            min_speakers=args.min_speakers, max_speakers=args.max_speakers,
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


def run_alignment_eval(args):
    """Run timestamp alignment evaluation."""
    if args.assemblyai:
        if not args.api_key:
            raise ValueError("AssemblyAI API key required (--api-key or ASSEMBLYAI_API_KEY env var)")
        model_name = f"AssemblyAI ({args.assemblyai_model})"
    elif args.model:
        model_name = args.model
    else:
        raise ValueError("Model argument required for alignment evaluation (or use --assemblyai)")

    if args.dataset is None:
        args.dataset = "librispeech-alignments"

    if args.dataset not in ALIGNMENT_DATASETS:
        raise ValueError(f"Unknown alignment dataset: {args.dataset}. Available: {ALIGNMENT_DATASETS}")

    cfg = DATASET_REGISTRY[args.dataset]
    split = args.split if args.split != "test" else cfg.default_split

    if args.output_dir is None:
        safe_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        args.output_dir = Path(f"outputs/alignment_eval_{args.dataset}_{safe_name}")

    dataset = load_eval_dataset(args.dataset, split, args.config)
    dataset_desc = f"{cfg.path} (config: {args.config or cfg.config}, split: {split})"

    if args.max_samples:
        dataset = dataset.take(args.max_samples)

    if args.assemblyai:
        evaluator = AssemblyAIAlignmentEvaluator(
            api_key=args.api_key, model=args.assemblyai_model, audio_field=cfg.audio_field,
            text_field=cfg.text_field, words_field=cfg.words_field,
        )
    else:
        evaluator = TimestampAlignmentEvaluator(
            model_path=model_name, audio_field=cfg.audio_field, text_field=cfg.text_field,
            words_field=cfg.words_field, user_prompt=args.user_prompt,
        )

    # Run evaluation
    print(f"Running timestamp alignment evaluation with {model_name}...")
    evaluator.evaluate(dataset, args.max_samples)
    metrics = evaluator.compute_metrics()

    # Save and print results
    output_path = args.output_dir / "results.txt"
    save_alignment_results(output_path, model_name, dataset_desc, evaluator.results, metrics)
    print_alignment_summary(model_name, dataset_desc, metrics, output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ASR models and speaker diarization")
    parser.add_argument(
        "model", nargs="?", help="Model path or endpoint URL (not needed for diarization)"
    )
    parser.add_argument(
        "--task",
        default="asr",
        choices=["asr", "diarization", "alignment"],
        help="Task to evaluate: 'asr' for speech recognition, 'diarization' for speaker diarization, 'alignment' for timestamp alignment",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset to evaluate on (default: loquacious for ASR, callhome for diarization, librispeech-alignments for alignment)",
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
    parser.add_argument(
        "--num-speakers", type=int, default=None, help="Exact number of speakers (diarization)"
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None, help="Min number of speakers (diarization)"
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None, help="Max number of speakers (diarization)"
    )
    args = parser.parse_args()

    # Handle diarization task
    if args.task == "diarization":
        run_diarization_eval(args)
        return

    # Handle alignment task
    if args.task == "alignment":
        run_alignment_eval(args)
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
        dataset = load_eval_dataset(args.dataset, split, args.config)
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
