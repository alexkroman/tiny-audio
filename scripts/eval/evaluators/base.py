"""Base evaluator classes and shared utilities."""

import os

import attrs
import jiwer
from rich.console import Console

from scripts.eval.audio import TextNormalizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
console = Console()

# AssemblyAI model options
ASSEMBLYAI_MODELS = {"best", "universal", "slam_1", "nano"}


def setup_assemblyai(
    api_key: str, model: str, speaker_labels: bool = False, base_url: str | None = None
):
    """Initialize AssemblyAI transcriber with given model."""
    import assemblyai as aai

    aai.settings.api_key = api_key
    if base_url:
        aai.settings.base_url = base_url
    if model not in ASSEMBLYAI_MODELS:
        raise ValueError(f"Invalid model '{model}'. Choose from: {ASSEMBLYAI_MODELS}")
    config = aai.TranscriptionConfig(
        speech_model=getattr(aai.types.SpeechModel, model), speaker_labels=speaker_labels
    )
    return aai.Transcriber(config=config)


# =============================================================================
# Result Types (using attrs for conciseness)
# =============================================================================


@attrs.define
class EvalResult:
    """Result of a single ASR sample evaluation."""

    prediction: str
    reference: str
    wer: float
    time: float


@attrs.define
class DiarizationResult:
    """Result of a single diarization evaluation."""

    der: float
    confusion: float
    missed: float
    false_alarm: float
    time: float
    num_speakers_ref: int
    num_speakers_hyp: int
    total: float = 0.0
    confusion_raw: float = 0.0
    missed_raw: float = 0.0
    false_alarm_raw: float = 0.0


@attrs.define
class AlignmentResult:
    """Result of a single timestamp alignment evaluation."""

    pred_starts: list[float]
    pred_ends: list[float]
    ref_starts: list[float]
    ref_ends: list[float]
    num_aligned_words: int
    num_ref_words: int
    num_pred_words: int
    time: float
    reference_text: str
    predicted_text: str


# =============================================================================
# Base Evaluator
# =============================================================================


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

            # Skip samples marked as inaudible
            if isinstance(reference, str) and "inaudible" in reference.lower():
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
            sample_wer = jiwer.wer(norm_ref, norm_pred) * 100 if norm_ref else 0.0

            result = EvalResult(prediction, reference, sample_wer, inference_time)
            self.results.append(result)

            print(f"Sample {processed}: WER={sample_wer:.1f}%, Time={inference_time:.2f}s")
            print(f"  Ref:  {norm_ref}")
            print(f"  Pred: {norm_pred}")

            # Checkpoint every 100 samples
            if processed % 100 == 0:
                self._print_checkpoint(processed)

        return self.results

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        preds = [self.normalizer.normalize(r.prediction) for r in self.results]
        refs = [self.normalizer.normalize(r.reference) for r in self.results]
        corpus_wer = jiwer.wer(refs, preds) * 100
        avg_time = sum(r.time for r in self.results) / len(self.results)
        console.print(
            f"\n[bold]CHECKPOINT @ {sample_count}[/bold]: WER={corpus_wer:.2f}%, Avg Time={avg_time:.2f}s\n"
        )

    def compute_metrics(self) -> dict:
        """Compute final metrics."""
        if not self.results:
            return {"wer": 0.0, "avg_time": 0.0, "num_samples": 0}

        preds = [self.normalizer.normalize(r.prediction) for r in self.results]
        refs = [self.normalizer.normalize(r.reference) for r in self.results]

        return {
            "wer": jiwer.wer(refs, preds) * 100,
            "avg_time": sum(r.time for r in self.results) / len(self.results),
            "num_samples": len(self.results),
        }
