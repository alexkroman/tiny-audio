"""Base evaluator classes and shared utilities."""

import os

import attrs
import jiwer
from rich.console import Console

from scripts.eval.audio import TextNormalizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
console = Console()

# AssemblyAI model options
ASSEMBLYAI_MODELS = {"best", "universal", "universal-3-pro"}


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
        speech_models=[model],
        speaker_labels=speaker_labels,
    )
    return aai.Transcriber(config=config)


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


def _is_skipped_reference(reference) -> bool:
    """Filter out unscoreable samples (TEDLIUM markers, inaudible)."""
    if not isinstance(reference, str):
        return False
    return reference.strip() == "ignore_time_segment_in_scoring" or "inaudible" in reference.lower()


class Evaluator:
    """Base evaluator with common evaluation loop logic."""

    def __init__(self, audio_field: str = "audio", text_field: str = "text", num_workers: int = 1):
        self.audio_field = audio_field
        self.text_field = text_field
        self.num_workers = num_workers
        self.normalizer = TextNormalizer()
        self.results: list[EvalResult] = []

    def transcribe(self, audio) -> tuple[str, float]:
        """Transcribe audio and return (text, inference_time). Override in subclass."""
        raise NotImplementedError

    def _process_sample(self, sample_data: tuple[int, dict]) -> tuple[int, EvalResult]:
        """Process a single sample. Returns (index, result) for ordering."""
        idx, sample = sample_data
        reference = sample["reference"]
        audio = sample["audio"]

        try:
            prediction, inference_time = self.transcribe(audio)
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            prediction, inference_time = "", 0.0

        norm_pred = self.normalizer.normalize(prediction)
        norm_ref = self.normalizer.normalize(reference)
        sample_wer = jiwer.wer(norm_ref, norm_pred) * 100 if norm_ref else 0.0

        return idx, EvalResult(prediction, reference, sample_wer, inference_time)

    def evaluate(self, dataset, max_samples: int | None = None) -> list[EvalResult]:
        """Run evaluation loop on dataset."""
        self.results = []

        if self.num_workers > 1:
            # Parallel processing requires pre-collecting samples
            samples_to_process = self._collect_samples(dataset, max_samples)
            self._evaluate_parallel(samples_to_process)
        else:
            # Sequential: process lazily to avoid slow collection for streaming datasets
            self._evaluate_sequential_lazy(dataset, max_samples)

        return self.results

    def _iter_dataset_samples(self, dataset):
        """Yield (audio, reference) for samples that pass the skip filter."""
        for sample in dataset:
            reference = sample[self.text_field]
            if _is_skipped_reference(reference):
                continue
            yield {"audio": sample[self.audio_field], "reference": reference}

    def _print_sample_log(self, prefix: str, idx: int, result: EvalResult) -> None:
        norm_pred = self.normalizer.normalize(result.prediction)
        norm_ref = self.normalizer.normalize(result.reference)
        print(f"{prefix}Sample {idx}: WER={result.wer:.1f}%, Time={result.time:.2f}s")
        print(f"  Ref:  {norm_ref}")
        print(f"  Pred: {norm_pred}")

    def _corpus_wer(self, results: list[EvalResult]) -> float:
        preds = [self.normalizer.normalize(r.prediction) for r in results]
        refs = [self.normalizer.normalize(r.reference) for r in results]
        return jiwer.wer(refs, preds) * 100

    def _collect_samples(self, dataset, max_samples: int | None) -> list[dict]:
        """Collect samples for parallel processing."""
        samples_to_process = []
        target = max_samples or "all"
        console.print(f"[dim]Collecting samples (target: {target})...[/dim]")
        for s in self._iter_dataset_samples(dataset):
            samples_to_process.append(s)
            count = len(samples_to_process)
            if count % 50 == 0 or count == max_samples:
                console.print(f"[dim]  Collected {count} samples...[/dim]")
            if max_samples and count >= max_samples:
                break

        console.print(
            f"[dim]Collected {len(samples_to_process)} samples, starting evaluation...[/dim]"
        )
        return samples_to_process

    def _evaluate_sequential_lazy(self, dataset, max_samples: int | None) -> None:
        """Run sequential evaluation lazily (no pre-collection)."""
        for idx, sample_data in enumerate(self._iter_dataset_samples(dataset), start=1):
            _, result = self._process_sample((idx, sample_data))
            self.results.append(result)
            self._print_sample_log("", idx, result)

            if idx % 100 == 0:
                avg_time = sum(r.time for r in self.results) / len(self.results)
                console.print(
                    f"\n[bold]CHECKPOINT @ {idx}[/bold]: "
                    f"WER={self._corpus_wer(self.results):.2f}%, Avg Time={avg_time:.2f}s\n"
                )

            if max_samples and idx >= max_samples:
                break

    def _evaluate_parallel(self, samples: list[dict]) -> None:
        """Run parallel evaluation using thread pool."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        console.print(f"[bold]Running parallel evaluation with {self.num_workers} workers[/bold]")

        results_map: dict[int, EvalResult] = {}
        completed = 0
        total = len(samples)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_sample, (idx, sample)): idx
                for idx, sample in enumerate(samples, 1)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results_map[idx] = result
                completed += 1

                self._print_sample_log(f"[{completed}/{total}] ", idx, result)

                if completed % 100 == 0:
                    corpus_wer = self._corpus_wer(list(results_map.values()))
                    console.print(
                        f"\n[bold]CHECKPOINT @ {completed}[/bold]: WER={corpus_wer:.2f}%\n"
                    )

        self.results = [results_map[i] for i in sorted(results_map.keys())]

    def compute_metrics(self) -> dict:
        """Compute final metrics."""
        if not self.results:
            return {"wer": 0.0, "avg_time": 0.0, "num_samples": 0}

        return {
            "wer": self._corpus_wer(self.results),
            "avg_time": sum(r.time for r in self.results) / len(self.results),
            "num_samples": len(self.results),
        }
