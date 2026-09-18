"""Base evaluator classes and shared utilities."""

import os
from enum import Enum

import attrs
import jiwer
from rich.console import Console

from scripts.eval.audio import TextNormalizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
console = Console()


class AssemblyAIModel(str, Enum):
    """AssemblyAI model options."""

    best = "best"
    universal = "universal"
    universal_3_pro = "universal-3-pro"
    # API name uses dashes for the decimal: "universal-3-5-pro", not "3.5".
    universal_3_5_pro = "universal-3-5-pro"


# Valid `model` values accepted by setup_assemblyai; the enum is the source.
ASSEMBLYAI_MODELS = {m.value for m in AssemblyAIModel}


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
    # Per-sample confidence stats from greedy decode (None when the evaluator
    # cannot expose per-token logits — external APIs, old in-tree pipelines).
    # `mean_top1_logprob`: average log-probability of emitted tokens.
    # `mean_margin`: average (top1 - top2) logprob gap across emitted tokens.
    # `num_tokens`: number of generated tokens contributing to the averages.
    # Aggregated to corpus-level in Evaluator.compute_metrics.
    mean_top1_logprob: float | None = None
    mean_margin: float | None = None
    num_tokens: int | None = None
    # Normalized forms, computed once in _process_sample. Everything that
    # scores or prints this result reads these instead of re-normalizing:
    # _corpus_wer runs at every checkpoint over all results so far, so
    # re-normalizing there made the eval loop quadratic in sample count.
    norm_prediction: str = attrs.field(
        default=attrs.Factory(lambda self: self.prediction, takes_self=True)
    )
    norm_reference: str = attrs.field(
        default=attrs.Factory(lambda self: self.reference, takes_self=True)
    )


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

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        """Transcribe audio, returning (text, inference_time, confidence).

        `confidence` is None for evaluators that cannot expose per-token
        logits (external APIs); evaluators with a scores-capable pipeline
        return a dict of per-sample stats. Override in subclass.
        """
        raise NotImplementedError

    def _process_sample(self, sample_data: tuple[int, dict]) -> tuple[int, EvalResult]:
        """Process a single sample. Returns (index, result) for ordering."""
        idx, sample = sample_data
        reference = sample["reference"]
        audio = sample["audio"]

        confidence: dict | None = None
        try:
            prediction, inference_time, confidence = self.transcribe(audio)
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            prediction, inference_time = "", 0.0

        norm_pred = self.normalizer.normalize(prediction)
        norm_ref = self.normalizer.normalize(reference)
        sample_wer = jiwer.wer(norm_ref, norm_pred) * 100 if norm_ref else 0.0
        confidence = confidence or {}

        return idx, EvalResult(
            prediction,
            reference,
            sample_wer,
            inference_time,
            mean_top1_logprob=confidence.get("mean_top1_logprob"),
            mean_margin=confidence.get("mean_margin"),
            num_tokens=confidence.get("num_tokens"),
            norm_prediction=norm_pred,
            norm_reference=norm_ref,
        )

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
        print(f"{prefix}Sample {idx}: WER={result.wer:.1f}%, Time={result.time:.2f}s")
        print(f"  Ref:  {result.norm_reference}")
        print(f"  Pred: {result.norm_prediction}")

    def _corpus_wer(self, results: list[EvalResult]) -> float:
        """Corpus WER over results whose reference survives normalization.

        `jiwer.wer` raises ValueError on an empty ground truth, and a
        reference can normalize to "" -- a disfluency-only utterance like
        "Uh." does. `_process_sample` already scores those as 0 rather than
        calling jiwer; without the same guard here a single such row takes
        down the whole run at the first checkpoint, after all the API spend.
        """
        pairs = [(r.norm_reference, r.norm_prediction) for r in results if r.norm_reference]
        if not pairs:
            return 0.0
        refs, preds = zip(*pairs, strict=True)
        return jiwer.wer(list(refs), list(preds)) * 100

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

        metrics = {
            "wer": self._corpus_wer(self.results),
            "avg_time": sum(r.time for r in self.results) / len(self.results),
            "num_samples": len(self.results),
        }

        # Corpus-level confidence aggregates: token-weighted means across samples
        # that have per-token stats (skips API evaluators that can't expose
        # logits). Weighting by num_tokens — not by sample count — gives the
        # right "average over emitted tokens" answer when sample lengths differ.
        weighted_logprob = 0.0
        weighted_margin = 0.0
        total_tokens = 0
        for r in self.results:
            if r.num_tokens and r.mean_top1_logprob is not None and r.mean_margin is not None:
                weighted_logprob += r.mean_top1_logprob * r.num_tokens
                weighted_margin += r.mean_margin * r.num_tokens
                total_tokens += r.num_tokens
        if total_tokens > 0:
            metrics["mean_top1_logprob"] = weighted_logprob / total_tokens
            metrics["mean_margin"] = weighted_margin / total_tokens
            metrics["total_tokens"] = total_tokens

        return metrics
