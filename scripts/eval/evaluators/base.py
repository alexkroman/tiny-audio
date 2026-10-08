"""Base evaluator classes and shared utilities."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import assemblyai as aai
import attrs
import jiwer
from rich.console import Console

from scripts.eval.audio import TextNormalizer
from scripts.eval.constants import ASSEMBLYAI_MODELS
from scripts.eval.formatting import compute_formatting_metrics
from scripts.eval.speaker_metrics import has_speakers, plain_text, speaker_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"
console = Console()


def setup_assemblyai(
    api_key: str, model: str, speaker_labels: bool = False, base_url: str | None = None
):
    """Initialize AssemblyAI transcriber with given model."""
    aai.settings.api_key = api_key
    if base_url:
        aai.settings.base_url = base_url
    if model not in ASSEMBLYAI_MODELS:
        msg = f"Invalid model '{model}'. Choose from: {ASSEMBLYAI_MODELS}"
        raise ValueError(msg)
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


def _scoring_text(text: str) -> str:
    """Text WER is computed on: speaker tokens (`<SPK_n>`) dropped, if present.

    Speaker-labelled references (`ami-speakers`) are scored for words here and
    for attribution by cpWER in `compute_metrics`; a system that emits no
    speaker tokens is scored on the same words.
    """
    return plain_text(text) if has_speakers(text) else text


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
        # Set per `evaluate` call: the dataset's references are `<SPK_n>`
        # speaker-attributed, so evaluators that can should emit speakers too.
        self.speakers = False
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

        norm_pred = self.normalizer.normalize(_scoring_text(prediction))
        norm_ref = self.normalizer.normalize(_scoring_text(reference))
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

    def _reset_run_state(self) -> None:
        """Clear per-dataset accumulators before an `evaluate` call.

        The base class only accumulates `results`. Subclasses that collect
        anything else across `transcribe` calls must extend this -- a single
        evaluator instance is now reused for every dataset in a sweep (see
        `scripts/eval/cli.py`), so anything not reset here silently pools
        across corpora.
        """
        self.results = []

    def evaluate(
        self,
        dataset,
        max_samples: int | None = None,
        *,
        audio_field: str | None = None,
        text_field: str | None = None,
        speakers: bool = False,
    ) -> list[EvalResult]:
        """Run evaluation loop on dataset.

        `audio_field` / `text_field` override the column names given at
        construction, for the duration of this call onward. They exist so one
        evaluator -- and therefore one loaded model, one Swift subprocess, one
        authorized SFSpeechRecognizer -- can be reused across datasets whose
        columns differ (`wav` vs `audio`, `text` vs `sentence` vs
        `transcript`). Omit both to keep the constructor's values.

        `speakers` marks a dataset whose references are speaker-attributed
        (`<SPK_1>text<SPK_2>text`): evaluators that can label speakers do so,
        and `compute_metrics` adds cpWER.
        """
        self._reset_run_state()
        self.speakers = speakers
        if audio_field is not None:
            self.audio_field = audio_field
        if text_field is not None:
            self.text_field = text_field

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

        # Casing / punctuation, scored on RAW text. `wer` above is computed
        # after Whisper's normalizer lowercases and strips punctuation from
        # both sides, so it is structurally blind to the formatting this model
        # exists to produce -- and blind to the truecase damage that reaches
        # 42% of the training labels. Keys are omitted for corpora whose
        # references are not cased or not punctuated (LibriSpeech is ALL-CAPS,
        # AMI/TEDLIUM/Peoples are mono-case), rather than reporting a
        # meaningless 0.0. See scripts/eval/formatting.py.
        metrics.update(
            compute_formatting_metrics(
                [(_scoring_text(r.reference), _scoring_text(r.prediction)) for r in self.results]
            )
        )
        metrics.update(self._speaker_metrics())

        return metrics

    def _speaker_metrics(self) -> dict:
        """cpWER and speaker-count accuracy, when the references carry speakers.

        cpWER matches hypothesis to reference speakers one-to-one to minimise
        word errors, so labels are permutation-free; `attribution_gap` is
        cpWER - WER, what speaker mistakes cost on top of recognition. Rates
        in percent, like `wer`.
        """
        if not any(has_speakers(r.reference) for r in self.results):
            return {}
        scores = speaker_metrics(
            [r.reference for r in self.results],
            [r.prediction for r in self.results],
            self.normalizer.normalize,
        )
        out = {}
        for key, value in scores.items():
            if key in ("n", "wer"):
                continue
            out[key] = value if key.startswith("speaker_count") else value * 100
        return out
