"""Collect `ta eval` sweeps into per-model metrics and pair them for comparison."""

import functools
from collections import defaultdict
from pathlib import Path
from typing import Literal, NamedTuple, Required, TypedDict

import jiwer
import numpy as np

from scripts.analysis.common import (
    Entity,
    console,
    entity_in_text,
    extract_dataset_name,
    normalize_text,
)
from scripts.eval.audio import TextNormalizer
from scripts.eval.speaker_metrics import cp_errors, has_speakers, scoring_text, speaker_count
from scripts.itn import merge_scores, score_sample
from scripts.utils import ResultSample, extract_model_from_dir, find_model_dirs, parse_results_file

# Run directories the evaluators that wrote them no longer exist for.
LEGACY_SUFFIXES = ("_diarization", "_alignment", "_mcq")

RunFloatKey = Literal["avg_time", "wer"]
RUN_FLOAT_KEYS: tuple[RunFloatKey, ...] = ("avg_time", "wer")


class SpeakerRow(NamedTuple):
    """One speaker-labelled row: cpWER errors, reference words, speaker count right."""

    errors: int
    ref_words: int
    count_hit: bool


class DatasetMetrics(TypedDict, total=False):
    """One dataset of one sweep: scored rows plus what metrics.txt reported."""

    refs: Required[list[str]]
    preds: Required[list[str]]
    raw_pairs: Required[list[tuple[str, str]]]
    # Parallel to `refs` when every row carries `<SPK_n>`; otherwise incomplete
    # and ignored.
    speaker_rows: Required[list[SpeakerRow]]
    avg_time: float
    wer: float
    wer_calculated: float
    ins_rate: float
    del_rate: float
    sub_rate: float
    cpwer: float
    speaker_count_acc: float
    attribution_gap: float


class EntityStats(TypedDict):
    """Reference entities seen, and how many the prediction reproduced."""

    found: int
    total: int


class ModelMetrics(TypedDict, total=False):
    """One model's sweep; the `corpus_*` keys are set by `recompute_matched_corpus`."""

    display_name: Required[str]
    sweep: Required[str]
    datasets: Required[dict[str, DatasetMetrics]]
    by_length: Required[dict[int, list[float]]]
    entity_errors: Required[dict[str, EntityStats]]
    itn: Required[dict[str, dict[str, int]]]
    itn_raw_samples: Required[int]
    avg_latency: float
    corpus_wer: float
    corpus_ins_rate: float
    corpus_datasets: list[str]
    corpus_excluded: dict[str, str]
    corpus_utt_errors: list[int]
    corpus_utt_ref_words: list[int]
    corpus_cpwer: float
    corpus_cp_datasets: list[str]
    corpus_cp_utt_errors: list[int]
    corpus_cp_utt_ref_words: list[int]


def parse_metrics_file(metrics_file: Path) -> dict[str, float | str]:
    """Parse a metrics.txt file into a dictionary."""
    result: dict[str, float | str] = {}
    for line in metrics_file.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


@functools.cache
def current_normalizer() -> TextNormalizer | None:
    """The normalizer `ta eval` scores with today, or None if unavailable.

    Built lazily and once: constructing it pulls the whisper-tiny tokenizer,
    which an offline box may not have. On failure the caller falls back to the
    normalized text stored in results.txt and says so, rather than quietly
    comparing two normalizations.
    """
    try:
        return TextNormalizer()
    except Exception as exc:  # any failure means "score as stored"
        console.print(
            f"[yellow]Could not build the eval normalizer ({exc}); scoring the stored "
            "normalized text instead. Sweeps made under different normalizer versions "
            "will not match.[/yellow]"
        )
        return None


def latest_sweep(dirs: list[Path]) -> tuple[list[Path], str]:
    """Keep only the newest sweep. Returns (dirs, run_id).

    A sweep is one `ta eval` invocation; every dataset it writes shares a
    `Run ID`. Selecting per DATASET instead -- which `find_model_dirs(
    latest=True)` does -- silently pairs runs from different sweeps: a
    partially re-run suite gave one model a corpus pool that was 53%
    LibriSpeech by reference word against 18% for the other, and put an n=100
    CommonVoice column next to an n=500 LibriSpeech column with nothing in the
    table to show it.

    Directories with no `Run ID` are dropped, not guessed at. Every number in
    the output then comes from one invocation of one checkpoint, which is the
    only basis on which the columns can be read together.
    """
    by_run: dict[str, list[Path]] = defaultdict(list)
    for d in dirs:
        mf = d / "metrics.txt"
        if not mf.exists():
            continue
        rid = parse_metrics_file(mf).get("run_id")
        if isinstance(rid, str) and rid:
            by_run[rid].append(d)
    if not by_run:
        return [], ""
    # Directory names start with a zero-padded UTC timestamp, so lexicographic
    # max is chronological max.
    newest = max(by_run.items(), key=lambda kv: max(d.name for d in kv[1]))
    return newest[1], newest[0]


def speaker_rows(ds: DatasetMetrics) -> list[SpeakerRow]:
    """The dataset's speaker rows, or [] unless every scored row has one."""
    rows = ds["speaker_rows"]
    return rows if rows and len(rows) == len(ds["refs"]) else []


def _set_speaker_rates(ds: DatasetMetrics, n: int | None) -> None:
    """cpWER, speaker-count accuracy and attribution gap over the first `n` rows.

    Same arithmetic as `scripts.eval.speaker_metrics.speaker_metrics`: errors
    and reference words are summed before dividing, so cpWER is a corpus rate.
    """
    rows = speaker_rows(ds)[:n]
    words = sum(r.ref_words for r in rows)
    if not words:
        return
    ds["cpwer"] = sum(r.errors for r in rows) / words * 100
    ds["speaker_count_acc"] = sum(r.count_hit for r in rows) / len(rows)
    if "wer_calculated" in ds:
        ds["attribution_gap"] = ds["cpwer"] - ds["wer_calculated"]


def set_error_rates(ds: DatasetMetrics, n: int | None = None) -> None:
    """WER, insertion/deletion/substitution and speaker rates (percent) over the first `n` rows."""
    refs, preds = ds["refs"][:n], ds["preds"][:n]
    if not refs:
        return
    output = jiwer.process_words(refs, preds)
    total = output.hits + output.substitutions + output.deletions
    if total == 0:
        return
    ds["wer_calculated"] = output.wer * 100
    ds["ins_rate"] = output.insertions / total * 100
    ds["del_rate"] = output.deletions / total * 100
    ds["sub_rate"] = output.substitutions / total * 100
    _set_speaker_rates(ds, n)


def _raw_pair(sample: ResultSample) -> tuple[str, str] | None:
    gt, pred = sample["ground_truth_raw"], sample["prediction_raw"]
    return None if gt is None or pred is None else (gt, pred)


def _speaker_row(ref: str, hyp: str, normalizer: TextNormalizer) -> SpeakerRow:
    """Score one speaker-labelled pair the way `ta eval`'s `_speaker_metrics` does."""
    errors, words = cp_errors(ref, hyp, normalizer.normalize)
    return SpeakerRow(errors, words, speaker_count(ref) == speaker_count(hyp))


def _add_sample(sample: ResultSample, ds: DatasetMetrics, by_length: dict[int, list[float]]):
    """Score one results.txt row into `ds` and the by-word-count buckets.

    Re-normalized here from the raw pair, NOT read off the `Ground Truth:` /
    `Prediction:` lines: those carry whatever scripts/eval/audio.TextNormalizer
    did on the day that sweep ran. Two sweeps three hours apart straddled
    f001a061, which added `\\bah\\b` removal and whitespace collapse. Their
    stored references then disagreed position-for-position on 5 of the 7 shared
    datasets, the matched corpus dropped all 5, and the "Corpus" cell quietly
    became tedlium+spgispeech alone -- the two easiest corpora.

    This is the same class `ta eval` scores with, so a run made under today's
    code reproduces the harness number and a stale one is carried onto today's
    convention instead of being dropped. Do NOT reach for `normalize_text`
    instead: that is the looser entity-matching normalizer, it expands "%" to
    " percent" and strips currency, and it disagreed with the harness by 0.22
    WER on earnings22.
    """
    tagged = _raw_pair(sample)
    pair = None if tagged is None else (scoring_text(tagged[0]), scoring_text(tagged[1]))
    if pair is not None:
        ds["raw_pairs"].append(pair)
    normalizer = current_normalizer() if pair is not None else None
    if pair is not None and normalizer is not None:
        ref, pred = normalizer.normalize(pair[0]), normalizer.normalize(pair[1])
    else:
        ref, pred = sample["ground_truth"], sample["prediction"]
    if not ref:
        return
    ds["refs"].append(ref)
    ds["preds"].append(pred)
    # cpWER needs the normalizer the harness scored with; without it the row
    # is left unscored and the dataset reports no cpWER rather than a guess.
    if tagged is not None and normalizer is not None and has_speakers(tagged[0]):
        ds["speaker_rows"].append(_speaker_row(*tagged, normalizer))
    # The file's per-sample WER belongs to the stored pair, so it is only
    # usable when that is what we scored.
    sample_wer = (
        jiwer.process_words([ref], [pred]).wer * 100 if normalizer is not None else sample["wer"]
    )
    by_length.setdefault(len(ref.split()), []).append(sample_wer)


def _collect_dataset(dir_path: Path, by_length: dict[int, list[float]]) -> DatasetMetrics:
    ds: DatasetMetrics = {"refs": [], "preds": [], "raw_pairs": [], "speaker_rows": []}
    metrics_file = dir_path / "metrics.txt"
    if metrics_file.exists():
        parsed = parse_metrics_file(metrics_file)
        for key in RUN_FLOAT_KEYS:
            value = parsed.get(key)
            if isinstance(value, float):
                ds[key] = value
    for sample in parse_results_file(dir_path / "results.txt"):
        _add_sample(sample, ds, by_length)
    set_error_rates(ds)
    return ds


def _add_avg_latency(metrics: ModelMetrics) -> None:
    latencies = [ds["avg_time"] for ds in metrics["datasets"].values() if "avg_time" in ds]
    if latencies:
        metrics["avg_latency"] = sum(latencies) / len(latencies)


def collect_model_metrics(
    model_pattern: str,
    outputs_dir: Path,
    exclude: list[str] | None = None,
    exclude_datasets: list[str] | None = None,
) -> ModelMetrics:
    """Collect all metrics for a model's newest sweep, across datasets.

    `exclude_datasets` drops runs before the newest sweep is chosen, so a
    later one-dataset run (say ami-speakers-long) doesn't shadow a full sweep.
    """
    model_dirs = find_model_dirs(outputs_dir, model_pattern, exclude, latest=True)
    skip = set(exclude_datasets or ())
    model_dirs = [d for d in model_dirs if extract_dataset_name(d.name) not in skip]
    # One sweep only -- see latest_sweep. Mixing them is how the corpus WER
    # ended up comparing different data between models.
    model_dirs, sweep_label = latest_sweep(model_dirs)

    metrics: ModelMetrics = {
        "display_name": (
            extract_model_from_dir(model_dirs[0].name) if model_dirs else model_pattern
        ),
        "sweep": sweep_label,
        "datasets": {},
        "by_length": {},
        "entity_errors": {},
        # Pattern-based ITN, scored on the raw (un-normalized) pair only.
        "itn": {},
        "itn_raw_samples": 0,
    }
    for dir_path in model_dirs:
        if dir_path.name.endswith(LEGACY_SUFFIXES) or not (dir_path / "results.txt").exists():
            continue
        dataset = extract_dataset_name(dir_path.name)
        metrics["datasets"][dataset] = _collect_dataset(dir_path, metrics["by_length"])
    _add_avg_latency(metrics)
    return metrics


def match_dataset_rows(
    model_metrics: dict[str, ModelMetrics],
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Rescore each dataset column over the rows every model in it shares.

    Measured case: smallest-pulse swept Loquacious at n=100 while tiny-audio
    had 1,000 rows, so the Loquacious column read 5.82 for tiny against a 6.29
    corpus cell that was the same model on the same dataset -- the first 100
    rows -- and smallest's 12.15 sat beside a number from 10x the data.

    Unlike the corpus, a dataset only needs the models that HAVE it: a model
    missing a dataset shows "-" there and does not shrink the column for the
    rest. Rows are matched by the same prefix rule as
    `recompute_matched_corpus` (the eval draw is a fixed-seed prefix), and a
    dataset whose references disagree keeps each model's full-sweep number and
    is reported, rather than silently compared across different rows.

    The rates are overwritten in place; `refs`/`preds` stay whole. Returns
    (truncated: ds -> sorted per-model row counts, unmatched: ds -> reason).
    """
    datasets: set[str] = set().union(*(m["datasets"] for m in model_metrics.values()))
    truncated: dict[str, list[int]] = {}
    unmatched: dict[str, str] = {}
    for ds in sorted(datasets):
        per_model = [
            m["datasets"][ds]
            for m in model_metrics.values()
            if ds in m["datasets"] and m["datasets"][ds]["refs"]
        ]
        counts = sorted({len(d["refs"]) for d in per_model})
        if len(counts) < 2:
            continue
        n = counts[0]
        if any(d["refs"][:n] != per_model[0]["refs"][:n] for d in per_model):
            unmatched[ds] = "references differ (different eval rows)"
            continue
        for d in per_model:
            set_error_rates(d, n)
        truncated[ds] = counts
    return truncated, unmatched


def dataset_wer(ds_data: DatasetMetrics) -> float | None:
    """Per-dataset WER, preferring the value recomputed from saved samples."""
    return ds_data.get("wer_calculated", ds_data.get("wer"))


def _shared_rows(
    model_metrics: dict[str, ModelMetrics],
) -> tuple[list[tuple[str, int]], dict[str, str]]:
    """(dataset, row count) every model can be paired on, and why the rest cannot."""
    shared = set.intersection(*(set(m["datasets"]) for m in model_metrics.values()))
    usable: list[tuple[str, int]] = []
    excluded: dict[str, str] = {}
    for ds in sorted(shared):
        per_model = [m["datasets"][ds] for m in model_metrics.values()]
        if any(not d["refs"] for d in per_model):
            excluded[ds] = "no scored rows"
            continue
        n = min(len(d["refs"]) for d in per_model)
        first = per_model[0]["refs"][:n]
        if any(d["refs"][:n] != first for d in per_model):
            # References are re-normalized from the raw pair at collection
            # time, so a mismatch here is a genuinely different draw -- unless
            # a run predates raw transcripts, in which case its normalization
            # is frozen at whatever shipped then and cannot be reconciled.
            excluded[ds] = (
                "a run has no raw transcripts, so its normalization cannot be reconciled"
                if any(len(d["raw_pairs"]) < len(d["refs"]) for d in per_model)
                else "references differ (different eval rows)"
            )
            continue
        usable.append((ds, n))
    return usable, excluded


def _rescore_formatting(
    m: ModelMetrics, usable: list[tuple[str, int]], ref_entities: dict[str, list[Entity]]
) -> None:
    """Entity recall and ITN over the paired rows only.

    Scored at collection time they compared a 1,200-sample sweep against a
    6,000-sample one, which put "PERSON 100% missed" (1 of 1) next to "58.3%"
    (7 of 12) as if the two were commensurable.
    """
    m["entity_errors"] = {}
    m["itn"] = {}
    m["itn_raw_samples"] = 0
    for ds, n in usable:
        for gt_raw, pred_raw in m["datasets"][ds]["raw_pairs"][:n]:
            m["itn_raw_samples"] += 1
            merge_scores(m["itn"], score_sample(gt_raw, pred_raw))
            for entity in ref_entities.get(normalize_text(gt_raw), ()):
                stats = m["entity_errors"].setdefault(entity["label"], {"found": 0, "total": 0})
                stats["total"] += 1
                stats["found"] += entity_in_text(entity["text"], pred_raw)


def _set_corpus_wer(m: ModelMetrics, refs: list[str], preds: list[str]) -> None:
    m.pop("corpus_wer", None)
    m.pop("corpus_ins_rate", None)
    if not refs:
        return
    out = jiwer.process_words(refs, preds)
    denom = out.hits + out.substitutions + out.deletions
    if denom:
        m["corpus_wer"] = out.wer * 100
        m["corpus_ins_rate"] = out.insertions / denom * 100
    # Per-utterance error and reference-length counts, kept so the corpus delta
    # between two models can carry a confidence interval. They are recorded
    # HERE because this is the only place the paired row set exists: same
    # datasets, same rows, same order for every model.
    per_utt = [jiwer.process_words([r], [p]) for r, p in zip(refs, preds, strict=True)]
    m["corpus_utt_errors"] = [o.substitutions + o.deletions + o.insertions for o in per_utt]
    m["corpus_utt_ref_words"] = [o.substitutions + o.deletions + o.hits for o in per_utt]


def _set_corpus_cpwer(m: ModelMetrics, usable: list[tuple[str, int]]) -> None:
    """Pooled cpWER over the paired rows of the speaker-labelled datasets."""
    for key in ("corpus_cpwer", "corpus_cp_utt_errors", "corpus_cp_utt_ref_words"):
        m.pop(key, None)
    rows = [r for ds, n in usable for r in speaker_rows(m["datasets"][ds])[:n]]
    words = sum(r.ref_words for r in rows)
    if not words:
        return
    m["corpus_cpwer"] = sum(r.errors for r in rows) / words * 100
    m["corpus_cp_utt_errors"] = [r.errors for r in rows]
    m["corpus_cp_utt_ref_words"] = [r.ref_words for r in rows]


def recompute_matched_corpus(
    model_metrics: dict[str, ModelMetrics], ref_entities: dict[str, list[Entity]] | None = None
) -> None:
    """Rebuild each model's `corpus_wer` over a subset every model shares.

    The per-model pooled WER is only meaningful when the models were scored on
    the same audio. Measured case: one model's pool was 53% LibriSpeech by
    reference word (its only two n=500 runs, and the two easiest corpora)
    against 18% for the other, which made a 5.70% "corpus WER" look like it
    beat 7.03% when neither number described the same data.

    So: keep only datasets every model has, truncate each to the shared row
    count, and require the references to match position-for-position. The eval
    draw is a deterministic prefix of a fixed-seed shuffle, so an n=100 run is
    the first 100 rows of the n=500 run of the same dataset -- truncation
    yields a genuinely paired comparison rather than an approximate one. A
    dataset whose references disagree after truncation is dropped rather than
    silently pooled.

    Sets `corpus_wer`, `corpus_ins_rate`, `corpus_datasets` and
    `corpus_excluded` on each model in place.
    """
    if not model_metrics:
        return
    usable, excluded = _shared_rows(model_metrics)
    for m in model_metrics.values():
        refs = [r for ds, n in usable for r in m["datasets"][ds]["refs"][:n]]
        preds = [p for ds, n in usable for p in m["datasets"][ds]["preds"][:n]]
        m["corpus_datasets"] = [ds for ds, _ in usable]
        m["corpus_excluded"] = excluded
        _rescore_formatting(m, usable, ref_entities or {})
        _set_corpus_wer(m, refs, preds)
    # cpWER pools only datasets every model has speaker rows for, so the
    # per-utterance lists stay paired for the bootstrap.
    cp_usable = [
        (ds, n)
        for ds, n in usable
        if all(speaker_rows(m["datasets"][ds]) for m in model_metrics.values())
    ]
    for m in model_metrics.values():
        m["corpus_cp_datasets"] = [ds for ds, _ in cp_usable]
        _set_corpus_cpwer(m, cp_usable)


def paired_bootstrap_delta(
    a_errors: list[int],
    a_ref_words: list[int],
    b_errors: list[int],
    b_ref_words: list[int],
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """95% CI on (WER_a - WER_b) in points, by paired utterance bootstrap.

    Why this has to exist: without a confidence interval every margin quoted in
    the experiment configs was a point estimate. That is how the same
    granite_qwen_top4 checkpoint came to be cited at 11.95 / 11.76 / 10.94 /
    10.50 across sweeps -- a ~1.5pt spread -- while recipes were being judged
    on differences smaller than that.

    PAIRED means both models are resampled on the SAME utterance indices, so
    the shared difficulty of the draw cancels. That is what makes this able to
    resolve a delta much finer than either model's own CI.

    WER is a ratio of corpus totals, not a mean of per-utterance rates, so the
    statistic recomputed on each resample is sum(errors)/sum(ref_words) over
    the resampled rows. Averaging per-utterance WER instead would silently
    reweight the corpus toward short references.
    """
    a_err = np.asarray(a_errors, dtype=np.float64)
    a_ref = np.asarray(a_ref_words, dtype=np.float64)
    b_err = np.asarray(b_errors, dtype=np.float64)
    b_ref = np.asarray(b_ref_words, dtype=np.float64)

    n = len(a_err)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float((a_err.sum() / a_ref.sum() - b_err.sum() / b_ref.sum()) * 100)

    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    # Chunked: a single (resamples, n) index matrix is 10k x 6k x 8B = 480 MB.
    chunk = max(1, min(resamples, 2_000_000 // n))
    done = 0
    while done < resamples:
        size = min(chunk, resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        a_wer = a_err[idx].sum(axis=1) / a_ref[idx].sum(axis=1)
        b_wer = b_err[idx].sum(axis=1) / b_ref[idx].sum(axis=1)
        deltas[done : done + size] = (a_wer - b_wer) * 100
        done += size

    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
