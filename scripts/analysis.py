#!/usr/bin/env python3
"""Analysis tools for ASR evaluation results."""

import functools
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from scripts.itn import ITN_CLASSES, contains_subsequence, merge_scores, score_sample
from scripts.utils import _extract_model_from_dir, find_model_dirs, parse_results_file

# Minimum reference spans before an ITN class gets its own column. At n<15 a
# single span is worth >6.7 points, so the cell reads as a measurement while
# being dominated by sampling. Raise it as the eval set grows.
MIN_CLASS_SUPPORT = 15

app = typer.Typer(help="Analyze and compare `ta eval` results.", add_completion=False)
console = Console()

KEYWORDS_FILE = "outputs/keywords.json"

# Shared help text: every analysis command reads the same run-directory layout,
# so the flags that select runs are spelled and described identically.
MODEL_ARG_HELP = "Model short name to analyze (text after the last '/', matched exactly)"
OUTPUT_DIR_HELP = "Directory containing `ta eval` results"
OutputDirOption = Annotated[
    Path,
    typer.Option("--output-dir", "-o", exists=True, file_okay=False, help=OUTPUT_DIR_HELP),
]
EXCLUDE_HELP = "Model name pattern to exclude (repeatable)"
LATEST_HELP = "Only use the most recent run per dataset"

# OntoNotes' seven numeric labels. The ITN table scores these spans off the raw
# reference, covers classes NER never labels at all (phone numbers, URLs,
# versions), and separates a formatting miss from a recognition miss -- so the
# entity table leaves them out and reports semantic recall only.
ITN_COVERED_ENTITY_TYPES = {
    "CARDINAL",
    "DATE",
    "MONEY",
    "ORDINAL",
    "PERCENT",
    "QUANTITY",
    "TIME",
}


def extract_dataset_name(dir_name: str) -> str:
    """Extract dataset name from output directory name.

    Handles formats:
      - {timestamp}_{model}_{dataset} -> dataset
      - {timestamp}_{model}_{dataset}_diarization -> dataset
      - {timestamp}_{model}_{dataset}_alignment -> dataset
      - {timestamp}_{model}_{dataset}_mcq -> dataset
    """
    parts = dir_name.split("_")
    if not parts:
        return "unknown"
    dataset = parts[-1]
    if dataset in ("diarization", "alignment", "mcq") and len(parts) > 1:
        dataset = parts[-2]
    return dataset


extract_model_name = _extract_model_from_dir


@functools.lru_cache(maxsize=65536)
def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = text.replace("%", " percent").replace("per cent", "percent")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def entity_in_text(entity_text: str, text: str) -> bool:
    """Check if entity appears in text (normalized comparison)."""
    norm_entity = normalize_text(entity_text)
    norm_text = normalize_text(text)
    if norm_entity in norm_text:
        return True

    # Check word-by-word match
    return contains_subsequence(norm_text.split(), norm_entity.split())


@app.command("high-wer")
def high_wer(
    model: Annotated[str, typer.Argument(help=MODEL_ARG_HELP)],
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="WER threshold (percent)")
    ] = 50.0,
    output_dir: OutputDirOption = Path("outputs"),
    exclude: Annotated[list[str] | None, typer.Option("--exclude", help=EXCLUDE_HELP)] = None,
    latest: Annotated[bool, typer.Option("--latest", help=LATEST_HELP)] = False,
    output_file: Annotated[
        Path | None, typer.Option("--output-file", help="Write the report here instead of stdout")
    ] = None,
):
    """Output ground truth and predictions for samples with WER above threshold."""
    model_dirs = find_model_dirs(output_dir, model, exclude, latest=latest)
    results_files = [d / "results.txt" for d in model_dirs if (d / "results.txt").exists()]

    if not results_files:
        console.print(f"[red]No results files found for pattern '{model}'[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(results_files)} results files for '{model}'")
    console.print(f"Filtering samples with WER >= {threshold}%\n")

    high_wer_samples = []

    for results_file in sorted(results_files):
        dataset = extract_dataset_name(results_file.parent.name)

        samples = parse_results_file(results_file)
        for sample in samples:
            if sample["wer"] >= threshold:
                high_wer_samples.append(
                    {
                        "dataset": dataset,
                        "sample_num": sample["sample_num"],
                        "wer": sample["wer"],
                        "ground_truth": sample["ground_truth"],
                        "prediction": sample["prediction"],
                    }
                )

    # Sort by WER descending
    high_wer_samples.sort(key=lambda x: x["wer"], reverse=True)

    console.print(f"Found {len(high_wer_samples)} samples with WER >= {threshold}%\n")

    # Build output
    lines = []
    lines.append(f"# High WER Samples (>= {threshold}%)")
    lines.append(f"# Model: {model}")
    lines.append(f"# Total: {len(high_wer_samples)} samples\n")

    for sample in high_wer_samples:
        lines.append("-" * 80)
        lines.append(
            f"Dataset: {sample['dataset']} | Sample: {sample['sample_num']} | WER: {sample['wer']:.1f}%"
        )
        lines.append(f"Ground Truth: {sample['ground_truth']}")
        lines.append(f"Prediction:   {sample['prediction']}")

    lines.append("-" * 80)
    output_text = "\n".join(lines)

    if output_file:
        output_file.write_text(output_text)
        console.print(f"Saved to [bold]{output_file}[/bold]")
    else:
        console.print(output_text)


@app.command("entity-errors")
def entity_errors(
    model: Annotated[str, typer.Argument(help=MODEL_ARG_HELP)],
    output_dir: OutputDirOption = Path("outputs"),
    exclude: Annotated[list[str] | None, typer.Option("--exclude", help=EXCLUDE_HELP)] = None,
    latest: Annotated[bool, typer.Option("--latest", help=LATEST_HELP)] = False,
    entity_type: Annotated[
        str, typer.Option("--entity-type", help="Filter by entity type (e.g., PERSON, ORG)")
    ] = "",
    output_file: Annotated[
        Path | None, typer.Option("--output-file", help="Write the report here instead of stdout")
    ] = None,
):
    """Output samples where entities were missed in the prediction."""
    # Load keywords file
    keywords_path = Path(KEYWORDS_FILE)
    if not keywords_path.exists():
        console.print(f"[red]Keywords file not found: {keywords_path}[/red]")
        console.print("Run 'analysis extract-entities' first to generate it.")
        raise typer.Exit(1)

    keywords = json.loads(keywords_path.read_text())
    ref_entities = {ref["text"]: ref["entities"] for ref in keywords["references"]}

    model_dirs = find_model_dirs(output_dir, model, exclude, latest=latest)
    results_files = [d / "results.txt" for d in model_dirs if (d / "results.txt").exists()]

    if not results_files:
        console.print(f"[red]No results files found for pattern '{model}'[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(results_files)} results files for '{model}'")
    if entity_type:
        console.print(f"Filtering for entity type: {entity_type}")

    error_samples = []

    for results_file in sorted(results_files):
        dataset = extract_dataset_name(results_file.parent.name)

        samples = parse_results_file(results_file)
        for sample in samples:
            gt = sample["ground_truth"]
            pred = sample["prediction"]
            if gt in ref_entities:
                entities = ref_entities[gt]
                # Filter by entity type if specified
                if entity_type:
                    entities = [e for e in entities if e["label"].upper() == entity_type.upper()]
                # Find entities that are missing from prediction
                missing_entities = [e for e in entities if not entity_in_text(e["text"], pred)]
                if missing_entities:
                    error_samples.append(
                        {
                            "dataset": dataset,
                            "sample_num": sample["sample_num"],
                            "ground_truth": gt,
                            "prediction": pred,
                            "missing_entities": missing_entities,
                        }
                    )

    console.print(f"Found {len(error_samples)} samples with missing entities\n")

    # Build output
    lines = []
    lines.append("# Entity Errors (Missing Entities)")
    lines.append(f"# Model: {model}")
    if entity_type:
        lines.append(f"# Entity Type: {entity_type}")
    lines.append(f"# Total: {len(error_samples)} samples\n")

    for sample in error_samples:
        entity_strs = [f"{e['text']} ({e['label']})" for e in sample["missing_entities"]]
        lines.append("-" * 80)
        lines.append(f"Dataset: {sample['dataset']} | Sample: {sample['sample_num']}")
        lines.append(f"Missing: {', '.join(entity_strs)}")
        lines.append(f"Ground Truth: {sample['ground_truth']}")
        lines.append(f"Prediction:   {sample['prediction']}")

    lines.append("-" * 80)
    output_text = "\n".join(lines)

    if output_file:
        output_file.write_text(output_text)
        console.print(f"Saved to [bold]{output_file}[/bold]")
    else:
        console.print(output_text)


@app.command("extract-entities")
def extract_entities(
    model: Annotated[str, typer.Argument(help=f"{MODEL_ARG_HELP} (omit to use every model)")] = "",
    output_dir: OutputDirOption = Path("outputs"),
    exclude: Annotated[list[str] | None, typer.Option("--exclude", help=EXCLUDE_HELP)] = None,
    min_count: Annotated[
        int, typer.Option("--min-count", help="Minimum entity count to include a type")
    ] = 20,
    latest: Annotated[bool, typer.Option("--latest", help=LATEST_HELP)] = False,
):
    """Extract named entities from reference texts and save to keywords.json.

    Entities are read off the *raw* reference. `EnglishTextNormalizer` output is
    lowercased and stripped of punctuation, and spaCy's NER leans on both -- run
    against normalized text it loses most PERSON/ORG spans and nearly all of the
    rarer labels (PRODUCT, WORK_OF_ART, FAC, EVENT, LAW). Samples written before
    raw transcripts were persisted carry no raw reference and are skipped rather
    than extracted from normalized text.

    References stay keyed by the normalized ground truth, which every run has;
    only extraction needs the raw form, since `entity_in_text` normalizes both
    sides before matching. Entity `start`/`end` offsets therefore index the raw
    reference, not the key.
    """
    import spacy

    console.print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    model_dirs = find_model_dirs(output_dir, model, exclude, latest=latest)
    results_files = [d / "results.txt" for d in model_dirs if (d / "results.txt").exists()]
    console.print(f"Found {len(results_files)} results files")

    all_references = {}
    entity_counts = defaultdict(int)
    missing_raw: set[str] = set()

    for results_file in sorted(results_files):
        samples = parse_results_file(results_file)

        for sample in samples:
            gt = sample["ground_truth"]
            if gt in all_references:
                continue
            gt_raw = sample.get("ground_truth_raw")
            if gt_raw is None:
                missing_raw.add(gt)
                continue

            doc = nlp(gt_raw)
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
            ]
            all_references[gt] = {"entities": entities}
            for ent in entities:
                entity_counts[ent["label"]] += 1

    # A reference skipped in one file may carry raw text in another.
    missing_raw -= set(all_references)

    valid_types = {t for t, c in entity_counts.items() if c >= min_count}

    keywords = {
        "total_references": len(all_references),
        "source_text": "raw",
        "references_missing_raw": len(missing_raw),
        "entity_counts_by_type": {t: c for t, c in entity_counts.items() if t in valid_types},
        "min_count_threshold": min_count,
        "excluded_types": {t: c for t, c in entity_counts.items() if t not in valid_types},
        "references": [
            {
                "text": gt,
                "entities": [e for e in data["entities"] if e["label"] in valid_types],
            }
            for gt, data in all_references.items()
            if any(e["label"] in valid_types for e in data["entities"])
        ],
    }

    keywords_path = Path(KEYWORDS_FILE)
    keywords_path.parent.mkdir(parents=True, exist_ok=True)
    keywords_path.write_text(json.dumps(keywords, indent=2))

    console.print(f"\nExtracted entities from {len(all_references)} unique references")
    console.print(f"References with entities: {len(keywords['references'])}")
    if missing_raw:
        console.print(
            f"[yellow]Skipped {len(missing_raw)} references with no raw transcript[/yellow] "
            "(re-run `ta eval` to capture them)."
        )
    if keywords["excluded_types"]:
        dropped = ", ".join(
            f"{t} ({c})"
            for t, c in sorted(keywords["excluded_types"].items(), key=lambda kv: -kv[1])
        )
        console.print(f"[dim]Below --min-count {min_count}: {dropped}[/dim]")
    console.print(f"Saved to [bold]{keywords_path}[/bold]")


# Datasets to exclude from comparison tables
EXCLUDED_DATASETS = {"classification", "expresso"}

# Short names for display
DATASET_SHORT_NAMES = {
    "earnings22": "Earnings22",
    "peoples": "Peoples",
    "ami": "AMI",
    "gigaspeech": "Gigaspeech",
    "commonvoice": "CV",
    "voxpopuli": "VoxPopuli",
    "loquacious": "Loquacious",
    "librispeech-other": "LS Other",
    "tedlium": "Tedlium",
    "librispeech": "LS Clean",
    "english-dialects-irish": "Irish",
    "english-dialects-scottish": "Scottish",
    "english-dialects-welsh": "Welsh",
    "english-dialects-northern": "Northern",
    "edacc": "EDACC",
}

# Canonical dataset order for comparison tables (display order above)
DATASET_ORDER = list(DATASET_SHORT_NAMES)


def parse_metrics_file(metrics_file: Path) -> dict:
    """Parse a metrics.txt file into a dictionary."""
    result = {}
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
def _current_normalizer():
    """The normalizer `ta eval` scores with today, or None if unavailable.

    Built lazily and once: constructing it pulls the whisper-tiny tokenizer,
    which an offline box may not have. On failure the caller falls back to the
    normalized text stored in results.txt and says so, rather than quietly
    comparing two normalizations.
    """
    try:
        from scripts.eval.audio import TextNormalizer

        return TextNormalizer()
    except Exception as exc:  # any failure means "score as stored"
        console.print(
            f"[yellow]Could not build the eval normalizer ({exc}); scoring the stored "
            "normalized text instead. Sweeps made under different normalizer versions "
            "will not match.[/yellow]"
        )
        return None


def _latest_sweep(dirs: list[Path]) -> tuple[list[Path], str]:
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


def collect_model_metrics(
    model_pattern: str, outputs_dir: Path, exclude: list[str] | None = None
) -> dict:
    """Collect all metrics for a model across datasets."""
    import jiwer

    model_dirs = find_model_dirs(outputs_dir, model_pattern, exclude, latest=True)
    # One sweep only -- see _latest_sweep. Mixing them is how the corpus WER
    # ended up comparing different data between models.
    model_dirs, sweep_label = _latest_sweep(model_dirs)

    display_name = extract_model_name(model_dirs[0].name) if model_dirs else model_pattern

    metrics = {
        "display_name": display_name,
        "sweep": sweep_label,
        "datasets": {},
        "by_length": defaultdict(lambda: {"wers": []}),
        "diarization": None,
        "alignment": None,
        "mcq": {},
        "entity_errors": defaultdict(lambda: {"found": 0, "total": 0}),
        # Pattern-based ITN, scored on the raw (un-normalized) pair only.
        "itn": {},
        "itn_raw_samples": 0,
    }

    all_refs = []
    all_preds = []
    all_latencies = []

    for dir_path in model_dirs:
        results_file = dir_path / "results.txt"
        metrics_file = dir_path / "metrics.txt"
        dir_name = dir_path.name

        if dir_name.endswith("_diarization"):
            if metrics_file.exists():
                metrics["diarization"] = parse_metrics_file(metrics_file)
            continue
        if dir_name.endswith("_alignment"):
            if metrics_file.exists():
                metrics["alignment"] = parse_metrics_file(metrics_file)
            continue
        if dir_name.endswith("_mcq"):
            if metrics_file.exists():
                dataset = extract_dataset_name(dir_name)
                metrics["mcq"][dataset] = parse_metrics_file(metrics_file)
            continue

        dataset = extract_dataset_name(dir_name)

        if not results_file.exists():
            continue

        ds_metrics = {
            "refs": [],
            "preds": [],
            "raw_pairs": [],
            "avg_time": None,
            "wer": None,
            "run_id": None,
        }

        if metrics_file.exists():
            parsed = parse_metrics_file(metrics_file)
            ds_metrics["run_id"] = parsed.get("run_id")
            avg_time = parsed.get("avg_time")
            if isinstance(avg_time, float):
                ds_metrics["avg_time"] = avg_time
                all_latencies.append(avg_time)
            wer = parsed.get("wer")
            if isinstance(wer, float):
                ds_metrics["wer"] = wer
            # Confidence stats (present only when the evaluator captured per-token
            # logits — currently LocalEvaluator with a scores-capable pipeline).
            for k in ("mean_top1_logprob", "mean_margin", "total_tokens"):
                v = parsed.get(k)
                if isinstance(v, (int, float)):
                    ds_metrics[k] = v

        for sample in parse_results_file(results_file):
            gt_raw = sample["ground_truth"]
            pred_raw = sample["prediction"]

            gt_unnorm = sample.get("ground_truth_raw")
            pred_unnorm = sample.get("prediction_raw")
            has_raw = gt_unnorm is not None and pred_unnorm is not None
            if has_raw:
                ds_metrics["raw_pairs"].append((gt_unnorm, pred_unnorm))

            # Re-normalized here from the raw pair, NOT read off the
            # `Ground Truth:` / `Prediction:` lines: those carry whatever
            # scripts/eval/audio.TextNormalizer did on the day that sweep ran.
            # Two sweeps three hours apart straddled f001a061, which added
            # `\bah\b` removal and whitespace collapse. Their stored
            # references then disagreed position-for-position on 5 of the 7
            # shared datasets, _recompute_matched_corpus dropped all 5, and the
            # "Corpus" cell quietly became tedlium+spgispeech alone -- the two
            # easiest corpora -- reading 2.50% beside per-dataset columns of
            # 9-30%. Re-normalizing reconciled all 7 exactly and moved the
            # older sweep's GigaSpeech WER 9.30 -> 8.41, the `ah` credit the
            # newer sweep had already been given.
            #
            # This is the same class `ta eval` scores with, so a run made under
            # today's code reproduces the harness number and a stale one is
            # carried onto today's convention instead of being dropped. Do NOT
            # reach for `normalize_text` instead: that is the looser
            # entity-matching normalizer, it expands "%" to " percent" (so
            # "25%" becomes two reference tokens) and strips currency, and it
            # disagreed with the harness by 0.22 WER on earnings22.
            normalizer = _current_normalizer() if has_raw else None
            if normalizer is not None:
                ref = normalizer.normalize(gt_unnorm)
                pred = normalizer.normalize(pred_unnorm)
            else:
                ref = gt_raw
                pred = pred_raw

            if ref:
                ds_metrics["refs"].append(ref)
                ds_metrics["preds"].append(pred)
                all_refs.append(ref)
                all_preds.append(pred)

                word_count = len(ref.split())
                # The file's per-sample WER belongs to the stored pair, so it
                # is only usable when that is what we scored.
                sample_wer = (
                    jiwer.process_words([ref], [pred]).wer * 100
                    if normalizer is not None
                    else sample.get("wer", 0)
                )
                metrics["by_length"][word_count]["wers"].append(sample_wer)

        if ds_metrics["refs"]:
            output = jiwer.process_words(ds_metrics["refs"], ds_metrics["preds"])
            total = output.hits + output.substitutions + output.deletions
            if total > 0:
                ds_metrics["wer_calculated"] = output.wer * 100
                ds_metrics["ins_rate"] = output.insertions / total * 100
                ds_metrics["del_rate"] = output.deletions / total * 100
                ds_metrics["sub_rate"] = output.substitutions / total * 100

        metrics["datasets"][dataset] = ds_metrics

    if all_refs:
        output = jiwer.process_words(all_refs, all_preds)
        total = output.hits + output.substitutions + output.deletions
        if total > 0:
            metrics["corpus_wer"] = output.wer * 100
            metrics["corpus_ins_rate"] = output.insertions / total * 100

    if all_latencies:
        metrics["avg_latency"] = sum(all_latencies) / len(all_latencies)

    # Corpus-level confidence aggregates: token-weighted means across datasets
    # whose metrics.txt carried per-token stats. Token-weighting matches the
    # per-evaluator aggregation in base.Evaluator.compute_metrics.
    weighted_lp = 0.0
    weighted_mg = 0.0
    corpus_tokens = 0
    for ds_data in metrics["datasets"].values():
        n = ds_data.get("total_tokens")
        lp = ds_data.get("mean_top1_logprob")
        mg = ds_data.get("mean_margin")
        if isinstance(n, (int, float)) and isinstance(lp, float) and isinstance(mg, float):
            weighted_lp += lp * n
            weighted_mg += mg * n
            corpus_tokens += int(n)
    if corpus_tokens > 0:
        metrics["corpus_mean_top1_logprob"] = weighted_lp / corpus_tokens
        metrics["corpus_mean_margin"] = weighted_mg / corpus_tokens

    return metrics


def _sort_key(value: str) -> float:
    """Extract numeric sort key from a formatted value like '12.34%' or '123' or '-'."""
    if value == "-":
        return float("inf")  # Put missing values at the end
    try:
        return float(value.rstrip("%"))
    except ValueError:
        return float("inf")


def _sort_key_desc(value: str) -> tuple[int, float]:
    """Descending-numeric sort key that still pushes missing values last.

    `-_sort_key(value)` does not work for descending order: _sort_key maps the
    "-" placeholder to +inf so it lands at the end of an *ascending* sort, and
    negating that sends it to the front instead. The leading flag keeps
    missing values last regardless of direction.
    """
    if value == "-":
        return (1, 0.0)
    try:
        return (0, -float(value.rstrip("%")))
    except ValueError:
        return (1, 0.0)


def _dataset_wer(ds_data: dict) -> float | None:
    """Per-dataset WER, preferring the value recomputed from saved samples."""
    wer = ds_data.get("wer_calculated")
    return ds_data.get("wer") if wer is None else wer


def _entity_support(model_metrics: dict) -> Counter:
    """Max reference-entity count per semantic type across the compared models."""
    support: Counter = Counter()
    for d in model_metrics.values():
        for t, st in d.get("entity_errors", {}).items():
            if t not in ITN_COVERED_ENTITY_TYPES:
                support[t] = max(support[t], st["total"])
    return support


def _entity_types_with_support(model_metrics: dict) -> set:
    return {t for t, n in _entity_support(model_metrics).items() if n >= MIN_CLASS_SUPPORT}


def _load_ref_entities() -> dict:
    """Reference text -> spaCy entities, keyed on the NORMALIZED reference.

    `ta analysis extract-entities` writes the map; the key is the normalized
    form because that is what `results.txt` stores first and what the
    extractor keyed on.
    """
    path = Path(KEYWORDS_FILE)
    if not path.exists():
        return {}
    keywords = json.loads(path.read_text())
    return {ref["text"]: ref["entities"] for ref in keywords["references"]}


def _recompute_matched_corpus(model_metrics: dict, ref_entities: dict | None = None) -> None:
    """Rebuild each model's `corpus_wer` over a subset every model shares.

    The per-model pooled WER is only meaningful when the models were scored on
    the same audio. `find_model_dirs(latest=True)` picks the newest directory
    PER DATASET, so a sweep that was re-run for some datasets and not others
    yields a different mix per model. Measured case: one model's pool was 53%
    LibriSpeech by reference word (its only two n=500 runs, and the two easiest
    corpora) against 18% for the other, which made a 5.70% "corpus WER" look
    like it beat 7.03% when neither number described the same data.

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
    import jiwer

    ref_entities = ref_entities or {}
    if not model_metrics:
        return
    shared = set.intersection(*(set(m["datasets"]) for m in model_metrics.values()))

    usable, excluded = [], {}
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

    for m in model_metrics.values():
        refs, preds = [], []
        for ds, n in usable:
            d = m["datasets"][ds]
            refs += d["refs"][:n]
            preds += d["preds"][:n]
        m["corpus_datasets"] = [ds for ds, _ in usable]
        m["corpus_excluded"] = excluded

        # Entity and ITN are rescored over the SAME rows, not over whatever
        # each model happened to have on disk. Scored at collection time they
        # compared a 1,200-sample sweep against a 6,000-sample one, which put
        # "PERSON 100% missed" (1 of 1) next to "58.3%" (7 of 12) as if the two
        # were commensurable.
        m["entity_errors"] = defaultdict(lambda: {"found": 0, "total": 0})
        m["itn"] = {}
        m["itn_raw_samples"] = 0
        for ds, n in usable:
            for gt_raw, pred_raw in m["datasets"][ds]["raw_pairs"][:n]:
                m["itn_raw_samples"] += 1
                merge_scores(m["itn"], score_sample(gt_raw, pred_raw))
                for entity in ref_entities.get(normalize_text(gt_raw), ()):
                    stats = m["entity_errors"][entity["label"]]
                    stats["total"] += 1
                    if entity_in_text(entity["text"], pred_raw):
                        stats["found"] += 1
        m.pop("corpus_wer", None)
        m.pop("corpus_ins_rate", None)
        if not refs:
            continue
        out = jiwer.process_words(refs, preds)
        denom = out.hits + out.substitutions + out.deletions
        if denom:
            m["corpus_wer"] = out.wer * 100
            m["corpus_ins_rate"] = out.insertions / denom * 100

        # Per-utterance error and reference-length counts, kept so the corpus
        # delta between two models can carry a confidence interval. They are
        # recorded HERE because this is the only place the paired row set
        # exists: same datasets, same rows, same order for every model.
        per_utt = [jiwer.process_words([r], [p]) for r, p in zip(refs, preds, strict=True)]
        m["corpus_utt_errors"] = [o.substitutions + o.deletions + o.insertions for o in per_utt]
        m["corpus_utt_ref_words"] = [o.substitutions + o.deletions + o.hits for o in per_utt]


def _paired_bootstrap_delta(
    a_errors: list[int],
    a_ref_words: list[int],
    b_errors: list[int],
    b_ref_words: list[int],
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """95% CI on (WER_a - WER_b) in points, by paired utterance bootstrap.

    Why this has to exist: nothing in this repo computed a confidence interval,
    so every margin quoted in the experiment configs was a point estimate.
    That is how the same granite_qwen_top4 checkpoint came to be cited at
    11.95 / 11.76 / 10.94 / 10.50 across sweeps -- a ~1.5pt spread -- while
    recipes were being judged on differences smaller than that.

    PAIRED means both models are resampled on the SAME utterance indices, so
    the shared difficulty of the draw cancels. That is what makes this able to
    resolve a delta much finer than either model's own CI.

    WER is a ratio of corpus totals, not a mean of per-utterance rates, so the
    statistic recomputed on each resample is sum(errors)/sum(ref_words) over
    the resampled rows. Averaging per-utterance WER instead would silently
    reweight the corpus toward short references.
    """
    import numpy as np

    a_err = np.asarray(a_errors, dtype=np.float64)
    a_ref = np.asarray(a_ref_words, dtype=np.float64)
    b_err = np.asarray(b_errors, dtype=np.float64)
    b_ref = np.asarray(b_ref_words, dtype=np.float64)

    n = len(a_err)
    point = (a_err.sum() / a_ref.sum() - b_err.sum() / b_ref.sum()) * 100 if n else float("nan")
    if n == 0:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    # Chunked: a single (resamples, n) index matrix is 10k x 6k x 8B = 480 MB.
    chunk = max(1, min(resamples, 2_000_000 // max(n, 1)))
    done = 0
    while done < resamples:
        size = min(chunk, resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        a_wer = a_err[idx].sum(axis=1) / a_ref[idx].sum(axis=1)
        b_wer = b_err[idx].sum(axis=1) / b_ref[idx].sum(axis=1)
        deltas[done : done + size] = (a_wer - b_wer) * 100
        done += size

    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _print_corpus_wer_cis(model_metrics: dict) -> None:
    """Corpus-WER deltas against the best model, each with a 95% CI.

    Ranks by corpus WER and reports every other model's gap to the leader.
    A CI that straddles zero means the sweep cannot separate the two, which is
    the finding -- reporting the point estimate alone is how a rounding error
    gets read as a win.
    """
    scored = {
        name: data
        for name, data in model_metrics.items()
        if data.get("corpus_wer") is not None and data.get("corpus_utt_errors")
    }
    if len(scored) < 2:
        return

    ranked = sorted(scored.items(), key=lambda kv: kv[1]["corpus_wer"])
    best_name, best = ranked[0]
    n_utts = len(best["corpus_utt_errors"])

    console.print("\n")
    table = Table(
        title=(
            f"Corpus WER delta vs {best['display_name'] if best.get('display_name') else best_name}"
            f"  (paired bootstrap, n={n_utts:,} utterances, 10k resamples)"
        )
    )
    table.add_column("Model", style="cyan")
    table.add_column("Corpus WER", justify="right")
    table.add_column("Δ vs best", justify="right", style="bold")
    table.add_column("95% CI", justify="right")
    table.add_column("Verdict")

    for name, data in ranked[1:]:
        if len(data["corpus_utt_errors"]) != n_utts:
            continue
        point, lo, hi = _paired_bootstrap_delta(
            data["corpus_utt_errors"],
            data["corpus_utt_ref_words"],
            best["corpus_utt_errors"],
            best["corpus_utt_ref_words"],
        )
        if lo > 0:
            verdict = "[green]separated[/green]"
        elif hi < 0:
            # Only reachable if corpus_wer and the paired statistic disagree,
            # which would mean the two were computed over different rows.
            verdict = "[red]inconsistent — check pairing[/red]"
        else:
            verdict = "[yellow]not separated[/yellow]"
        table.add_row(
            data.get("display_name", name),
            f"{data['corpus_wer']:.2f}%",
            f"{point:+.2f}",
            f"[{lo:+.2f}, {hi:+.2f}]",
            verdict,
        )

    console.print(table)
    console.print(
        "[dim]Paired: both models resampled on the same utterance indices, so the "
        "draw's difficulty cancels. 'not separated' means this sweep cannot tell "
        "the two apart — collect more samples rather than reporting the point "
        "estimate.[/dim]"
    )


@app.command("compare")
def compare(
    models: Annotated[list[str], typer.Argument(help=f"{MODEL_ARG_HELP}s to compare")],
    output_dir: OutputDirOption = Path("outputs"),
    exclude: Annotated[list[str] | None, typer.Option("--exclude", help=EXCLUDE_HELP)] = None,
):
    """Generate comprehensive comparison tables for multiple models."""
    # Collect metrics for all models
    model_metrics = {}
    for model in models:
        console.print(f"Collecting metrics for '{model}'...")
        model_metrics[model] = collect_model_metrics(model, output_dir, exclude)

    # Every model must have contributed a sweep, or the table has nothing
    # coherent to show. Runs predating Run IDs are excluded by _latest_sweep.
    stale = [m for m, d in model_metrics.items() if not d["datasets"]]
    if stale:
        console.print(
            f"[red]No Run ID found for: {', '.join(stale)}[/red]\n"
            "[yellow]`ta analysis compare` only reads runs that carry a Run ID, so that "
            "every column comes from one sweep of one checkpoint. Re-run "
            "`ta eval -m <model> -d all -n <N>` to produce one.[/yellow]"
        )
        raise typer.Exit(1)
    for model, data in model_metrics.items():
        console.print(
            f"[dim]{data.get('display_name', model)}: sweep {data.get('sweep')} "
            f"({len(data['datasets'])} datasets)[/dim]"
        )

    # Corpus WER is only comparable across models scored on the same rows;
    # the per-dataset columns below are fine as-is.
    _recompute_matched_corpus(model_metrics, _load_ref_entities())
    sample = next(iter(model_metrics.values()), {})
    if sample.get("corpus_excluded"):
        for ds, why in sorted(sample["corpus_excluded"].items()):
            console.print(f"[yellow]Corpus excludes {ds}: {why}[/yellow]")
    mixed = {
        ds: sorted(
            {len(m["datasets"][ds]["refs"]) for m in model_metrics.values() if ds in m["datasets"]}
        )
        for ds in sample.get("corpus_datasets", [])
    }
    ragged = {ds: ns for ds, ns in mixed.items() if len(ns) > 1}
    if ragged:
        console.print(
            "[yellow]Corpus truncated to the shared row count on: "
            + ", ".join(f"{ds} {ns}" for ds, ns in sorted(ragged.items()))
            + "[/yellow]"
        )

    # Get all datasets present across models (excluding certain datasets)
    all_datasets = set()
    for m in model_metrics.values():
        all_datasets.update(m["datasets"].keys())
    all_datasets -= EXCLUDED_DATASETS

    # Order datasets according to canonical order
    ordered_datasets = [d for d in DATASET_ORDER if d in all_datasets]
    ordered_datasets += [d for d in sorted(all_datasets) if d not in DATASET_ORDER]

    def print_dataset_table(
        title: str,
        summary_column: str,
        summary_key: str,
        dataset_value: Callable[[dict], float | None],
        fmt: Callable[[float], str],
    ) -> None:
        """Model x dataset table with a leading summary column, best first."""
        console.print("\n")
        table = Table(title=title)
        table.add_column("Model", style="cyan")
        table.add_column(summary_column, justify="right", style="bold")
        for ds in ordered_datasets:
            table.add_column(DATASET_SHORT_NAMES.get(ds, ds), justify="right")

        rows = []
        for model, data in model_metrics.items():
            summary = data.get(summary_key)
            row = [
                data.get("display_name", model),
                fmt(summary) if summary is not None else "-",
            ]
            for ds in ordered_datasets:
                value = dataset_value(data["datasets"].get(ds, {}))
                row.append(fmt(value) if value is not None else "-")
            rows.append(row)

        for row in sorted(rows, key=lambda r: _sort_key(r[1])):
            table.add_row(*row)

        console.print(table)

    print_dataset_table(
        "Latency (ms)",
        "Average",
        "avg_latency",
        lambda ds: ds.get("avg_time"),
        lambda v: f"{v * 1000:.0f}",
    )
    # The corpus column pools only the datasets every model shares on identical
    # rows, which can be far fewer than the columns beside it. Spelling the
    # coverage into the title keeps a 2-of-12 pool from reading as a
    # whole-suite number: the excluded corpora were the hard ones, so the cell
    # came out below every dataset in its own table.
    pooled = sample.get("corpus_datasets", [])
    names = ", ".join(DATASET_SHORT_NAMES.get(ds, ds) for ds in pooled)
    coverage = (
        f"corpus = {len(pooled)}/{len(ordered_datasets)} datasets"
        f"{f' ({names})' if 0 < len(pooled) <= 4 else ''}"
        f", {len(sample.get('corpus_utt_errors', [])):,} rows"
    )
    print_dataset_table(
        f"Accuracy by WER ({coverage})",
        "Corpus",
        "corpus_wer",
        _dataset_wer,
        lambda v: f"{v:.2f}%",
    )
    print_dataset_table(
        f"Insertion Rate (Hallucination Proxy) ({coverage})",
        "Corpus",
        "corpus_ins_rate",
        lambda ds: ds.get("ins_rate"),
        lambda v: f"{v:.2f}%",
    )

    _print_corpus_wer_cis(model_metrics)

    # === WER by Word Count Table ===
    console.print("\n")
    wc_table = Table(title="WER by Word Count")
    wc_table.add_column("Model", style="cyan")
    for i in range(1, 11):
        wc_table.add_column(f"{i} word{'s' if i > 1 else ''}", justify="right")

    rows = []
    for model, data in model_metrics.items():
        display_name = data.get("display_name", model)
        row = [display_name]
        for wc in range(1, 11):
            wc_data = data["by_length"].get(wc, {})
            wers = wc_data.get("wers", [])
            if wers:
                avg_wer = sum(wers) / len(wers)
                row.append(f"{avg_wer:.1f}%")
            else:
                row.append("-")
        rows.append(row)

    for row in sorted(rows, key=lambda r: _sort_key(r[1])):
        wc_table.add_row(*row)

    console.print(wc_table)

    # === Confidence Table ===
    # Skip entirely when no model captured per-token logits (e.g. only API
    # evaluators compared) — keeps the output clean for non-tiny-audio runs.
    has_confidence = any(
        m.get("corpus_mean_top1_logprob") is not None for m in model_metrics.values()
    )
    if has_confidence:
        console.print("\n")
        conf_table = Table(title="Confidence (mean top-1 logprob and top1-top2 margin per token)")
        conf_table.add_column("Model", style="cyan")
        conf_table.add_column("Corpus top1", justify="right", style="bold")
        conf_table.add_column("Corpus margin", justify="right", style="bold")
        for ds in ordered_datasets:
            conf_table.add_column(DATASET_SHORT_NAMES.get(ds, ds), justify="right")

        rows = []
        for model, data in model_metrics.items():
            display_name = data.get("display_name", model)
            corpus_lp = data.get("corpus_mean_top1_logprob")
            corpus_mg = data.get("corpus_mean_margin")
            row = [
                display_name,
                f"{corpus_lp:.3f}" if isinstance(corpus_lp, float) else "-",
                f"{corpus_mg:.3f}" if isinstance(corpus_mg, float) else "-",
            ]
            for ds in ordered_datasets:
                ds_data = data["datasets"].get(ds, {})
                lp = ds_data.get("mean_top1_logprob")
                mg = ds_data.get("mean_margin")
                if isinstance(lp, float) and isinstance(mg, float):
                    # Format: "lp / mg" — both small negative numbers, fits in a cell.
                    row.append(f"{lp:.2f}/{mg:.2f}")
                else:
                    row.append("-")
            rows.append(row)

        # Sort by corpus margin descending — wider margin = more decisive
        # model = better, so best-first here matches best-first in the
        # ascending WER / latency tables above.
        for row in sorted(rows, key=lambda r: _sort_key_desc(r[2])):
            conf_table.add_row(*row)

        console.print(conf_table)

    # === Diarization Table ===
    has_diarization = any(m.get("diarization") for m in model_metrics.values())
    if has_diarization:
        console.print("\n")
        diar_table = Table(title="Diarization")
        diar_table.add_column("Model", style="cyan")
        diar_table.add_column("DER", justify="right")
        diar_table.add_column("Confusion", justify="right")
        diar_table.add_column("Missed", justify="right")
        diar_table.add_column("False Alarm", justify="right")

        rows = []
        for model, data in model_metrics.items():
            display_name = data.get("display_name", model)
            diar = data.get("diarization", {})
            if diar:
                rows.append(
                    [
                        display_name,
                        f"{diar.get('der', 0):.2f}%",
                        f"{diar.get('confusion', 0):.2f}%",
                        f"{diar.get('missed', 0):.2f}%",
                        f"{diar.get('false_alarm', 0):.2f}%",
                    ]
                )
            else:
                rows.append([display_name, "-", "-", "-", "-"])

        for row in sorted(rows, key=lambda r: _sort_key(r[1])):
            diar_table.add_row(*row)

        console.print(diar_table)

    # === Alignment Table ===
    has_alignment = any(m.get("alignment") for m in model_metrics.values())
    if has_alignment:
        console.print("\n")
        align_table = Table(title="Timestamp Alignment")
        align_table.add_column("Model", style="cyan")
        align_table.add_column("Median AE (ms)", justify="right")

        rows = []
        for model, data in model_metrics.items():
            display_name = data.get("display_name", model)
            align = data.get("alignment", {})
            if align:
                mae = align.get("mae", 0)
                rows.append([display_name, f"{mae * 1000:.1f}"])
            else:
                rows.append([display_name, "-"])

        for row in sorted(rows, key=lambda r: _sort_key(r[1])):
            align_table.add_row(*row)

        console.print(align_table)

    # === MCQ/Audio Understanding Table ===
    # Collect all MCQ datasets across models
    all_mcq_datasets = set()
    for m in model_metrics.values():
        all_mcq_datasets.update(m["mcq"].keys())

    if all_mcq_datasets:
        console.print("\n")
        mcq_table = Table(title="Audio Understanding (MCQ Accuracy)")
        mcq_table.add_column("Model", style="cyan")
        for mcq_ds in sorted(all_mcq_datasets):
            mcq_table.add_column(mcq_ds.upper(), justify="right")

        rows = []
        for model, data in model_metrics.items():
            display_name = data.get("display_name", model)
            row = [display_name]
            for mcq_ds in sorted(all_mcq_datasets):
                mcq_data = data["mcq"].get(mcq_ds, {})
                accuracy = mcq_data.get("accuracy")
                if accuracy is not None:
                    # Handle both float and string (e.g., "19.00%") formats
                    if isinstance(accuracy, str):
                        row.append(accuracy if "%" in accuracy else f"{accuracy}%")
                    else:
                        row.append(f"{accuracy:.2f}%")
                else:
                    row.append("-")
            rows.append(row)

        # Sort by first MCQ column (highest accuracy first for MCQ)
        for row in sorted(rows, key=lambda r: -_sort_key(r[1]) if len(r) > 1 else 0):
            mcq_table.add_row(*row)

        console.print(mcq_table)

    # === Entity Errors Table ===
    # Get all entity types across models, minus the numeric ones the ITN table
    # already covers (see ITN_COVERED_ENTITY_TYPES) -- what is left is semantic
    # recall: did the model get the name at all.
    all_entity_types = set()
    for m in model_metrics.values():
        all_entity_types.update(m["entity_errors"].keys())
    all_entity_types -= ITN_COVERED_ENTITY_TYPES

    if all_entity_types and not _entity_types_with_support(model_metrics):
        best = _entity_support(model_metrics).most_common(4)
        console.print(
            f"\n[yellow]Entity table skipped: no semantic type reaches {MIN_CLASS_SUPPORT} "
            "reference entities on the compared rows"
            + (f" (best: {', '.join(f'{t}={n}' for t, n in best)})" if best else "")
            + ".[/yellow]\n[dim]outputs/keywords.json only covers the references it was generated "
            "from; regenerate it with `ta analysis extract-entities` against the current eval "
            "pool to restore coverage.[/dim]"
        )
    elif all_entity_types:
        # Order entity types by frequency
        entity_type_order = [
            "GPE",
            "PERSON",
            "ORG",
            "NORP",
            "LOC",
            "FAC",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
        ]
        # Same support floor as the ITN table: below it a single entity moves
        # the cell by >6 points, which is how "PERSON 100% missed" (1 of 1)
        # came to sit next to "58.3%" (7 of 12) as though they were the same
        # kind of number.
        support = Counter()
        for d in model_metrics.values():
            for t, st in d["entity_errors"].items():
                support[t] = max(support[t], st["total"])
        keep = {t for t in all_entity_types if support[t] >= MIN_CLASS_SUPPORT}
        ent_dropped = sorted(
            (t for t in all_entity_types if 0 < support[t] < MIN_CLASS_SUPPORT),
            key=lambda t: -support[t],
        )
        ordered_entity_types = [t for t in entity_type_order if t in keep]
        ordered_entity_types += [t for t in sorted(keep) if t not in entity_type_order]

        console.print("\n")
        entity_table = Table(title="Missed Entity Errors (semantic types; numeric types in ITN)")
        entity_table.add_column("Model", style="cyan")
        entity_table.add_column("Average", justify="right", style="bold")
        for etype in ordered_entity_types:
            entity_table.add_column(etype, justify="right")

        rows = []
        for model, data in model_metrics.items():
            display_name = data.get("display_name", model)
            row = [display_name]
            # Average over the displayed types only, so it is not dominated by
            # the numeric labels this table no longer shows.
            shown = [
                data["entity_errors"][t] for t in ordered_entity_types if t in data["entity_errors"]
            ]
            total_found = sum(e["found"] for e in shown)
            total_entities = sum(e["total"] for e in shown)
            if total_entities > 0:
                avg_err = (total_entities - total_found) / total_entities * 100
                row.append(f"{avg_err:.2f}%")
            else:
                row.append("-")

            for etype in ordered_entity_types:
                stats = data["entity_errors"].get(etype, {"found": 0, "total": 0})
                if stats["total"] > 0:
                    err = (stats["total"] - stats["found"]) / stats["total"] * 100
                    row.append(f"{err:.2f}%")
                else:
                    row.append("-")
            rows.append(row)

        for row in sorted(rows, key=lambda r: _sort_key(r[1])):
            entity_table.add_row(*row)

        console.print(entity_table)
        if ordered_entity_types:
            console.print(
                "[dim]Reference entities per type: "
                + ", ".join(f"{t}={support[t]}" for t in ordered_entity_types)
                + "[/dim]"
            )
        if ent_dropped:
            console.print(
                f"[dim]Hidden (fewer than {MIN_CLASS_SUPPORT} reference entities): "
                + ", ".join(f"{t}={support[t]}" for t in ent_dropped)
                + "[/dim]"
            )
        console.print(
            "[dim]% of reference entities absent from the prediction. "
            "Numeric types (dates, money, counts) are scored in the ITN table below, "
            "which separates a formatting miss from a recognition miss.[/dim]"
        )

    # === ITN Formatting Table (raw text, pattern-based) ===
    # Only runs written after raw-transcript persistence landed contribute here;
    # older runs have no un-normalized text and are reported as missing rather
    # than scored against normalized text, which carries no formatting signal.
    scored = {m: d for m, d in model_metrics.items() if d.get("itn_raw_samples")}
    skipped = [m for m, d in model_metrics.items() if not d.get("itn_raw_samples")]

    if scored:
        class_order = [name for name, _ in ITN_CLASSES]
        # Drop classes with too few reference spans to carry a percentage.
        # Below the threshold a single span moves the number by >6 points, so
        # the column reads like a measurement while being noise: `fraction`
        # had n=1 (rendering one sample as "0.0%"), `title_abbrev` and
        # `acronym_dotted` n=4. Support is pooled across models because every
        # model is scored on the same references, so the count is a property
        # of the eval set rather than of any one system.
        support = Counter()
        for d in scored.values():
            for c, st in d["itn"].items():
                support[c] = max(support[c], st["total"])
        ordered = [c for c in class_order if support[c] >= MIN_CLASS_SUPPORT]
        dropped = sorted(
            (c for c in class_order if 0 < support[c] < MIN_CLASS_SUPPORT),
            key=lambda c: -support[c],
        )

        console.print("\n")
        itn_table = Table(
            title="ITN Formatting (% of reference spans reproduced exactly, raw text)"
        )
        itn_table.add_column("Model", style="cyan")
        itn_table.add_column("Exact", justify="right", style="bold")
        itn_table.add_column("Fmt-only err", justify="right")
        itn_table.add_column("n", justify="right", style="dim")
        for cls in ordered:
            itn_table.add_column(cls, justify="right")

        rows = []
        for model, data in scored.items():
            display_name = data.get("display_name", model)
            stats = data["itn"]
            total = sum(v["total"] for v in stats.values())
            exact = sum(v["exact"] for v in stats.values())
            loose = sum(v["loose"] for v in stats.values())
            row = [
                display_name,
                f"{100 * exact / total:.2f}%" if total else "-",
                f"{100 * (loose - exact) / total:.2f}%" if total else "-",
                str(total),
            ]
            for cls in ordered:
                v = stats.get(cls, {"total": 0, "exact": 0})
                row.append(f"{100 * v['exact'] / v['total']:.1f}%" if v["total"] else "-")
            rows.append(row)

        # Higher Exact% is better, so sort descending -- via _sort_key_desc, which
        # keeps "-" rows last (plain reverse=True would float them to the top).
        for row in sorted(rows, key=lambda r: _sort_key_desc(r[1])):
            itn_table.add_row(*row)

        console.print(itn_table)
        console.print(
            "[dim]Exact = value and formatting both reproduced. "
            "Fmt-only err = value present, formatting differs (e.g. '1250' for '1,250'). "
            "The remainder is recognition error.[/dim]"
        )
        console.print(
            "[dim]Reference spans per class: "
            + ", ".join(f"{c}={support[c]}" for c in ordered)
            + "[/dim]"
        )
        if dropped:
            console.print(
                f"[dim]Hidden (fewer than {MIN_CLASS_SUPPORT} reference spans): "
                + ", ".join(f"{c}={support[c]}" for c in dropped)
                + "[/dim]"
            )

    if skipped:
        console.print(
            f"\n[dim]No raw transcripts for ITN scoring: {', '.join(sorted(skipped))} "
            "(re-run `ta eval` to capture them).[/dim]"
        )


if __name__ == "__main__":
    app()
