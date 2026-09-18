#!/usr/bin/env python3
"""Analysis tools for ASR evaluation results."""

import functools
import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from scripts.itn import ITN_CLASSES, contains_subsequence, merge_scores, score_sample
from scripts.utils import _extract_model_from_dir, find_model_dirs, parse_results_file

app = typer.Typer(help="Analysis tools for ASR evaluation results")
console = Console()

KEYWORDS_FILE = "outputs/keywords.json"

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
    model: str = typer.Argument(..., help="Model pattern to analyze"),
    threshold: float = typer.Option(50.0, "--threshold", "-t", help="WER threshold (percent)"),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", help="Directory containing eval results"
    ),
    exclude: list[str] = typer.Option([], help="Patterns to exclude"),
    latest: bool = typer.Option(False, "--latest", help="Only use most recent run per dataset"),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file (default: stdout)"
    ),
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
    model: str = typer.Argument(..., help="Model pattern to analyze"),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", help="Directory containing eval results"
    ),
    exclude: list[str] = typer.Option([], help="Patterns to exclude"),
    latest: bool = typer.Option(False, "--latest", help="Only use most recent run per dataset"),
    entity_type: str = typer.Option(
        "", "--type", "-t", help="Filter by entity type (e.g., PERSON, ORG)"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file (default: stdout)"
    ),
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
    model: str = typer.Option("", help="Model pattern to extract from (empty for all)"),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", help="Directory containing eval results"
    ),
    exclude: list[str] = typer.Option([], help="Patterns to exclude"),
    min_count: int = typer.Option(20, help="Minimum entity count to include a type"),
    latest: bool = typer.Option(False, "--latest", help="Only use most recent run per dataset"),
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


def collect_model_metrics(model_pattern: str, outputs_dir: Path, exclude: list[str]) -> dict:
    """Collect all metrics for a model across datasets."""
    import jiwer

    model_dirs = find_model_dirs(outputs_dir, model_pattern, exclude, latest=True)

    display_name = extract_model_name(model_dirs[0].name) if model_dirs else model_pattern

    metrics = {
        "display_name": display_name,
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

    keywords_path = Path(KEYWORDS_FILE)
    ref_entities = {}
    if keywords_path.exists():
        keywords = json.loads(keywords_path.read_text())
        ref_entities = {ref["text"]: ref["entities"] for ref in keywords["references"]}

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

        ds_metrics = {"refs": [], "preds": [], "avg_time": None, "wer": None}

        if metrics_file.exists():
            parsed = parse_metrics_file(metrics_file)
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
            if gt_unnorm is not None and pred_unnorm is not None:
                metrics["itn_raw_samples"] += 1
                merge_scores(metrics["itn"], score_sample(gt_unnorm, pred_unnorm))
            ref = normalize_text(gt_raw)
            pred = normalize_text(pred_raw)

            if ref:
                ds_metrics["refs"].append(ref)
                ds_metrics["preds"].append(pred)
                all_refs.append(ref)
                all_preds.append(pred)

                word_count = len(ref.split())
                metrics["by_length"][word_count]["wers"].append(sample.get("wer", 0))

                if gt_raw in ref_entities:
                    for entity in ref_entities[gt_raw]:
                        entity_type = entity["label"]
                        entity_text = entity["text"]

                        metrics["entity_errors"][entity_type]["total"] += 1
                        if entity_in_text(entity_text, pred_raw):
                            metrics["entity_errors"][entity_type]["found"] += 1

        if ds_metrics["refs"]:
            output = jiwer.process_words(ds_metrics["refs"], ds_metrics["preds"])
            total = output.hits + output.substitutions + output.deletions
            if total > 0:
                ds_metrics["wer_calculated"] = (
                    (output.substitutions + output.deletions + output.insertions) / total * 100
                )
                ds_metrics["ins_rate"] = output.insertions / total * 100
                ds_metrics["del_rate"] = output.deletions / total * 100
                ds_metrics["sub_rate"] = output.substitutions / total * 100

        metrics["datasets"][dataset] = ds_metrics

    if all_refs:
        output = jiwer.process_words(all_refs, all_preds)
        total = output.hits + output.substitutions + output.deletions
        if total > 0:
            metrics["corpus_wer"] = (
                (output.substitutions + output.deletions + output.insertions) / total * 100
            )
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


@app.command("compare")
def compare(
    models: list[str] = typer.Argument(..., help="Model patterns to compare"),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", help="Directory containing eval results"
    ),
    exclude: list[str] = typer.Option([], help="Patterns to exclude from matching"),
):
    """Generate comprehensive comparison tables for multiple models."""

    if not models:
        console.print("[red]Please provide at least one model pattern[/red]")
        raise typer.Exit(1)

    # Collect metrics for all models
    model_metrics = {}
    for model in models:
        console.print(f"Collecting metrics for '{model}'...")
        model_metrics[model] = collect_model_metrics(model, output_dir, exclude)

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
    print_dataset_table(
        "Accuracy by WER",
        "Corpus",
        "corpus_wer",
        _dataset_wer,
        lambda v: f"{v:.2f}%",
    )
    print_dataset_table(
        "Insertion Rate (Hallucination Proxy)",
        "Corpus",
        "corpus_ins_rate",
        lambda ds: ds.get("ins_rate"),
        lambda v: f"{v:.2f}%",
    )

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

    if all_entity_types:
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
        ordered_entity_types = [t for t in entity_type_order if t in all_entity_types]
        ordered_entity_types += [t for t in sorted(all_entity_types) if t not in entity_type_order]

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
        present = {c for d in scored.values() for c, st in d["itn"].items() if st["total"]}
        ordered = [c for c in class_order if c in present]

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

    if skipped:
        console.print(
            f"\n[dim]No raw transcripts for ITN scoring: {', '.join(sorted(skipped))} "
            "(re-run `ta eval` to capture them).[/dim]"
        )


if __name__ == "__main__":
    app()
