#!/usr/bin/env python3
"""Analysis tools for ASR evaluation results.

Commands:
    extract-entities - Extract named entities from reference texts for comparison
    compare          - Generate comprehensive comparison tables for multiple models
"""

import contextlib
import json
import re
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from scripts.utils import find_model_dirs, parse_results_file

app = typer.Typer(help="Analysis tools for ASR evaluation results")
console = Console()

KEYWORDS_FILE = "outputs/keywords.json"

# =============================================================================
# Shared utilities
# =============================================================================


def extract_dataset_name(dir_name: str) -> str:
    """Extract dataset name from output directory name.

    Handles formats:
      - {timestamp}_{model}_{dataset} -> dataset
      - {timestamp}_{model}_{dataset}_diarization -> dataset
      - {timestamp}_{model}_{dataset}_alignment -> dataset
    """
    parts = dir_name.split("_")
    if not parts:
        return "unknown"
    dataset = parts[-1]
    if dataset in ("diarization", "alignment") and len(parts) > 1:
        dataset = parts[-2]
    return dataset


def extract_model_name(dir_name: str) -> str:
    """Extract model name from output directory name.

    Format: {timestamp}_{model}_{dataset}[_suffix]
    """
    parts = dir_name.split("_")
    return parts[2] if len(parts) >= 3 else "unknown"


WORD_TO_NUM = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "billion": "1000000000",
    "first": "1st",
    "second": "2nd",
    "third": "3rd",
    "fourth": "4th",
    "fifth": "5th",
    "sixth": "6th",
    "seventh": "7th",
    "eighth": "8th",
    "ninth": "9th",
    "tenth": "10th",
}


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = text.replace("%", " percent").replace("per cent", "percent")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_numbers(text: str) -> str:
    """Convert number words to digits for flexible matching."""
    text = normalize_text(text)
    return " ".join(WORD_TO_NUM.get(w, w) for w in text.split())


def entity_in_text(entity_text: str, text: str) -> bool:
    """Check if entity appears in text (normalized comparison)."""
    norm_entity = normalize_text(entity_text)
    norm_text = normalize_text(text)
    if norm_entity in norm_text:
        return True

    num_entity = normalize_numbers(entity_text)
    num_text = normalize_numbers(text)
    if num_entity in num_text:
        return True

    entity_words = num_entity.split()
    text_words = num_text.split()
    if len(entity_words) <= len(text_words):
        for i in range(len(text_words) - len(entity_words) + 1):
            if text_words[i : i + len(entity_words)] == entity_words:
                return True
    return False


# Entity types that should be checked for ITN (Inverse Text Normalization)
ITN_ENTITY_TYPES = {"CARDINAL", "DATE", "TIME", "MONEY", "PERCENT", "ORDINAL", "QUANTITY"}


def entity_itn_correct(entity_text: str, text: str) -> bool:
    """Check if entity appears with correct formatting (ITN accuracy).

    This is stricter than entity_in_text - checks for exact formatted match.
    Example: "$25" should appear as "$25", not "twenty five dollars".
    """
    # Case-insensitive but format-preserving check
    entity_lower = entity_text.lower()
    text_lower = text.lower()

    # Direct substring match (preserves formatting like $, %, :, etc.)
    if entity_lower in text_lower:
        return True

    # Check with minor punctuation variations (e.g., "3:00" vs "3.00")
    entity_normalized = entity_lower.replace(":", ".").replace(",", "")
    text_normalized = text_lower.replace(":", ".").replace(",", "")
    return entity_normalized in text_normalized


# =============================================================================
# extract-entities command
# =============================================================================


@app.command("high-wer")
def high_wer(
    model: str = typer.Argument(..., help="Model pattern to analyze"),
    threshold: float = typer.Option(50.0, "--threshold", "-t", help="WER threshold (percent)"),
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
    exclude: list[str] = typer.Option([], help="Patterns to exclude"),
    latest: bool = typer.Option(False, "--latest", help="Only use most recent run per dataset"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
):
    """Output ground truth and predictions for samples with WER above threshold."""
    model_dirs = find_model_dirs(outputs_dir, model, exclude, latest=latest)
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
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
    exclude: list[str] = typer.Option([], help="Patterns to exclude"),
    latest: bool = typer.Option(False, "--latest", help="Only use most recent run per dataset"),
    entity_type: str = typer.Option(
        "", "--type", "-t", help="Filter by entity type (e.g., PERSON, ORG)"
    ),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
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

    model_dirs = find_model_dirs(outputs_dir, model, exclude, latest=latest)
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
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
    exclude: list[str] = typer.Option([], help="Patterns to exclude"),
    min_count: int = typer.Option(20, help="Minimum entity count to include a type"),
    latest: bool = typer.Option(False, "--latest", help="Only use most recent run per dataset"),
):
    """Extract named entities from reference texts and save to keywords.json."""
    import spacy

    console.print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    model_dirs = find_model_dirs(outputs_dir, model, exclude, latest=latest)
    results_files = [d / "results.txt" for d in model_dirs if (d / "results.txt").exists()]
    console.print(f"Found {len(results_files)} results files")

    all_references = {}
    entity_counts = defaultdict(int)

    for results_file in sorted(results_files):
        samples = parse_results_file(results_file)

        for sample in samples:
            gt = sample["ground_truth"]
            if gt not in all_references:
                doc = nlp(gt)
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

    valid_types = {t for t, c in entity_counts.items() if c >= min_count}

    keywords = {
        "total_references": len(all_references),
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
    keywords_path.parent.mkdir(exist_ok=True)
    keywords_path.write_text(json.dumps(keywords, indent=2))

    console.print(f"\nExtracted entities from {len(all_references)} unique references")
    console.print(f"References with entities: {len(keywords['references'])}")
    console.print(f"Saved to [bold]{keywords_path}[/bold]")


# =============================================================================
# compare command - comprehensive model comparison
# =============================================================================

# Canonical dataset order for comparison tables
DATASET_ORDER = [
    "earnings22",
    "peoples",
    "ami",
    "gigaspeech",
    "commonvoice",
    "loquacious",
    "librispeech-other",
    "tedlium",
    "librispeech",
    "english-dialects-irish",
    "english-dialects-scottish",
    "english-dialects-welsh",
    "english-dialects-northern",
    "edacc",
    "switchboard",
]

# Short names for display
DATASET_SHORT_NAMES = {
    "earnings22": "Earnings22",
    "peoples": "Peoples",
    "ami": "AMI",
    "gigaspeech": "Gigaspeech",
    "commonvoice": "CV",
    "loquacious": "Loquacious",
    "librispeech-other": "LS Other",
    "tedlium": "Tedlium",
    "librispeech": "LS Clean",
    "english-dialects-irish": "Irish",
    "english-dialects-scottish": "Scottish",
    "english-dialects-welsh": "Welsh",
    "english-dialects-northern": "Northern",
    "edacc": "EDACC",
    "switchboard": "Switchboard",
}


def collect_model_metrics(model_pattern: str, outputs_dir: Path, exclude: list[str]) -> dict:
    """Collect all metrics for a model across datasets."""
    import jiwer

    model_dirs = find_model_dirs(outputs_dir, model_pattern, exclude, latest=True)

    # Extract display name from folder structure
    display_name = extract_model_name(model_dirs[0].name) if model_dirs else model_pattern

    metrics = {
        "display_name": display_name,
        "datasets": {},
        "by_length": defaultdict(lambda: {"samples": [], "wers": []}),
        "diarization": None,
        "alignment": None,
        "entity_errors": defaultdict(lambda: {"found": 0, "total": 0}),
        "itn_errors": defaultdict(lambda: {"correct": 0, "total": 0}),
    }

    all_refs = []
    all_preds = []
    all_latencies = []

    # Load keywords for entity analysis
    keywords_path = Path(KEYWORDS_FILE)
    ref_entities = {}
    if keywords_path.exists():
        keywords = json.loads(keywords_path.read_text())
        ref_entities = {ref["text"]: ref["entities"] for ref in keywords["references"]}

    for dir_path in model_dirs:
        results_file = dir_path / "results.txt"
        metrics_file = dir_path / "metrics.txt"
        dir_name = dir_path.name

        # Check for diarization/alignment results (special handling)
        if dir_name.endswith("_diarization"):
            if metrics_file.exists():
                metrics["diarization"] = parse_metrics_file(metrics_file)
            continue
        if dir_name.endswith("_alignment"):
            if metrics_file.exists():
                metrics["alignment"] = parse_metrics_file(metrics_file)
            continue

        dataset = extract_dataset_name(dir_name)

        if not results_file.exists():
            continue

        ds_metrics = {"refs": [], "preds": [], "avg_time": None, "wer": None}

        # Parse metrics.txt
        if metrics_file.exists():
            for line in metrics_file.read_text().splitlines():
                if line.startswith("avg_time:"):
                    try:
                        ds_metrics["avg_time"] = float(line.split(":")[1].strip())
                        all_latencies.append(ds_metrics["avg_time"])
                    except ValueError:
                        pass
                elif line.startswith("wer:"):
                    with contextlib.suppress(ValueError):
                        ds_metrics["wer"] = float(line.split(":")[1].strip())

        # Parse results
        for sample in parse_results_file(results_file):
            ref = normalize_text(sample["ground_truth"])
            pred = normalize_text(sample["prediction"])
            gt_raw = sample["ground_truth"]
            pred_raw = sample["prediction"]

            if ref:
                ds_metrics["refs"].append(ref)
                ds_metrics["preds"].append(pred)
                all_refs.append(ref)
                all_preds.append(pred)

                # Track by word count
                word_count = len(ref.split())
                wer = sample.get("wer", 0)
                metrics["by_length"][word_count]["samples"].append(sample)
                metrics["by_length"][word_count]["wers"].append(wer)

                # Track entity errors and ITN errors
                if gt_raw in ref_entities:
                    for entity in ref_entities[gt_raw]:
                        entity_type = entity["label"]
                        entity_text = entity["text"]

                        # Entity detection (normalized)
                        found = entity_in_text(entity_text, pred_raw)
                        metrics["entity_errors"][entity_type]["total"] += 1
                        if found:
                            metrics["entity_errors"][entity_type]["found"] += 1

                        # ITN accuracy (format preservation)
                        if entity_type in ITN_ENTITY_TYPES:
                            metrics["itn_errors"][entity_type]["total"] += 1
                            if entity_itn_correct(entity_text, pred_raw):
                                metrics["itn_errors"][entity_type]["correct"] += 1

        # Calculate detailed error breakdown
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

    # Calculate corpus-level metrics
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

    return metrics


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


@app.command("compare")
def compare(
    models: list[str] = typer.Argument(..., help="Model patterns to compare"),
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
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
        model_metrics[model] = collect_model_metrics(model, outputs_dir, exclude)

    # Get all datasets present across models
    all_datasets = set()
    for m in model_metrics.values():
        all_datasets.update(m["datasets"].keys())

    # Order datasets according to canonical order
    ordered_datasets = [d for d in DATASET_ORDER if d in all_datasets]
    ordered_datasets += [d for d in sorted(all_datasets) if d not in DATASET_ORDER]

    # === Latency Table ===
    console.print("\n")
    latency_table = Table(title="Latency (ms)")
    latency_table.add_column("Model", style="cyan")
    latency_table.add_column("Average", justify="right", style="bold")
    for ds in ordered_datasets:
        latency_table.add_column(DATASET_SHORT_NAMES.get(ds, ds), justify="right")

    for model, data in model_metrics.items():
        display_name = data.get("display_name", model)
        row = [display_name]
        avg_lat = data.get("avg_latency")
        row.append(f"{avg_lat * 1000:.0f}" if avg_lat else "-")
        for ds in ordered_datasets:
            ds_data = data["datasets"].get(ds, {})
            lat = ds_data.get("avg_time")
            row.append(f"{lat * 1000:.0f}" if lat else "-")
        latency_table.add_row(*row)

    console.print(latency_table)

    # === WER Table ===
    console.print("\n")
    wer_table = Table(title="Accuracy by WER")
    wer_table.add_column("Model", style="cyan")
    wer_table.add_column("Corpus", justify="right", style="bold")
    for ds in ordered_datasets:
        wer_table.add_column(DATASET_SHORT_NAMES.get(ds, ds), justify="right")

    for model, data in model_metrics.items():
        display_name = data.get("display_name", model)
        row = [display_name]
        corpus_wer = data.get("corpus_wer")
        row.append(f"{corpus_wer:.2f}%" if corpus_wer else "-")
        for ds in ordered_datasets:
            ds_data = data["datasets"].get(ds, {})
            wer = ds_data.get("wer_calculated") or ds_data.get("wer")
            row.append(f"{wer:.2f}%" if wer else "-")
        wer_table.add_row(*row)

    console.print(wer_table)

    # === Insertion Rate Table ===
    console.print("\n")
    ins_table = Table(title="Insertion Rate (Hallucination Proxy)")
    ins_table.add_column("Model", style="cyan")
    ins_table.add_column("Corpus", justify="right", style="bold")
    for ds in ordered_datasets:
        ins_table.add_column(DATASET_SHORT_NAMES.get(ds, ds), justify="right")

    for model, data in model_metrics.items():
        display_name = data.get("display_name", model)
        row = [display_name]
        avg_ins = data.get("corpus_ins_rate")
        row.append(f"{avg_ins:.2f}%" if avg_ins else "-")
        for ds in ordered_datasets:
            ds_data = data["datasets"].get(ds, {})
            ins = ds_data.get("ins_rate")
            row.append(f"{ins:.2f}%" if ins else "-")
        ins_table.add_row(*row)

    console.print(ins_table)

    # === WER by Word Count Table ===
    console.print("\n")
    wc_table = Table(title="WER by Word Count")
    wc_table.add_column("Model", style="cyan")
    wc_table.add_column("Corpus", justify="right", style="bold")
    for i in range(1, 11):
        wc_table.add_column(f"{i} word{'s' if i > 1 else ''}", justify="right")

    for model, data in model_metrics.items():
        display_name = data.get("display_name", model)
        corpus_wer = data.get("corpus_wer")
        row = [display_name]
        row.append(f"{corpus_wer:.2f}%" if corpus_wer else "-")
        for wc in range(1, 11):
            wc_data = data["by_length"].get(wc, {})
            wers = wc_data.get("wers", [])
            if wers:
                avg_wer = sum(wers) / len(wers)
                row.append(f"{avg_wer:.1f}%")
            else:
                row.append("-")
        wc_table.add_row(*row)

    console.print(wc_table)

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

        for model, data in model_metrics.items():
            display_name = data.get("display_name", model)
            diar = data.get("diarization", {})
            if diar:
                diar_table.add_row(
                    display_name,
                    f"{diar.get('der', 0):.2f}%",
                    f"{diar.get('confusion', 0):.2f}%",
                    f"{diar.get('missed', 0):.2f}%",
                    f"{diar.get('false_alarm', 0):.2f}%",
                )
            else:
                diar_table.add_row(display_name, "-", "-", "-", "-")

        console.print(diar_table)

    # === Alignment Table ===
    has_alignment = any(m.get("alignment") for m in model_metrics.values())
    if has_alignment:
        console.print("\n")
        align_table = Table(title="Timestamp Alignment")
        align_table.add_column("Model", style="cyan")
        align_table.add_column("MAE (ms)", justify="right")
        align_table.add_column("Alignment Error", justify="right")

        for model, data in model_metrics.items():
            display_name = data.get("display_name", model)
            align = data.get("alignment", {})
            if align:
                mae = align.get("mae", 0)
                align_err = align.get("alignment_error", 0)
                align_table.add_row(
                    display_name,
                    f"{mae * 1000:.1f}",
                    f"{align_err * 100:.2f}%",
                )
            else:
                align_table.add_row(display_name, "-", "-")

        console.print(align_table)

    # === Entity Errors Table ===
    # Get all entity types across models
    all_entity_types = set()
    for m in model_metrics.values():
        all_entity_types.update(m["entity_errors"].keys())

    if all_entity_types:
        # Order entity types by frequency
        entity_type_order = [
            "CARDINAL",
            "DATE",
            "GPE",
            "PERSON",
            "ORG",
            "NORP",
            "ORDINAL",
            "TIME",
            "QUANTITY",
            "LOC",
            "MONEY",
            "PERCENT",
        ]
        ordered_entity_types = [t for t in entity_type_order if t in all_entity_types]
        ordered_entity_types += [t for t in sorted(all_entity_types) if t not in entity_type_order]

        console.print("\n")
        entity_table = Table(title="Missed Entity Errors")
        entity_table.add_column("Model", style="cyan")
        entity_table.add_column("Average", justify="right", style="bold")
        for etype in ordered_entity_types:
            entity_table.add_column(etype, justify="right")

        for model, data in model_metrics.items():
            display_name = data.get("display_name", model)
            row = [display_name]
            # Calculate average entity error rate
            total_found = sum(e["found"] for e in data["entity_errors"].values())
            total_entities = sum(e["total"] for e in data["entity_errors"].values())
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
            entity_table.add_row(*row)

        console.print(entity_table)

    # === ITN Formatting Errors Table ===
    all_itn_types = set()
    for m in model_metrics.values():
        all_itn_types.update(m["itn_errors"].keys())

    if all_itn_types:
        itn_type_order = ["CARDINAL", "DATE", "TIME", "MONEY", "PERCENT", "ORDINAL", "QUANTITY"]
        ordered_itn_types = [t for t in itn_type_order if t in all_itn_types]
        ordered_itn_types += [t for t in sorted(all_itn_types) if t not in itn_type_order]

        console.print("\n")
        itn_table = Table(title="ITN Formatting Errors")
        itn_table.add_column("Model", style="cyan")
        itn_table.add_column("Average", justify="right", style="bold")
        for itype in ordered_itn_types:
            itn_table.add_column(itype, justify="right")

        for model, data in model_metrics.items():
            display_name = data.get("display_name", model)
            row = [display_name]
            # Calculate average ITN error rate
            total_correct = sum(e["correct"] for e in data["itn_errors"].values())
            total_itn = sum(e["total"] for e in data["itn_errors"].values())
            if total_itn > 0:
                avg_itn_err = (total_itn - total_correct) / total_itn * 100
                row.append(f"{avg_itn_err:.2f}%")
            else:
                row.append("-")

            for itype in ordered_itn_types:
                stats = data["itn_errors"].get(itype, {"correct": 0, "total": 0})
                if stats["total"] > 0:
                    err = (stats["total"] - stats["correct"]) / stats["total"] * 100
                    row.append(f"{err:.2f}%")
                else:
                    row.append("-")
            itn_table.add_row(*row)

        console.print(itn_table)


def cli():
    """Entry point for pyproject.toml."""
    app()


if __name__ == "__main__":
    app()
