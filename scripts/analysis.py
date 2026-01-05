#!/usr/bin/env python3
"""Analysis tools for ASR evaluation results.

Commands:
    by-length    - WER breakdown by utterance length
    keyword      - Named entity/keyword WER analysis
"""

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


# =============================================================================
# by-length command
# =============================================================================


@app.command("by-length")
def by_length(
    model: str = typer.Argument(..., help="Model name pattern to match"),
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
    max_words: int = typer.Option(10, help="Max word count to show individually"),
    exclude: list[str] = typer.Option([], help="Patterns to exclude from matching"),
):
    """Calculate WER broken down by reference utterance length."""
    if not outputs_dir.exists():
        console.print(f"[red]Error: {outputs_dir} does not exist[/red]")
        raise typer.Exit(1)

    model_dirs = find_model_dirs(outputs_dir, model, exclude)
    if not model_dirs:
        console.print(f"[red]No output directories found matching '{model}'[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(model_dirs)} directories matching '{model}'")

    # Collect all samples
    all_samples = []
    for dir_path in model_dirs:
        results_file = dir_path / "results.txt"
        if results_file.exists():
            samples = parse_results_file(results_file)
            for s in samples:
                s["dataset"] = dir_path.name.split("_")[1]
            all_samples.extend(samples)

    if not all_samples:
        console.print("[red]No samples found in results files[/red]")
        raise typer.Exit(1)

    # Group by word count
    by_wc = defaultdict(list)
    for s in all_samples:
        wc = min(s["word_count"], max_words + 1)
        by_wc[wc].append(s)

    # Build table
    table = Table(title=f"WER by Utterance Length: {model}")
    table.add_column("Words", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("Perfect", justify="right")
    table.add_column("Failures", justify="right")

    for wc in sorted(by_wc.keys()):
        samples = by_wc[wc]
        n = len(samples)
        avg_wer = sum(s["wer"] for s in samples) / n
        perfect = sum(1 for s in samples if s["wer"] == 0)
        failures = sum(1 for s in samples if s["wer"] == 100)
        label = f"{wc}" if wc <= max_words else f"{max_words}+"
        table.add_row(
            label,
            str(n),
            f"{avg_wer:.1f}%",
            f"{perfect} ({perfect / n * 100:.1f}%)",
            f"{failures} ({failures / n * 100:.1f}%)",
        )

    # Total row
    n = len(all_samples)
    avg_wer = sum(s["wer"] for s in all_samples) / n
    perfect = sum(1 for s in all_samples if s["wer"] == 0)
    failures = sum(1 for s in all_samples if s["wer"] == 100)
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{n}[/bold]",
        f"[bold]{avg_wer:.1f}%[/bold]",
        f"[bold]{perfect} ({perfect / n * 100:.1f}%)[/bold]",
        f"[bold]{failures} ({failures / n * 100:.1f}%)[/bold]",
    )

    console.print(table)


# =============================================================================
# keyword commands
# =============================================================================

keyword_app = typer.Typer(help="Named entity/keyword WER analysis")
app.add_typer(keyword_app, name="keyword")


@keyword_app.command("extract")
def keyword_extract(
    model: str = typer.Option("tiny-audio", help="Model pattern to extract from"),
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
    exclude: list[str] = typer.Option(
        ["glm", "moe", "mosa", "qformer", "swiglu", "stage2"], help="Patterns to exclude"
    ),
    min_count: int = typer.Option(20, help="Minimum entity count to include a type"),
):
    """Extract named entities from reference texts and save to keywords.json."""
    import spacy

    console.print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    model_dirs = find_model_dirs(outputs_dir, model, exclude)
    results_files = [d / "results.txt" for d in model_dirs if (d / "results.txt").exists()]
    console.print(f"Found {len(results_files)} results files")

    all_references = {}
    entity_counts = defaultdict(int)

    for results_file in sorted(results_files):
        dataset = results_file.parent.name.split("_")[1]
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
                all_references[gt] = {"entities": entities, "datasets": []}
                for ent in entities:
                    entity_counts[ent["label"]] += 1
            all_references[gt]["datasets"].append(dataset)

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
                "datasets": list(set(data["datasets"])),
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


@keyword_app.command("calculate")
def keyword_calculate(
    model: str = typer.Argument(..., help="Model name pattern to match"),
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
    exclude: list[str] = typer.Option([], help="Patterns to exclude from matching"),
    by_type: bool = typer.Option(False, "--by-type", help="Show breakdown by entity type"),
):
    """Calculate keyword WER for a model using pre-extracted entities."""
    keywords_path = Path(KEYWORDS_FILE)
    if not keywords_path.exists():
        console.print(f"[red]Keywords file not found: {keywords_path}[/red]")
        console.print("Run 'analysis keyword extract' first")
        raise typer.Exit(1)

    keywords = json.loads(keywords_path.read_text())
    ref_entities = {ref["text"]: ref["entities"] for ref in keywords["references"]}

    model_dirs = find_model_dirs(outputs_dir, model, exclude)
    if not model_dirs:
        console.print(f"[red]No output directories found matching '{model}'[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(model_dirs)} directories matching '{model}'")

    results_by_type = defaultdict(lambda: {"found": 0, "total": 0})
    misses = []

    for dir_path in sorted(model_dirs):
        results_file = dir_path / "results.txt"
        if not results_file.exists():
            continue

        for sample in parse_results_file(results_file):
            gt, pred = sample["ground_truth"], sample["prediction"]
            if gt in ref_entities:
                for entity in ref_entities[gt]:
                    found = entity_in_text(entity["text"], pred)
                    results_by_type[entity["label"]]["total"] += 1
                    if found:
                        results_by_type[entity["label"]]["found"] += 1
                    else:
                        misses.append(
                            {"entity": entity["text"], "type": entity["label"], "pred": pred}
                        )

    total_found = sum(r["found"] for r in results_by_type.values())
    total_entities = sum(r["total"] for r in results_by_type.values())

    if total_entities == 0:
        console.print("[red]No entities found in model outputs[/red]")
        raise typer.Exit(1)

    # Results table
    table = Table(title=f"Keyword WER: {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total entities", str(total_entities))
    table.add_row(
        "Correctly transcribed", f"{total_found} ({total_found / total_entities * 100:.1f}%)"
    )
    table.add_row("Keyword WER", f"{100 - total_found / total_entities * 100:.1f}%")
    console.print(table)

    if by_type:
        type_table = Table(title="By Entity Type")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Found", justify="right")
        type_table.add_column("Total", justify="right")
        type_table.add_column("Accuracy", justify="right")
        type_table.add_column("WER", justify="right")

        for etype in sorted(results_by_type.keys(), key=lambda x: -results_by_type[x]["total"]):
            stats = results_by_type[etype]
            acc = stats["found"] / stats["total"] * 100 if stats["total"] > 0 else 0
            type_table.add_row(
                etype, str(stats["found"]), str(stats["total"]), f"{acc:.1f}%", f"{100 - acc:.1f}%"
            )
        console.print(type_table)

    if misses:
        console.print("\n[bold]Sample missed entities (up to 10):[/bold]")
        for miss in misses[:10]:
            console.print(f"  [{miss['type']}] '{miss['entity']}'")
            console.print(f"    [dim]Pred: {miss['pred'][:80]}...[/dim]")


def cli():
    """Entry point for pyproject.toml."""
    app()


if __name__ == "__main__":
    app()
