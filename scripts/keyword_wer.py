#!/usr/bin/env python3
"""Calculate WER for named entities/keywords in ASR outputs.

Two-step process:
1. Extract entities from reference texts once: --extract
2. Calculate keyword WER for a model: <model_pattern>
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import typer

app = typer.Typer()

KEYWORDS_FILE = "outputs/keywords.json"


def parse_results_file(results_path: Path) -> list[dict]:
    """Parse a results.txt file and return list of samples."""
    samples = []
    content = results_path.read_text()
    blocks = content.split("-" * 80)

    for block in blocks:
        sample_match = re.search(r"Sample (\d+) - WER: ([\d.]+)%", block)
        gt_match = re.search(r"Ground Truth: (.+?)(?:\n|$)", block)
        pred_match = re.search(r"Prediction:\s*(.+?)(?:\n|$)", block)

        if sample_match and gt_match and pred_match:
            wer = float(sample_match.group(2))
            ground_truth = gt_match.group(1).strip()
            prediction = pred_match.group(1).strip()

            samples.append(
                {
                    "sample_num": int(sample_match.group(1)),
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "wer": wer,
                }
            )

    return samples


def extract_entities(text: str, nlp) -> list[dict]:
    """Extract named entities from text using spaCy."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )
    return entities


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

NUM_TO_WORD = {v: k for k, v in WORD_TO_NUM.items()}


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase, remove punctuation, normalize whitespace
    text = text.lower()
    # Normalize percent symbols
    text = text.replace("%", " percent").replace("per cent", "percent")
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_numbers(text: str) -> str:
    """Convert number words to digits for flexible matching."""
    text = normalize_text(text)
    words = text.split()
    result = []
    for word in words:
        if word in WORD_TO_NUM:
            result.append(WORD_TO_NUM[word])
        else:
            result.append(word)
    return " ".join(result)


def entity_in_text(entity_text: str, text: str) -> bool:
    """Check if entity appears in text (normalized comparison).

    Tries multiple normalization strategies:
    1. Basic text normalization
    2. Number word -> digit conversion (e.g., "seventy one" -> "71")
    """
    norm_entity = normalize_text(entity_text)
    norm_text = normalize_text(text)

    # Direct match
    if norm_entity in norm_text:
        return True

    # Try with number normalization (both directions)
    num_entity = normalize_numbers(entity_text)
    num_text = normalize_numbers(text)

    if num_entity in num_text:
        return True

    # Try matching individual number words to digits
    # e.g., entity "71%" should match "seventy one percent"
    entity_words = num_entity.split()
    text_words = num_text.split()

    # Check if all entity words appear in sequence in text
    if len(entity_words) <= len(text_words):
        for i in range(len(text_words) - len(entity_words) + 1):
            if text_words[i : i + len(entity_words)] == entity_words:
                return True

    return False


@app.command()
def extract(
    model: str = typer.Option(
        "tiny-audio", help="Model pattern to extract from (any model works, refs are same)"
    ),
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
    exclude: list[str] = typer.Option(
        ["glm", "moe", "mosa", "qformer", "swiglu", "stage2"], help="Patterns to exclude"
    ),
    min_count: int = typer.Option(20, help="Minimum entity count to include a type"),
):
    """Extract all named entities from reference texts and save to keywords.json."""
    import spacy

    typer.echo("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Find matching model directories
    model_dirs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        if model.lower() in d.name.lower() and not any(
            ex.lower() in d.name.lower() for ex in exclude
        ):
            model_dirs.append(d)

    results_files = [d / "results.txt" for d in model_dirs if (d / "results.txt").exists()]
    typer.echo(f"Found {len(results_files)} results files")

    # Collect all unique reference texts with their entities
    all_references = {}  # text -> {entities, datasets}
    entity_counts = defaultdict(int)

    for results_file in sorted(results_files):
        dataset = results_file.parent.name.split("_")[1]
        samples = parse_results_file(results_file)

        for sample in samples:
            gt = sample["ground_truth"]
            if gt not in all_references:
                entities = extract_entities(gt, nlp)
                all_references[gt] = {
                    "entities": entities,
                    "datasets": [],
                }
                for ent in entities:
                    entity_counts[ent["label"]] += 1

            all_references[gt]["datasets"].append(dataset)

    # Filter out entity types with low counts
    valid_types = {t for t, c in entity_counts.items() if c >= min_count}
    filtered_counts = {t: c for t, c in entity_counts.items() if t in valid_types}
    excluded_types = {t: c for t, c in entity_counts.items() if t not in valid_types}

    if excluded_types:
        typer.echo(
            f"\nExcluding types with < {min_count} entities: {', '.join(excluded_types.keys())}"
        )

    # Build keyword list
    keywords = {
        "total_references": len(all_references),
        "entity_counts_by_type": filtered_counts,
        "min_count_threshold": min_count,
        "excluded_types": excluded_types,
        "references": [],
    }

    for gt_text, data in all_references.items():
        # Filter entities to only include valid types
        filtered_entities = [e for e in data["entities"] if e["label"] in valid_types]
        if filtered_entities:
            keywords["references"].append(
                {
                    "text": gt_text,
                    "entities": filtered_entities,
                    "datasets": list(set(data["datasets"])),
                }
            )

    # Save to file
    keywords_path = Path(KEYWORDS_FILE)
    keywords_path.parent.mkdir(exist_ok=True)
    with keywords_path.open("w") as f:
        json.dump(keywords, f, indent=2)

    typer.echo(f"\nExtracted entities from {len(all_references)} unique references")
    typer.echo(f"References with entities: {len(keywords['references'])}")
    typer.echo("\nEntity counts by type:")
    for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        typer.echo(f"  {label}: {count}")
    typer.echo(f"\nSaved to {keywords_path}")


@app.command()
def calculate(
    model: str = typer.Argument(..., help="Model name pattern to match"),
    outputs_dir: Path = typer.Option(Path("outputs"), help="Directory containing eval results"),
    exclude: list[str] = typer.Option([], help="Patterns to exclude from matching"),
    by_type: bool = typer.Option(False, help="Show breakdown by entity type"),
):
    """Calculate keyword WER for a model using pre-extracted entities."""
    keywords_path = Path(KEYWORDS_FILE)
    if not keywords_path.exists():
        typer.echo(f"Keywords file not found: {keywords_path}")
        typer.echo("Run 'keyword-wer extract' first to extract entities")
        raise typer.Exit(1)

    with keywords_path.open() as f:
        keywords = json.load(f)

    # Build lookup: reference text -> entities
    ref_entities = {}
    for ref in keywords["references"]:
        ref_entities[ref["text"]] = ref["entities"]

    # Find model directories
    model_dirs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        if model.lower() in d.name.lower() and not any(
            ex.lower() in d.name.lower() for ex in exclude
        ):
            model_dirs.append(d)

    if not model_dirs:
        typer.echo(f"No output directories found matching '{model}'")
        raise typer.Exit(1)

    typer.echo(f"Found {len(model_dirs)} directories matching '{model}'")

    # Calculate keyword accuracy
    results_by_type = defaultdict(lambda: {"found": 0, "total": 0})
    all_entity_results = []

    for dir_path in sorted(model_dirs):
        results_file = dir_path / "results.txt"
        if not results_file.exists():
            continue

        samples = parse_results_file(results_file)

        for sample in samples:
            gt = sample["ground_truth"]
            pred = sample["prediction"]

            if gt in ref_entities:
                for entity in ref_entities[gt]:
                    entity_text = entity["text"]
                    entity_type = entity["label"]
                    found = entity_in_text(entity_text, pred)

                    results_by_type[entity_type]["total"] += 1
                    if found:
                        results_by_type[entity_type]["found"] += 1

                    all_entity_results.append(
                        {
                            "entity": entity_text,
                            "type": entity_type,
                            "found": found,
                            "ground_truth": gt,
                            "prediction": pred,
                        }
                    )

    # Calculate overall stats
    total_found = sum(r["found"] for r in results_by_type.values())
    total_entities = sum(r["total"] for r in results_by_type.values())

    if total_entities == 0:
        typer.echo("No entities found in model outputs")
        raise typer.Exit(1)

    overall_accuracy = total_found / total_entities * 100
    keyword_wer = 100 - overall_accuracy

    typer.echo(f"\nKeyword WER: {model}")
    typer.echo("=" * 60)
    typer.echo(f"Total entities: {total_entities}")
    typer.echo(f"Correctly transcribed: {total_found} ({overall_accuracy:.1f}%)")
    typer.echo(f"Keyword WER: {keyword_wer:.1f}%")

    if by_type:
        typer.echo("\nBy Entity Type:")
        typer.echo("-" * 60)
        typer.echo(f"{'Type':<12} {'Found':<10} {'Total':<10} {'Accuracy':<10} {'WER':<10}")
        typer.echo("-" * 60)

        for etype in sorted(results_by_type.keys(), key=lambda x: -results_by_type[x]["total"]):
            stats = results_by_type[etype]
            acc = stats["found"] / stats["total"] * 100 if stats["total"] > 0 else 0
            wer = 100 - acc
            typer.echo(
                f"{etype:<12} {stats['found']:<10} {stats['total']:<10} {acc:<10.1f} {wer:<10.1f}"
            )

    # Show some misses
    misses = [r for r in all_entity_results if not r["found"]]
    if misses:
        typer.echo("\nSample missed entities (showing up to 10):")
        typer.echo("-" * 60)
        for miss in misses[:10]:
            typer.echo(f"  [{miss['type']}] '{miss['entity']}'")
            typer.echo(f"    Pred: {miss['prediction'][:80]}...")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Calculate WER for named entities/keywords."""
    if ctx.invoked_subcommand is None:
        typer.echo("Use 'extract' to build keyword list or 'calculate <model>' to compute WER")
        typer.echo("Run with --help for more info")


if __name__ == "__main__":
    app()
