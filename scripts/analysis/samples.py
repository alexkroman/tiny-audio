"""Per-sample reports: high-WER rows and rows that missed a reference entity."""

from pathlib import Path
from typing import Annotated

import typer

from scripts.analysis import common
from scripts.analysis.common import (
    MODEL_ARG_HELP,
    Entity,
    ExcludeOption,
    OutputDirOption,
    console,
    entity_in_text,
    extract_dataset_name,
)
from scripts.utils import find_model_dirs, parse_results_file

LatestOption = Annotated[
    bool, typer.Option("--latest", help="Only use the most recent run per dataset")
]
OutputFileOption = Annotated[
    Path | None, typer.Option("--output-file", help="Write the report here instead of stdout")
]
RULE = "-" * 80


def _results_files(
    model: str, output_dir: Path, exclude: list[str] | None, latest: bool
) -> list[Path]:
    """Every results.txt for `model`, exiting with an error when there are none."""
    model_dirs = find_model_dirs(output_dir, model, exclude, latest=latest)
    results_files = sorted(d / "results.txt" for d in model_dirs if (d / "results.txt").exists())
    if not results_files:
        console.print(f"[red]No results files found for pattern '{model}'[/red]")
        raise typer.Exit(1)
    console.print(f"Found {len(results_files)} results files for '{model}'")
    return results_files


def _emit(lines: list[str], output_file: Path | None) -> None:
    """Write the report to `output_file`, or print it."""
    output_text = "\n".join([*lines, RULE])
    if output_file:
        output_file.write_text(output_text)
        console.print(f"Saved to [bold]{output_file}[/bold]")
    else:
        console.print(output_text)


def high_wer(
    model: Annotated[str, typer.Argument(help=MODEL_ARG_HELP)],
    threshold: Annotated[float, typer.Option("--threshold", help="WER threshold (percent)")] = 50.0,
    output_dir: OutputDirOption = Path("outputs"),
    exclude: ExcludeOption = None,
    latest: LatestOption = False,
    output_file: OutputFileOption = None,
) -> None:
    """Output ground truth and predictions for samples with WER above threshold."""
    console.print(f"Filtering samples with WER >= {threshold}%\n")
    rows = [
        (extract_dataset_name(results_file.parent.name), sample)
        for results_file in _results_files(model, output_dir, exclude, latest)
        for sample in parse_results_file(results_file)
        if sample["wer"] >= threshold
    ]
    rows.sort(key=lambda row: row[1]["wer"], reverse=True)
    console.print(f"Found {len(rows)} samples with WER >= {threshold}%\n")

    lines = [
        f"# High WER Samples (>= {threshold}%)",
        f"# Model: {model}",
        f"# Total: {len(rows)} samples\n",
    ]
    for dataset, sample in rows:
        lines += [
            RULE,
            f"Dataset: {dataset} | Sample: {sample['sample_num']} | WER: {sample['wer']:.1f}%",
            f"Ground Truth: {sample['ground_truth']}",
            f"Prediction:   {sample['prediction']}",
        ]
    _emit(lines, output_file)


def _missing_entities(entities: list[Entity], prediction: str, entity_type: str) -> list[Entity]:
    """Entities of `entity_type` (or any type, when empty) absent from `prediction`."""
    return [
        e
        for e in entities
        if (not entity_type or e["label"].upper() == entity_type.upper())
        and not entity_in_text(e["text"], prediction)
    ]


def entity_errors(
    model: Annotated[str, typer.Argument(help=MODEL_ARG_HELP)],
    output_dir: OutputDirOption = Path("outputs"),
    exclude: ExcludeOption = None,
    latest: LatestOption = False,
    entity_type: Annotated[
        str, typer.Option("--entity-type", help="Filter by entity type (e.g., PERSON, ORG)")
    ] = "",
    output_file: OutputFileOption = None,
) -> None:
    """Output samples where entities were missed in the prediction."""
    ref_entities = common.load_ref_entities()
    if not ref_entities:
        console.print(f"[red]No reference entities in {common.KEYWORDS_FILE}[/red]")
        raise typer.Exit(1)
    if entity_type:
        console.print(f"Filtering for entity type: {entity_type}")

    lines: list[str] = []
    count = 0
    for results_file in _results_files(model, output_dir, exclude, latest):
        dataset = extract_dataset_name(results_file.parent.name)
        for sample in parse_results_file(results_file):
            entities = ref_entities.get(sample["ground_truth"], [])
            missing = _missing_entities(entities, sample["prediction"], entity_type)
            if not missing:
                continue
            count += 1
            lines += [
                RULE,
                f"Dataset: {dataset} | Sample: {sample['sample_num']}",
                "Missing: " + ", ".join(f"{e['text']} ({e['label']})" for e in missing),
                f"Ground Truth: {sample['ground_truth']}",
                f"Prediction:   {sample['prediction']}",
            ]
    console.print(f"Found {count} samples with missing entities\n")

    header = ["# Entity Errors (Missing Entities)", f"# Model: {model}"]
    if entity_type:
        header.append(f"# Entity Type: {entity_type}")
    header.append(f"# Total: {count} samples\n")
    _emit(header + lines, output_file)
