"""The model x dataset tables `ta analysis compare` prints."""

from collections.abc import Callable, Iterable

from rich.table import Table

from scripts.analysis.common import console, sort_key, sort_key_desc
from scripts.analysis.metrics import DatasetMetrics, ModelMetrics, dataset_wer

# Datasets left out of the comparison tables.
EXCLUDED_DATASETS = {"classification", "expresso"}

# Short names for display, in canonical column order.
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
}

Models = dict[str, ModelMetrics]


def ordered_datasets(model_metrics: Models) -> list[str]:
    """Every dataset any model has, canonical ones first."""
    present: set[str] = set().union(*(m["datasets"] for m in model_metrics.values()))
    present -= EXCLUDED_DATASETS
    known = [d for d in DATASET_SHORT_NAMES if d in present]
    return known + sorted(present - set(known))


def _new_table(title: str, columns: Iterable[str], bold_first: int = 0) -> Table:
    """Table with a cyan Model column, then right-aligned `columns`."""
    console.print("\n")
    table = Table(title=title)
    table.add_column("Model", style="cyan")
    for i, name in enumerate(columns):
        table.add_column(name, justify="right", style="bold" if i < bold_first else None)
    return table


def _print_rows(table: Table, rows: list[list[str]], col: int = 1, descending: bool = False):
    """Add `rows` best first by column `col`, missing ("-") values last."""
    key = sort_key_desc if descending else sort_key
    for row in sorted(rows, key=lambda r: key(r[col])):
        table.add_row(*row)
    console.print(table)


def _pct(value: float | None, digits: int = 2) -> str:
    """Format a percentage to `digits` places, or "-" when there is no value."""
    return "-" if value is None else f"{value:.{digits}f}%"


def print_dataset_table(
    model_metrics: Models,
    datasets: list[str],
    title: str,
    summary: tuple[str, Callable[[ModelMetrics], float | None]],
    dataset_value: Callable[[DatasetMetrics], float | None],
    fmt: Callable[[float], str],
) -> None:
    """Model x dataset table with a leading summary column, best first."""
    summary_column, summary_value = summary
    names = [DATASET_SHORT_NAMES.get(ds, ds) for ds in datasets]
    table = _new_table(title, [summary_column, *names], bold_first=1)
    rows: list[list[str]] = []
    for data in model_metrics.values():
        cells = [data["datasets"].get(ds) for ds in datasets]
        values = [summary_value(data)] + [None if c is None else dataset_value(c) for c in cells]
        rows.append([data["display_name"], *("-" if v is None else fmt(v) for v in values)])
    _print_rows(table, rows)


def print_word_count_table(model_metrics: Models) -> None:
    """Mean per-utterance WER for references of 1-10 words."""
    table = _new_table(
        "WER by Word Count", [f"{i} word{'s' if i > 1 else ''}" for i in range(1, 11)]
    )
    rows: list[list[str]] = []
    for data in model_metrics.values():
        cells = [data["display_name"]]
        for wc in range(1, 11):
            wers = data["by_length"].get(wc, [])
            cells.append(_pct(sum(wers) / len(wers) if wers else None, digits=1))
        rows.append(cells)
    _print_rows(table, rows)


def print_speaker_tables(model_metrics: Models, datasets: list[str]) -> None:
    """cpWER and speaker counting, per speaker-labelled dataset, as `ta eval` scored them.

    Scored here from the `<SPK_n>` raw transcripts with the harness's own
    `cp_errors`, over the rows the models share (see `match_dataset_rows`).
    """
    for ds in datasets:
        runs = [
            (m["display_name"], m["datasets"][ds])
            for m in model_metrics.values()
            if ds in m["datasets"] and "cpwer" in m["datasets"][ds]
        ]
        if not runs:
            continue
        table = _new_table(
            f"Speaker Attribution: {DATASET_SHORT_NAMES.get(ds, ds)}",
            ["cpWER", "WER", "Attribution gap", "Speaker count acc"],
            bold_first=1,
        )
        rows: list[list[str]] = []
        for name, d in runs:
            acc = d.get("speaker_count_acc")
            gap = d.get("attribution_gap")
            rows.append(
                [
                    name,
                    _pct(d.get("cpwer")),
                    _pct(dataset_wer(d)),
                    "-" if gap is None else f"{gap:+.2f}",
                    "-" if acc is None else f"{acc * 100:.1f}%",
                ]
            )
        _print_rows(table, rows)
        console.print(
            "[dim]cpWER = WER after optimally mapping hypothesis to reference speakers; "
            "attribution gap = cpWER - WER, the cost of speaker errors alone.[/dim]"
        )
