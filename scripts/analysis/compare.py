"""`ta analysis compare`: one sweep per model, paired row-for-row, in comparison tables."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from scripts.analysis import tables
from scripts.analysis.common import (
    MODEL_ARG_HELP,
    ExcludeOption,
    OutputDirOption,
    console,
    load_ref_entities,
)
from scripts.analysis.metrics import (
    ModelMetrics,
    collect_model_metrics,
    dataset_wer,
    match_dataset_rows,
    paired_bootstrap_delta,
    recompute_matched_corpus,
)


def _verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "[green]separated[/green]"
    if hi < 0:
        # Only reachable if corpus_wer and the paired statistic disagree,
        # which would mean the two were computed over different rows.
        return "[red]inconsistent — check pairing[/red]"
    return "[yellow]not separated[/yellow]"


# (corpus rate, per-utterance errors, per-utterance reference words)
CorpusSeries = tuple[float, list[int], list[int]]


def _wer_series(m: ModelMetrics) -> CorpusSeries | None:
    if "corpus_wer" not in m or not m.get("corpus_utt_errors"):
        return None
    return m["corpus_wer"], m.get("corpus_utt_errors", []), m.get("corpus_utt_ref_words", [])


def _cpwer_series(m: ModelMetrics) -> CorpusSeries | None:
    if "corpus_cpwer" not in m:
        return None
    errors = m.get("corpus_cp_utt_errors", [])
    return m["corpus_cpwer"], errors, m.get("corpus_cp_utt_ref_words", [])


def print_corpus_cis(
    model_metrics: dict[str, ModelMetrics],
    metric: str,
    series: Callable[[ModelMetrics], CorpusSeries | None],
) -> None:
    """Corpus-`metric` deltas against the best model, each with a 95% CI.

    Ranks by the corpus rate and reports every other model's gap to the leader.
    A CI that straddles zero means the sweep cannot separate the two, which is
    the finding -- reporting the point estimate alone is how a rounding error
    gets read as a win.
    """
    scored = [(m["display_name"], s) for m in model_metrics.values() if (s := series(m))]
    if len(scored) < 2:
        return
    scored.sort(key=lambda item: item[1][0])
    best_name, (_, best_errors, best_words) = scored[0]
    n_utts = len(best_errors)

    console.print("\n")
    table = Table(
        title=(
            f"Corpus {metric} delta vs {best_name}"
            f"  (paired bootstrap, n={n_utts:,} utterances, 10k resamples)"
        )
    )
    table.add_column("Model", style="cyan")
    table.add_column(f"Corpus {metric}", justify="right")
    table.add_column("Δ vs best", justify="right", style="bold")
    table.add_column("95% CI", justify="right")
    table.add_column("Verdict")

    for name, (rate, errors, words) in scored[1:]:
        if len(errors) != n_utts:
            continue
        point, lo, hi = paired_bootstrap_delta(errors, words, best_errors, best_words)
        table.add_row(
            name, f"{rate:.2f}%", f"{point:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", _verdict(lo, hi)
        )

    console.print(table)
    console.print(
        "[dim]Paired: both models resampled on the same utterance indices, so the "
        "draw's difficulty cancels. 'not separated' means this sweep cannot tell "
        "the two apart — collect more samples rather than reporting the point "
        "estimate.[/dim]"
    )


def _collect(models: list[str], output_dir: Path, exclude: list[str] | None):
    """Each model's newest sweep, exiting when a model has none."""
    model_metrics: dict[str, ModelMetrics] = {}
    for model in models:
        console.print(f"Collecting metrics for '{model}'...")
        model_metrics[model] = collect_model_metrics(model, output_dir, exclude)

    # Every model must have contributed a sweep, or the table has nothing
    # coherent to show. Runs predating Run IDs are excluded by latest_sweep.
    stale = [m for m, d in model_metrics.items() if not d["datasets"]]
    if stale:
        console.print(
            f"[red]No Run ID found for: {', '.join(stale)}[/red]\n"
            "[yellow]`ta analysis compare` only reads runs that carry a Run ID, so that "
            "every column comes from one sweep of one checkpoint. Re-run "
            "`ta eval -m <model> -d all -n <N>` to produce one.[/yellow]"
        )
        raise typer.Exit(1)
    for data in model_metrics.values():
        console.print(
            f"[dim]{data['display_name']}: sweep {data['sweep']} "
            f"({len(data['datasets'])} datasets)[/dim]"
        )
    return model_metrics


def _pair_rows(model_metrics: dict[str, ModelMetrics]) -> None:
    """Restrict the corpus and every dataset column to rows all models share."""
    # Corpus WER is only comparable across models scored on the same rows.
    recompute_matched_corpus(model_metrics, load_ref_entities())
    sample = next(iter(model_metrics.values()))
    for ds, why in sorted(sample.get("corpus_excluded", {}).items()):
        console.print(f"[yellow]Corpus excludes {ds}: {why}[/yellow]")
    # Same for every dataset column, so a column never compares an n=100 sweep
    # against n=1000.
    truncated, unmatched = match_dataset_rows(model_metrics)
    if truncated:
        console.print(
            "[yellow]Truncated to the shared row count on: "
            + ", ".join(f"{ds} {ns}" for ds, ns in truncated.items())
            + "[/yellow]"
        )
    for ds, why in unmatched.items():
        console.print(f"[yellow]{ds} not row-matched, full sweeps shown: {why}[/yellow]")


def _coverage(sample: ModelMetrics, n_datasets: int) -> str:
    """How much of the suite the corpus column pools.

    It pools only the datasets every model shares on identical rows, which can
    be far fewer than the columns beside it. Spelling the coverage into the
    title keeps a 2-of-12 pool from reading as a whole-suite number: the
    excluded corpora were the hard ones, so the cell came out below every
    dataset in its own table.
    """
    pooled = sample.get("corpus_datasets", [])
    names = ", ".join(tables.DATASET_SHORT_NAMES.get(ds, ds) for ds in pooled)
    listed = f" ({names})" if 0 < len(pooled) <= 4 else ""
    rows = len(sample.get("corpus_utt_errors", []))
    return f"corpus = {len(pooled)}/{n_datasets} datasets{listed}, {rows:,} rows"


def compare(
    models: Annotated[list[str], typer.Argument(help=f"{MODEL_ARG_HELP}s to compare")],
    output_dir: OutputDirOption = Path("outputs"),
    exclude: ExcludeOption = None,
) -> None:
    """Generate comprehensive comparison tables for multiple models."""
    model_metrics = _collect(models, output_dir, exclude)
    _pair_rows(model_metrics)
    datasets = tables.ordered_datasets(model_metrics)
    coverage = _coverage(next(iter(model_metrics.values())), len(datasets))

    tables.print_dataset_table(
        model_metrics,
        datasets,
        "Latency (ms)",
        ("Average", lambda m: m.get("avg_latency")),
        lambda ds: ds.get("avg_time"),
        lambda v: f"{v * 1000:.0f}",
    )
    tables.print_dataset_table(
        model_metrics,
        datasets,
        f"Accuracy by WER ({coverage})",
        ("Corpus", lambda m: m.get("corpus_wer")),
        dataset_wer,
        lambda v: f"{v:.2f}%",
    )
    tables.print_speaker_tables(model_metrics, datasets)
    print_corpus_cis(model_metrics, "cpWER", _cpwer_series)
    tables.print_dataset_table(
        model_metrics,
        datasets,
        f"Insertion Rate (Hallucination Proxy) ({coverage})",
        ("Corpus", lambda m: m.get("corpus_ins_rate")),
        lambda ds: ds.get("ins_rate"),
        lambda v: f"{v:.2f}%",
    )
    print_corpus_cis(model_metrics, "WER", _wer_series)
    tables.print_word_count_table(model_metrics)
    tables.print_entity_table(model_metrics)
    tables.print_itn_table(model_metrics)
