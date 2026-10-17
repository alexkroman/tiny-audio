"""Write `ta eval` run directories for the analysis tests.

`ta analysis` reads what `ta eval` leaves under outputs/: one directory per
(model, dataset) run holding metrics.txt (with the sweep's Run ID) and
results.txt (one block per sample).
"""

from pathlib import Path


def write_run(
    root: Path,
    ts: str,
    model: str,
    ds: str,
    run_id: str | None,
    pairs: list[tuple[str, str]],
    wer: float = 0.0,
) -> Path:
    """Write one run of `model` on `ds` scoring (reference, prediction) `pairs`."""
    d = root / f"{ts}_{model}_{ds}"
    d.mkdir(parents=True)
    header = f"Model: {model}\nDataset: {ds}\nTimestamp: {ts}\n"
    if run_id:
        header += f"Run ID: {run_id}\n"
    (d / "metrics.txt").write_text(header + "-" * 40 + f"\nwer: {wer}\nnum_samples: {len(pairs)}\n")
    (d / "results.txt").write_text(
        "".join(
            f"Sample {i} - WER: 0.00%\nGround Truth: {r}\nPrediction: {p}\n"
            f"Ground Truth Raw: {r}\nPrediction Raw: {p}\n" + "-" * 80 + "\n"
            for i, (r, p) in enumerate(pairs, 1)
        )
    )
    return d
