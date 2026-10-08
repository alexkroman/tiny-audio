"""Analyze and compare `ta eval` results."""

import typer

from scripts.analysis.compare import compare
from scripts.analysis.samples import entity_errors, high_wer

app = typer.Typer(
    name="analysis",
    help="Analyze and compare `ta eval` results.",
    no_args_is_help=True,
    add_completion=False,
)

app.command(name="compare")(compare)
app.command(name="high-wer")(high_wer)
app.command(name="entity-errors")(entity_errors)
