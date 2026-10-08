#!/usr/bin/env python3
"""Debug CLI."""

import typer

from scripts.debug.check_gradient_flow import main as check_gradient_flow_command

app = typer.Typer(
    name="debug",
    help="Check gradient flow through a model.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Check gradient flow through a model."""
    # A callback keeps `ta debug` a group with `check-gradient-flow` as a
    # subcommand; with one command and no callback, Typer would collapse it.


app.command(name="check-gradient-flow")(check_gradient_flow_command)

if __name__ == "__main__":
    app()
