#!/usr/bin/env python3
"""Debug and analysis CLI.

Commands:
    analyze-weights  Analyze model weights for training health
"""

import typer

from scripts.debug.analyze_weights import main as analyze_weights_command

app = typer.Typer(
    name="debug",
    help="Debug and analysis tools",
    no_args_is_help=True,
)

app.command(name="analyze-weights", help="Analyze model weights for training health")(
    analyze_weights_command
)

if __name__ == "__main__":
    app()
