#!/usr/bin/env python3
"""Debug and analysis CLI."""

import typer

from scripts.debug.analyze_lora import main as analyze_lora_command
from scripts.debug.analyze_weights import main as analyze_weights_command
from scripts.debug.check_gradient_flow import main as check_gradient_flow_command
from scripts.debug.compare_to_base import main as compare_to_base_command

app = typer.Typer(
    name="debug",
    help="Inspect the weights and gradients of a checkpoint.",
    no_args_is_help=True,
    add_completion=False,
)

app.command(name="analyze-lora")(analyze_lora_command)
app.command(name="analyze-weights")(analyze_weights_command)
app.command(name="compare-to-base")(compare_to_base_command)
app.command(name="check-gradient-flow")(check_gradient_flow_command)

if __name__ == "__main__":
    app()
