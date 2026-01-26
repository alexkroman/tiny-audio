#!/usr/bin/env python3
"""Debug and analysis CLI.

Commands:
    check-moe        Check MoE model for router health
    check-mosa       Check MOSA model for router collapse
    analyze-lora     Analyze LoRA adapter weights
    analyze-weights  Analyze model weights for training health
"""

import typer

from scripts.debug.analyze_lora import main as analyze_lora_command
from scripts.debug.analyze_weights import main as analyze_weights_command
from scripts.debug.check_moe import main as check_moe_command
from scripts.debug.check_mosa import main as check_mosa_command

app = typer.Typer(
    name="debug",
    help="Debug and analysis tools",
    no_args_is_help=True,
)

app.command(name="check-moe", help="Check MoE model for router health")(check_moe_command)
app.command(name="check-mosa", help="Check MOSA model for router collapse")(check_mosa_command)
app.command(name="analyze-lora", help="Analyze LoRA adapter weights")(analyze_lora_command)
app.command(name="analyze-weights", help="Analyze model weights for training health")(
    analyze_weights_command
)

if __name__ == "__main__":
    app()
