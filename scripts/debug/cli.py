#!/usr/bin/env python3
"""Debug and analysis CLI.

Commands:
    check-mosa     Check MOSA model for router collapse
    analyze-lora   Analyze LoRA adapter weights
"""

import typer

from scripts.debug.analyze_lora import main as analyze_lora_command
from scripts.debug.check_mosa import main as check_mosa_command

app = typer.Typer(
    name="debug",
    help="Debug and analysis tools",
    no_args_is_help=True,
)

app.command(name="check-mosa", help="Check MOSA model for router collapse")(check_mosa_command)
app.command(name="analyze-lora", help="Analyze LoRA adapter weights")(analyze_lora_command)

if __name__ == "__main__":
    app()
