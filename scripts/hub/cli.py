#!/usr/bin/env python3
"""HuggingFace Hub CLI.

Commands:
    push    Push model files to HuggingFace Hub
"""

import typer

from scripts.hub.push import main as push_command

app = typer.Typer(
    name="hub",
    help="HuggingFace Hub operations",
    no_args_is_help=True,
)

app.command(name="push", help="Push model files to HuggingFace Hub")(push_command)

if __name__ == "__main__":
    app()
