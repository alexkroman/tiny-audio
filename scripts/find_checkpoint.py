#!/usr/bin/env python3
"""Find the latest checkpoint on a remote training server."""

import subprocess
import sys

import typer


def main(
    host: str = typer.Argument(..., help="Remote server IP or hostname"),
    port: int = typer.Argument(22, help="SSH port"),
):
    """Find the latest checkpoint on a remote training server."""
    cmd = [
        "ssh",
        "-p",
        str(port),
        f"root@{host}",
        "find /workspace/outputs -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -1",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        raise typer.Exit(1)

    checkpoint = result.stdout.strip()
    if checkpoint:
        print(checkpoint)
    else:
        print("No checkpoints found", file=sys.stderr)
        raise typer.Exit(1)


def cli():
    """Entry point for poetry script."""
    typer.run(main)


if __name__ == "__main__":
    cli()
