#!/usr/bin/env python3
"""Update model config on Hugging Face Hub based on local Hydra config."""

import os
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi, hf_hub_download
from rich.console import Console

app = typer.Typer(help="Update model config on Hugging Face Hub")
console = Console()


@app.command()
def main(
    repo_id: Annotated[
        str,
        typer.Option("--repo-id", "-r", help="Hugging Face repository ID"),
    ],
    config_path: Annotated[
        str,
        typer.Option(
            "--config",
            "-c",
            help="Path to Hydra config file (e.g., configs/experiments/omni.yaml)",
        ),
    ] = "configs/experiments/omni.yaml",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show changes without uploading"),
    ] = False,
):
    """Update model config on Hub based on local Hydra config.

    Reads the local Hydra experiment config and updates the corresponding
    fields in the model's config.json on the Hub.

    Examples:
        # Update config for omni model
        ta hub update-config -r mazesmazes/tiny-audio-omni

        # Dry run to see changes
        ta hub update-config -r mazesmazes/tiny-audio-omni --dry-run
    """
    import json

    from omegaconf import OmegaConf

    # Check for HF_TOKEN
    if not os.environ.get("HF_TOKEN") and not dry_run:
        console.print("[red]Error: HF_TOKEN environment variable must be set[/red]")
        raise typer.Exit(1)

    # Load local Hydra config
    if not Path(config_path).exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Loading local config: {config_path}[/dim]")
    local_config = OmegaConf.load(config_path)
    model_config = local_config.get("model", {})

    if not model_config:
        console.print("[red]Error: No 'model' section in config[/red]")
        raise typer.Exit(1)

    # Download current config.json from Hub
    console.print(f"[dim]Downloading config.json from {repo_id}[/dim]")
    try:
        config_file = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            token=os.environ.get("HF_TOKEN"),
        )
        with Path(config_file).open() as f:
            hub_config = json.load(f)
    except Exception as e:
        console.print(f"[red]Error downloading config.json: {e}[/red]")
        raise typer.Exit(1) from e

    # Fields to sync from local config to Hub
    sync_fields = [
        "enable_thinking",
        "do_sample",
        "max_new_tokens",
        "top_k",
        "top_p",
        "temperature",
        "repetition_penalty",
        "projector_type",
        "projector_pool_stride",
        "projector_hidden_dim",
    ]

    # Track changes
    changes = []
    for field in sync_fields:
        if field in model_config:
            local_value = model_config[field]
            hub_value = hub_config.get(field)
            if hub_value != local_value:
                changes.append((field, hub_value, local_value))
                hub_config[field] = local_value

    if not changes:
        console.print("[green]No changes needed - config is already in sync[/green]")
        return

    # Show changes
    console.print("\n[bold]Config changes:[/bold]")
    for field, old_val, new_val in changes:
        console.print(f"  {field}: [red]{old_val}[/red] -> [green]{new_val}[/green]")

    if dry_run:
        console.print("\n[yellow]Dry run - no changes uploaded[/yellow]")
        return

    # Upload updated config
    import tempfile

    api = HfApi(token=os.environ["HF_TOKEN"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hub_config, f, indent=2)
        temp_path = f.name

    try:
        console.print(f"\n[dim]Uploading updated config.json to {repo_id}[/dim]")
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo="config.json",
            repo_id=repo_id,
            commit_message=f"Update config: {', '.join(f[0] for f in changes)}",
        )
        console.print("[green]Config updated successfully![/green]")
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    app()
