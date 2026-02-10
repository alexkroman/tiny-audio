#!/usr/bin/env python3
"""Assemble a full S2S model from a base ASRModel + AudioHead weights.

Downloads the base ASRModel (encoder + projector + LLM), loads AudioHead weights
from a separate Hub repo or local checkpoint, and pushes the assembled model.

Can be run multiple times safely — just overwrites the target repo.
Does not modify the AudioHead weights repo, so training can continue.

Usage:
    ta assemble-s2s --base-model mazesmazes/tiny-audio-omni \
                    --audio-head mazesmazes/tiny-audio-s2s \
                    --output mazesmazes/tiny-audio-s2s-full

    # From a local checkpoint instead of Hub
    ta assemble-s2s --base-model mazesmazes/tiny-audio-omni \
                    --audio-head ./outputs/audio_head/checkpoint-500 \
                    --output mazesmazes/tiny-audio-s2s-full
"""

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(help="Assemble full S2S model from base ASRModel + AudioHead")
console = Console()


@app.command()
def main(
    base_model: Annotated[
        str,
        typer.Option("--base-model", "-b", help="Base ASRModel (HF repo ID)"),
    ] = "mazesmazes/tiny-audio-omni",
    audio_head: Annotated[
        str,
        typer.Option("--audio-head", "-a", help="AudioHead weights (HF repo ID or local path)"),
    ] = "mazesmazes/tiny-audio-head",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output HF repo ID for assembled model"),
    ] = "mazesmazes/tiny-audio-s2s-full",
):
    """Assemble full S2S model by loading base ASRModel and attaching AudioHead weights."""
    if not os.environ.get("HF_TOKEN"):
        console.print("[red]Error: HF_TOKEN environment variable must be set[/red]")
        raise typer.Exit(1)

    from safetensors.torch import load_file

    # Step 1: Load AudioHead config
    console.print(f"\n[bold]Loading AudioHead from {audio_head}[/bold]")
    audio_head_path = Path(audio_head)

    if audio_head_path.exists():
        # Local path
        config_path = audio_head_path / "audio_head_config.json"
        weights_path = audio_head_path / "audio_head.safetensors"
        if not weights_path.exists():
            # Try Trainer checkpoint format (model.safetensors with unprefixed keys)
            weights_path = audio_head_path / "model.safetensors"
    else:
        # Download from Hub
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(audio_head, "audio_head_config.json"))
        try:
            weights_path = Path(hf_hub_download(audio_head, "audio_head.safetensors"))
        except Exception:
            weights_path = Path(hf_hub_download(audio_head, "model.safetensors"))

    if not weights_path.exists():
        console.print(f"[red]Error: No weights found at {audio_head}[/red]")
        raise typer.Exit(1)

    # Load config
    if config_path.exists():
        with config_path.open() as f:
            ah_config = json.load(f)
        console.print(f"  AudioHead config: {ah_config}")
    else:
        console.print(
            "[yellow]  No audio_head_config.json found, using defaults from base model[/yellow]"
        )
        ah_config = {}

    # Load weights
    ah_state = load_file(weights_path)
    console.print(f"  Loaded {len(ah_state)} weight tensors from {weights_path.name}")

    # Step 2: Load base ASRModel with AudioHead enabled
    console.print(f"\n[bold]Loading base model from {base_model}[/bold]")

    from tiny_audio.asr_config import ASRConfig
    from tiny_audio.asr_modeling import ASRModel

    base_config = ASRConfig.from_pretrained(base_model)
    base_config.use_audio_head = True

    # Apply AudioHead config overrides
    for k, v in ah_config.items():
        setattr(base_config, k, v)

    model = ASRModel.from_pretrained(base_model, config=base_config)
    console.print(f"  Model loaded (audio_head present: {model.audio_head is not None})")

    # Step 3: Load AudioHead weights into the model
    console.print("\n[bold]Attaching AudioHead weights[/bold]")

    # Keys may or may not have "audio_head." prefix depending on source
    has_prefix = any(k.startswith("audio_head.") for k in ah_state)

    if has_prefix:
        # Strip prefix for loading into AudioHead module directly
        ah_state_clean = {k.removeprefix("audio_head."): v for k, v in ah_state.items()}
    else:
        ah_state_clean = ah_state

    result = model.audio_head.load_state_dict(ah_state_clean, strict=False)
    if result.missing_keys:
        console.print(f"  [yellow]Missing keys: {result.missing_keys}[/yellow]")
    if result.unexpected_keys:
        console.print(f"  [yellow]Unexpected keys: {result.unexpected_keys}[/yellow]")
    console.print("  AudioHead weights loaded successfully")

    # Step 4: Push assembled model to Hub
    console.print(f"\n[bold]Pushing assembled model to {output}[/bold]")
    model.push_to_hub(output, commit_message="Assembled S2S model (base + AudioHead)")
    console.print(f"\n[green]Done! Model available at https://huggingface.co/{output}[/green]")


if __name__ == "__main__":
    app()
