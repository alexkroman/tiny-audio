#!/usr/bin/env python3
"""Push custom model files to Hugging Face Hub."""

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    repo_id: Annotated[
        str,
        typer.Option("--repo-id", "-r", help="Hugging Face repository ID"),
    ] = "mazesmazes/tiny-audio",
    branch: Annotated[
        str,
        typer.Option("--branch", "-b", help="Branch to push to"),
    ] = "main",
    checkpoint_dir: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint-dir",
            exists=True,
            file_okay=False,
            help="Checkpoint directory to copy tokenizer files from",
        ),
    ] = None,
    hf_token: Annotated[
        str,
        typer.Option("--hf-token", envvar="HF_TOKEN", help="Hugging Face write token"),
    ] = ...,
):
    """Push model files to Hugging Face Hub."""
    api = HfApi(token=hf_token)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Excluding tokenizer_config.json from LFS prevents the "Invalid JSON"
        # configuration warning on the Hub.
        (temp_path / ".gitattributes").write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
            "*.bin filter=lfs diff=lfs merge=lfs -text\n"
            "tokenizer_config.json -filter -diff -merge text\n"
        )
        console.print("Created .gitattributes (excludes tokenizer_config.json from LFS)")

        custom_files = [
            "asr_config.py",
            "asr_modeling.py",
            "asr_processing.py",
            "asr_pipeline.py",
            "projectors.py",
            "alignment.py",
            "diarization.py",
            "handler.py",
        ]
        for filename in custom_files:
            src = Path("tiny_audio") / filename
            if src.exists():
                shutil.copy2(src, temp_path / filename)
                suffix = " (for Inference Endpoints)" if filename == "handler.py" else ""
                console.print(f"Copied {src} to staging{suffix}")
            else:
                console.print(f"[yellow]Warning: {src} not found, skipping[/yellow]")

        # MODEL_CARD.md is published as README.md on the Hub.
        if Path("MODEL_CARD.md").exists():
            shutil.copy2("MODEL_CARD.md", temp_path / "README.md")
            console.print("Copied MODEL_CARD.md as README.md to staging")

        if Path("requirements.txt").exists():
            shutil.copy2("requirements.txt", temp_path / "requirements.txt")
            console.print("Copied requirements.txt to staging")

        if checkpoint_dir:
            for filename in (
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "added_tokens.json",
            ):
                src = checkpoint_dir / filename
                if src.exists():
                    shutil.copy2(src, temp_path / filename)
                    console.print(f"Copied {src} to staging")

        console.print(f"\nUploading to {repo_id}...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            commit_message="Update custom model files, README, and requirements",
        )

        console.print(f"\n[green]Successfully pushed to https://huggingface.co/{repo_id}[/green]")


if __name__ == "__main__":
    app()
