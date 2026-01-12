#!/usr/bin/env python3
"""Push custom model files to Hugging Face Hub."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer
from huggingface_hub import HfApi
from rich.console import Console

app = typer.Typer(help="Push model files to Hugging Face Hub")
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
        Optional[str],
        typer.Option(
            "--checkpoint-dir",
            "-c",
            help="Path to checkpoint directory to copy tokenizer files from",
        ),
    ] = None,
):
    """Push model files to Hugging Face Hub."""
    # Check for HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        console.print("[red]Error: HF_TOKEN environment variable must be set[/red]")
        raise typer.Exit(1)

    api = HfApi(token=os.environ["HF_TOKEN"])

    # Create a temporary directory for staging files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create .gitattributes to prevent tokenizer_config.json from being stored in LFS
        # This fixes "Configuration Parsing Warning: Invalid JSON" on Hub
        gitattributes_content = """*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
tokenizer_config.json -filter -diff -merge text
"""
        gitattributes_path = temp_path / ".gitattributes"
        gitattributes_path.write_text(gitattributes_content)
        console.print("Created .gitattributes (excludes tokenizer_config.json from LFS)")

        # Copy all custom ASR Python files for trust_remote_code support
        # Include handler.py for Inference Endpoints support
        custom_files = [
            "asr_config.py",
            "asr_modeling.py",
            "asr_processing.py",
            "asr_pipeline.py",
            "projectors.py",
            "handler.py",  # For Inference Endpoints
        ]
        for filename in custom_files:
            src = Path("tiny_audio") / filename
            dst = temp_path / filename
            if src.exists():
                shutil.copy2(src, dst)
                if filename == "handler.py":
                    console.print(f"Copied {src} to staging (for Inference Endpoints)")
                else:
                    console.print(f"Copied {src} to staging")
            else:
                console.print(f"[yellow]Warning: {src} not found, skipping[/yellow]")

        # Copy MODEL_CARD.md as README.md
        model_card_src = Path("MODEL_CARD.md")
        if model_card_src.exists():
            readme_dst = temp_path / "README.md"
            shutil.copy2(model_card_src, readme_dst)
            console.print(f"Copied {model_card_src} as README.md to staging")

        # Copy requirements.txt for model dependencies
        requirements_src = Path("requirements.txt")
        if requirements_src.exists():
            requirements_dst = temp_path / "requirements.txt"
            shutil.copy2(requirements_src, requirements_dst)
            console.print(f"Copied {requirements_src} to staging")

        # Copy tokenizer files from checkpoint directory if provided
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            tokenizer_files = [
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "added_tokens.json",
            ]
            for filename in tokenizer_files:
                src = checkpoint_path / filename
                if src.exists():
                    dst = temp_path / filename
                    shutil.copy2(src, dst)
                    console.print(f"Copied {src} to staging")

        # Upload files to Hub
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
