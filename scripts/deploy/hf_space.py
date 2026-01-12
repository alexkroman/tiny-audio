#!/usr/bin/env python3
"""
Deploy the demo application to a Hugging Face Space.

This script uploads the demo files (app.py, requirements.txt, README.md)
and optionally the wav_outputs directory to a Hugging Face Space.

Usage:
    poetry run deploy-hf
    poetry run deploy-hf --repo-id YOUR_USERNAME/YOUR_SPACE
    poetry run deploy-hf --delete-existing
"""

from pathlib import Path

import typer
from huggingface_hub import HfApi, upload_folder

app = typer.Typer(help="Deploy demo to Hugging Face Space")


def extract_repo_id(repo_id_or_url: str) -> str:
    """Extract repo_id from a URL or return as-is if already a repo_id."""
    if repo_id_or_url.startswith("https://huggingface.co/spaces/"):
        return repo_id_or_url.replace("https://huggingface.co/spaces/", "").rstrip("/")
    return repo_id_or_url


@app.command()
def deploy(
    repo_id: str = typer.Option(
        "mazesmazes/tiny-audio",
        "--repo-id",
        "-r",
        help="HuggingFace Space repo ID (e.g., username/space-name)",
    ),
    demo_dir: Path = typer.Option(
        Path("demo"),
        "--demo-dir",
        "-d",
        help="Path to demo directory",
    ),
    delete_existing: bool = typer.Option(
        False,
        "--delete-existing",
        help="Delete files in Space that are not in demo_dir",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Create Space as private (if creating new)",
    ),
):
    """Deploy demo files to a Hugging Face Space."""
    repo_id = extract_repo_id(repo_id)

    # Validate demo directory
    if not demo_dir.exists():
        raise typer.BadParameter(f"Demo directory not found: {demo_dir}")

    required_files = ["app.py", "requirements.txt", "README.md"]
    missing = [f for f in required_files if not (demo_dir / f).exists()]
    if missing:
        raise typer.BadParameter(f"Required files not found: {', '.join(missing)}")

    typer.echo(f"\nDeploying to Hugging Face Space: {repo_id}")
    typer.echo(f"Demo directory: {demo_dir.absolute()}")

    api = HfApi()

    # Create Space if it doesn't exist
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        typer.echo(f"Space '{repo_id}' exists, uploading files...")
    except Exception:
        typer.echo(f"Creating new Space '{repo_id}'...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
        )

    # Upload the demo folder
    typer.echo("\nUploading files...")
    upload_folder(
        folder_path=str(demo_dir),
        repo_id=repo_id,
        repo_type="space",
        delete_patterns=["*"] if delete_existing else None,
        commit_message="Deploy demo to HF Space",
    )

    typer.echo("\nSuccessfully deployed to Hugging Face Space!")
    typer.echo(f"Your Space is available at: https://huggingface.co/spaces/{repo_id}")
    typer.echo("\nNote: The Space may take a few minutes to build and become available.")


def main():
    """Entry point for pyproject.toml scripts."""
    app()


if __name__ == "__main__":
    app()
