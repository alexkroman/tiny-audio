#!/usr/bin/env python3
"""Deploy the demo application to a Hugging Face Space."""

from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi, RepoUrl, upload_folder

app = typer.Typer(add_completion=False)


def extract_repo_id(repo_id_or_url: str) -> str:
    """Return the `owner/name` Space id for a plain id or any huggingface.co Space URL.

    `RepoUrl` handles trailing slashes, `/tree/<branch>` suffixes and custom
    `HF_ENDPOINT` hosts; a URL that points at a model or dataset is rejected
    instead of being passed through to a Space API call that would fail later.
    """
    if "://" not in repo_id_or_url:
        return repo_id_or_url.strip("/")
    parsed = RepoUrl(repo_id_or_url)
    if parsed.repo_type != "space":
        raise typer.BadParameter(f"Not a Space: {repo_id_or_url}")
    return parsed.repo_id


@app.command()
def deploy(
    repo_id: Annotated[
        str,
        typer.Option(
            "--repo-id", "-r", help="HuggingFace Space repo ID (e.g., username/space-name)"
        ),
    ] = "mazesmazes/tiny-audio",
    demo_dir: Annotated[
        Path,
        typer.Option(
            "--demo-dir", exists=True, file_okay=False, help="Path to the demo directory to upload"
        ),
    ] = Path("demo"),
    delete_existing: Annotated[
        bool,
        typer.Option(
            "--delete-existing", help="Delete files in the Space that are not in --demo-dir"
        ),
    ] = False,
    private: Annotated[
        bool, typer.Option("--private", help="Create the Space as private (if creating new)")
    ] = False,
):
    """Deploy demo files to a Hugging Face Space."""
    repo_id = extract_repo_id(repo_id)

    required_files = ["app.py", "requirements.txt", "README.md"]
    missing = [f for f in required_files if not (demo_dir / f).exists()]
    if missing:
        raise typer.BadParameter(
            f"required files not found: {', '.join(missing)}", param_hint="--demo-dir"
        )

    typer.echo(f"\nDeploying to Hugging Face Space: {repo_id}")
    typer.echo(f"Demo directory: {demo_dir.absolute()}")

    # `exist_ok` makes this a no-op on an existing Space; a real auth or
    # network failure surfaces here instead of being swallowed.
    HfApi().create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
    )

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


if __name__ == "__main__":
    app()
