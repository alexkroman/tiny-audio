#!/usr/bin/env python3
"""Push custom model files to Hugging Face Hub."""

from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import CommitOperationAdd, HfApi, get_token
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
        str | None,
        typer.Option(
            "--checkpoint-dir",
            "-c",
            help="Path to checkpoint directory to copy tokenizer files from",
        ),
    ] = None,
):
    """Push model files to Hugging Face Hub."""
    # HfApi resolves HF_TOKEN and the `hf auth login` cache itself; fail early
    # with a clear message rather than on the first authenticated call.
    if get_token() is None:
        console.print("[red]Error: not logged in (set HF_TOKEN or run `hf auth login`)[/red]")
        raise typer.Exit(1)

    # One atomic commit built from CommitOperationAdd entries. `path_in_repo`
    # is independent of the local path, so MODEL_CARD.md publishes as
    # README.md without staging a temp directory of copies first.
    operations: list[CommitOperationAdd] = [
        # Excluding tokenizer_config.json from LFS prevents the "Invalid JSON"
        # configuration warning on the Hub.
        CommitOperationAdd(
            path_in_repo=".gitattributes",
            path_or_fileobj=(
                b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
                b"*.bin filter=lfs diff=lfs merge=lfs -text\n"
                b"tokenizer_config.json -filter -diff -merge text\n"
            ),
        )
    ]
    console.print("Added .gitattributes (excludes tokenizer_config.json from LFS)")

    def _add(src: Path, path_in_repo: str | None = None, note: str = "") -> None:
        if not src.exists():
            console.print(f"[yellow]Warning: {src} not found, skipping[/yellow]")
            return
        operations.append(
            CommitOperationAdd(path_in_repo=path_in_repo or src.name, path_or_fileobj=str(src))
        )
        console.print(f"Added {src} as {path_in_repo or src.name}{note}")

    for filename in (
        "asr_config.py",
        "asr_modeling.py",
        "asr_processing.py",
        "asr_pipeline.py",
        "projectors.py",
        "alignment.py",
        "diarization.py",
        "handler.py",
    ):
        _add(
            Path("tiny_audio") / filename,
            note=" (for Inference Endpoints)" if filename == "handler.py" else "",
        )

    # MODEL_CARD.md is published as README.md on the Hub.
    _add(Path("MODEL_CARD.md"), "README.md")
    _add(Path("requirements.txt"))

    if checkpoint_dir:
        for filename in (
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ):
            _add(Path(checkpoint_dir) / filename)

    console.print(f"\nUploading to {repo_id}...")
    HfApi().create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        operations=operations,
        commit_message="Update custom model files, README, and requirements",
    )

    console.print(f"\n[green]Successfully pushed to https://huggingface.co/{repo_id}[/green]")


if __name__ == "__main__":
    app()
