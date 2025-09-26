#!/usr/bin/env python3
"""Push modeling.py and MODEL_CARD.md to Hugging Face Hub."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Push model files to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mazesmazes/tiny-audio",
        help="Hugging Face repository ID (default: mazesmazes/tiny-audio)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Branch to push to (default: main)",
    )
    args = parser.parse_args()

    # Check for HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN environment variable must be set")

    api = HfApi(token=os.environ["HF_TOKEN"])

    # Create a temporary directory for staging files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy modeling.py
        modeling_src = Path("src/modeling.py")
        modeling_dst = temp_path / "modeling.py"
        shutil.copy2(modeling_src, modeling_dst)
        print(f"✓ Copied {modeling_src} to staging")

        # Copy MODEL_CARD.md as README.md
        model_card_src = Path("MODEL_CARD.md")
        readme_dst = temp_path / "README.md"
        shutil.copy2(model_card_src, readme_dst)
        print(f"✓ Copied {model_card_src} as README.md to staging")

        # Upload files to Hub
        print(f"\nUploading to {args.repo_id}...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.branch,
            commit_message="Update modeling.py and README.md",
        )

        print(f"\n✅ Successfully pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
