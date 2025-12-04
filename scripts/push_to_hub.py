#!/usr/bin/env python3
"""Push custom model files to Hugging Face Hub."""

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

        # Copy all custom ASR Python files for trust_remote_code support
        # Include handler.py for Inference Endpoints support
        custom_files = [
            "asr_config.py",
            "asr_modeling.py",
            "asr_processing.py",
            "asr_pipeline.py",
            "moe_projector.py",
            "residual_projector.py",
            "swiglu_projector.py",
            "handler.py",  # For Inference Endpoints
        ]
        for filename in custom_files:
            src = Path("src") / filename
            dst = temp_path / filename
            if src.exists():
                shutil.copy2(src, dst)
                if filename == "handler.py":
                    print(f"✓ Copied {src} to staging (for Inference Endpoints)")
                else:
                    print(f"✓ Copied {src} to staging")
            else:
                print(f"⚠ Warning: {src} not found, skipping")

        # Copy MODEL_CARD.md as README.md
        model_card_src = Path("MODEL_CARD.md")
        if model_card_src.exists():
            readme_dst = temp_path / "README.md"
            shutil.copy2(model_card_src, readme_dst)
            print(f"✓ Copied {model_card_src} as README.md to staging")

        # Copy requirements.txt for model dependencies
        requirements_src = Path("requirements.txt")
        if requirements_src.exists():
            requirements_dst = temp_path / "requirements.txt"
            shutil.copy2(requirements_src, requirements_dst)
            print(f"✓ Copied {requirements_src} to staging")

        # Upload files to Hub
        print(f"\nUploading to {args.repo_id}...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.branch,
            commit_message="Update custom model files, README, and requirements",
        )

        print(f"\n✅ Successfully pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
