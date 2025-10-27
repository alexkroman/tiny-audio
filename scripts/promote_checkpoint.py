#!/usr/bin/env python3
"""
Promote a checkpoint to the root directory and push to HuggingFace Hub.

This script loads a model from a checkpoint directory (local or on Hub),
saves it properly, and pushes to the Hub root directory.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Promote checkpoint to root and push to HuggingFace Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (local path like 'outputs/last-checkpoint')",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder in HuggingFace repo (e.g., 'last-checkpoint' to load from hub subfolder)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mazesmazes/tiny-audio",
        help="HuggingFace repository ID (default: mazesmazes/tiny-audio)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only save locally, don't push to hub",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/promoted_model",
        help="Local directory to save the promoted model (default: outputs/promoted_model)",
    )
    args = parser.parse_args()

    # Check for HF_TOKEN if pushing to hub
    if not args.local_only and not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN environment variable must be set for pushing to hub")

    if args.subfolder:
        print(f"Loading model from: {args.checkpoint}/{args.subfolder}")
    else:
        print(f"Loading model from checkpoint: {args.checkpoint}")

    # Import after path setup
    from asr_modeling import ASRModel

    # Load kwargs for subfolder support
    load_kwargs = {"subfolder": args.subfolder} if args.subfolder else {}

    try:
        # Load the model - config should now have all required dimensions
        model = ASRModel.from_pretrained(args.checkpoint, **load_kwargs)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

    print(f"✓ Model loaded successfully")

    # Save locally first
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to: {output_path}")
    model.save_pretrained(output_path)
    print(f"✓ Model saved locally")

    # Push to hub if requested
    if not args.local_only:
        print(f"\nPushing to HuggingFace Hub: {args.repo_id}")

        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])

        # List files that should be deleted (old model files in root)
        # This ensures we're starting fresh with the promoted checkpoint
        files_to_delete = [
            "projector.safetensors",
            "model.safetensors",
            "pytorch_model.bin",
            "config.json",
            "generation_config.json",
        ]

        print("  Cleaning old files from root directory...")
        for file_path in files_to_delete:
            try:
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=args.repo_id,
                    repo_type="model",
                    commit_message=f"Remove old {file_path} before promotion",
                )
                print(f"    Deleted: {file_path}")
            except Exception:
                # File might not exist, which is fine
                pass

        # Upload the entire directory
        print("  Uploading promoted checkpoint files...")
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Promote checkpoint to root directory",
        )

        print(f"✓ Successfully pushed to https://huggingface.co/{args.repo_id}")
        print(f"\nNow users can load with:")
        print(f'  model = ASRModel.from_pretrained("{args.repo_id}")')
    else:
        print(f"\nModel saved locally at: {output_path}")
        print(f"To push to hub, run without --local-only flag")


if __name__ == "__main__":
    main()
