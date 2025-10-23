#!/usr/bin/env python3
"""
Check what's available in a HuggingFace Hub model repository.
"""

import argparse
from huggingface_hub import list_repo_files, hf_hub_download, HfApi
import json

def check_hub_model(repo_id: str):
    """Check what files and checkpoints are available in a Hub repository."""

    print(f"\nChecking repository: {repo_id}\n")

    try:
        # List all files in the repo
        api = HfApi()
        files = list_repo_files(repo_id)

        # Categorize files
        checkpoints = []
        model_files = []
        config_files = []
        other_files = []

        for file in files:
            if file.startswith("checkpoint-"):
                # Extract checkpoint number
                parts = file.split("/")
                if len(parts) > 0 and parts[0].startswith("checkpoint-"):
                    checkpoint_name = parts[0]
                    if checkpoint_name not in checkpoints:
                        checkpoints.append(checkpoint_name)
            elif file.endswith(".safetensors") or file.endswith(".bin"):
                model_files.append(file)
            elif file.endswith(".json"):
                config_files.append(file)
            else:
                other_files.append(file)

        # Sort checkpoints by number
        checkpoints.sort(key=lambda x: int(x.replace("checkpoint-", "")))

        print("=" * 60)
        print("AVAILABLE CHECKPOINTS:")
        print("=" * 60)
        if checkpoints:
            for checkpoint in checkpoints:
                print(f"  - {repo_id}/{checkpoint}")
            print(f"\nLatest checkpoint: {repo_id}/{checkpoints[-1]}")
        else:
            print("  No checkpoints found")

        print("\n" + "=" * 60)
        print("MODEL FILES:")
        print("=" * 60)
        for file in model_files[:10]:  # Show first 10
            print(f"  - {file}")
        if len(model_files) > 10:
            print(f"  ... and {len(model_files) - 10} more")

        print("\n" + "=" * 60)
        print("CONFIG FILES:")
        print("=" * 60)
        for file in config_files:
            print(f"  - {file}")

        # Check if it's a proper model (has config.json in root)
        if "config.json" in files:
            print("\n✓ This appears to be a valid model repository")
            print("\nTo resume training, use one of these:")
            print(f"  resume_from_checkpoint: {repo_id}  # Load from root")
            if checkpoints:
                print(f"  resume_from_checkpoint: {repo_id}/{checkpoints[-1]}  # Latest checkpoint")
        else:
            print("\n⚠ No config.json found in root - may not be a complete model")
            if checkpoints:
                print(f"\nTry using a specific checkpoint:")
                print(f"  resume_from_checkpoint: {repo_id}/{checkpoints[-1]}")

        # Check for model card
        if "README.md" in files:
            print("\n✓ Model card (README.md) found")

    except Exception as e:
        print(f"Error accessing repository: {e}")
        print("\nMake sure:")
        print("  1. The repository exists")
        print("  2. You have access (if private)")
        print("  3. You're logged in: huggingface-cli login")

def main():
    parser = argparse.ArgumentParser(description="Check HuggingFace Hub model repository")
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace Hub repository ID (e.g., 'username/model-name')"
    )

    args = parser.parse_args()
    check_hub_model(args.repo_id)

if __name__ == "__main__":
    main()