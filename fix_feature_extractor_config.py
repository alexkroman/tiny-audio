#!/usr/bin/env python3
"""
Fix the feature extractor configuration in the HuggingFace model repository.
This updates the preprocessor_config.json to use 128 mel bins instead of 80.
"""

from huggingface_hub import HfApi
from transformers import WhisperFeatureExtractor
import tempfile
import json
from pathlib import Path

repo_id = "mazesmazes/tiny-audio"

print(f"Fixing feature extractor config for {repo_id}...")

# Create a Whisper feature extractor with 128 mel bins
fe = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-large-v3-turbo",
    feature_size=128
)

# Save it to a temp directory
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Save the feature extractor
    fe.save_pretrained(tmpdir)

    # Check what was saved
    config_file = tmpdir / "preprocessor_config.json"
    with open(config_file) as f:
        config = json.load(f)

    print(f"\nNew preprocessor_config.json:")
    print(json.dumps(config, indent=2))

    # Upload to HuggingFace
    api = HfApi()

    try:
        api.upload_file(
            path_or_fileobj=str(config_file),
            path_in_repo="preprocessor_config.json",
            repo_id=repo_id,
            commit_message="Fix: Update feature extractor to use 128 mel bins for Whisper V3 Turbo",
        )
        print(f"\n✓ Successfully uploaded preprocessor_config.json to {repo_id}")

        # Also upload to last-checkpoint subfolder
        api.upload_file(
            path_or_fileobj=str(config_file),
            path_in_repo="last-checkpoint/preprocessor_config.json",
            repo_id=repo_id,
            commit_message="Fix: Update feature extractor to use 128 mel bins for Whisper V3 Turbo (last-checkpoint)",
        )
        print(f"✓ Successfully uploaded to last-checkpoint subfolder")

    except Exception as e:
        print(f"\n✗ Error uploading: {e}")
        print("\nAlternatively, manually upload this file:")
        print(f"File: {config_file}")
        print(f"Destination: https://huggingface.co/{repo_id}/tree/main")
