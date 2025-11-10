#!/usr/bin/env python3
"""
Fix encoder weights by reloading from base model.
WARNING: This will discard any encoder training progress.
"""

from huggingface_hub import HfApi
from transformers import WhisperModel
import tempfile
from pathlib import Path
from safetensors.torch import save_file

repo_id = "mazesmazes/tiny-audio"

print(f"Fixing encoder weights for {repo_id}...")
print("⚠️  WARNING: This will replace encoder weights with base Whisper V3 Turbo")
print("⚠️  Any encoder fine-tuning progress will be lost")

response = input("\nContinue? (yes/no): ")
if response.lower() != "yes":
    print("Aborted")
    exit(0)

# Load the correct encoder
encoder = WhisperModel.from_pretrained("openai/whisper-large-v3-turbo")

# Save just the encoder weights
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Get encoder state dict
    encoder_state = encoder.state_dict()

    # Save as safetensors
    encoder_file = tmpdir / "encoder.safetensors"
    save_file(encoder_state, str(encoder_file))

    print(f"\nEncoder weights shape check:")
    print(f"  Conv1: {encoder.encoder.conv1.weight.shape}")
    print(f"  Config num_mel_bins: {encoder.config.num_mel_bins}")

    # Upload to HuggingFace
    api = HfApi()

    try:
        api.upload_file(
            path_or_fileobj=str(encoder_file),
            path_in_repo="encoder.safetensors",
            repo_id=repo_id,
            commit_message="Fix: Replace encoder with correct Whisper V3 Turbo weights (128 mel bins)",
        )
        print(f"\n✓ Successfully uploaded encoder.safetensors to {repo_id}")

        # Also upload encoder config
        encoder_config_file = tmpdir / "encoder_config.json"
        encoder.config.save_pretrained(tmpdir)

        # The config is saved as config.json, rename it
        import shutil
        shutil.move(str(tmpdir / "config.json"), str(encoder_config_file))

        api.upload_file(
            path_or_fileobj=str(encoder_config_file),
            path_in_repo="encoder_config.json",
            repo_id=repo_id,
            commit_message="Fix: Update encoder config for Whisper V3 Turbo (128 mel bins)",
        )
        print(f"✓ Successfully uploaded encoder config")

    except Exception as e:
        print(f"\n✗ Error uploading: {e}")
        import traceback
        traceback.print_exc()
