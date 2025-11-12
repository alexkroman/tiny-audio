#!/usr/bin/env python3
"""Download sample audio files for course exercises.

This script downloads a few small audio samples from LibriSpeech
to help students test their code without needing to find their own audio files.
"""

import urllib.request
from pathlib import Path


def download_samples():
    """Download sample audio files to samples/ directory."""
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)

    print("📥 Downloading sample audio files for course exercises...\n")

    # LibriSpeech test-clean samples (public domain audiobooks)
    # These are small, clear speech samples perfect for testing
    samples = [
        {
            "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
            "name": "librispeech-samples.tar.gz",
            "description": "LibriSpeech test-clean samples (346 MB)",
        }
    ]

    for sample in samples:
        filename = samples_dir / sample["name"]

        if filename.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue

        print(f"Downloading {sample['description']}...")
        print(f"  URL: {sample['url']}")
        print(f"  Destination: {filename}")

        try:
            urllib.request.urlretrieve(sample["url"], filename)
            print(f"✓ Downloaded {filename}\n")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}\n")
            continue

    print("\n📦 Extracting archives...")
    import tarfile

    tar_path = samples_dir / "librispeech-samples.tar.gz"
    if tar_path.exists():
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(samples_dir)
        print(f"✓ Extracted to {samples_dir}/LibriSpeech/test-clean/")

        # Find a specific audio file to use as default
        test_clean_dir = samples_dir / "LibriSpeech" / "test-clean"
        if test_clean_dir.exists():
            # Find the first .flac file
            first_audio = next(test_clean_dir.rglob("*.flac"), None)
            if first_audio:
                # Create a symlink for easy access
                default_link = Path("test.wav")
                if default_link.exists() or default_link.is_symlink():
                    default_link.unlink()

                # Copy instead of symlink for cross-platform compatibility
                import shutil

                shutil.copy(first_audio, "test.wav")
                print(f"\n✓ Created test.wav from {first_audio.name}")
                print("  You can now run the course exercises!")

    print("\n✅ Sample download complete!")
    print("\nℹ️  Sample audio files are in: samples/LibriSpeech/test-clean/")
    print("ℹ️  Default test file: test.wav")
    print("\n🎓 You can now run the course exercises:")
    print("   poetry run python test_inference.py")
    print("   poetry run python explore_audio.py")
    print("   etc.")


if __name__ == "__main__":
    download_samples()
