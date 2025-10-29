#!/usr/bin/env python3
"""Verify that the Tiny Audio course setup is complete and working.

This script checks:
1. Python version
2. Required packages are installed
3. Sample audio files exist
4. Model can be loaded

Run this after initial setup to ensure everything is working.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version is 3.10+."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False


def check_packages():
    """Check required packages are installed."""
    print("\n📦 Checking required packages...")
    packages = [
        "torch",
        "transformers",
        "librosa",
        "matplotlib",
        "accelerate",
        "peft",
        "datasets",
    ]

    all_ok = True
    for package in packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ❌ {package} not installed")
            all_ok = False

    return all_ok


def check_audio_samples():
    """Check if sample audio files exist."""
    print("\n🎵 Checking for sample audio files...")

    test_wav = Path("test.wav")
    samples_dir = Path("samples/LibriSpeech/test-clean")

    if test_wav.exists():
        print(f"   ✓ test.wav exists")
        has_test = True
    else:
        print(f"   ⚠️  test.wav not found (run: poetry run download-samples)")
        has_test = False

    if samples_dir.exists():
        audio_files = list(samples_dir.rglob("*.flac"))
        if audio_files:
            print(f"   ✓ Found {len(audio_files)} sample files in {samples_dir}")
            has_samples = True
        else:
            print(f"   ⚠️  No audio files in {samples_dir}")
            has_samples = False
    else:
        print(f"   ⚠️  {samples_dir} not found (run: poetry run download-samples)")
        has_samples = False

    return has_test or has_samples


def check_model_loading():
    """Try to load the model config (doesn't download full model)."""
    print("\n🤖 Checking model loading...")

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            "mazesmazes/tiny-audio", trust_remote_code=True
        )
        print(f"   ✓ Model config loaded successfully")
        print(f"      Audio encoder: {config.audio_model_id}")
        print(f"      Language model: {config.text_model_id}")
        return True
    except Exception as e:
        print(f"   ❌ Error loading model config: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Tiny Audio Course Setup Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("Sample Audio Files", check_audio_samples),
        ("Model Loading", check_model_loading),
    ]

    results = {}
    for name, check_func in checks:
        results[name] = check_func()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {name}")

    print("\n" + "=" * 60)

    if all_passed:
        print("✅ All checks passed! You're ready to start the course.")
        print("\n🎓 Next steps:")
        print("   1. Read: docs/QUICKSTART.md")
        print("   2. Start: docs/course/1-introduction-and-setup.md")
        print("   3. Run: poetry run python test_inference.py")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        if not results["Required Packages"]:
            print("   - Run: poetry install")
        if not results["Sample Audio Files"]:
            print("   - Run: poetry run download-samples")
        return 1


if __name__ == "__main__":
    sys.exit(main())
