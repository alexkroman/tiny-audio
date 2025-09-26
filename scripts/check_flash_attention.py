#!/usr/bin/env python3
"""Check Flash Attention availability and compatibility."""

import sys


def check_flash_attention():
    """Diagnose Flash Attention installation and compatibility."""
    print("🔍 Checking Flash Attention availability...\n")

    # Check PyTorch
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            capability = torch.cuda.get_device_capability()
            print(f"   Compute capability: SM {capability[0]}.{capability[1]}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False

    print()

    # Check Flash Attention
    try:
        import flash_attn

        print(f"✅ Flash Attention installed: {flash_attn.__version__}")

        # Check compatibility
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 7 and capability[1] >= 5:
                print("✅ GPU is compatible with Flash Attention (SM 7.5+)")
                return True
            print(
                f"⚠️ GPU may not be fully compatible (SM {capability[0]}.{capability[1]} < 7.5)"
            )
            print("   Flash Attention works best on Turing (RTX 2080) or newer GPUs")
            return False
        print("⚠️ CUDA not available - Flash Attention requires GPU")
        return False

    except ImportError as e:
        print(f"❌ Flash Attention not installed: {e}")
        print("\nTo install Flash Attention:")
        print("  pip install flash-attn --no-build-isolation")
        print("  or")
        print("  uv pip install flash-attn --no-build-isolation")
        return False

    print()
    return True


if __name__ == "__main__":
    success = check_flash_attention()
    sys.exit(0 if success else 1)
