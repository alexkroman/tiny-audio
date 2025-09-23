#!/usr/bin/env python3
"""
Deploy the demo application to a Hugging Face Space.

This script creates a clean deployment of only the necessary demo files
(app.py, requirements.txt, README.md) to a Hugging Face Space, avoiding
the upload of the entire repository.

Usage:
    # Deploy to default space (mazesmazes/tiny-audio)
    python scripts/deploy_to_hf_space.py

    # Deploy to a custom space
    python scripts/deploy_to_hf_space.py --space-url https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE

    # Deploy with force push (overwrites existing space)
    python scripts/deploy_to_hf_space.py --force
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)

    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    return result


def deploy_to_space(
    space_url: str = "https://huggingface.co/spaces/mazesmazes/tiny-audio",
    force: bool = False,
    demo_dir: Path = Path("demo")
) -> None:
    """Deploy demo files to a Hugging Face Space.

    Args:
        space_url: The URL of the Hugging Face Space
        force: Whether to force push (overwrite existing content)
        demo_dir: Path to the demo directory containing files to deploy
    """
    # Validate demo directory
    if not demo_dir.exists():
        raise FileNotFoundError(f"Demo directory not found: {demo_dir}")

    required_files = ["app.py", "requirements.txt", "README.md"]
    for file in required_files:
        if not (demo_dir / file).exists():
            raise FileNotFoundError(f"Required file not found: {demo_dir / file}")

    print(f"\n🚀 Deploying to Hugging Face Space: {space_url}")
    print(f"📁 Demo directory: {demo_dir.absolute()}")

    # Create temporary directory for deployment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\n📋 Creating temporary deployment directory: {temp_path}")

        # Copy demo files to temp directory
        print("\n📦 Copying demo files...")
        for file in required_files:
            src = demo_dir / file
            dst = temp_path / file
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {file}")

        # Initialize git repository
        print("\n🔧 Initializing git repository...")
        run_command(["git", "init"], cwd=temp_path)
        run_command(["git", "config", "user.email", "noreply@example.com"], cwd=temp_path)
        run_command(["git", "config", "user.name", "HF Space Deploy"], cwd=temp_path)

        # Add and commit files
        print("\n📝 Committing files...")
        run_command(["git", "add", "."], cwd=temp_path)
        run_command(["git", "commit", "-m", "Deploy demo to HF Space"], cwd=temp_path)

        # Add remote and push
        print(f"\n🌐 Adding remote: {space_url}")
        run_command(["git", "remote", "add", "origin", space_url], cwd=temp_path)

        # Push to space
        push_cmd = ["git", "push", "origin", "main"]
        if force:
            push_cmd.append("--force")
            print("\n⚠️  Force pushing to Space (overwriting existing content)...")
        else:
            print("\n📤 Pushing to Space...")

        result = run_command(push_cmd, cwd=temp_path, check=False)

        if result.returncode != 0:
            if "failed to push some refs" in result.stderr and not force:
                print("\n❌ Push failed: Space already has content.")
                print("   Use --force to overwrite existing content.")
                raise SystemExit(1)
            elif result.returncode != 0:
                print(f"\n❌ Push failed: {result.stderr}")
                raise SystemExit(1)

        print("\n✅ Successfully deployed to Hugging Face Space!")
        print(f"🔗 Your Space will be available at: {space_url.replace('.git', '')}")
        print("\n📝 Note: The Space may take a few minutes to build and become available.")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy demo application to a Hugging Face Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy to default space
  %(prog)s

  # Deploy to custom space
  %(prog)s --space-url https://huggingface.co/spaces/username/my-space

  # Force push (overwrite existing)
  %(prog)s --force
        """
    )

    parser.add_argument(
        "--space-url",
        type=str,
        default="https://huggingface.co/spaces/mazesmazes/tiny-audio",
        help="URL of the Hugging Face Space (default: mazesmazes/tiny-audio)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force push to overwrite existing Space content"
    )

    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("demo"),
        help="Path to demo directory (default: demo)"
    )

    args = parser.parse_args()

    try:
        deploy_to_space(
            space_url=args.space_url,
            force=args.force,
            demo_dir=args.demo_dir
        )
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()