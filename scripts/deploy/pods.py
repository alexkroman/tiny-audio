"""Create RunPod pods: the capacity-retrying create loop, and inference pods for `ta serve`."""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import typer

from scripts.deploy import gpu_catalog

# Every pod exposes SSH for `ta runpod deploy` and one HTTP port for
# `ta runpod serve`, reachable at https://<pod-id>-<port>.proxy.runpod.net.
SERVE_PORT = 8000
POD_PORTS = f"22/tcp,{SERVE_PORT}/http"
# GPU preference for an inference pod, best first. ~7.5 GB of bf16 weights fit
# a 24 GB card even at batch 64, so the pick is throughput per dollar on
# well-supported kernels: Ada/Ampere only, since flash-linear-attention's
# delta-rule kernels have open issues on Hopper and are unconfirmed on
# Blackwell (RTX 5090 / PRO 6000).
SERVE_GPUS = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA A40",
)
# Image + python env + ~12 GB of model weights in the HF cache, with headroom.
SERVE_DISK_GB = 80


def ssh_public_key() -> Path:
    """The key every pod is created with (`ta runpod deploy` logs in with it); exit 1 if missing."""
    pubkey = Path("~/.ssh/id_ed25519.pub").expanduser()
    if not pubkey.exists():
        print(f"Missing {pubkey}; `ta runpod deploy` authenticates with that key.")
        raise typer.Exit(1)
    return pubkey


def create_first_available(
    pod_name: str, gpu_ids: Sequence[str], *, image: str, disk_gb: int, pubkey: Path
) -> str:
    """Create a pod on the first of `gpu_ids` with capacity; its id, or exit 1.

    RunPod's catalog reports `available: true` for GPU types that still fail
    to create ("There are no longer any instances available..."), so each
    candidate is tried in turn rather than trusting the catalog.
    """
    for gpu_id in gpu_ids:
        print(f"trying {gpu_id}... ", end="", flush=True)
        result = subprocess.run(
            check=False,
            args=[
                "runpodctl",
                "pod",
                "create",
                "--name",
                pod_name,
                "--gpu-id",
                gpu_id,
                "--image",
                image,
                "--container-disk-in-gb",
                str(disk_gb),
                "--ports",
                POD_PORTS,
                "--env",
                json.dumps({"SSH_PUBLIC_KEY": pubkey.read_text().strip()}),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        blob = result.stdout + result.stderr
        if "no longer any instances" in blob or '"error"' in blob:
            print("no capacity")
            continue
        try:
            pod = json.loads(blob[blob.index("{") :])
            pod_id: str = pod["id"]
        except Exception:
            print(f"unexpected response:\n{blob[:400]}")
            continue
        print(f"created {pod_id}")
        return pod_id

    print("\nEvery candidate GPU type was out of capacity. Retry shortly.")
    raise typer.Exit(1)


def provision_serve_command(
    name: str | None, image: str, max_attempts: int, dry_run: bool
) -> str | None:
    """Create an inference pod for `ta runpod serve` on the best available SERVE_GPUS type."""
    listed = {gpu_id for _, gpu_id in gpu_catalog.available_gpus(0)}
    # Listed types first, in preference order; the rest after, since the
    # catalog's availability is racy in both directions.
    candidates = sorted(SERVE_GPUS, key=lambda g: g not in listed)[:max_attempts]
    print(f"\ninference pod: {SERVE_DISK_GB} GB disk, HTTP on port {SERVE_PORT}")
    print(f"candidates: {', '.join(candidates)}\n")

    pubkey = ssh_public_key()
    if dry_run:
        return None

    pod_id = create_first_available(
        name or "tiny-audio-serve",
        candidates,
        image=image,
        disk_gb=SERVE_DISK_GB,
        pubkey=pubkey,
    )
    print("\nNext:")
    print(f"  poetry run ta runpod wait {pod_id}          # prints <ip> <port>")
    print("  poetry run ta runpod deploy <ip> <port>")
    print("  poetry run ta runpod serve <ip> <port> --no-attach")
    print(f"  # then: https://{pod_id}-{SERVE_PORT}.proxy.runpod.net")
    print(f"  runpodctl pod delete {pod_id}               # when finished\n")
    return pod_id
