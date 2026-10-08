"""RunPod GPU and datacenter catalog queries used to place a pod.

Wraps `runpodctl gpu list` / `runpodctl datacenter list` and the hand-kept
set of datacenters that support network volumes, so `plan` and `provision`
can pick a GPU type that both fits the VRAM estimate and can be placed.
"""

from __future__ import annotations

import functools
import json
import subprocess


def runpodctl_json(*args: str) -> str:
    """Stdout of `runpodctl <args> -o json` (never raises on a non-zero exit)."""
    return subprocess.run(
        ["runpodctl", *args, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout


def available_gpus(min_vram_gib: float) -> list[tuple[int, str]]:
    """GPUs the catalog claims are available with enough VRAM, smallest first.

    Smallest-first approximates cheapest-first; `runpodctl gpu list` exposes no
    price field. The returned order is a candidate list rather than a choice,
    because `available` is not a promise -- see `provision`.
    """
    out = runpodctl_json("gpu", "list")
    catalog = json.loads(out[out.index("[") :])
    fitting = [
        (g["memoryInGb"], g["gpuId"])
        for g in catalog
        if g.get("available") and g.get("memoryInGb", 0) >= min_vram_gib
    ]
    return sorted(set(fitting))


# `runpodctl datacenter list` reports stock per GPU per datacenter. Observed
# values are High / Medium / Low / "".
#
# "Low" outranks "": a reported status of any kind means the datacenter is
# actually offering that GPU, whereas an empty string carries no stock signal
# at all and is the weaker bet. (An earlier revision of this file had these
# two the other way round on the theory that "Low" was an explicit scarcity
# warning. It is not -- it is stock information, and stock information beats
# none.) `pod create` failing with "no longer any instances" is still normal
# on any of them; this only orders which to try first.
STOCK_RANK = {"High": 0, "Medium": 1, "Low": 2, "": 3}

# Datacenters that actually support network volumes. This is a SEPARATE and
# much smaller set than "datacenters that have the GPU", and nothing in
# `runpodctl datacenter list` exposes it -- suggesting a datacenter that has
# an H100 but no volume support gets you:
#   create network volume: Data center "AP-IN-1" not found or does not
#   support network volumes. Available data centers: ...
# which is how this list was obtained (2026-09-18). The error enumerates the
# supported set, so the cheap way to refresh it is to run
# `runpodctl network-volume create --name x --size 1 --data-center-id NOPE`
# and read the message; it fails without creating anything.
#
# Concretely, 6 of the 13 datacenters offering an H100 80GB HBM3 do NOT
# support network volumes: AP-IN-1, CA-MTL-1, US-GA-2, US-KS-2, US-MO-1,
# US-NE-1. Filtering matters.
NETWORK_VOLUME_DATACENTERS = frozenset(
    [
        "AP-IN-2",
        "AP-JP-1",
        "CA-MTL-3",
        "CA-MTL-4",
        "EU-FR-1",
        "EU-NL-1",
        "EU-RO-1",
        "EUR-IS-1",
        "EUR-IS-3",
        "EUR-NO-1",
        "EUR-NO-2",
        "US-CA-2",
        "US-CO-1",
        "US-IL-1",
        "US-MO-2",
        "US-NC-2",
        "US-TX-3",
    ]
)


@functools.cache
def datacenter_catalog() -> tuple:
    """`runpodctl datacenter list`, fetched once per process.

    Cached because GPU selection probes this for every candidate GPU type, and
    shelling out ~30 times would dominate the runtime of a command that
    otherwise only reads HTTP headers. Returns a tuple so the cache key is
    hashable; an empty tuple means the CLI was unavailable.
    """
    try:
        out = runpodctl_json("datacenter", "list")
        return tuple(json.loads(out[out.index("[") :]))
    except Exception:
        return ()


def datacenters_for_gpu(
    gpu_id: str, require_network_volume: bool = False
) -> list[tuple[str, str, str]]:
    """Datacenters offering `gpu_id`, most likely to fill first.

    Returns (datacenter_id, location, stock_status). A network volume is bound
    to one datacenter and a pod can only mount a volume in its own, so the
    volume has to be created where the GPU actually is -- picking the wrong
    one means creating the volume, failing to place the pod, and deleting it
    again.

    With `require_network_volume`, the result is additionally filtered to
    datacenters that support network volumes at all. That is a strictly
    smaller set which no API field exposes; see NETWORK_VOLUME_DATACENTERS.
    """
    catalog = datacenter_catalog()
    hits = [
        (dc["id"], dc.get("location", "?"), gpu.get("stockStatus", ""))
        for dc in catalog
        for gpu in dc.get("gpuAvailability", [])
        if gpu.get("gpuId") == gpu_id
        and (not require_network_volume or dc["id"] in NETWORK_VOLUME_DATACENTERS)
    ]
    return sorted(hits, key=lambda h: (STOCK_RANK.get(h[2], 9), h[0]))
