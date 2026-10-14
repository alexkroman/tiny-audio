#!/usr/bin/env python3
"""Build `notsofar-test`: the 129 meetings of NOTSOFAR-1's full eval set.

NOTSOFAR-1 (Vinnikov et al., Interspeech 2024; CC BY 4.0,
huggingface.co/datasets/microsoft/NOTSOFAR) records unscripted office meetings
of 4-8 people on several far-field devices at once. Each row here is one
meeting: one single-channel device's recording, the reference as `<SPK_n>`
turns from the human transcript, and the meeting and device ids.

    python -m scripts.eval.build_notsofar --push mazesmazes/notsofar-test
    ta eval -d notsofar-test ...
"""

import io
import json
import re
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import soundfile
import typer
from huggingface_hub import HfApi, hf_hub_download
from rich.console import Console

from scripts.eval.build_ami_meetings import publish_meetings
from scripts.eval.speaker_metrics import serialize_turns

app = typer.Typer(add_completion=False)
console = Console()

REPO_ID = "microsoft/NOTSOFAR"
# The full eval set with ground truth: 129 meetings, speakers disjoint from train.
VERSION = "eval_set/240825.1_eval_full_with_GT"

_NAME = re.compile(r"</?PName>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]*/>")


def clean_text(text: str) -> str:
    """Transcript text without annotation tags: '<ST/> Hi <PName>Bob</PName> <FILL/>' -> 'Hi Bob'.

    Spoken fillers (um, uh, Mm-hmm) are words in the text and stay; `<FILL/>`,
    `<ST/>` (sentence truncation), `<UNKNOWN/>` and the other tags are dropped.
    """
    return " ".join(_TAG.sub(" ", _NAME.sub("", text)).split())


def reference_text(transcript: list[dict[str, Any]]) -> str:
    """`<SPK_n>` turns from the transcript's segments, by start time.

    Segments are one speaker each but not sorted in the file; sorting by start
    keeps every speaker's own segments in order, which is what cpWER reads.
    """
    segments = sorted(transcript, key=lambda s: (float(s["start_time"]), float(s["end_time"])))
    return serialize_turns((s["speaker_id"], clean_text(s["text"])) for s in segments)


def meeting_devices(files: list[str], root: str, devices: str) -> dict[str, list[str]]:
    """Meeting -> its recordings matching the `devices` prefix (`<dev>/ch0.wav`), sorted."""
    out: dict[str, list[str]] = {}
    for f in files:
        parts = f.removeprefix(root).split("/")
        if len(parts) == 3 and parts[1].startswith(devices) and parts[2] == "ch0.wav":
            out.setdefault(parts[0], []).append(f)
    return {m: sorted(paths) for m, paths in sorted(out.items())}


def download_file(path: str) -> Path:
    return Path(hf_hub_download(REPO_ID, path, repo_type="dataset"))


def first_live_recording(paths: list[str]) -> tuple[str, bytes]:
    """The first recording that is not silent: a few devices recorded nothing."""
    for path in paths:
        data = download_file(path).read_bytes()
        audio, sr = soundfile.read(io.BytesIO(data), dtype="float32")
        if sr != 16000:
            msg = f"{path} is {sr} Hz; NOTSOFAR ships 16 kHz"
            raise ValueError(msg)
        if np.any(audio):
            return path, data
        console.print(f"  [yellow]{path} is silent, trying the next device[/yellow]")
    msg = f"every device is silent: {paths}"
    raise RuntimeError(msg)


@app.command()
def main(
    out: Annotated[
        Path | None, typer.Option("--out", help="Also write the test split here as Parquet")
    ] = None,
    push: Annotated[
        str | None, typer.Option("--push", help="Push to this Hub dataset repo (private)")
    ] = None,
    devices: Annotated[
        str,
        typer.Option(
            "--devices",
            help="Device folder prefix; the first matching device by name is used per meeting",
        ),
    ] = "sc_",
) -> None:
    """Download NOTSOFAR-1's full eval set and build the `notsofar-test` eval set."""
    if out is None and push is None:
        msg = "pass --out, --push or both"
        raise typer.BadParameter(msg)
    root = f"benchmark-datasets/{VERSION}/MTG/"
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    recordings = meeting_devices(files, root, devices)
    console.print(f"{len(recordings)} meetings in {VERSION}")
    rows = []
    for meeting, paths in recordings.items():
        transcript = json.loads(download_file(f"{root}{meeting}/gt_transcription.json").read_text())
        path, audio = first_live_recording(paths)
        device = path.split("/")[-2]
        console.print(f"{meeting}: {len(transcript)} segments, {device}")
        rows.append(
            {
                "audio": {"bytes": audio, "path": f"{meeting}.{device}.wav"},
                "text": reference_text(transcript),
                "meeting_id": meeting,
                "device": device,
            }
        )
    publish_meetings(rows, out, push)


if __name__ == "__main__":
    app()
