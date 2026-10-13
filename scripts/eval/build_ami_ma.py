#!/usr/bin/env python3
"""Build `ami-ma`: the 24 full AMI meetings AssemblyAI's MA tool scores diarization on.

The MA tool's `ami_diarization_der_cpwer_11_2025` set is these meetings
(DeepLearning `research/projects/diarization/evaluation/eval_datasets/ami.py`),
the pyannote AMI test split: each meeting's real Mix-Headset recording, with
references from AMI's manual word annotations. This downloads both from the AMI
corpus site and writes one row per meeting: the audio, the reference as
`<SPK_n>` turns, and the meeting id.

    python -m scripts.eval.build_ami_ma --push mazesmazes/ami-ma-meetings
    ta eval -d ami-ma ...
"""

import heapq
import io
import itertools
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
import typer
from datasets import Audio, Dataset
from rich.console import Console

from scripts.eval.speaker_metrics import serialize_turns

app = typer.Typer(add_completion=False)
console = Console()

# Same order as the MA tool's `AMIDataset._items`.
MEETINGS = [
    "TS3007b", "TS3007c", "EN2002b", "TS3007a", "EN2002a", "ES2014c",
    "TS3003b", "TS3003d", "EN2002d", "IS1009c", "EN2002c", "TS3003c",
    "ES2014b", "ES2014d", "ES2004d", "IS1009a", "ES2004c", "IS1009b",
    "ES2004a", "TS3007d", "ES2004b", "ES2014a", "IS1009d", "TS3003a",
]  # fmt: skip

AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
)
ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


class Word(TypedDict):
    """One reference word: text, AMI speaker letter, start time in seconds."""

    text: str
    speaker: str
    start: float


def parse_words(annotations: zipfile.ZipFile, meeting: str) -> list[Word]:
    """A meeting's reference words, kept exactly as the MA tool's `parse_ami_annos` keeps them.

    Every `<w>` with start and end times and text, except punctuation; speakers
    are the file letters A-Z. Words are returned per speaker in file order, and
    the speakers in letter order, as the MA tool builds them.
    """
    names = set(annotations.namelist())
    words: list[Word] = []
    for letter in (chr(c) for c in range(ord("A"), ord("Z") + 1)):
        name = f"words/{meeting}.{letter}.words.xml"
        if name not in names:
            continue
        # AMI's own annotation release, fetched from the corpus site over HTTPS.
        for child in ET.fromstring(annotations.read(name)):  # nosec B314
            if child.attrib.get("punc", "false") == "true":
                continue
            if (
                child.tag == "w"
                and "starttime" in child.attrib
                and "endtime" in child.attrib
                and child.text
            ):
                words.append(
                    {
                        "text": child.text,
                        "speaker": letter,
                        "start": float(child.attrib["starttime"]),
                    }
                )
    return words


def reference_text(words: list[Word]) -> str:
    """`<SPK_n>` turns in time order, each speaker's words still in file order.

    `heapq.merge` interleaves the speakers by start time without reordering any
    one speaker's words, so the word sequence cpWER reads per speaker is exactly
    the MA tool's.
    """
    per_speaker = [list(group) for _, group in itertools.groupby(words, key=lambda w: w["speaker"])]
    ordered = heapq.merge(*per_speaker, key=lambda w: w["start"])
    return serialize_turns((w["speaker"], w["text"]) for w in ordered)


def _download(url: str) -> bytes:
    console.print(f"  {url}")
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.content


@app.command()
def main(
    out: Annotated[
        Path | None, typer.Option("--out", help="Also write the test split here as Parquet")
    ] = None,
    push: Annotated[
        str | None, typer.Option("--push", help="Push to this Hub dataset repo (private)")
    ] = None,
) -> None:
    """Download the 24 MA-tool AMI meetings and build the `ami-ma` eval set."""
    if out is None and push is None:
        msg = "pass --out, --push or both"
        raise typer.BadParameter(msg)
    console.print("AMI manual annotations:")
    annotations = zipfile.ZipFile(io.BytesIO(_download(ANNOTATIONS_URL)))
    rows = []
    for meeting in MEETINGS:
        words = parse_words(annotations, meeting)
        if not words:
            msg = f"no reference words for {meeting} in {ANNOTATIONS_URL}"
            raise RuntimeError(msg)
        console.print(f"{meeting}: {len(words)} words")
        audio = _download(AUDIO_URL.format(m=meeting))
        rows.append(
            {
                "audio": {"bytes": audio, "path": f"{meeting}.Mix-Headset.wav"},
                "text": reference_text(words),
                "meeting_id": meeting,
            }
        )
    ds = Dataset.from_list(rows).cast_column("audio", Audio())
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(out / "test-00000-of-00001.parquet"))
        console.print(f"Wrote {len(ds)} meetings to {out}")
    if push is not None:
        ds.push_to_hub(push, split="test", private=True)
        console.print(f"Pushed {len(ds)} meetings to {push}")


if __name__ == "__main__":
    app()
