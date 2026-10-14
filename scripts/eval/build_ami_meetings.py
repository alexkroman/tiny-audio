#!/usr/bin/env python3
"""Build `ami-meetings-24`: the 24 full meetings of the pyannote AMI test split.

Each meeting's real Mix-Headset recording, with references from AMI's manual
word annotations. This downloads both from the AMI corpus site and writes one
row per meeting: the audio, the reference as `<SPK_n>` turns, and the meeting id.

    python -m scripts.eval.build_ami_meetings --push mazesmazes/ami-meetings-24
    ta eval -d ami-meetings-24 ...
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
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from scripts.eval.speaker_metrics import serialize_turns

app = typer.Typer(add_completion=False)
console = Console()

# The pyannote AMI test split.
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
CACHE_DIR = Path.home() / ".cache" / "ami-meetings"


class Word(TypedDict):
    """One reference word: text, AMI speaker letter, start time in seconds."""

    text: str
    speaker: str
    start: float


def parse_words(annotations: zipfile.ZipFile, meeting: str) -> list[Word]:
    """A meeting's reference words: every timed `<w>` with text, except punctuation.

    Speakers are the file letters A-Z. Words are returned per speaker in file
    order, and the speakers in letter order.
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
    one speaker's words, so cpWER reads each speaker's words in file order.
    """
    per_speaker = [list(group) for _, group in itertools.groupby(words, key=lambda w: w["speaker"])]
    ordered = heapq.merge(*per_speaker, key=lambda w: w["start"])
    return serialize_turns((w["speaker"], w["text"]) for w in ordered)


# The mirror also answers a burst of retries with a transient 403 (throttling),
# so retry that and 429/5xx alongside dropped connections.
_RETRY_STATUS = {403, 429, 500, 502, 503, 504}


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(12),
    wait=wait_exponential_jitter(initial=5, max=120),
    before_sleep=lambda state: console.print(
        f"  [yellow]retrying after {state.outcome and state.outcome.exception()!r}[/yellow]"
    ),
)
def _fetch_to(url: str, part: Path) -> None:
    """Stream `url` into `part`, resuming from its current size with a Range request.

    The AMI mirror drops long transfers mid-body; each retry picks up where the
    last one stopped instead of starting the file over.
    """
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as response:
        if response.status_code == 416:  # already have every byte
            return
        response.raise_for_status()
        # A server that ignores Range answers 200 with the whole file: start over.
        mode = "ab" if response.status_code == 206 else "wb"
        with part.open(mode) as f:
            for chunk in response.iter_bytes():
                f.write(chunk)


def _download(url: str, cache_dir: Path) -> bytes:
    """`url`'s bytes, cached in `cache_dir` so a rerun skips finished files."""
    console.print(f"  {url}")
    path = cache_dir / url.rsplit("/", 1)[-1]
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        _fetch_to(url, part)
        part.rename(path)
    return path.read_bytes()


@app.command()
def main(
    out: Annotated[
        Path | None, typer.Option("--out", help="Also write the test split here as Parquet")
    ] = None,
    push: Annotated[
        str | None, typer.Option("--push", help="Push to this Hub dataset repo (private)")
    ] = None,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Keep downloads here; reruns skip finished files")
    ] = CACHE_DIR,
) -> None:
    """Download the 24 AMI meetings and build the `ami-meetings-24` eval set."""
    if out is None and push is None:
        msg = "pass --out, --push or both"
        raise typer.BadParameter(msg)
    console.print("AMI manual annotations:")
    annotations = zipfile.ZipFile(io.BytesIO(_download(ANNOTATIONS_URL, cache_dir)))
    rows = []
    for meeting in MEETINGS:
        words = parse_words(annotations, meeting)
        if not words:
            msg = f"no reference words for {meeting} in {ANNOTATIONS_URL}"
            raise RuntimeError(msg)
        console.print(f"{meeting}: {len(words)} words")
        audio = _download(AUDIO_URL.format(m=meeting), cache_dir)
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
