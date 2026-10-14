#!/usr/bin/env python3
"""Build `notsofar-train`: NOTSOFAR-1 train + dev1 as far-field multi-speaker windows.

NOTSOFAR-1 (Vinnikov et al., Interspeech 2024; CC BY 4.0,
huggingface.co/datasets/microsoft/NOTSOFAR) records unscripted office meetings
of 4-8 people on several far-field devices at once, with human transcripts
whose words are timed on the shared meeting timeline.

A row is a window of one device's recording, cut by the inference chunker
(`chunk_bounds`: the quietest 100 ms frame between a minimum and maximum
length, drawn per recording from 3-18 s), labelled with EVERY word spoken
inside it -- all speakers, overlap included -- so the target is what is
audible, as single-stream inference sees it. Per-speaker segments would label
one voice and leave out the other: half of NOTSOFAR's segment audio has a
second speaker over it.

- Words come from the cleaned transcript text (cased, punctuated, spoken
  fillers kept, annotation tags dropped), timed by matching it to
  `word_timing`. A word is in a window when at least half of it is; words are
  grouped by transcript segment in start order, as the eval references are,
  so two overlapping speakers are not interleaved mid-phrase.
- A window is dropped if it touches an `<UNKNOWN/>` (unintelligible) span,
  holds no words, or has under 80% of its words timed (its edges are guesses).
- 240825.1_train (72 meetings) and 240825.1_dev1 (36): 14 rooms, sharing no
  meeting, room or speaker with the eval set `notsofar-test`
  (240825.1_eval_full_with_GT), which is never read here. dev1 shares speakers
  with train, so it stops being a clean eval once this trains. Dev-set-2
  (240415.2) is challenge-only and is refused.
- `--devices-per-meeting` single-channel devices per meeting (default 2, picked
  at random per meeting): the same words under different microphones, so more
  devices are acoustic augmentation over fixed text, not new data.

    python -m scripts.build_notsofar_train --push mazesmazes/notsofar-train
"""

import io
import json
import random
import re
from pathlib import Path
from typing import Annotated, Any, NamedTuple

import numpy as np
import numpy.typing as npt
import soundfile
import typer
from huggingface_hub import HfApi
from rich.console import Console

from scripts.eval.build_ami_meetings import publish_meetings
from scripts.eval.build_notsofar import REPO_ID, clean_text, download_file, meeting_devices
from scripts.eval.build_notsofar import VERSION as EVAL_VERSION
from tiny_audio.asr_processing import chunk_bounds

app = typer.Typer(add_completion=False)
console = Console()

VERSIONS = ("train_set/240825.1_train", "dev_set/240825.1_dev1")
SAMPLE_RATE = 16000
# Window length range: chunk_bounds' min/max are drawn per recording from these.
MIN_S = (3.0, 8.0)
MAX_S = (10.0, 18.0)
# A word belongs to a window when at least this share of it lies inside.
EDGE_KEEP = 0.5
# Windows with fewer of their words timed than this are dropped.
MIN_TIMED = 0.8
_UNKNOWN = "<unknown/>"


class Word(NamedTuple):
    """One transcript word on the meeting timeline."""

    text: str
    start: float
    end: float
    timed: bool  # False: matched no `word_timing` entry; times interpolated
    segment: int  # its segment's position in start order


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def segment_words(segment: dict[str, Any], index: int) -> list[Word]:
    """A segment's cleaned-text words, timed from its `word_timing`.

    The cleaned text (cased, punctuated) is matched in order to the lowercase,
    unpunctuated `word_timing` entries, looking a few entries ahead; a stray
    punctuation token joins the word before it. An unmatched word is placed
    between its neighbours' times and marked untimed.
    """
    timing = [
        (_norm(w), float(s), float(e))
        for w, s, e in segment.get("word_timing") or []
        if not w.startswith("<")
    ]
    tokens: list[str] = []
    for tok in clean_text(segment["text"]).split():
        if tokens and not _norm(tok):
            tokens[-1] += tok
        else:
            tokens.append(tok)
    times: list[tuple[float, float] | None] = [None] * len(tokens)
    j = 0
    for k, tok in enumerate(tokens):
        for look in range(j, min(j + 4, len(timing))):
            if timing[look][0] == _norm(tok):
                times[k] = (timing[look][1], timing[look][2])
                j = look + 1
                break
    words: list[Word] = []
    last_end = float(segment["start_time"])
    for k, (tok, t) in enumerate(zip(tokens, times, strict=True)):
        if t is None:
            nxt = next((x[0] for x in times[k + 1 :] if x is not None), last_end)
            words.append(Word(tok, last_end, max(last_end, nxt), False, index))
        else:
            words.append(Word(tok, t[0], t[1], True, index))
            last_end = t[1]
    return words


def unknown_spans(segment: dict[str, Any]) -> list[tuple[float, float]]:
    """Where a segment's `<UNKNOWN/>` words are; the whole segment if they are untimed."""
    timed = [
        (float(s), float(e))
        for w, s, e in segment.get("word_timing") or []
        if w.lower() == _UNKNOWN
    ]
    if timed:
        return timed
    if _UNKNOWN in segment["text"].lower():
        return [(float(segment["start_time"]), float(segment["end_time"]))]
    return []


def meeting_words(
    transcript: list[dict[str, Any]],
) -> tuple[list[Word], list[tuple[float, float]]]:
    """Every word of a meeting, and its `<UNKNOWN/>` spans; segments in start order."""
    segments = sorted(transcript, key=lambda s: (float(s["start_time"]), float(s["end_time"])))
    words: list[Word] = []
    blocked: list[tuple[float, float]] = []
    for i, seg in enumerate(segments):
        words += segment_words(seg, i)
        blocked += unknown_spans(seg)
    return words, blocked


def window_text(words: list[Word], start: float, end: float) -> tuple[str, int, int]:
    """(text, words kept, words timed) of the window `[start, end)` seconds.

    A word is kept when at least EDGE_KEEP of it lies inside (a zero-length word
    when it starts inside). Kept words are grouped by segment, segments in start
    order, so two overlapping speakers' words are not interleaved mid-phrase.
    """
    kept: list[Word] = []
    for w in words:
        if w.end <= w.start:
            inside = start <= w.start < end
        else:
            inside = (min(w.end, end) - max(w.start, start)) / (w.end - w.start) >= EDGE_KEEP
        if inside:
            kept.append(w)
    kept.sort(key=lambda w: w.segment)  # stable: each segment's words stay in order
    return " ".join(w.text for w in kept), len(kept), sum(w.timed for w in kept)


def window_rows(
    words: list[Word],
    blocked: list[tuple[float, float]],
    audio: npt.NDArray[np.float32],
    meeting: str,
    device: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """One row per usable window of one device's `audio`."""
    min_s = rng.uniform(*MIN_S)
    max_s = rng.uniform(max(MAX_S[0], min_s + 2.0), MAX_S[1])
    rows: list[dict[str, Any]] = []
    for lo, hi in chunk_bounds(audio, SAMPLE_RATE, max_s, min_s):
        start, end = lo / SAMPLE_RATE, hi / SAMPLE_RATE
        if any(s < end and e > start for s, e in blocked):
            continue
        text, kept, timed = window_text(words, start, end)
        if not kept or timed < MIN_TIMED * kept:
            continue
        flac = io.BytesIO()
        soundfile.write(flac, audio[lo:hi], SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        rows.append(
            {
                "audio": {"bytes": flac.getvalue(), "path": f"{meeting}.{device}.{lo}.flac"},
                "text": text,
                "meeting_id": meeting,
                "device": device,
                "audio_duration": (hi - lo) / SAMPLE_RATE,
            }
        )
    return rows


def live_recordings(
    paths: list[str], count: int, rng: random.Random
) -> list[tuple[str, npt.NDArray[np.float32]]]:
    """Up to `count` non-silent recordings from `paths`, in random order."""
    order = paths[:]
    rng.shuffle(order)
    picked: list[tuple[str, npt.NDArray[np.float32]]] = []
    for path in order:
        data, sr = soundfile.read(str(download_file(path)), dtype="float32")
        audio = np.asarray(data, dtype=np.float32)
        if sr != SAMPLE_RATE:
            msg = f"{path} is {sr} Hz; NOTSOFAR ships 16 kHz"
            raise ValueError(msg)
        if np.any(audio):
            picked.append((path, audio))
        if len(picked) == count:
            break
    return picked


@app.command()
def main(
    out: Annotated[
        Path | None, typer.Option("--out", help="Also write the train split here as Parquet")
    ] = None,
    push: Annotated[
        str | None, typer.Option("--push", help="Push to this Hub dataset repo (private)")
    ] = None,
    devices_per_meeting: Annotated[
        int, typer.Option("--devices-per-meeting", help="Single-channel devices per meeting")
    ] = 2,
    seed: Annotated[int, typer.Option("--seed", help="Seeds device picks and window lengths")] = 0,
) -> None:
    """Download NOTSOFAR-1 train + dev1 and build the `notsofar-train` training set."""
    if out is None and push is None:
        msg = "pass --out, --push or both"
        raise typer.BadParameter(msg)
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    eval_root = f"benchmark-datasets/{EVAL_VERSION}/MTG/"
    eval_meetings = {
        f.removeprefix(eval_root).split("/")[0] for f in files if f.startswith(eval_root)
    }
    rows: list[dict[str, Any]] = []
    for version in VERSIONS:
        if "240415.2" in version:
            msg = "NOTSOFAR Dev-set-2 is licensed for the challenge only"
            raise ValueError(msg)
        root = f"benchmark-datasets/{version}/MTG/"
        recordings = meeting_devices(files, root, "sc_")
        leaked = recordings.keys() & eval_meetings
        if leaked:
            msg = f"{version} shares meetings with {EVAL_VERSION}: {sorted(leaked)}"
            raise RuntimeError(msg)
        console.print(f"{version}: {len(recordings)} meetings")
        for meeting, paths in recordings.items():
            transcript = json.loads(
                download_file(f"{root}{meeting}/gt_transcription.json").read_text()
            )
            words, blocked = meeting_words(transcript)
            rng = random.Random(f"{seed}:{meeting}")
            n = len(rows)
            for path, audio in live_recordings(paths, devices_per_meeting, rng):
                rows += window_rows(words, blocked, audio, meeting, path.split("/")[-2], rng)
            console.print(f"  {meeting}: {len(rows) - n} windows")
    console.print(f"{len(rows)} rows, {sum(r['audio_duration'] for r in rows) / 3600:.1f} h")
    publish_meetings(rows, out, push, split="train")


if __name__ == "__main__":
    app()
