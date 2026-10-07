"""NOTSOFAR-1 far-field meetings as short real-chunk rows (scripts/speaker_asr/chunks.py).

AMI rows are SUMMED close-talk headset clips: exact per-speaker labels, but none of
the room -- reverb, distance, crosstalk, real acoustic overlap -- that long-form
evaluation audio has. NOTSOFAR-1 (Vinnikov et al., Interspeech 2024; CC BY 4.0,
huggingface.co/datasets/microsoft/NOTSOFAR) records real, unscripted office meetings
of 4-8 people on several far-field devices at once, with human transcripts carrying
WORD timings on the shared meeting timeline. So a training chunk is simply a 3-8 s
slice of one device's recording, cut at quiet points as inference cuts it, and its
target comes from the words inside it -- no mixing and no forced alignment.

- Every single-channel device (`sc_*/ch0.wav`) of a meeting is its own recording:
  same words, different room position and microphone.
- Transcript text is cased, punctuated and keeps disfluencies (um, uh, Mm-hmm,
  repetitions), as AMI's self-distilled targets do. Annotation tags are dropped
  (`<ST/>` sentence truncation, `<FILL/>`, `<FILLlaugh/>`, `<BA/>`, ...), and
  `<PName>Bob</PName>` keeps the name.
- A chunk overlapping an `<UNKNOWN/>` (unintelligible) word is dropped: its audio
  holds speech the target cannot spell.
- Only 240825.1_train and the eval sets are open-licensed; Dev-set-2 is challenge-only
  and is never fetched. dev1 shares speakers with train, so held-out rows come from
  eval_small_with_GT (speakers disjoint from train).
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.speaker_asr.chunks import ShortChunkConfig, plan_recording_chunks

log = logging.getLogger(__name__)

_NAME = re.compile(r"</?PName>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]*/>")
_UNKNOWN = "<unknown/>"


@dataclass(frozen=True)
class NotsofarConfig:
    """configs/speaker_asr `notsofar:` -- see that section for the fields."""

    repo_id: str = "microsoft/NOTSOFAR"
    train_version: str = "train_set/240825.1_train"
    eval_version: str | None = "eval_set/240629.1_eval_small_with_GT"
    local_dir: str = "data/notsofar"
    # Single-channel devices to use per meeting (glob on the device folder name).
    devices: str = "sc_*"
    eval_rows: int = 400


def subset_root(version: str) -> str:
    """Repo path of one subset's meetings, e.g. benchmark-datasets/train_set/240825.1_train/MTG."""
    return f"benchmark-datasets/{version}/MTG"


def download(ncfg: NotsofarConfig, version: str) -> Path:
    """Fetch one subset's transcripts + single-channel recordings (not the 7-ch arrays or CT)."""
    from huggingface_hub import snapshot_download

    if "240415.2" in version:
        raise ValueError(
            "NOTSOFAR Dev-set-2 is licensed for the challenge only; do not train on it"
        )
    root = subset_root(version)
    snapshot_download(
        ncfg.repo_id,
        repo_type="dataset",
        local_dir=ncfg.local_dir,
        allow_patterns=[
            f"{root}/*/gt_transcription.json",
            f"{root}/*/devices.json",
            f"{root}/*/{ncfg.devices}/ch0.wav",
        ],
    )
    return Path(ncfg.local_dir) / root


def clean_text(text: str) -> str:
    """Transcript text without annotation tags: '<ST/> Hi <PName>Bob</PName> <FILL/>' -> 'Hi Bob'."""
    return " ".join(_TAG.sub(" ", _NAME.sub("", text)).split())


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def segment_words(segment: dict) -> tuple[list[tuple], list[tuple[float, float]]]:
    """(timed words relative to the segment start, absolute <UNKNOWN/> spans) of one segment.

    Words are the cleaned transcript's tokens (cased, punctuated); times come from
    `word_timing` (lowercased, unpunctuated), matched in order. A token with no match
    within a few timing entries is left unaligned and interpolated, as
    chunks.timed_words does for alignment gaps.
    """
    start = float(segment["start_time"])
    timing, unknown = [], []
    for w, s, e in segment.get("word_timing") or []:
        if w.lower() == _UNKNOWN:
            unknown.append((float(s), float(e)))
        elif not w.startswith("<"):
            timing.append((_norm(w), float(s), float(e)))
    tokens: list[str] = []
    for tok in clean_text(segment["text"]).split():
        if tokens and not _norm(tok):  # a stray ',' or '-' token: punctuation of the word before
            tokens[-1] += tok
        else:
            tokens.append(tok)
    times: list[tuple | None] = [None] * len(tokens)
    j = 0
    for k, tok in enumerate(tokens):
        key = _norm(tok)
        for look in range(j, min(j + 4, len(timing))):
            if timing[look][0] == key:
                times[k] = timing[look][1:]
                j = look + 1
                break
    out, last_end = [], start
    for k, (tok, t) in enumerate(zip(tokens, times, strict=True)):
        if t is None:
            later = [x for x in times[k + 1 :] if x is not None]
            nxt = later[0][0] if later else last_end
            out.append((tok, last_end - start, max(last_end, nxt) - start, False))
        else:
            out.append((tok, t[0] - start, t[1] - start, True))
            last_end = t[1]
    return out, unknown


def meeting_utterances(transcript: list[dict]) -> tuple[list[dict], list[tuple[float, float]]]:
    """chunks.plan_recording_chunks utterances for one meeting, and its <UNKNOWN/> spans."""
    utts, blocked = [], []
    for seg in transcript:
        words, unknown = segment_words(seg)
        blocked += unknown
        if words:
            utts.append(
                {"speaker": seg["speaker_id"], "start": float(seg["start_time"]), "words": words}
            )
    return utts, blocked


def drop_blocked(rows: list[dict], blocked: list[tuple[float, float]]) -> list[dict]:
    """Rows whose [start_s, end_s) touches no <UNKNOWN/> span."""
    return [r for r in rows if not any(s < r["end_s"] and e > r["start_s"] for s, e in blocked)]


def build_notsofar_rows(
    ncfg: NotsofarConfig, scfg: ShortChunkConfig, version: str, max_speakers: int, seed: int = 0
) -> list[dict]:
    """Short chunk rows over every device recording of a subset (downloaded on first use).

    Each row's `recording` is relative to ncfg.local_dir (SpeakerASRDataset's recordings_root).
    """
    import soundfile as sf

    root = download(ncfg, version)
    local = Path(ncfg.local_dir)
    rng = random.Random(seed)
    rows, n_devices = [], 0
    for meeting in sorted(p for p in root.iterdir() if p.is_dir()):
        transcript = json.loads((meeting / "gt_transcription.json").read_text())
        utts, blocked = meeting_utterances(transcript)
        for wav in sorted(meeting.glob(f"{ncfg.devices}/ch0.wav")):
            audio, sr = sf.read(str(wav), dtype="float32")
            if sr != 16000:
                raise ValueError(f"{wav} is {sr} Hz; NOTSOFAR ships 16 kHz")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if not np.any(audio):
                continue  # a dead device (train logs list a few removed white-noise/blank recordings)
            group = f"notsofar:{meeting.name}:{wav.parent.name}"
            rec = str(wav.relative_to(local))
            rows += drop_blocked(
                plan_recording_chunks(utts, audio, scfg, group, rec, rng, max_speakers), blocked
            )
            n_devices += 1
    log.info("notsofar %s: %d chunks from %d device recordings", version, len(rows), n_devices)
    return rows
