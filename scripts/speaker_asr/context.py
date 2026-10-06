"""Labelled context prefixes: voice samples + the previous chunk, so labels persist.

A 30 s window alone cannot know that its first voice is the meeting's third
speaker. Long-form decoding therefore gives the model what it has already
heard, as audio IN FRONT of the new chunk with its labelled transcript as a
forced prefix of the assistant turn:

    audio:   [sample: speaker 1][sample: speaker 2] [previous chunk's tail] [new chunk]
    prefix:  <SPK_1>sample text<SPK_2>sample text<SPK_2>tail text<SPK_1>tail text<CONTINUE>
    target:  <SPK_2>new words<SPK_3>a new voice<SPK_1>...

- Voice samples (one short clip per known speaker) carry identity across any
  distance: a speaker returning after 40 minutes is matched by listening to
  their sample, not by an external embedding model.
- The previous chunk's tail carries the conversation across the cut, so a
  turn that spans two chunks keeps its label.
- Known speakers are numbered 1..m in order of first appearance; a voice
  matching none of them opens m+1. Long-form decoding maps these positional
  labels back to recording-level speaker ids (scripts/speaker_asr/longform.py).

Training rows come from consecutive windows of one AMI meeting, whose
speaker ids are global: "known" speakers are those heard in earlier windows,
their samples are earlier utterances, the tail is the previous window's end.
The prefix is masked from the loss; only the new chunk is supervised.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass

from scripts.speaker_asr.metrics import CONTEXT_END, SPEAKER_TOKEN, serialize_turns


@dataclass(frozen=True)
class ContextConfig:
    """Prefix layout, shared by training rows and long-form decoding. Seconds."""

    # Voice-sample length range; samples are whole utterances (training) or
    # word spans (decoding) inside it.
    memory_s: tuple[float, float] = (2.0, 6.0)
    memory_gap_s: float = 0.4
    # At most this many samples; beyond it the most recently heard speakers win.
    max_memory: int = 6
    # Tail of the previous chunk carried over, and how often (training only).
    carry_s: float = 8.0
    carry_prob: float = 0.8
    # Silence between the prefix audio and the new chunk.
    live_gap_s: float = 0.3
    # Training rows longer than this (prefix + window) are dropped.
    max_audio_s: float = 90.0
    # Long-form decoding: chunks are cut at the quietest point between these.
    min_chunk_s: float = 10.0
    chunk_s: float = 25.0


def prompt_speakers(first_seen: dict, last_seen: dict, has_sample, max_memory: int) -> list:
    """Speakers to put in the prefix, in first-appearance order (= labels 1..m).

    Only speakers with a voice sample can be listed; with more than
    `max_memory`, the most recently heard are kept. Training and decoding
    both use this, so the numbering the model learns is the one it gets.
    """
    known = [s for s in first_seen if has_sample(s)]
    if len(known) > max_memory:
        keep = set(sorted(known, key=lambda s: -last_seen[s])[:max_memory])
        known = [s for s in known if s in keep]
    return sorted(known, key=lambda s: first_seen[s])


def build_context_rows(
    rows: list[dict],
    starts: dict[str, float],
    texts: dict[str, str],
    cfg: ContextConfig,
    seed: int = 0,
) -> list[dict]:
    """Pool rows -> rows whose audio and assistant turn start with a context prefix.

    `starts` maps utterance id -> its start on the meeting timeline (to order
    windows); `texts` utterance id -> transcript. Each output row keeps the
    pool row's shape (`parts` as JSON, now including the prefix audio,
    `tail_s`, `duration_s`, `target`, `n_speakers`) plus `prefix` and
    `n_memory`. Deterministic in `seed`.
    """
    rng = random.Random(seed)
    lo, hi = cfg.memory_s
    by_group: dict[str, list] = defaultdict(list)
    for row in rows:
        parts = json.loads(row["parts"])
        by_group[row["group"]].append((min(starts[p["id"]] for p in parts), row, parts))

    def text(part: dict) -> str:
        return texts.get(part["id"], "").strip()

    out = []
    for group in sorted(by_group):
        first_seen: dict[str, int] = {}
        last_seen: dict[str, int] = {}
        samples: dict[str, list[dict]] = {}  # only speakers with >= 1 sample
        prev = None
        for index, (_, row, parts) in enumerate(sorted(by_group[group], key=lambda t: t[0])):
            known = prompt_speakers(first_seen, last_seen, samples.__contains__, cfg.max_memory)
            labels = {s: i for i, s in enumerate(known, 1)}
            audio, prefix, cursor = [], "", 0.0
            for speaker in known:
                clip = rng.choice(samples[speaker])
                audio.append({**clip, "offset_s": round(cursor, 3)})
                prefix += SPEAKER_TOKEN.format(labels[speaker]) + text(clip)
                cursor += clip["dur_s"] + cfg.memory_gap_s
            if prev is not None and labels and rng.random() < cfg.carry_prob:
                prev_row, prev_parts = prev
                cut = prev_row["duration_s"] - cfg.carry_s
                carry = [
                    p
                    for p in prev_parts
                    if p["offset_s"] >= cut and p["speaker"] in labels and text(p)
                ]
                if carry:
                    base = min(p["offset_s"] for p in carry)
                    carry.sort(key=lambda p: (p["offset_s"], p["dur_s"]))
                    audio += [
                        {**p, "offset_s": round(cursor + p["offset_s"] - base, 3)} for p in carry
                    ]
                    prefix += serialize_turns([(p["speaker"], text(p)) for p in carry], labels)
                    cursor += max(p["offset_s"] - base + p["dur_s"] for p in carry)
            if audio:
                cursor += cfg.live_gap_s
            live = sorted(parts, key=lambda p: (p["offset_s"], p["dur_s"]))
            target = serialize_turns([(p["speaker"], text(p)) for p in live], labels)
            duration = cursor + row["duration_s"]

            for p in live:  # this window becomes history for the next
                first_seen.setdefault(p["speaker"], len(first_seen))
                last_seen[p["speaker"]] = index
                if lo <= p["dur_s"] <= hi and text(p):
                    samples.setdefault(p["speaker"], []).append(p)
            prev = (row, parts)

            if not target or duration > cfg.max_audio_s:
                continue
            audio += [{**p, "offset_s": round(cursor + p["offset_s"], 3)} for p in parts]
            out.append(
                {
                    **row,
                    "parts": json.dumps(audio),
                    "duration_s": round(duration, 3),
                    "prefix": prefix + CONTEXT_END,
                    "target": target,
                    "n_speakers": len({s for s, t in ((p["speaker"], text(p)) for p in live) if t}),
                    "n_memory": len(known),
                    # the live window's utterances: the target's turns come from these alone
                    "live_ids": json.dumps([p["id"] for p in live]),
                }
            )
    return out
