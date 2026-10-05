"""Junction windows: the same voice minutes apart, so the labeller learns to verify.

Pool windows (data.plan_windows) are contiguous: two clips of one person in a
window are almost always seconds apart, mid-conversation. Long-form linking
(scripts/speaker_asr/longform.py and friends) asks a different question --
is the voice in this chunk the one heard four minutes ago? -- and the model
only ever answers it at a JUNCTION: audio [A][gap][B], transcript
`<SPK_1>A...` followed by either more words (same voice) or `<SPK_2>` (new).

A junction window stitches k whole utterances drawn at least
`min_separation_s` apart in one meeting, each next clip's speaker chosen so
"a speaker already in the window" and "a new speaker" are equally likely
(`same_prob`). Same-meeting negatives share room and microphones, so only the
voice can tell them apart; same-speaker positives differ in energy, topic and
posture. The target is the ordinary serialized `<SPK_n>` transcript, so these
rows train the same objective, through the same dataset and collator, as the
pool's windows: verification is labelling on a harder window.

Rows have the pool row shape (parts JSON at mix offsets, target, n_speakers,
duration_s, tail_s) and are built at load time from the utterance index, like
context rows: no pool rebuild, not part of the pool signature.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass

from scripts.speaker_asr.data import format_target


@dataclass(frozen=True)
class JunctionConfig:
    """Junction-window layout (configs/speaker_asr `junction:`). Seconds."""

    # Share of the training mix that is junction windows (the rest: pool windows).
    ratio: float = 0.35
    # Clips per window, drawn uniformly in this range.
    clips: tuple[int, int] = (2, 4)
    # A clip is one whole utterance whose length is in this range.
    clip_s: tuple[float, float] = (2.0, 6.0)
    # Silence between clips.
    gap_s: tuple[float, float] = (0.3, 1.0)
    # Every pair of clips in a window starts at least this far apart in the meeting.
    min_separation_s: float = 60.0
    # Chance that the next clip is a speaker already in the window.
    same_prob: float = 0.5
    # Junction windows decoded at each eval (eval_junction/*); 0 = none.
    eval_rows: int = 200


def junction_count(n_pool: int, ratio: float) -> int:
    """Junction rows to add to `n_pool` pool rows so they make up `ratio` of the mix."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"junction ratio must be in [0, 1), got {ratio}")
    return round(n_pool * ratio / (1.0 - ratio))


def _eligible(utts: list[dict], texts: dict[str, str], cfg: JunctionConfig) -> dict:
    """group -> speaker -> utterances usable as clips (length in range, has words)."""
    lo, hi = cfg.clip_s
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for u in utts:
        if lo <= u["end"] - u["start"] <= hi and texts.get(u["id"], "").strip():
            out[u["group"]][u["speaker"]].append(u)
    return {g: dict(s) for g, s in out.items() if len(s) >= 2}


def _far(u: dict, chosen: list[dict], min_sep: float) -> bool:
    return all(abs(u["start"] - c["start"]) >= min_sep for c in chosen)


def _sample_window(rng, speakers: dict, cfg: JunctionConfig, max_speakers: int) -> list[dict]:
    """Clip utterances for one window, in playback order (may be shorter than drawn k)."""
    k = rng.randint(*cfg.clips)
    first_speaker = rng.choice(sorted(speakers))
    chosen = [rng.choice(speakers[first_speaker])]
    in_window = [first_speaker]
    while len(chosen) < k:
        same = rng.random() < cfg.same_prob
        options = []
        for want_same in (same, not same):  # fall back to the other kind if infeasible
            pool = (
                in_window
                if want_same
                else (
                    [s for s in sorted(speakers) if s not in in_window]
                    if len(in_window) < max_speakers
                    else []
                )
            )
            options = [
                (s, u)
                for s in pool
                for u in speakers[s]
                if u not in chosen and _far(u, chosen, cfg.min_separation_s)
            ]
            if options:
                break
        if not options:
            break
        speaker, u = rng.choice(options)
        chosen.append(u)
        if speaker not in in_window:
            in_window.append(speaker)
    return chosen


def build_junction_rows(
    utts: list[dict],
    texts: dict[str, str],
    cfg: JunctionConfig,
    n_rows: int,
    max_speakers: int = 4,
    max_audio_s: float = 30.0,
    seed: int = 0,
) -> list[dict]:
    """`n_rows` junction windows over `utts` (row, id, group, speaker, start, end).

    Meetings are drawn in proportion to their usable clips. Each row records
    `n_same`/`n_new`: how many junctions continued a speaker already in the
    window vs opened a new one. Deterministic in `seed`.
    """
    rng = random.Random(seed)
    groups = _eligible(utts, texts, cfg)
    if not groups or n_rows <= 0:
        return []
    names = sorted(groups)
    weights = [sum(len(v) for v in groups[g].values()) for g in names]
    rows, attempts = [], 0
    while len(rows) < n_rows and attempts < 20 * n_rows:
        attempts += 1
        group = rng.choices(names, weights)[0]
        clips = _sample_window(rng, groups[group], cfg, max_speakers)
        if len(clips) < 2:
            continue
        parts, cursor, seen, n_same = [], 0.0, set(), 0
        for i, u in enumerate(clips):
            if i:
                cursor += rng.uniform(*cfg.gap_s)
                n_same += u["speaker"] in seen
            dur = u["end"] - u["start"]
            parts.append(
                {
                    "row": int(u["row"]),
                    "id": u["id"],
                    "speaker": u["speaker"],
                    "offset_s": round(cursor, 3),
                    "dur_s": round(dur, 3),
                }
            )
            seen.add(u["speaker"])
            cursor += dur
        if cursor > max_audio_s:
            continue
        target, n_speakers = format_target(parts, texts)
        rows.append(
            {
                "key": f"junction:{group}:{len(rows)}",
                "group": group,
                "parts": json.dumps(parts),
                "tail_s": 0.0,
                "duration_s": round(cursor, 3),
                "target": target,
                "n_speakers": n_speakers,
                "n_parts": len(parts),
                "n_same": n_same,
                "n_new": len(parts) - 1 - n_same,
            }
        )
    return rows
