"""Score every system's first endpoint on the same turns, with one labeler.

    poetry run python -m scripts.turn_aware.score_endpoints \
        outputs/turn_aware_local/replay200_v3_onset/replay.csv -o outputs/turn_aware_local/scored

Reads the `ta turn-aware replay` CSV (our fires) and the dataset's `endpoints`
table (the services' fires) for the same turns, then attributes each first fire
to the PAUSE that caused it, measured from the audio:

    at turn end        fired after the caller's last speech frame
    complete pause     fired in (or reported late from) a pause at a chunk
                       boundary whose chunk ends . ? !  -- the dataset labels
                       this moment a correct fire (`chunk_cut`)
    mid-sentence pause same, at a chunk that stops mid-sentence: a real cutoff
    within-chunk pause a pause inside one TTS chunk: a real cutoff
    during speech      the caller was talking and no pause explains the fire
                       within the allowance below: a real cutoff
    never              no fire

The published table used a fixed 1.2 s window after each chunk END. That
mislabels a fire made 1.3 s into a 1.8 s pause (the caller still silent) as a
cutoff. Here a fire that lands while the caller is silent is always credited to
the pause it landed in, however late. Only a fire that lands after the caller
resumed needs a lateness allowance, and two are reported:

    strict    0: a fire that arrives while the caller is talking is a cutoff
              (it is one from the caller's point of view)
    generous  the system's own p90 reply latency: a late report of the pause
              the caller just left is credited to that pause

Latency is measured on at-turn-end fires, from the caller's last speech frame
(not the end of the file, which adds the TTS clip's own trailing silence).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.turn_aware.data import TurnAudioStore, ends_mid_sentence, load_split
from tiny_audio.turns.model import SAMPLE_RATE, SILENCE_RMS

FRAME_S = 0.02
# Shortest silence counted as a pause. AssemblyAI's `min_turn_silence` is
# 128 ms on `balanced`/`min_latency`, so anything shorter cannot trigger it.
MIN_PAUSE_S = 0.1
# A chunk END (a TTS clip boundary) belongs to a pause if it falls inside it,
# give or take this much: the clip's own tail silence is part of the pause.
BOUNDARY_TOL_S = 0.05
OURS = "turn-aware-qwen3-asr"
CUTOFFS = ("mid-sentence pause", "within-chunk pause", "during speech")


def speech_segments(audio: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) seconds of speech, split wherever >= MIN_PAUSE_S is silent."""
    n = int(FRAME_S * SAMPLE_RATE)
    frames = len(audio) // n
    rms = np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1))
    loud = np.flatnonzero(rms > SILENCE_RMS)
    if not len(loud):
        return []
    gap = round(MIN_PAUSE_S / FRAME_S)
    breaks = np.flatnonzero(np.diff(loud) > gap)
    starts = np.concatenate([[loud[0]], loud[breaks + 1]])
    ends = np.concatenate([loud[breaks], [loud[-1]]]) + 1
    return [(s * FRAME_S, e * FRAME_S) for s, e in zip(starts, ends, strict=True)]


def pause_kind(start: float, end: float, turn: pd.Series) -> tuple[str, str]:
    """What sort of pause (start, end) is, and the chunk text it follows."""
    for end_s, text in zip(turn.chunk_ends_s[:-1], turn.chunks[:-1], strict=False):
        if start - BOUNDARY_TOL_S <= end_s <= end + BOUNDARY_TOL_S:
            kind = "mid-sentence pause" if ends_mid_sentence(text) else "complete pause"
            return kind, text
    return "within-chunk pause", ""


def classify(fire: float, segs: list[tuple[float, float]], turn: pd.Series, allow_s: float):
    """(category, pause_start, pause_end, chunk_text) for one first fire."""
    if fire is None or np.isnan(fire):
        return "never", np.nan, np.nan, ""
    if fire >= segs[-1][1]:
        return "at turn end", np.nan, np.nan, ""
    # The latest pause that began before the fire: pause i runs from the end
    # of segment i to the start of segment i + 1.
    i = int(np.searchsorted([e for _, e in segs], fire, side="right")) - 1
    if i < 0:
        return "during speech", np.nan, np.nan, ""
    start, end = segs[i][1], segs[i + 1][0]
    if fire >= end + allow_s:  # the caller resumed and the fire is not a late report
        return "during speech", start, end, ""
    kind, text = pause_kind(start, end, turn)
    return kind, start, end, text


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main(
    replay_csv: Path,
    output_dir: Path = typer.Option(..., "--output-dir", "-o"),
    fire_col: str = typer.Option("fire_s_tau0", help="Which replay column is ours"),
):
    from scipy.stats import binomtest

    ours = pd.read_csv(replay_csv)
    keys = sorted(ours.turn_key)
    turns = load_split("turns", "test").remove_columns(["audio"]).to_pandas()
    turns = turns.set_index("turn_key").loc[keys]
    endpoints = load_split("endpoints", "test").to_pandas()
    endpoints = endpoints[endpoints.turn_key.isin(keys)]
    fires = {OURS: dict(zip(ours.turn_key, ours[fire_col], strict=True))}
    for arm, grp in endpoints.groupby("arm"):
        fires[arm] = dict(zip(grp.turn_key, grp.first_endpoint_s.astype(float), strict=True))

    store = TurnAudioStore("test")
    segs = {k: speech_segments(store.get(k)) for k in keys}
    speech_end = {k: segs[k][-1][1] for k in keys}

    rows, summary = [], {}
    for system, by_key in fires.items():
        fire = np.array([by_key[k] for k in keys], dtype=float)
        end = np.array([speech_end[k] for k in keys])
        on_time = fire >= end
        latency = fire[on_time] - end[on_time]
        p90 = float(np.percentile(latency, 90))
        cats = {}
        for scoring, allow in (("strict", 0.0), ("generous", p90)):
            cats[scoring] = [classify(by_key[k], segs[k], turns.loc[k], allow) for k in keys]
        for j, k in enumerate(keys):
            s, g = cats["strict"][j], cats["generous"][j]
            rows.append(
                {
                    "turn_key": k,
                    "system": system,
                    "fire_s": fire[j],
                    "speech_end_s": speech_end[k],
                    "audio_duration_s": float(turns.loc[k].audio_duration_s),
                    "category_strict": s[0],
                    "category_generous": g[0],
                    "pause_start_s": g[1],
                    "pause_end_s": g[2],
                    "chunk_before_pause": g[3],
                }
            )
        n = len(keys)
        summary[system] = {
            "n": n,
            "latency_from_speech_end_s": {
                q: float(np.percentile(latency, int(q[1:]))) for q in ("p50", "p90", "p99")
            },
            "latency_from_file_end_p50_s": float(
                np.median(fire[on_time] - turns.audio_duration_s.to_numpy()[on_time])
            ),
            "generous_allowance_s": p90,
            **{
                scoring: pd.Series([c[0] for c in cs]).value_counts().to_dict()
                for scoring, cs in cats.items()
            },
        }
        for scoring in ("strict", "generous"):
            k_cut = sum(summary[system][scoring].get(c, 0) for c in CUTOFFS)
            lo, hi = wilson(k_cut, n)
            summary[system][scoring]["real_cutoffs"] = k_cut
            summary[system][scoring]["real_cutoffs_ci95"] = [lo, hi]

    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "per_turn_fires.csv", index=False)

    cut = {
        (sc, s): set(table[(table.system == s) & table[f"category_{sc}"].isin(CUTOFFS)].turn_key)
        for sc in ("strict", "generous")
        for s in fires
    }
    for scoring in ("strict", "generous"):
        for system in fires:
            if system == OURS:
                continue
            a, b = cut[(scoring, OURS)], cut[(scoring, system)]
            only_ours, only_them = len(a - b), len(b - a)
            p = binomtest(only_ours, only_ours + only_them).pvalue if only_ours + only_them else 1.0
            summary[system][scoring]["vs_ours"] = {
                "only_ours": only_ours,
                "only_service": only_them,
                "both": len(a & b),
                "sign_test_p": p,
            }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    order = ["at turn end", "complete pause", *CUTOFFS, "never"]
    for scoring in ("strict", "generous"):
        print(f"\n{scoring}")
        counts = table.groupby("system")[f"category_{scoring}"].value_counts().unstack(fill_value=0)
        counts = counts.reindex(columns=order, fill_value=0)
        counts["real cutoffs"] = counts[list(CUTOFFS)].sum(axis=1)
        counts["total"] = counts[order].sum(axis=1)
        print(counts.to_string())
    print("\nlatency from end of speech (at-turn-end fires)")
    for system, s in summary.items():
        lat = s["latency_from_speech_end_s"]
        print(f"  {system:30s} p50 {lat['p50']:.2f}  p90 {lat['p90']:.2f}  p99 {lat['p99']:.2f}")


if __name__ == "__main__":
    typer.run(main)
