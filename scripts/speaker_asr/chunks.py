"""Short real-timeline chunks: train the labeller on exactly what long-form inference feeds it.

Clustered long-form decoding (scripts/speaker_asr/clustered.py) cuts a recording into
3-8 s chunks at the quietest frame (longform.chunk_bounds) and asks the model to label
speakers inside each chunk. No earlier recipe trained on that input: pool windows
squeeze silences to <= 1.5 s and run 8-30 s; context rows add prefixes. Rows here:

1. Each meeting is mixed at its REAL timeline (every headset segment at its time).
2. It is cut with chunk_bounds over a range drawn per meeting around 3-8 s, so the
   cuts fall where inference would put them -- often mid-utterance.
3. A chunk's audio is every utterance overlapping it, trimmed to the chunk (parts carry
   clip_start_s / clip_end_s), i.e. exactly the meeting audio of that span.
4. Its target needs word times: every utterance's transcript is force-aligned on its
   clean headset clip (cached). An utterance contributes the words at least
   `edge_keep` inside the chunk; utterances are ordered by start (first-in first-out,
   as data.format_target), so the target format is unchanged: `<SPK_1>...<SPK_2>...`.

Rows have the pool row shape, so SpeakerASRDataset/SpeakerASRCollator take them as is.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.speaker_asr.metrics import serialize_turns

SAMPLE_RATE = 16000
# Shortest clip worth force-aligning (0.1 s); wav2vec2's first conv needs more than its kernel.
MIN_ALIGN_SAMPLES = SAMPLE_RATE // 10


@dataclass(frozen=True)
class ShortChunkConfig:
    """Short real-chunk rows (configs/speaker_asr `short_chunks:`). Seconds."""

    # chunk_bounds(min_s, max_s) is drawn per meeting from these ranges.
    min_s: tuple[float, float] = (2.5, 3.5)
    max_s: tuple[float, float] = (6.5, 9.0)
    # An edge word is kept when at least this share of it lies inside the chunk.
    edge_keep: float = 0.5
    # Chunks whose kept words are aligned less than this share are dropped.
    min_aligned: float = 0.8
    # Share of each training mix taken from the recipe's existing rows (rehearsal).
    rehearsal: float = 0.15
    # Loss weight on <SPK_n> targets (boundaries are what linking depends on).
    speaker_loss_weight: float = 3.0
    # Sampling weight of chunks with >= 2 speakers or >= 3 turns (1 = uniform).
    busy_oversample: float = 2.0
    # Chunks decoded at each eval (eval_short/*); 0 = none.
    eval_rows: int = 400
    # Share of training chunks with random gain (+-6 dB) and noise (SNR 10-30 dB).
    augment_prob: float = 0.3
    # Keep chunks with no words, with an empty target (the model learns no speech -> no output).
    keep_empty: bool = True
    # Segments shorter than this take AMI's HUMAN transcript, restyled: transcribing a
    # 0.3 s "mm-hmm" clip on its own, the base model writes "Yeah." -- the first short
    # run learned that and inserted ~30 "yeah"s per meeting. 0 = self-transcripts only.
    human_text_below_s: float = 1.0


def human_style(text: str) -> str:
    """AMI's uppercase human transcript -> the targets' style: 'MM-HMM' -> 'Mm-hmm.'."""
    t = " ".join(text.lower().split())
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".?!" else t + "."


def target_texts(meta, self_texts: dict[str, str], below_s: float) -> dict[str, str]:
    """utterance id -> target text: self-transcript, or the restyled human one when shorter than below_s."""
    texts = dict(self_texts)
    if below_s > 0:
        dur = (meta["end"] - meta["start"]).to_numpy()
        for uid, human, d in zip(meta["id"], meta["text"].fillna(""), dur, strict=True):
            if d < below_s and human.strip():
                texts[uid] = human_style(human)
    return texts


def timed_words(text: str, aligned: list[dict]) -> list[tuple]:
    """[(word, start, end, aligned)] for every word of `text`, unaligned ones interpolated.

    `aligned` is ForcedAligner.align output (it skips words with no alignable
    characters); an unaligned word sits between its neighbours' times.
    """
    words = text.split()
    times: list[tuple | None] = [None] * len(words)
    j = 0
    for item in aligned:
        while j < len(words) and words[j] != item["word"]:
            j += 1
        if j == len(words):
            break
        times[j] = (float(item["start"]), float(item["end"]))
        j += 1
    out, last_end = [], 0.0
    for k, (w, t) in enumerate(zip(words, times, strict=True)):
        if t is None:
            nxt = next(
                (times[i][0] for i in range(k + 1, len(words)) if times[i] is not None), last_end
            )
            t_s = (last_end, max(last_end, nxt))
            out.append((w, t_s[0], t_s[1], False))
        else:
            out.append((w, t[0], t[1], True))
            last_end = t[1]
    return out


def align_utterances(store, texts: dict[str, str], ids: list[str], cache: Path, align=None) -> dict:
    """utterance id -> [(word, start, end, aligned)] (seconds from the utterance start), cached as JSONL."""
    known: dict = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a line cut short by a crash: redo that utterance
                continue
            known[rec["id"]] = [tuple(w) for w in rec["words"]]
    # an utterance whose text changed since it was cached (e.g. now a human transcript) is redone
    known = {i: w for i, w in known.items() if [x[0] for x in w] == texts.get(i, "").split()}
    todo = [i for i in ids if i not in known and texts.get(i, "").strip()]
    if todo:
        import logging

        log = logging.getLogger(__name__)
        if align is None:
            from tiny_audio.alignment import ForcedAligner

            align = ForcedAligner.align
        index = dict(zip(store.meta["id"], store.meta["row"], strict=True))
        log.info("aligning %d utterances (cache %s, %d already done)", len(todo), cache, len(known))
        failed = 0
        with cache.open("a") as fh:
            for n, uid in enumerate(todo, 1):
                clip = store.get(index[uid])
                aligned = []
                # AMI has empty / few-sample segments: too short for wav2vec2's first conv.
                if len(clip) >= MIN_ALIGN_SAMPLES:
                    try:
                        aligned = align(clip, texts[uid])
                    except Exception as e:  # one bad segment must not kill an hour-long pass
                        failed += 1
                        log.warning("alignment failed for %s (%d samples): %s", uid, len(clip), e)
                words = timed_words(
                    texts[uid], aligned
                )  # unaligned words: chunks using them get dropped
                known[uid] = words
                fh.write(json.dumps({"id": uid, "words": words}) + "\n")
                if n % 2000 == 0:
                    fh.flush()
                    log.info("aligned %d / %d utterances (%d failed)", n, len(todo), failed)
    return known


def plan_meeting_chunks(
    parts: list[dict],
    words: dict,
    audio: np.ndarray,
    cfg: ShortChunkConfig,
    group: str,
    rng: random.Random,
) -> list[dict]:
    """Pool-shaped rows for one meeting.

    `parts` are data.meeting_parts entries (row, id, speaker, offset_s, dur_s at real
    time); `words` maps utterance id -> timed words; `audio` is the meeting mix.
    """
    from scripts.speaker_asr.longform import chunk_bounds

    min_s, max_s = rng.uniform(*cfg.min_s), rng.uniform(*cfg.max_s)
    rows = []
    for k, (s, e) in enumerate(chunk_bounds(audio, max_s, min_s)):
        cs, ce = s / SAMPLE_RATE, e / SAMPLE_RATE
        overlapping = sorted(
            (p for p in parts if p["offset_s"] < ce and p["offset_s"] + p["dur_s"] > cs),
            key=lambda p: (p["offset_s"], p["dur_s"]),
        )
        mix_parts, turns, kept, aligned = [], [], 0, 0
        for p in overlapping:
            clip_start = max(0.0, cs - p["offset_s"])
            clip_end = min(p["dur_s"], ce - p["offset_s"])
            mix_parts.append({
                "row": int(p["row"]), "id": p["id"], "speaker": p["speaker"],
                "offset_s": round(max(0.0, p["offset_s"] - cs), 3),
                "dur_s": round(clip_end - clip_start, 3),
                "clip_start_s": round(clip_start, 3), "clip_end_s": round(clip_end, 3),
            })  # fmt: skip
            inside = []
            for w, ws, we, ok in words.get(p["id"], []):
                a, b = p["offset_s"] + ws, p["offset_s"] + we
                frac = 1.0 if b <= a else max(0.0, min(b, ce) - max(a, cs)) / (b - a)
                if frac >= cfg.edge_keep and (b > a or cs <= a < ce):
                    inside.append(w)
                    kept += 1
                    aligned += ok
            if inside:
                turns.append((p["speaker"], " ".join(inside)))
        if kept and aligned < cfg.min_aligned * kept:
            continue  # word times unreliable: the target could be wrong at the edges
        if not turns and not cfg.keep_empty:
            continue
        # No words at all (breath, noise, bleed, silence) -> an EMPTY target: about a quarter
        # of real 3-8 s inference chunks look like this, and a model never shown one
        # hallucinates a word into each ("yeah" in the first short run).
        target = serialize_turns(turns)
        last_end = max((p["offset_s"] + p["dur_s"] for p in mix_parts), default=0.0)
        rows.append(
            {
                "key": f"short:{group}:{k}",
                "group": group,
                "parts": json.dumps(mix_parts),
                # pad the chunk's trailing silence: inference audio includes it
                "tail_s": round(max(0.0, (ce - cs) - last_end), 3),
                "duration_s": round(ce - cs, 3),
                "target": target,
                "n_speakers": len({spk for spk, _ in turns}),
                "n_parts": len(mix_parts),
                "n_turns": target.count("<SPK_"),
                "short": True,
            }
        )
    return rows


def build_short_rows(
    cfg, split: str, store, texts: dict[str, str], scfg: ShortChunkConfig, seed: int = 0
) -> list[dict]:
    """Short real-chunk rows for every meeting of a split (alignments cached in the pool dir).

    `texts` are the self-transcripts; segments shorter than scfg.human_text_below_s take
    the human transcript instead (target_texts).
    """
    from hydra.utils import to_absolute_path

    from scripts.speaker_asr.data import meeting_parts, mix_window

    pool_dir = Path(to_absolute_path(cfg.data.pool_dir))
    texts = target_texts(store.meta, texts, scfg.human_text_below_s)
    words = align_utterances(
        store, texts, list(store.meta["id"]), pool_dir / f"words-{split}.jsonl"
    )
    rng = random.Random(seed)
    rows = []
    for group in sorted(store.meta["group"].unique()):
        parts = meeting_parts(store.meta, group)
        rows += plan_meeting_chunks(parts, words, mix_window(parts, store), scfg, group, rng)
    return rows


def busy_weights(rows: list[dict], factor: float) -> list[float]:
    """Sampling weight per row: `factor` for chunks with >= 2 speakers or >= 3 turns, else 1."""
    return [
        factor if (r.get("n_speakers", 1) >= 2 or r.get("n_turns", 1) >= 3) else 1.0 for r in rows
    ]


def augment(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random gain (+-6 dB) and white noise at 10-30 dB SNR."""
    out = audio * (10 ** (rng.uniform(-6, 6) / 20))
    power = float(np.mean(out**2)) + 1e-10
    noise = rng.normal(0, np.sqrt(power / (10 ** (rng.uniform(10, 30) / 10))), size=out.shape)
    return np.clip(out + noise, -1.0, 1.0).astype(np.float32)
