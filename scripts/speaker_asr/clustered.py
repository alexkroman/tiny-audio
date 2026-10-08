"""Long-form speaker-attributed transcription: short chunks, speaker embeddings, global clustering.

    result = transcribe_clustered(model, processor, audio_16k)
    result.text     # '<SPK_1>...<SPK_2>...' over the whole recording
    result.turns    # [Turn(speaker, text, start, end), ...]

1. Cut the recording into short chunks (3-8 s) at quiet points (longform.chunk_bounds).
   Short chunks hold few speakers, so the speaker-ASR model rarely merges two voices.
2. Decode each chunk independently (the empty-memory `<CONTINUE>` prompt on a
   context-trained checkpoint) and drop runaway repetitions.
3. Force-align the words; every run of one label in one chunk is a turn.
4. Embed each turn and each chunk-local speaker with ECAPA-TDNN (all of its speech).
5. Units = chunk-local speakers, split when their long turns clearly sound like
   different people; units with almost no speech join the previous speaker.
6. Link units across the recording: tiny_audio.diarization.SpectralCluster on cosine
   affinities (speaker count by eigengap) over units with >= `cluster_min_s` of speech,
   labels of one chunk kept distinct, then clusters holding < `tiny_share` of the words
   are set aside and the rest re-clustered (a stray cluster would otherwise force two
   real speakers to share one). Short and set-aside units are then placed on the
   nearest speaker their chunk does not already use.

Compared with context-prefix linking (longform.transcribe_long) on all 16
ami-speakers-long test meetings: cpWER 23.28 vs 104 for prefix linking (WER 20.48;
U3-Pro 23.58). On 6 untouched AMI IB meetings: 24.14 with the `short` labeller,
24.24 with `context` (U3-Pro 27.55).
Model, aligner and embedder are injectable, so the logic is testable without them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from scripts.eval.speaker_metrics import CONTEXT_END, parse_turns
from scripts.speaker_asr.longform import LongFormResult, Turn, chunk_bounds

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ClusterConfig:
    """Knobs of transcribe_clustered (validated on AMI; see module docstring)."""

    chunk_min_s: float = 3.0
    chunk_max_s: float = 8.0
    # A word repeated more than this many times in a row is a decoding loop.
    loop_max_run: int = 3
    # Units with less speech than this get no embedding and join the previous speaker.
    min_unit_s: float = 0.25
    # Split a unit when its turns of >= split_min_s sound different ((1 - cos) / 2 > split_tau).
    split_tau: float = 0.3
    split_min_s: float = 2.0
    # Spectral clustering: speaker-count search range and affinity pruning.
    min_speakers: int = 3
    max_speakers: int = 5
    pval: float = 0.06
    # Clusters holding less than this share of the words are re-assigned after re-clustering.
    tiny_share: float = 0.03
    # Only units with at least this much speech choose the speaker count and centroids;
    # shorter ones are placed afterwards. Sub-second units (a "Yes.", an "Okay.") carry
    # too little voice to cluster on and tip the eigengap: on 6 untouched AMI IB meetings
    # this took cpWER 25.66 -> 24.14 for the short labeller (24.53 -> 24.24 for context);
    # 0.75-1.5 s is a plateau.
    cluster_min_s: float = 1.0
    max_new_tokens: int = 320


# ------------------------------------------------------------------ decoding helpers


def collapse_loops(text: str, max_run: int = 3) -> str:
    """Drop a word repeated more than `max_run` times in a row; speaker tokens kept."""
    out = []
    for label, chunk in parse_turns(text):
        kept, prev, run = [], None, 0
        for w in chunk.split():
            key = re.sub(r"\W", "", w.lower())
            run = run + 1 if key == prev else 1
            prev = key
            if run <= max_run:
                kept.append(w)
        out.append((label, " ".join(kept)))
    return "".join((f"<SPK_{label}>" if label else "") + t for label, t in out)


def timed_words(text: str, aligned: list[dict], offset_s: float) -> list[tuple]:
    """[(word, label, start, end, aligned)] for every decoded word, times on the recording.

    `aligned` is ForcedAligner.align output for the chunk's words in order (it skips
    words with no alignable characters); unaligned words borrow the previous word's time
    and are flagged, so they are transcribed but never cut as speech.
    """
    flat = [(w, label) for label, t in parse_turns(text) if t.strip() for w in t.split()]
    times: list[tuple | None] = [None] * len(flat)
    j = 0
    for item in aligned:
        while j < len(flat) and flat[j][0] != item["word"]:
            j += 1
        if j == len(flat):
            break
        times[j] = (item["start"], item["end"])
        j += 1
    out, last = [], (0.0, 0.0)
    for (w, label), found in zip(flat, times, strict=True):
        last = found or last
        out.append((w, label, offset_s + last[0], offset_s + last[1], found is not None))
    return out


def chunk_turns(words: list[tuple], chunk: int) -> list[dict]:
    """Runs of one label within one chunk's words -> turns."""
    turns, run = [], []
    for w in [*words, None]:
        if run and (w is None or w[1] != run[0][1]):
            turns.append(
                {
                    "chunk": chunk,
                    "label": run[0][1],
                    "words": run,
                    "text": " ".join(x[0] for x in run),
                }
            )
            run = []
        if w is not None:
            run.append(w)
    return turns


def speech(audio: np.ndarray, words: list[tuple]) -> np.ndarray | None:
    """The aligned words' audio, concatenated (unaligned words are skipped)."""
    pieces = [
        audio[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)]
        for _, _, a, b, ok in words
        if ok and b > a
    ]
    return np.concatenate(pieces) if pieces else None


# ------------------------------------------------------------------ units and linking


def build_units(turns: list[dict], cfg: ClusterConfig, embed: Callable) -> list[dict]:
    """Chunk-local speakers, split where their long turns clearly disagree.

    Each turn carries `dur`, `emb` (or None) and `speech` (its aligned audio or None);
    each unit is {chunk, label, turns: [turn indices], emb, dur}. An unsplit unit is
    embedded from ALL of its speech concatenated (more audio, steadier vector); a split
    part takes the duration-weighted mean of its long turns' vectors.
    """
    by_unit: dict[tuple, list[int]] = {}
    for i, t in enumerate(turns):
        by_unit.setdefault((t["chunk"], t["label"]), []).append(i)
    units = []
    for (chunk, label), idx in by_unit.items():
        dur = sum(turns[i]["dur"] for i in idx)
        long_ = [
            i for i in idx if turns[i]["emb"] is not None and turns[i]["dur"] >= cfg.split_min_s
        ]
        groups = [[i] for i in long_]
        while len(groups) > 1:  # average-linkage agglomeration of the long turns
            best, pair = None, None
            for a in range(len(groups)):
                for b in range(a + 1, len(groups)):
                    d = (1 - float(_mean_emb(turns, groups[a]) @ _mean_emb(turns, groups[b]))) / 2
                    if best is None or d < best:
                        best, pair = d, (a, b)
            if best > cfg.split_tau:
                break
            groups[pair[0]] += groups.pop(pair[1])
        if len(groups) <= 1:
            pieces = [turns[i]["speech"] for i in idx if turns[i]["speech"] is not None]
            emb = embed(np.concatenate(pieces)) if pieces else None
            if dur >= cfg.min_unit_s and emb is not None:
                units.append({"chunk": chunk, "label": label, "turns": idx, "emb": emb, "dur": dur})
            continue
        member = {i: g for g, grp in enumerate(groups) for i in grp}
        assigned = {g: list(grp) for g, grp in enumerate(groups)}
        for i in idx:  # short turns join the group of the nearest long turn in time
            if i not in member:
                near = min(long_, key=lambda j: abs(_mid(turns[j]) - _mid(turns[i])))
                assigned[member[near]].append(i)
        for g, grp in assigned.items():
            units.append({"chunk": chunk, "label": label, "turns": grp, "emb": _mean_emb(turns, groups[g]),
                          "dur": sum(turns[i]["dur"] for i in grp)})  # fmt: skip
    return units


def _mean_emb(turns: list[dict], idx: list[int]) -> np.ndarray | None:
    if not idx:
        return None
    w = np.array([turns[i]["dur"] for i in idx]) + 1e-6
    v = (np.stack([turns[i]["emb"] for i in idx]) * w[:, None]).sum(0)
    return v / (np.linalg.norm(v) + 1e-9)


def _mid(turn: dict) -> float:
    return (turn["words"][0][2] + turn["words"][-1][3]) / 2


def affinity(units: list[dict]) -> np.ndarray:
    """Cosine affinity in [0, 1]; 0 between different labels of one chunk (cannot-link)."""
    e = np.stack([u["emb"] for u in units])
    a = 1.0 - (1.0 - e @ e.T) / 2.0
    for i in range(len(units)):
        for j in range(len(units)):
            same_chunk = units[i]["chunk"] == units[j]["chunk"]
            if i != j and same_chunk and units[i]["label"] != units[j]["label"]:
                a[i, j] = 0.0
    return a


def spectral(a: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    """Repo SpectralCluster on a precomputed affinity (eigengap speaker count)."""
    from tiny_audio.diarization import SpectralCluster

    class _Precomputed(SpectralCluster):
        def get_sim_mat(self, embeddings):
            return np.array(embeddings, dtype=float)

    if len(a) < 6:
        return np.zeros(len(a), dtype=int)
    sc = _Precomputed(min_num_spks=cfg.min_speakers, max_num_spks=cfg.max_speakers, pval=cfg.pval)
    return np.asarray(sc(a.copy()))


def keep_chunks_distinct(units: list[dict], a: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Give the units of each chunk distinct clusters (Hungarian on mean affinity)."""
    from scipy.optimize import linear_sum_assignment

    labels = labels.copy()
    by_chunk: dict[int, list[int]] = {}
    for i, u in enumerate(units):
        by_chunk.setdefault(u["chunk"], []).append(i)
    for idx in by_chunk.values():
        if len(idx) < 2 or len({labels[i] for i in idx}) == len(idx):
            continue
        clusters = sorted(set(labels.tolist()))
        score = np.zeros((len(idx), len(clusters) + len(idx)))
        for x, i in enumerate(idx):
            for y, c in enumerate(clusters):
                members = [j for j in range(len(units)) if labels[j] == c and j not in idx]
                score[x, y] = float(np.mean(a[i, members])) if members else 0.0
            score[x, len(clusters) + x] = 0.05  # a fresh cluster only if nothing fits
        rows, cols = linear_sum_assignment(-score)
        fresh = labels.max() + 1
        for x, y in zip(rows, cols, strict=True):
            labels[idx[x]] = clusters[y] if y < len(clusters) else fresh + x
    return labels


def link_units(units: list[dict], words_per_unit: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    """Cluster label per unit.

    Units with >= cfg.cluster_min_s of speech are clustered (spectral + chunk constraint,
    minus tiny clusters, re-clustered); every other unit is then placed on a centroid,
    within each chunk on speakers no other unit of that chunk already holds.
    """
    if not units:
        return np.zeros(0, dtype=int)
    keep = [i for i, u in enumerate(units) if u["dur"] >= cfg.cluster_min_s]
    if len(keep) < 6:  # too little reliable speech to choose a speaker count from
        keep = list(range(len(units)))
    labels = np.full(len(units), -1)
    labels[keep] = _cluster([units[i] for i in keep], cfg)
    if cfg.tiny_share:
        total = words_per_unit[keep].sum() or 1.0
        tiny = {c for c in set(labels[keep].tolist())
                if words_per_unit[[i for i in keep if labels[i] == c]].sum() / total < cfg.tiny_share}  # fmt: skip
        if tiny and len(set(labels[keep].tolist())) > len(tiny):
            keep = [i for i in keep if labels[i] not in tiny]
            labels = np.full(len(units), -1)
            labels[keep] = _cluster([units[i] for i in keep], cfg)
    embs = np.stack([u["emb"] for u in units])
    centroids = {}
    for c in set(labels[keep].tolist()):
        m = [i for i in keep if labels[i] == c]
        v = (embs[m] * words_per_unit[m, None]).sum(0)
        centroids[c] = v / (np.linalg.norm(v) + 1e-9)
    ids = sorted(centroids)
    return place_units(units, labels, embs @ np.stack([centroids[c] for c in ids]).T, ids)


def _cluster(units: list[dict], cfg: ClusterConfig) -> np.ndarray:
    a = affinity(units)
    return keep_chunks_distinct(units, a, spectral(a, cfg))


def place_units(units: list[dict], labels: np.ndarray, sims: np.ndarray, ids: list) -> np.ndarray:
    """Give every unlabelled unit (-1) a speaker from `ids`, by similarity `sims` [unit, speaker].

    Within a chunk, unlabelled units take distinct speakers that the chunk's labelled
    units do not hold (Hungarian); when there are not enough of those, the best match.
    """
    from scipy.optimize import linear_sum_assignment

    labels = labels.copy()
    by_chunk: dict[int, list[int]] = {}
    for i, u in enumerate(units):
        by_chunk.setdefault(u["chunk"], []).append(i)
    for idx in by_chunk.values():
        free = [i for i in idx if labels[i] < 0]
        if not free:
            continue
        taken = {int(labels[i]) for i in idx if labels[i] >= 0}
        cols = [k for k, c in enumerate(ids) if c not in taken]
        if len(cols) < len(free):
            for i in free:
                labels[i] = ids[int(np.argmax(sims[i]))]
            continue
        rows, picked = linear_sum_assignment(-sims[np.ix_(free, cols)])
        for x, y in zip(rows, picked, strict=True):
            labels[free[x]] = ids[cols[y]]
    return labels


# ------------------------------------------------------------------ end to end


def transcribe_clustered(
    model,
    processor,
    audio: np.ndarray,
    cfg: ClusterConfig | None = None,
    align: Callable | None = None,
    embed: Callable | None = None,
    decode: Callable | None = None,
) -> LongFormResult:
    """Speaker-attributed transcript of 16 kHz mono `audio` of any length.

    `decode(chunk_audio) -> '<SPK_n>...'` defaults to speaker-ASR greedy decoding;
    `align(audio, text) -> [{word, start, end}]` to ForcedAligner.align;
    `embed(audio) -> unit vector or None` to ECAPA-TDNN (tiny_audio.diarization).
    """
    cfg = cfg or ClusterConfig()
    audio = np.asarray(audio, dtype=np.float32)
    decode = decode or _speaker_asr_decoder(model, processor, cfg)
    if align is None:
        from tiny_audio.alignment import ForcedAligner

        align = ForcedAligner.align
    embed = embed or ecapa_embedder()

    bounds = chunk_bounds(audio, cfg.chunk_max_s, cfg.chunk_min_s)
    turns: list[dict] = []
    for c, (s, e) in enumerate(bounds):
        text = collapse_loops(decode(audio[s:e]), cfg.loop_max_run)
        flat = " ".join(t for _, t in parse_turns(text) if t.strip())
        words = timed_words(text, align(audio[s:e], flat) if flat else [], s / SAMPLE_RATE)
        turns += chunk_turns(words, c)
    for t in turns:
        t["speech"] = speech(audio, t["words"])
        t["dur"] = len(t["speech"]) / SAMPLE_RATE if t["speech"] is not None else 0.0
        t["emb"] = embed(t["speech"])
    units = build_units(turns, cfg, embed)
    words = np.array([sum(len(turns[i]["words"]) for i in u["turns"]) for u in units], dtype=float)
    labels = link_units(units, words, cfg)
    owner = {i: int(lab) for u, lab in zip(units, labels.tolist(), strict=True) for i in u["turns"]}
    out, last = [], 0
    for i, t in enumerate(turns):
        last = owner.get(i, last)  # too little speech to embed: continue the previous speaker
        out.append(Turn(last, t["text"], t["words"][0][2], t["words"][-1][3]))
    return LongFormResult(_merge(out), [(s / SAMPLE_RATE, e / SAMPLE_RATE) for s, e in bounds])


def _merge(turns: list[Turn]) -> list[Turn]:
    out: list[Turn] = []
    for t in turns:
        if out and out[-1].speaker == t.speaker:
            out[-1] = Turn(t.speaker, f"{out[-1].text} {t.text}", out[-1].start, t.end)
        else:
            out.append(t)
    return out


def _speaker_asr_decoder(model, processor, cfg: ClusterConfig) -> Callable:
    from scripts.speaker_asr.model import n_speaker_tokens, transcribe_speakers

    n = n_speaker_tokens(processor)
    prefix = [CONTEXT_END] if CONTEXT_END in processor.tokenizer.get_vocab() else None

    def decode(chunk: np.ndarray) -> str:
        return transcribe_speakers(
            model, processor, [chunk], n, cfg.max_new_tokens, prefixes=prefix
        )[0]

    return decode


def ecapa_embedder(min_s: float = 0.25, pad_s: float = 1.5, max_s: float = 20.0) -> Callable:
    """Unit ECAPA-TDNN vector of a clip (reflect-padded to pad_s), or None if under min_s."""
    import torch

    from tiny_audio.diarization import SpeakerDiarizer

    model = SpeakerDiarizer._get_ecapa_model()

    def embed(clip: np.ndarray | None) -> np.ndarray | None:
        if clip is None or len(clip) < int(min_s * SAMPLE_RATE):
            return None
        clip = clip[: int(max_s * SAMPLE_RATE)]
        if len(clip) < int(pad_s * SAMPLE_RATE):
            clip = np.pad(clip, (0, int(pad_s * SAMPLE_RATE) - len(clip)), mode="reflect")
        with torch.no_grad():
            v = (
                model.encode_batch(torch.from_numpy(np.asarray(clip, np.float32)).unsqueeze(0))
                .reshape(-1)
                .cpu()
                .numpy()
            )
        return v / (np.linalg.norm(v) + 1e-9)

    return embed
