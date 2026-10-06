"""Tests for clustered long-form decoding: short chunks + embeddings + global clustering."""

from __future__ import annotations

import numpy as np

from scripts.speaker_asr.clustered import (
    ClusterConfig,
    build_units,
    chunk_turns,
    collapse_loops,
    keep_chunks_distinct,
    link_units,
    timed_words,
    transcribe_clustered,
)
from scripts.speaker_asr.metrics import cp_errors, parse_turns, serialize_turns

SR = 16000


def test_collapse_loops_caps_runs_and_keeps_speaker_tokens():
    text = "<SPK_1>yeah yeah yeah yeah yeah ok<SPK_2>no"
    assert collapse_loops(text, 3) == "<SPK_1>yeah yeah yeah ok<SPK_2>no"


def test_timed_words_offsets_times_and_flags_unaligned_words():
    text = "<SPK_1>hello 25 world"
    aligned = [
        {"word": "hello", "start": 0.1, "end": 0.4},
        {"word": "world", "start": 0.6, "end": 0.9},
    ]
    words = timed_words(text, aligned, offset_s=10.0)
    assert [w[0] for w in words] == ["hello", "25", "world"]
    assert words[0][2:] == (10.1, 10.4, True)
    assert words[1][2:] == (10.1, 10.4, False)  # unaligned: borrows the previous time, flagged
    assert words[2][4]


def test_chunk_turns_are_runs_of_one_label():
    words = [("a", 1, 0, 1, True), ("b", 1, 1, 2, True), ("c", 2, 2, 3, True), ("d", 1, 3, 4, True)]
    turns = chunk_turns(words, chunk=7)
    assert [(t["label"], t["text"]) for t in turns] == [(1, "a b"), (2, "c"), (1, "d")]
    assert {t["chunk"] for t in turns} == {7}


def _unit_vec(k, dim=8, noise=0.0, rng=None):
    v = np.zeros(dim)
    v[k] = 1.0
    if noise:
        v = v + noise * rng.normal(size=dim)
    return v / np.linalg.norm(v)


def test_build_units_splits_a_label_whose_long_turns_sound_like_two_people():
    a, b = _unit_vec(0), _unit_vec(1)
    turns = [
        {
            "chunk": 0,
            "label": 1,
            "dur": 3.0,
            "emb": a,
            "speech": np.ones(10),
            "words": [("x", 1, 0.0, 3.0, True)],
        },
        {
            "chunk": 0,
            "label": 1,
            "dur": 3.0,
            "emb": b,
            "speech": np.ones(10),
            "words": [("y", 1, 4.0, 7.0, True)],
        },
        {
            "chunk": 0,
            "label": 1,
            "dur": 0.5,
            "emb": None,
            "speech": None,
            "words": [("z", 1, 7.1, 7.5, True)],
        },
    ]
    units = build_units(turns, ClusterConfig(split_tau=0.3, split_min_s=2.0), embed=lambda x: a)
    assert len(units) == 2
    assert sorted(len(u["turns"]) for u in units) == [
        1,
        2,
    ]  # the short turn joins its nearest long turn


def test_build_units_embeds_an_unsplit_unit_from_all_its_speech():
    seen = []
    turns = [{"chunk": 0, "label": 1, "dur": 1.0, "emb": _unit_vec(0), "speech": np.full(100, 0.5),
              "words": [("x", 1, 0.0, 1.0, True)]} for _ in range(2)]  # fmt: skip
    build_units(turns, ClusterConfig(), embed=lambda x: (seen.append(len(x)), _unit_vec(0))[1])
    assert seen == [200]  # both turns' speech, concatenated


def test_keep_chunks_distinct_separates_two_labels_of_one_chunk():
    units = [
        {"chunk": 0, "label": 1},
        {"chunk": 0, "label": 2},
        {"chunk": 1, "label": 1},
        {"chunk": 1, "label": 2},
    ]
    a = np.array(
        [[1, 0, 0.9, 0.1], [0, 1, 0.8, 0.2], [0.9, 0.8, 1, 0], [0.1, 0.2, 0, 1]], dtype=float
    )
    labels = keep_chunks_distinct(units, a, np.array([0, 0, 0, 1]))
    assert labels[0] != labels[1]


def test_link_units_reassigns_a_stray_cluster_instead_of_merging_two_speakers():
    rng = np.random.default_rng(0)
    units, words = [], []
    for chunk in range(40):  # 4 real speakers, ~10 units each
        k = chunk % 4
        units.append({"chunk": chunk, "label": 1, "emb": _unit_vec(k, noise=0.05, rng=rng)})
        words.append(30.0)
    units.append(
        {"chunk": 40, "label": 1, "emb": _unit_vec(6, noise=0.05, rng=rng)}
    )  # one stray fragment
    words.append(2.0)
    labels = link_units(units, np.array(words), ClusterConfig())
    real = [labels[i] for i in range(40)]
    assert len(set(real)) == 4  # the four real speakers stay apart
    assert all(real[i] == real[i % 4] for i in range(40))
    assert labels[40] in set(real)  # the stray unit joins a real speaker


def _synthetic_meeting(rng):
    """4 speakers taking turns; a speaker's audio is a constant level only they use."""
    order = [0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 3, 2, 0, 3, 1, 2] * 2
    pieces, turns = [], []
    for i, spk in enumerate(order):
        n = int(rng.uniform(2.0, 4.0) * SR)
        pieces += [np.full(n, 0.1 * (spk + 1), np.float32), np.zeros(int(0.6 * SR), np.float32)]
        turns.append((spk, f"w{i}a w{i}b w{i}c"))
    return np.concatenate(pieces), turns


def _segments(clip):
    """(start_sample, end_sample, level) of the non-silent runs in a clip."""
    on = np.abs(clip) > 1e-3
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
    return [(s, e, round(float(clip[s]), 2)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def test_transcribe_clustered_recovers_speakers_across_chunks():
    rng = np.random.default_rng(0)
    audio, ref_turns = _synthetic_meeting(rng)
    # word texts per audio segment, in order, so the fake decoder can read them back
    seg_words = {}
    for (s, _e, lvl), (_spk, text) in zip(_segments(audio), ref_turns, strict=True):
        seg_words[(s, round(lvl, 2))] = text

    def find_text(clip_start, s, lvl):
        for (gs, glvl), text in seg_words.items():
            if glvl == lvl and gs <= clip_start + s + SR // 2 and clip_start + s <= gs + 8 * SR:
                return text
        return "x"

    starts = {}

    def decode(clip):
        # labels by first appearance of each level within the chunk (what the model does)
        start = starts["next"]
        labels, out = {}, []
        for s, _e, lvl in _segments(clip):
            labels.setdefault(lvl, len(labels) + 1)
            out.append(f"<SPK_{labels[lvl]}>" + find_text(start, s, lvl))
        return "".join(out)

    def align(clip, text):
        words, segs = text.split(), _segments(clip)
        per = max(1, len(words) // max(1, len(segs)))
        out = []
        for k, w in enumerate(words):
            s, e, _ = segs[min(k // per, len(segs) - 1)]
            out.append({"word": w, "start": s / SR, "end": e / SR})
        return out

    def embed(clip):
        if clip is None or len(clip) < SR // 4:
            return None
        return _unit_vec(round(float(np.median(clip)) * 10) % 8, noise=0.02, rng=rng)

    # chunk_bounds is deterministic: precompute chunk starts for the fake decoder
    from scripts.speaker_asr.longform import chunk_bounds

    cfg = ClusterConfig()
    queue = [s for s, _ in chunk_bounds(audio, cfg.chunk_max_s, cfg.chunk_min_s)]

    def decode_tracked(clip):
        starts["next"] = queue.pop(0)
        return decode(clip)

    result = transcribe_clustered(
        None, None, audio, cfg, align=align, embed=embed, decode=decode_tracked
    )
    ref = serialize_turns(ref_turns)
    errors, n = cp_errors(ref, result.text)
    assert len({t.speaker for t in result.turns}) == 4
    assert errors / n < 0.1
    assert len(parse_turns(result.text)) >= 16
