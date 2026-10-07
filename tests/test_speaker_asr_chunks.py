"""Tests for short real-timeline chunk rows (the labeller retrain for clustered long-form decoding)."""

from __future__ import annotations

import json
import random
from dataclasses import fields

import numpy as np
import pytest
import torch

from scripts.speaker_asr.chunks import (
    ShortChunkConfig,
    augment,
    busy_weights,
    plan_meeting_chunks,
    plan_recording_chunks,
    timed_words,
)
from scripts.speaker_asr.metrics import parse_turns

SR = 16000


def test_timed_words_interpolates_unaligned_words_between_neighbours():
    aligned = [
        {"word": "hello", "start": 0.1, "end": 0.4},
        {"word": "world", "start": 0.9, "end": 1.2},
    ]
    words = timed_words("hello 25 world", aligned)
    assert words[0] == ("hello", 0.1, 0.4, True)
    assert words[1] == ("25", 0.4, 0.9, False)
    assert words[2] == ("world", 0.9, 1.2, True)


def _meeting():
    """Two speakers alternating 2 s utterances with 1 s silences; words spaced 0.5 s apart."""
    parts, words, pieces = [], {}, []
    t = 0.0
    for i in range(8):
        spk = "A" if i % 2 == 0 else "B"
        uid = f"u{i}"
        parts.append({"row": i, "id": uid, "speaker": spk, "offset_s": t, "dur_s": 2.0})
        words[uid] = [(f"{uid}w{k}", 0.5 * k, 0.5 * k + 0.4, True) for k in range(4)]
        pieces += [np.full(2 * SR, 0.3, np.float32), np.zeros(SR, np.float32)]
        t += 3.0
    return parts, words, np.concatenate(pieces)


def test_chunks_follow_quiet_points_and_targets_keep_utterance_order():
    parts, words, audio = _meeting()
    cfg = ShortChunkConfig(min_s=(3.0, 3.0), max_s=(7.0, 7.0))
    rows = plan_meeting_chunks(parts, words, audio, cfg, "m1", random.Random(0))
    assert rows
    for row in rows:
        assert 0 < row["duration_s"] <= 7.0 + 1e-6
        turns = parse_turns(row["target"])
        assert turns[0][0] == 1  # numbered from <SPK_1>
        assert row["n_speakers"] == len({lab for lab, _ in turns})
    all_words = " ".join(t for r in rows for _, t in parse_turns(r["target"])).split()
    assert sorted(all_words) == sorted(
        w for ws in words.values() for w, *_ in ws
    )  # every word, once


def test_a_chunk_edge_keeps_only_words_mostly_inside_and_trims_the_audio():
    parts = [{"row": 0, "id": "u", "speaker": "A", "offset_s": 0.0, "dur_s": 4.0}]
    words = {
        "u": [
            ("a", 0.0, 1.0, True),
            ("b", 1.0, 2.0, True),
            ("c", 2.0, 3.0, True),
            ("d", 3.0, 4.0, True),
        ]
    }
    audio = np.concatenate([np.full(int(2.2 * SR), 0.3, np.float32), np.zeros(10, np.float32),
                            np.full(int(1.8 * SR), 0.3, np.float32)])  # fmt: skip
    rows = plan_meeting_chunks(
        parts,
        words,
        audio,
        ShortChunkConfig(min_s=(2.0, 2.0), max_s=(2.5, 2.5)),
        "m",
        random.Random(0),
    )
    first = rows[0]
    assert parse_turns(first["target"])[0][1] == "a b"  # 'c' is mostly after the cut
    part = json.loads(first["parts"])[0]
    assert part["offset_s"] + part["dur_s"] + first["tail_s"] == pytest.approx(
        first["duration_s"], abs=1e-3
    )
    assert part["clip_start_s"] == 0.0
    assert part["clip_end_s"] == pytest.approx(first["duration_s"], abs=1e-3)


def test_chunks_with_too_few_aligned_words_are_dropped():
    parts, words, audio = _meeting()
    unaligned = {k: [(w, s, e, False) for w, s, e, _ in v] for k, v in words.items()}
    assert (
        plan_meeting_chunks(parts, unaligned, audio, ShortChunkConfig(), "m1", random.Random(0))
        == []
    )


def test_mix_window_plays_only_the_trimmed_span_of_a_part():
    from scripts.speaker_asr.data import mix_window

    class Store:
        def get(self, row):
            return np.arange(SR * 2, dtype=np.float32) / (SR * 2)

    part = {"row": 0, "offset_s": 0.0, "clip_start_s": 0.5, "clip_end_s": 1.0}
    out = mix_window([part], Store())
    assert len(out) == SR // 2
    assert out[0] == pytest.approx(0.25, abs=1e-4)


def test_busy_chunks_get_the_oversampling_weight():
    rows = [
        {"n_speakers": 1, "n_turns": 1},
        {"n_speakers": 2, "n_turns": 2},
        {"n_speakers": 1, "n_turns": 3},
    ]
    assert busy_weights(rows, 2.0) == [1.0, 2.0, 2.0]


def test_augment_changes_level_and_stays_in_range():
    x = np.full(SR, 0.2, np.float32)
    y = augment(x, np.random.default_rng(0))
    assert y.shape == x.shape
    assert np.abs(y).max() <= 1.0
    assert not np.allclose(x, y)


def test_weighted_lm_loss_upweights_speaker_targets():
    from scripts.speaker_asr.train import weighted_lm_loss

    torch.manual_seed(0)
    logits = torch.randn(1, 5, 10)
    labels = torch.tensor([[-100, 3, 7, 3, 2]])  # token 7 is a speaker token
    plain = weighted_lm_loss(logits, labels, torch.tensor([7]), 1.0)
    heavy = weighted_lm_loss(logits, labels, torch.tensor([7]), 5.0)
    ce = torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:], reduction="none")
    assert plain == pytest.approx(float(ce.mean()), abs=1e-5)
    assert heavy == pytest.approx(float((ce * torch.tensor([1, 5, 1, 1])).sum() / 8), abs=1e-5)


def test_short_chunk_config_defaults_match_the_base_config():
    from scripts.speaker_asr.config import load_config, short_chunk_config

    assert short_chunk_config(load_config()) == ShortChunkConfig()
    assert {f.name for f in fields(ShortChunkConfig)} <= set(load_config().short_chunks)


def test_short_preset_warm_starts_from_the_context_labeller():
    from scripts.speaker_asr.config import load_config, pool_signature

    context, short = load_config(["+experiment=context"]), load_config(["+experiment=short"])
    assert short.short_chunks.enabled
    assert not context.short_chunks.enabled
    assert short.context.enabled  # keeps the <CONTINUE> prompt and 8 speaker tokens
    assert short.model.model_id == context.hub_model_id
    assert pool_signature(short) == pool_signature(context)
    assert short.hub_model_id != context.hub_model_id


def test_align_utterances_skips_empty_clips_and_survives_aligner_errors(tmp_path):
    import pandas as pd

    from scripts.speaker_asr.chunks import align_utterances

    class Store:
        meta = pd.DataFrame({"id": ["empty", "bad", "ok"], "row": [0, 1, 2]})

        def get(self, row):
            return [np.zeros(0, np.float32), np.zeros(SR, np.float32), np.zeros(SR, np.float32)][
                row
            ]

    def align(clip, text):
        if text == "boom":
            raise RuntimeError("aligner exploded")
        return [{"word": "fine", "start": 0.1, "end": 0.5}]

    texts = {"empty": "nothing here", "bad": "boom", "ok": "fine"}
    cache = tmp_path / "words.jsonl"
    words = align_utterances(Store(), texts, ["empty", "bad", "ok"], cache, align=align)
    assert all(not w[3] for w in words["empty"])  # never sent to the aligner, all unaligned
    assert all(not w[3] for w in words["bad"])
    assert words["ok"] == [("fine", 0.1, 0.5, True)]
    assert len(cache.read_text().splitlines()) == 3  # cached: a rerun resumes, nothing redone
    again = align_utterances(Store(), texts, ["empty", "bad", "ok"], cache, align=lambda *a: 1 / 0)
    assert again == words


def test_short_segments_take_the_restyled_human_transcript():
    import pandas as pd

    from scripts.speaker_asr.chunks import human_style, target_texts

    assert human_style("MM-HMM") == "Mm-hmm."
    assert human_style("WHAT DO YOU THINK?") == "What do you think?"
    meta = pd.DataFrame({"id": ["short", "long"], "start": [0.0, 0.0], "end": [0.4, 3.0],
                         "text": ["MM-HMM", "THE REMOTE IS YELLOW"]})  # fmt: skip
    self_texts = {"short": "Yeah.", "long": "The remote is yellow."}
    texts = target_texts(meta, self_texts, below_s=1.0)
    assert texts == {"short": "Mm-hmm.", "long": "The remote is yellow."}
    assert target_texts(meta, self_texts, below_s=0) == self_texts


def test_alignment_cache_redoes_an_utterance_whose_text_changed(tmp_path):
    import pandas as pd

    from scripts.speaker_asr.chunks import align_utterances

    class Store:
        meta = pd.DataFrame({"id": ["u"], "row": [0]})

        def get(self, row):
            return np.zeros(SR, np.float32)

    calls = []

    def align(clip, text):
        calls.append(text)
        return [
            {"word": w, "start": 0.1 * k, "end": 0.1 * k + 0.05} for k, w in enumerate(text.split())
        ]

    cache = tmp_path / "words.jsonl"
    align_utterances(Store(), {"u": "Yeah."}, ["u"], cache, align=align)
    align_utterances(Store(), {"u": "Yeah."}, ["u"], cache, align=align)  # cached: not redone
    words = align_utterances(Store(), {"u": "Mm-hmm."}, ["u"], cache, align=align)
    assert calls == ["Yeah.", "Mm-hmm."]
    assert [w[0] for w in words["u"]] == ["Mm-hmm."]


def test_wordless_chunks_are_kept_with_an_empty_target():
    parts = [{"row": 0, "id": "u", "speaker": "A", "offset_s": 0.0, "dur_s": 2.0}]
    words = {"u": [("hi", 0.2, 0.6, True), ("there", 0.8, 1.2, True)]}
    # 2 s of speech then 6 s of silence: the second chunk holds no words
    audio = np.concatenate([np.full(2 * SR, 0.3, np.float32), np.zeros(10, np.float32),
                            np.full(int(0.2 * SR), 1e-4, np.float32), np.zeros(6 * SR, np.float32)])  # fmt: skip
    cfg = ShortChunkConfig(min_s=(2.0, 2.0), max_s=(5.0, 5.0))
    rows = plan_meeting_chunks(parts, words, audio, cfg, "m", random.Random(0))
    assert any(r["target"] == "" and r["n_speakers"] == 0 for r in rows)
    dropped = plan_meeting_chunks(
        parts,
        words,
        audio,
        ShortChunkConfig(min_s=(2.0, 2.0), max_s=(5.0, 5.0), keep_empty=False),
        "m",
        random.Random(0),
    )
    assert all(r["target"] for r in dropped)


def _recording_utts(parts, words):
    return [
        {"speaker": p["speaker"], "start": p["offset_s"], "words": words[p["id"]]} for p in parts
    ]


def test_recording_chunks_slice_the_recording_and_match_the_mixed_meeting_targets():
    parts, words, audio = _meeting()
    cfg = ShortChunkConfig(min_s=(3.0, 3.0), max_s=(7.0, 7.0))
    mixed = plan_meeting_chunks(parts, words, audio, cfg, "m1", random.Random(0))
    rec = plan_recording_chunks(
        _recording_utts(parts, words), audio, cfg, "m1", "m1/ch0.wav", random.Random(0)
    )
    # same cut points, same targets: only where the audio comes from differs
    assert [r["target"] for r in rec] == [r["target"] for r in mixed]
    for r in rec:
        assert r["recording"] == "m1/ch0.wav"
        assert r["parts"] == "[]"
        assert r["end_s"] - r["start_s"] == pytest.approx(r["duration_s"])


def test_recording_chunks_with_more_voices_than_speaker_tokens_are_dropped():
    utts = [
        {"speaker": f"S{i}", "start": 0.5 * i, "words": [(f"w{i}", 0.0, 0.3, True)]}
        for i in range(6)
    ]
    audio = np.full(4 * SR, 0.3, np.float32)
    cfg = ShortChunkConfig(min_s=(3.5, 3.5), max_s=(4.0, 4.0))
    assert (
        plan_recording_chunks(utts, audio, cfg, "m", "m.wav", random.Random(0), max_speakers=4)
        == []
    )
    kept = plan_recording_chunks(utts, audio, cfg, "m", "m.wav", random.Random(0), max_speakers=6)
    assert kept
    assert kept[0]["n_speakers"] == 6


def test_dataset_reads_a_recording_row_as_a_slice(tmp_path):
    import soundfile as sf

    from scripts.speaker_asr.data import SpeakerASRDataset, read_slice

    wav = np.arange(3 * SR, dtype=np.float32) / (3 * SR)
    (tmp_path / "m").mkdir()
    sf.write(tmp_path / "m" / "ch0.wav", wav, SR, subtype="FLOAT")
    assert np.allclose(read_slice(tmp_path / "m" / "ch0.wav", 1.0, 2.0), wav[SR : 2 * SR])

    class _Store:
        def verify(self, rows):
            pass

    row = {"recording": "m/ch0.wav", "start_s": 0.5, "end_s": 1.5, "parts": "[]", "tail_s": 0.0,
           "target": "<SPK_1>hi"}  # fmt: skip
    item = SpeakerASRDataset([row], _Store(), recordings_root=str(tmp_path))[0]
    assert np.allclose(item["audio"], wav[SR // 2 : 3 * SR // 2])
    assert item["target"] == "<SPK_1>hi"
