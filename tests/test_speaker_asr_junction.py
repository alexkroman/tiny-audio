"""Tests for speaker-ASR junction windows: same voice minutes apart, balanced same/new."""

from __future__ import annotations

import itertools
import json
from dataclasses import fields

import pytest

from scripts.eval.speaker_metrics import parse_turns
from scripts.speaker_asr.data import format_target
from scripts.speaker_asr.junction import JunctionConfig, build_junction_rows, junction_count


def meeting(group="m1", speakers="ABCD", minutes=30, every_s=7.0, dur_s=3.0, row0=0):
    """Utterances cycling through `speakers`, one every `every_s` seconds."""
    utts, texts = [], {}
    n = int(minutes * 60 / every_s)
    for i in range(n):
        row = row0 + i
        uid = f"{group}-u{i}"
        start = i * every_s
        utts.append(
            {"row": row, "id": uid, "group": group, "speaker": f"{group}-{speakers[i % len(speakers)]}",
             "start": start, "end": start + dur_s}
        )  # fmt: skip
        texts[uid] = f"words of {uid}."
    return utts, texts


@pytest.fixture
def corpus():
    a_utts, a_texts = meeting("m1")
    b_utts, b_texts = meeting("m2", speakers="XYZ", row0=10_000)
    return a_utts + b_utts, {**a_texts, **b_texts}


def parts_of(row):
    return json.loads(row["parts"])


def test_rows_have_the_pool_row_shape_and_the_ordinary_target(corpus):
    utts, texts = corpus
    rows = build_junction_rows(utts, texts, JunctionConfig(), 50)
    assert len(rows) == 50
    for row in rows:
        parts = parts_of(row)
        assert {"key", "group", "parts", "tail_s", "duration_s", "target", "n_speakers"} <= set(row)
        target, n_speakers = format_target(parts, texts)
        assert row["target"] == target
        assert row["n_speakers"] == n_speakers
        assert parse_turns(row["target"])[0][0] == 1  # numbered from <SPK_1>


def test_clips_are_whole_in_range_utterances_far_apart_in_one_meeting(corpus):
    utts, texts = corpus
    by_id = {u["id"]: u for u in utts}
    cfg = JunctionConfig(min_separation_s=90.0)
    for row in build_junction_rows(utts, texts, cfg, 100):
        parts = parts_of(row)
        assert cfg.clips[0] <= len(parts) <= cfg.clips[1]
        assert {by_id[p["id"]]["group"] for p in parts} == {row["group"]}
        for p in parts:
            assert cfg.clip_s[0] <= p["dur_s"] <= cfg.clip_s[1]
        for p, q in itertools.combinations(parts, 2):
            assert abs(by_id[p["id"]]["start"] - by_id[q["id"]]["start"]) >= 90.0


def test_clips_are_laid_out_in_order_with_gaps_in_range(corpus):
    utts, texts = corpus
    cfg = JunctionConfig(gap_s=(0.3, 1.0))
    for row in build_junction_rows(utts, texts, cfg, 50):
        parts = parts_of(row)
        assert parts[0]["offset_s"] == 0.0
        for p, q in itertools.pairwise(parts):
            gap = q["offset_s"] - (p["offset_s"] + p["dur_s"])
            assert 0.3 - 1e-3 <= gap <= 1.0 + 1e-3
        assert row["duration_s"] == pytest.approx(
            parts[-1]["offset_s"] + parts[-1]["dur_s"], abs=1e-3
        )


def test_same_and_new_speaker_junctions_are_balanced(corpus):
    utts, texts = corpus
    rows = build_junction_rows(utts, texts, JunctionConfig(same_prob=0.5), 400)
    same = sum(r["n_same"] for r in rows)
    new = sum(r["n_new"] for r in rows)
    assert 0.4 < same / (same + new) < 0.6
    for row in rows:  # the recorded counts match the parts
        speakers, n_same = set(), 0
        for p in parts_of(row):
            n_same += p["speaker"] in speakers
            speakers.add(p["speaker"])
        assert row["n_same"] == n_same
        assert row["n_new"] == len(speakers) - 1


def test_same_prob_steers_the_mix(corpus):
    utts, texts = corpus
    # <= 3 clips: every meeting here has 3+ speakers, so "all new" is always feasible
    cfg = JunctionConfig(clips=(2, 3))
    all_same = build_junction_rows(utts, texts, JunctionConfig(clips=(2, 3), same_prob=1.0), 50)
    all_new = build_junction_rows(utts, texts, JunctionConfig(clips=cfg.clips, same_prob=0.0), 50)
    assert all(r["n_new"] == 0 for r in all_same)
    assert all(r["n_same"] == 0 for r in all_new)


def test_an_infeasible_kind_falls_back_to_the_other_instead_of_dropping():
    utts, texts = meeting("pair", speakers="AB")  # only two voices
    rows = build_junction_rows(utts, texts, JunctionConfig(clips=(4, 4), same_prob=0.0), 20)
    assert rows
    assert all(len(parts_of(r)) == 4 for r in rows)
    assert all(r["n_new"] == 1 and r["n_same"] == 2 for r in rows)


def test_speaker_cap_and_audio_cap_hold(corpus):
    utts, texts = corpus
    cfg = JunctionConfig(clips=(4, 4), same_prob=0.0, clip_s=(2.0, 6.0))
    for row in build_junction_rows(utts, texts, cfg, 50, max_speakers=2, max_audio_s=30.0):
        assert row["n_speakers"] <= 2
        assert row["duration_s"] <= 30.0


def test_meetings_without_two_usable_speakers_are_skipped():
    solo, texts = meeting("solo", speakers="A")
    silent, silent_texts = meeting("quiet", speakers="AB", row0=5_000)
    silent_texts = dict.fromkeys(silent_texts, "")  # no words: no usable clips
    rows = build_junction_rows(solo + silent, {**texts, **silent_texts}, JunctionConfig(), 10)
    assert rows == []


def test_build_is_deterministic_in_seed(corpus):
    utts, texts = corpus
    a = build_junction_rows(utts, texts, JunctionConfig(), 30, seed=3)
    assert a == build_junction_rows(utts, texts, JunctionConfig(), 30, seed=3)
    assert a != build_junction_rows(utts, texts, JunctionConfig(), 30, seed=4)


def test_junction_count_hits_the_requested_share():
    assert junction_count(650, 0.35) == 350
    assert junction_count(100, 0.0) == 0
    with pytest.raises(ValueError, match="ratio"):
        junction_count(100, 1.0)


# ----------------------------------------------------------------- config


def test_junction_config_defaults_match_the_base_config():
    from scripts.speaker_asr.config import junction_config, load_config

    assert junction_config(load_config()) == JunctionConfig()
    assert {f.name for f in fields(JunctionConfig)} <= set(load_config().junction)


def test_junction_preset_is_v1_plus_junction_windows():
    from scripts.speaker_asr.config import load_config, n_speaker_tokens, pool_signature

    v1, junction = load_config(["+experiment=v1"]), load_config(["+experiment=junction"])
    assert not v1.junction.enabled
    assert junction.junction.enabled
    assert not junction.context.enabled
    assert pool_signature(junction) == pool_signature(v1)  # same published pool
    assert junction.model == v1.model  # same LoRA, tokens and decode budget
    assert n_speaker_tokens(junction) == n_speaker_tokens(v1)
    assert junction.hub_model_id != v1.hub_model_id
    assert junction.training.output_dir != v1.training.output_dir
