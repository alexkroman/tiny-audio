"""Tests for scripts.turn_aware.data: pool construction, labels, and metrics."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from scripts.turn_aware.data import (
    PoolConfig,
    assemble_audio,
    assistant_labels,
    build_pool,
    endpoint_summary,
    expand_observation,
    marker_metrics,
    stratified_subset,
    target_text,
    trim_tail_silence,
)
from tiny_audio.turns import END_OF_TURN, SAMPLE_RATE


def _speech_then_silence(speech_s: float, silence_s: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    speech = (0.1 * rng.standard_normal(int(speech_s * SAMPLE_RATE))).astype(np.float32)
    return np.concatenate([speech, np.zeros(int(silence_s * SAMPLE_RATE), dtype=np.float32)])


def _obs(key="k@1.0", label=1, speech_end_s=2.0, kind="complete", agent_turn="What is your name?"):
    return {
        "key": key,
        "turn_key": key.split("@")[0],
        "kind": kind,
        "label": label,
        "speech_end_s": speech_end_s,
        "target": "my name is jo",
        "agent_turn": agent_turn,
    }


class TestAudio:
    def test_trim_keeps_80ms_past_last_speech(self):
        audio = _speech_then_silence(1.0, 0.5)
        end = trim_tail_silence(audio)
        assert end == pytest.approx(1.08 * SAMPLE_RATE, abs=0.02 * SAMPLE_RATE)

    def test_trim_all_silence_is_zero(self):
        assert trim_tail_silence(np.zeros(SAMPLE_RATE, dtype=np.float32)) == 0

    def test_assemble_lengths_and_zero_padding(self):
        turn = _speech_then_silence(3.0, 0.0)
        row = {"speech_end_s": 1.5, "lead_s": 0.5, "tail_s": 0.7}
        out = assemble_audio(turn, row)
        assert len(out) == int(2.7 * SAMPLE_RATE)
        assert out.dtype == np.float32
        assert not out[: int(0.5 * SAMPLE_RATE)].any()
        assert not out[-int(0.7 * SAMPLE_RATE) :].any()
        np.testing.assert_array_equal(
            out[int(0.5 * SAMPLE_RATE) : int(2.0 * SAMPLE_RATE)], turn[: int(1.5 * SAMPLE_RATE)]
        )


class TestExpandObservation:
    def test_negative_is_one_silenced_hold(self):
        cfg = PoolConfig(lead_sil_prob=0.0, hold_copies=())
        out = expand_observation(_obs(label=0, kind="payload_cut"), cfg, random.Random(0))
        assert [e["schema"] for e in out] == ["hold_sil"]
        assert not out[0]["fire"]
        assert cfg.hold_tail_s[0] <= out[0]["tail_s"] <= cfg.hold_tail_s[1]

    def test_positive_pairs_fire_with_identical_nosil_twin(self):
        cfg = PoolConfig(nosil_prob=1.0, long_tail_prob=0.0, lead_sil_prob=0.0)
        fire, nosil = expand_observation(_obs(), cfg, random.Random(0))
        assert (fire["schema"], fire["fire"]) == ("fire_sil", True)
        assert (nosil["schema"], nosil["fire"], nosil["tail_s"]) == ("fire_nosil", False, 0.0)
        # Only the tail may differ between the pair.
        for field in ("turn_key", "speech_end_s", "lead_s", "text", "ctx"):
            assert fire[field] == nosil[field]

    def test_long_tail_positive(self):
        cfg = PoolConfig(nosil_prob=0.0, long_tail_prob=1.0)
        (ex,) = expand_observation(_obs(), cfg, random.Random(0))
        assert ex["schema"] == "fire_long"
        assert ex["fire"]
        assert cfg.long_tail_s[0] <= ex["tail_s"] <= cfg.long_tail_s[1]

    @pytest.mark.parametrize(("ctx_prob", "expected"), [(1.0, "What is your name?"), (0.0, "")])
    def test_context_gate(self, ctx_prob, expected):
        ex, *_ = expand_observation(_obs(), PoolConfig(ctx_prob=ctx_prob), random.Random(0))
        assert ex["ctx"] == expected


class TestBuildPool:
    def _observations(self, n=200):
        return [
            _obs(key=f"t{i}@1.0", label=i % 2, kind="complete" if i % 2 else "word_cut")
            for i in range(n)
        ]

    def test_deterministic_in_seed(self):
        obs = self._observations()
        assert build_pool(obs, PoolConfig(), seed=3) == build_pool(
            list(reversed(obs)), PoolConfig(), seed=3
        )
        assert build_pool(obs, PoolConfig(), seed=3) != build_pool(obs, PoolConfig(), seed=4)

    def test_filters_long_and_empty_and_adds_silence_only(self):
        obs = [
            *self._observations(),
            _obs(key="long@1", speech_end_s=40.0),
            _obs(key="empty@1", speech_end_s=0.0),
        ]
        pool = build_pool(obs, PoolConfig(silence_only_frac=0.1), seed=0)
        assert not any(e["turn_key"] in ("long", "empty") for e in pool)
        silence = [e for e in pool if e["schema"] == "silence_only"]
        speech = [e for e in pool if e["schema"] != "silence_only"]
        assert len(silence) == round(0.1 * len(speech))
        assert all(e["text"] == "" and not e["fire"] and e["turn_key"] == "" for e in silence)

    def test_stratified_subset_balances_schemas(self):
        pool = build_pool(self._observations(1000), PoolConfig(), seed=0)
        sub = stratified_subset(pool, 40)
        counts = {s: sum(e["schema"] == s for e in sub) for s in {e["schema"] for e in sub}}
        assert len(set(counts.values())) == 1


def test_target_text_appends_marker_only_on_fire():
    assert target_text({"text": "hi there", "fire": True}) == f"hi there{END_OF_TURN}"
    assert target_text({"text": "hi there", "fire": False}) == "hi there"


class TestAssistantLabels:
    ASR, IM_END, PAD = 50, 51, 0

    def test_supervises_after_asr_text_through_im_end(self):
        # [pad, prompt..., <asr_text>, t1, t2, <|im_end|>, "\n"]
        ids = torch.tensor([[self.PAD, 7, 8, self.ASR, 11, 12, self.IM_END, 13]])
        mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]])
        labels = assistant_labels(ids, mask, self.ASR, self.IM_END)
        assert labels.tolist() == [[-100, -100, -100, -100, 11, 12, self.IM_END, -100]]

    def test_empty_transcript_supervises_im_end_only(self):
        ids = torch.tensor([[7, self.ASR, self.IM_END, 13]])
        labels = assistant_labels(ids, torch.ones_like(ids), self.ASR, self.IM_END)
        assert labels.tolist() == [[-100, -100, self.IM_END, -100]]


class TestMetrics:
    def test_marker_metrics(self):
        rows = [
            {"fire": True, "schema": "fire_sil", "kind": "complete", "text": "a b"},
            {"fire": True, "schema": "fire_sil", "kind": "chunk_cut", "text": "c d"},
            {"fire": False, "schema": "hold_sil", "kind": "payload_cut", "text": "e f"},
            {"fire": False, "schema": "silence_only", "kind": "silence_only", "text": ""},
        ]
        preds = [("a b", True), ("c x", False), ("e f", True), ("", False)]
        m = marker_metrics(rows, [f for _, f in preds], [t for t, _ in preds])
        assert m["fire_precision"] == pytest.approx(0.5)
        assert m["fire_recall"] == pytest.approx(0.5)
        assert m["acc_schema/fire_sil"] == pytest.approx(0.5)
        assert m["acc_kind/payload_cut"] == 0.0
        assert m["acc_schema/silence_only"] == 1.0
        assert m["text_wer"] == pytest.approx(1 / 6)

    def test_endpoint_summary(self):
        s = endpoint_summary([1.0, 5.3, 5.5, float("nan")], [5.0, 5.0, 5.0, 5.0])
        assert s["n"] == 4
        assert s["cut_early"] == 0.25
        assert s["missed"] == 0.25
        assert s["latency_p50_s"] == pytest.approx(0.4)


class TestThresholdSweep:
    def test_sweep_trades_false_fires_for_recall(self):
        from scripts.turn_aware.data import threshold_sweep

        rows = [
            {"fire": True, "schema": "fire_sil", "kind": "complete", "text": "a"},
            {"fire": True, "schema": "fire_sil", "kind": "complete", "text": "b"},
            {"fire": False, "schema": "hold_sil", "kind": "payload_cut", "text": "c"},
            {"fire": False, "schema": "hold_sil", "kind": "payload_cut", "text": "d"},
        ]
        margins = [6.0, 1.5, 1.0, float("nan")]
        greedy, strict = threshold_sweep(rows, margins, [0.0, 2.0])
        assert (greedy["fire_recall"], greedy["acc_kind/payload_cut"]) == (1.0, 0.5)
        assert (strict["fire_recall"], strict["acc_kind/payload_cut"]) == (0.5, 1.0)
        assert strict["fire_precision"] == 1.0


def test_hold_copies_oversample_costly_kinds_with_fresh_tails():
    cfg = PoolConfig(lead_sil_prob=0.0, hold_copies=(("payload_cut", 3),))
    out = expand_observation(_obs(label=0, kind="payload_cut"), cfg, random.Random(0))
    assert [e["key"] for e in out] == ["k@1.0#hold_sil", "k@1.0#hold_sil1", "k@1.0#hold_sil2"]
    assert len({e["tail_s"] for e in out}) == 3
    assert len(expand_observation(_obs(label=0, kind="word_cut"), cfg, random.Random(0))) == 1


class TestMinePauses:
    @pytest.mark.parametrize(
        ("text", "mid"),
        [
            ("which I'd like to pay now with", True),
            ("my card—", True),
            ("It's 706,", True),
            ("That's all.", False),
            ('He said "stop."', False),
            ("Is that right?", False),
            ("", False),
        ],
    )
    def test_ends_mid_sentence(self, text, mid):
        from scripts.turn_aware.data import ends_mid_sentence

        assert ends_mid_sentence(text) is mid

    def test_mines_only_unlabelled_interior_mid_sentence_pauses(self):
        from scripts.turn_aware.data import MINED_KIND, mine_pauses

        turn = {
            "turn_key": "t1",
            "style": "afterthought",
            "domain": "banking",
            "chunks": ["Hi, I'd like to pay", "my bill today.", "Oh, and the", "late fee."],
            "chunk_ends_s": [1.5, 3.0, 4.2, 5.0],
        }
        # 4.2 is mid-sentence but already labelled; 3.0 ends a sentence; 5.0 is the turn end
        mined = mine_pauses([turn], labelled={("t1", 4.2)})
        assert [(m["at_s"], m["label"], m["kind"]) for m in mined] == [(1.5, 0, MINED_KIND)]
        assert mined[0]["key"] == "t1@1.5#pause"
        assert mined[0]["text"] == "Hi, I'd like to pay"


def test_hold_targets_lose_terminal_punct_but_fire_targets_keep_it():
    cfg = PoolConfig(hold_copies=(), nosil_prob=1.0, long_tail_prob=0.0)
    hold = expand_observation({**_obs(label=0), "target": "It's 706, 555."}, cfg, random.Random(0))
    assert hold[0]["text"] == "It's 706, 555"
    fire, nosil = expand_observation({**_obs(), "target": "That's all."}, cfg, random.Random(0))
    assert fire["text"] == nosil["text"] == "That's all."
    off = PoolConfig(hold_copies=(), strip_hold_punct=False)
    assert (
        expand_observation({**_obs(label=0), "target": "Wait?"}, off, random.Random(0))[0]["text"]
        == "Wait?"
    )


def test_transcript_cache_survives_a_truncated_last_line(tmp_path, monkeypatch):
    import json

    import pandas as pd

    import scripts.turn_aware.cli as cli

    cache = tmp_path / "transcripts-train.jsonl"
    cache.write_text(json.dumps({"key": "a", "text": "hello"}) + '\n{"key": "b", "te')
    obs = pd.DataFrame([{"key": "a", "turn_key": "t", "speech_end_s": 1.0}])
    # Every key is cached, so no model is loaded; the bad line must not raise.
    assert cli._self_transcripts(obs, store=None, model_id="x", cache=cache, batch_size=1) == {
        "a": "hello"
    }
    assert cache.read_text().endswith("\n")  # next appended record starts cleanly


def test_find_intra_pauses_skips_edges_short_gaps_and_boundaries():
    from scripts.turn_aware.data import find_intra_pauses

    rng = np.random.default_rng(0)

    def speech(s):
        return (0.1 * rng.standard_normal(int(s * SAMPLE_RATE))).astype(np.float32)

    def sil(s):
        return np.zeros(int(s * SAMPLE_RATE), dtype=np.float32)

    # lead sil | 1.0 speech | 0.5 pause (t=1.5) | 1.0 speech | 0.2 gap | 0.5 speech
    # | 0.6 pause at a chunk end (t=3.7) | 1.0 speech | trailing sil
    audio = np.concatenate(
        [
            sil(0.5),
            speech(1.0),
            sil(0.5),
            speech(1.0),
            sil(0.2),
            speech(0.5),
            sil(0.6),
            speech(1.0),
            sil(1.0),
        ]
    )
    pauses = find_intra_pauses(audio, avoid_s=[3.8])
    assert pauses == [pytest.approx(1.5, abs=0.03)]
    assert find_intra_pauses(audio, avoid_s=[]) == [
        pytest.approx(1.5, abs=0.03),
        pytest.approx(3.7, abs=0.03),
    ]


def test_with_context_modes():
    from scripts.turn_aware.data import with_context

    rows = [
        {"turn_key": "t1", "ctx": ""},
        {"turn_key": "t2", "ctx": "Can I get your number?"},
        {"turn_key": "", "ctx": ""},  # silence-only row
    ]
    meta = {
        "t1": {"agent_turn": "Did that fix it?"},
        "t2": {"agent_turn": "Can I get your number?"},
    }
    assert with_context(rows, "pool", meta) is rows
    assert [r["ctx"] for r in with_context(rows, "never", meta)] == ["", "", ""]
    assert [r["ctx"] for r in with_context(rows, "always", meta)] == [
        "Did that fix it?",
        "Can I get your number?",
        "",
    ]
    with pytest.raises(ValueError, match="context mode"):
        with_context(rows, "sometimes", meta)
