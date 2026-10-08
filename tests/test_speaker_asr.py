"""Tests for scripts.speaker_asr: window planning, targets, cpWER, presets, decode splitting."""

from __future__ import annotations

import json
import re
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.speaker_asr.data import (
    SAMPLE_RATE,
    WindowConfig,
    build_pool,
    format_target,
    mix_window,
    plan_windows,
    subset,
)

# Deterministic windows: fixed length, no lead/tail.
FIXED = WindowConfig(window_s=(30.0, 30.0), lead_s=(0.0, 0.0), tail_s=(0.0, 0.0))


def utt(i, speaker, start, end, group="m1"):
    return {"row": i, "id": f"u{i}", "group": group, "speaker": speaker, "start": start, "end": end}


# ----------------------------------------------------------------- windows


def test_real_gaps_kept_and_long_gaps_squeezed():
    utts = [utt(0, "A", 100.0, 102.0), utt(1, "B", 102.5, 104.0), utt(2, "A", 120.0, 121.0)]
    (w,) = plan_windows(utts, FIXED)
    offsets = [p["offset_s"] for p in w["parts"]]
    assert offsets == [0.0, 2.5, 4.0 + FIXED.max_gap_s]
    assert w["duration_s"] == pytest.approx(4.0 + FIXED.max_gap_s + 1.0)


def test_overlap_is_measured_from_the_latest_end():
    # B backchannels inside A's long turn; C starts after A ends.
    utts = [utt(0, "A", 0.0, 10.0), utt(1, "B", 3.0, 3.5), utt(2, "C", 10.2, 12.0)]
    (w,) = plan_windows(utts, FIXED)
    assert [p["offset_s"] for p in w["parts"]] == [0.0, 3.0, 10.2]


def test_no_overlap_pushes_utterances_apart():
    cfg = WindowConfig(**{**FIXED.__dict__, "overlap": False})
    utts = [utt(0, "A", 0.0, 10.0), utt(1, "B", 3.0, 3.5)]
    (w,) = plan_windows(utts, cfg)
    assert w["parts"][1]["offset_s"] == pytest.approx(10.0 + cfg.min_gap_s)


def test_window_closes_before_a_speaker_beyond_the_limit():
    cfg = WindowConfig(**{**FIXED.__dict__, "max_speakers": 2})
    utts = [utt(0, "A", 0, 1), utt(1, "B", 1, 2), utt(2, "A", 2, 3), utt(3, "C", 3, 4)]
    w1, w2 = plan_windows(utts, cfg)
    assert [p["speaker"] for p in w1["parts"]] == ["A", "B", "A"]
    assert [p["speaker"] for p in w2["parts"]] == ["C"]
    assert w2["parts"][0]["offset_s"] == 0.0  # a new window starts at its own lead


def test_window_respects_its_length_and_drops_overlong_utterances():
    cfg = WindowConfig(**{**FIXED.__dict__, "window_s": (5.0, 5.0)})
    utts = [utt(0, "A", 0, 3), utt(1, "B", 3, 6), utt(2, "A", 10, 50)]
    windows = plan_windows(utts, cfg)
    assert [[p["id"] for p in w["parts"]] for w in windows] == [["u0"], ["u1"]]
    assert all(w["duration_s"] <= cfg.max_audio_s for w in windows)


def test_windows_never_mix_meetings_and_short_fragments_are_dropped():
    utts = [utt(0, "A", 0, 1, "m1"), utt(1, "B", 1, 2, "m2"), utt(2, "B", 2, 2.1, "m2")]
    windows = plan_windows(utts, FIXED)
    assert sorted(w["group"] for w in windows) == ["m1", "m2"]
    assert all(p["id"] != "u2" for w in windows for p in w["parts"])  # 0.1 s < min_utt_s


def test_plan_windows_is_deterministic_in_seed():
    utts = [utt(i, "AB"[i % 2], 2.0 * i, 2.0 * i + 1.5) for i in range(40)]
    cfg = WindowConfig()
    assert plan_windows(utts, cfg, seed=3) == plan_windows(utts, cfg, seed=3)
    assert plan_windows(utts, cfg, seed=3) != plan_windows(utts, cfg, seed=4)


# ----------------------------------------------------------------- targets


def test_target_numbers_speakers_by_first_appearance_and_merges_runs():
    parts = [
        {"id": "a", "speaker": "FEE041", "offset_s": 0.0, "dur_s": 1.0},
        {"id": "b", "speaker": "MEE007", "offset_s": 1.2, "dur_s": 1.0},
        {"id": "c", "speaker": "MEE007", "offset_s": 2.4, "dur_s": 1.0},
        {"id": "d", "speaker": "FEE041", "offset_s": 3.5, "dur_s": 1.0},
    ]
    texts = {"a": "Hello.", "b": "Hi there.", "c": "How are you?", "d": "Fine."}
    target, n = format_target(parts, texts)
    assert target == "<SPK_1>Hello.<SPK_2>Hi there. How are you?<SPK_1>Fine."
    assert n == 2


def test_target_orders_by_offset_and_skips_empty_transcripts():
    parts = [
        {"id": "late", "speaker": "A", "offset_s": 2.0, "dur_s": 1.0},
        {"id": "silent", "speaker": "C", "offset_s": 0.5, "dur_s": 0.3},
        {"id": "early", "speaker": "B", "offset_s": 0.0, "dur_s": 1.0},
    ]
    target, n = format_target(parts, {"late": "Yes.", "early": "So.", "silent": " "})
    assert target == "<SPK_1>So.<SPK_2>Yes."
    assert n == 2


def test_build_pool_drops_wordless_windows_and_serialises_parts():
    windows = [
        {"key": "m:a", "group": "m", "tail_s": 0.0, "duration_s": 1.0,
         "parts": [{"id": "a", "speaker": "A", "offset_s": 0.0, "dur_s": 1.0, "row": 0}]},
        {"key": "m:b", "group": "m", "tail_s": 0.0, "duration_s": 1.0,
         "parts": [{"id": "b", "speaker": "A", "offset_s": 0.0, "dur_s": 1.0, "row": 1}]},
    ]  # fmt: skip
    (row,) = build_pool(windows, {"a": "Okay."})
    assert row["target"] == "<SPK_1>Okay."
    assert json.loads(row["parts"])[0]["row"] == 0
    assert (row["n_speakers"], row["n_parts"]) == (1, 1)


def test_subset_balances_speaker_counts():
    rows = [{"n_speakers": 1}] * 50 + [{"n_speakers": 3}] * 5
    picked = subset(rows, 10)
    assert sum(r["n_speakers"] == 3 for r in picked) == 5


# ----------------------------------------------------------------- audio


def test_mix_window_sums_parts_at_their_offsets_and_never_clips():
    clips = {0: np.full(SAMPLE_RATE, 0.6, np.float32), 1: np.full(SAMPLE_RATE, 0.6, np.float32)}
    store = SimpleNamespace(get=lambda row: clips[row])
    parts = [{"row": 0, "offset_s": 0.0}, {"row": 1, "offset_s": 0.5}]
    out = mix_window(parts, store, tail_s=0.25)
    assert len(out) == int(1.75 * SAMPLE_RATE)
    assert np.abs(out).max() <= 1.0
    assert out[0] == pytest.approx(out[-int(0.25 * SAMPLE_RATE) - 1])  # one voice each side
    assert out[int(0.75 * SAMPLE_RATE)] > out[0]  # overlap is louder
    assert out[-1] == 0.0  # tail silence


# ----------------------------------------------------------------- config


def test_presets_resolve_to_their_own_identity():
    from scripts.speaker_asr.config import load_config

    v1, tower = load_config(["+experiment=v1"]), load_config(["+experiment=tower"])
    assert v1.data.pool_dir == tower.data.pool_dir == "data/speaker_asr"  # same data
    assert v1.training.output_dir != tower.training.output_dir
    assert v1.hub_model_id != tower.hub_model_id
    assert "audio_tower" not in v1.model.lora_target_modules
    assert "audio_tower" in tower.model.lora_target_modules
    assert load_config().hub_model_id is None  # no preset never pushes


def test_window_config_defaults_match_the_base_config():
    from scripts.speaker_asr.config import load_config, window_config

    assert window_config(load_config()) == WindowConfig()
    assert {f.name for f in fields(WindowConfig)} <= set(load_config().pool)


def test_pool_signature_ignores_the_cache_location_but_not_the_recipe():
    from scripts.speaker_asr.config import load_config, pool_signature

    base = pool_signature(load_config())
    assert pool_signature(load_config(["pool.transcript_cache=/elsewhere"])) == base
    assert pool_signature(load_config(["pool.max_gap_s=3.0"])) != base
    assert pool_signature(load_config(["data.data_files='sdm/{split}-*.parquet'"])) != base


def _qwen3_asr_module_names() -> list[str]:
    import torch
    from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

    config = Qwen3ASRConfig(
        audio_config={"d_model": 32, "encoder_layers": 1, "encoder_attention_heads": 2,
                      "encoder_ffn_dim": 64, "output_dim": 32, "num_mel_bins": 16,
                      "downsample_hidden_size": 8},
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                     "vocab_size": 128},
    )  # fmt: skip
    with torch.device("meta"):
        model = Qwen3ASRForConditionalGeneration(config)
    return [name for name, _ in model.named_modules()]


@pytest.mark.parametrize("preset", ["v1", "tower"])
def test_lora_patterns_hit_the_intended_modules(preset):
    from scripts.speaker_asr.config import load_config

    pattern = load_config([f"+experiment={preset}"]).model.lora_target_modules
    hits = [n for n in _qwen3_asr_module_names() if re.fullmatch(pattern, n)]
    decoder = [n for n in hits if "language_model" in n]
    tower = [n for n in hits if "audio_tower" in n]
    assert len(decoder) == 7  # q/k/v/o + gate/up/down in the one decoder layer
    assert len(tower) == (4 if preset == "tower" else 0)
    assert not any("multi_modal_projector" in n or n.endswith("lm_head") for n in hits)


# ----------------------------------------------------------------- model


def test_split_at_speakers_keeps_speaker_tokens_only():
    from scripts.speaker_asr.model import split_at_speakers

    vocab = {1: "Hi", 2: " there", 3: "Yo", 9: ""}  # 9 stands in for <|im_end|>

    def decode(ids):
        return "".join(vocab[i] for i in ids)

    speakers = {100: 1, 101: 2}
    assert split_at_speakers([100, 1, 2, 101, 3, 9], speakers, decode) == (
        "<SPK_1>Hi there<SPK_2>Yo"
    )
    assert split_at_speakers([1, 2, 9], speakers, decode) == "Hi there"


def test_register_speaker_tokens_returns_only_the_new_ids():
    from scripts.speaker_asr.model import register_speaker_tokens

    class Tok:
        def __init__(self):
            self.vocab = {"<asr_text>": 0}

        def get_vocab(self):
            return dict(self.vocab)

        def add_tokens(self, tokens, special_tokens=False):
            assert special_tokens
            for t in tokens:
                self.vocab[t] = len(self.vocab)

        def convert_tokens_to_ids(self, tokens):
            return [self.vocab[t] for t in tokens]

    processor = SimpleNamespace(tokenizer=Tok())
    assert register_speaker_tokens(processor, 3) == [1, 2, 3]
    assert {"<SPK_1>", "<SPK_2>", "<SPK_3>", "<asr_text>"} == set(processor.tokenizer.vocab)
    assert register_speaker_tokens(processor, 3) == []
    # A context-trained run adds <CONTINUE> (and only that) to the same tokenizer.
    assert register_speaker_tokens(processor, 3, context=True) == [4]
    assert "<CONTINUE>" in processor.tokenizer.vocab


# ----------------------------------------------------------------- deploy


def test_runpod_script_runs_speaker_asr_pool_then_training_with_same_overrides():
    from scripts.deploy.runpod import build_speaker_asr_train_script

    overrides = ["+experiment=tower", "training.max_steps=10"]
    script = build_speaker_asr_train_script("token", None, None, overrides, 64)
    joined = " ".join(overrides)
    assert f"speaker-asr build-pool --split train --split validation --batch-size 64 {joined}" in (
        script
    )
    assert f"&& python -m scripts.speaker_asr.train {joined}" in script
    assert "turn-aware" not in script


# ----------------------------------------------------------------- ta eval


def test_qwen3_asr_checkpoints_are_detected_from_config(tmp_path):
    from scripts.eval.cli import _is_qwen3_asr

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_asr"}))
    assert _is_qwen3_asr(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "asr_model"}))
    assert not _is_qwen3_asr(str(tmp_path))
    assert not _is_qwen3_asr(str(tmp_path / "missing"))


# ----------------------------------------------------------------- pool on the Hub


def _fake_hub(monkeypatch, files: dict[str, str], tmp_path):
    from scripts.speaker_asr import hub

    def download(repo, filename):
        if filename not in files:
            return None
        path = tmp_path / "remote" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[filename])
        return path

    monkeypatch.setattr(hub, "_download", download)
    return hub


def test_pull_takes_a_matching_pool_and_records_its_signature(monkeypatch, tmp_path):
    from scripts.speaker_asr.config import load_config, pool_signature
    from scripts.turn_aware.config import pool_is_current

    sig = json.loads(json.dumps(pool_signature(load_config())))
    hub = _fake_hub(
        monkeypatch,
        {
            "signatures/train.json": json.dumps(sig),
            "train.parquet": "PARQUET",
            "transcripts-train.jsonl": json.dumps({"key": "u1", "text": "Hi."}) + "\n",
        },
        tmp_path,
    )
    pool_dir, cache = tmp_path / "pool", tmp_path / "pool" / "transcripts-train.jsonl"
    pool_dir.mkdir()
    assert hub.pull("repo", "train", pool_dir, cache, sig) == "pulled"
    assert (pool_dir / "train.parquet").read_text() == "PARQUET"
    assert pool_is_current(pool_dir, "train", sig)
    assert "u1" in cache.read_text()


def test_pull_reuses_only_transcripts_when_settings_differ(monkeypatch, tmp_path):
    from scripts.speaker_asr.config import load_config, pool_signature

    remote = json.loads(json.dumps(pool_signature(load_config(["pool.max_gap_s=3.0"]))))
    wanted = json.loads(json.dumps(pool_signature(load_config())))
    files = {
        "signatures/train.json": json.dumps(remote),
        "train.parquet": "PARQUET",
        "transcripts-train.jsonl": json.dumps({"key": "u1", "text": "Hi."}) + "\n",
    }
    hub = _fake_hub(monkeypatch, files, tmp_path)
    cache = tmp_path / "transcripts-train.jsonl"
    cache.write_text(json.dumps({"key": "u0", "text": "Yes."}) + "\n")
    assert hub.pull("repo", "train", tmp_path, cache, wanted) == "transcripts"
    assert not (tmp_path / "train.parquet").exists()
    assert [json.loads(line)["key"] for line in cache.read_text().splitlines()] == ["u0", "u1"]

    other_model = json.loads(json.dumps(remote))
    other_model["pool"]["target_model"] = "someone/else"
    files["signatures/train.json"] = json.dumps(other_model)
    assert hub.pull("repo", "train", tmp_path, cache, wanted) == "none"


def test_hub_repo_is_not_part_of_the_pool_signature():
    from scripts.speaker_asr.config import load_config, pool_signature

    assert pool_signature(load_config(["+experiment=v1"])) == pool_signature(
        load_config(["+experiment=v1", "pool.hub_repo=null"])
    )


def test_store_refuses_parts_whose_row_moved():
    import pandas as pd

    from scripts.speaker_asr.data import UtteranceStore

    store = UtteranceStore.__new__(UtteranceStore)
    store.meta = pd.DataFrame({"id": ["u0", "u1"]})
    good = {"parts": json.dumps([{"row": 1, "id": "u1"}])}
    store.verify([good])
    with pytest.raises(ValueError, match="different utterance"):
        store.verify([{"parts": json.dumps([{"row": 0, "id": "u1"}])}])
    with pytest.raises(ValueError, match="different utterance"):
        store.verify([{"parts": json.dumps([{"row": 5, "id": "u1"}])}])


def test_runpod_pool_script_builds_every_split_and_stops():
    from scripts.deploy.runpod import build_speaker_asr_pool_script

    script = build_speaker_asr_pool_script("t", ["train", "test"], ["+experiment=v1"], 128)
    assert (
        "speaker-asr build-pool --split train --split test --batch-size 128 +experiment=v1\n"
        in script
    )
    assert "scripts.speaker_asr.train" not in script
    assert "&&" not in script.split("build-pool")[1].split("EXIT_CODE")[0]
