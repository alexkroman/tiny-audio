"""Tests for NOTSOFAR-1 far-field chunk rows (scripts/speaker_asr/notsofar.py)."""

from __future__ import annotations

import json
from dataclasses import fields
from unittest import mock

import numpy as np
import pytest

from scripts.speaker_asr import notsofar as nsf
from scripts.speaker_asr.chunks import ShortChunkConfig
from scripts.speaker_asr.metrics import parse_turns

SR = 16000


def _seg(speaker, start, end, text, timing):
    return {"speaker_id": speaker, "start_time": start, "end_time": end, "text": text,
            "word_timing": timing, "ct_wav_file_name": "close_talk/CT_1.wav"}  # fmt: skip


def test_clean_text_drops_annotation_tags_and_keeps_names():
    text = "<ST/> OK I tried <FILL/> I tried <PName>Bob</PName> <FILLlaugh/> um, Mm-hmm. <ST/>"
    assert nsf.clean_text(text) == "OK I tried I tried Bob um, Mm-hmm."


def test_segment_words_keep_cased_punctuated_tokens_with_word_timing():
    seg = _seg("Sophie", 6.0, 9.0, "So AI in schools? It's <ST/>",
               [["so", 6.1, 6.3], ["ai", 6.3, 6.7], ["in", 6.7, 6.9], ["schools", 6.9, 7.4],
                ["it's", 7.4, 7.6]])  # fmt: skip
    words, unknown = nsf.segment_words(seg)
    assert [w[0] for w in words] == ["So", "AI", "in", "schools?", "It's"]
    assert words[3][1:] == pytest.approx((0.9, 1.4, True))  # relative to the segment start
    assert unknown == []


def test_a_stray_punctuation_token_joins_the_word_before():
    seg = _seg("A", 0.0, 2.0, "yes , right", [["yes", 0.1, 0.4], ["right", 0.6, 0.9]])
    words, _ = nsf.segment_words(seg)
    assert [w[0] for w in words] == ["yes,", "right"]
    assert all(w[3] for w in words)


def test_unmatched_tokens_are_interpolated_and_unknown_spans_reported():
    seg = _seg("A", 10.0, 13.0, "I 'cause <UNKNOWN/> know",
               [["i", 10.1, 10.2], ["<unknown/>", 10.5, 11.0], ["know", 11.2, 11.5]])  # fmt: skip
    words, unknown = nsf.segment_words(seg)
    assert [(w[0], w[3]) for w in words] == [("I", True), ("'cause", False), ("know", True)]
    assert words[1][1] == pytest.approx(0.2)
    assert words[1][2] == pytest.approx(1.2)
    assert unknown == [(10.5, 11.0)]


def test_chunks_touching_unintelligible_speech_are_dropped():
    rows = [{"start_s": 0.0, "end_s": 4.0}, {"start_s": 4.0, "end_s": 8.0}]
    assert nsf.drop_blocked(rows, [(5.0, 5.5)]) == [rows[0]]


def test_dev_set_2_is_never_downloaded():
    with pytest.raises(ValueError, match="challenge only"):
        nsf.download(nsf.NotsofarConfig(), "dev_set/240415.2_dev")


def test_build_rows_slice_every_device_and_skip_dead_ones(tmp_path):
    import soundfile as sf

    version = "train_set/test"
    meeting = tmp_path / nsf.subset_root(version) / "MTG_1"
    for dev, level in (("sc_a", 0.3), ("sc_b", 0.2), ("sc_dead", 0.0)):
        (meeting / dev).mkdir(parents=True)
        # 2 s of speech, 1 s of silence, repeated: chunk_bounds cuts in the silences
        audio = np.tile(np.r_[np.full(2 * SR, level), np.zeros(SR)], 4).astype(np.float32)
        sf.write(meeting / dev / "ch0.wav", audio, SR)
    segs = [_seg("A" if i % 2 == 0 else "B", 3.0 * i, 3.0 * i + 2.0, f"word{i} more{i}.",
                 [[f"word{i}", 3.0 * i + 0.2, 3.0 * i + 0.8], [f"more{i}", 3.0 * i + 1.0, 3.0 * i + 1.6]])
            for i in range(4)]  # fmt: skip
    (meeting / "gt_transcription.json").write_text(json.dumps(segs))
    ncfg = nsf.NotsofarConfig(local_dir=str(tmp_path))
    cfg = ShortChunkConfig(min_s=(3.0, 3.0), max_s=(7.0, 7.0))
    with mock.patch.object(nsf, "download", return_value=tmp_path / nsf.subset_root(version)):
        rows = nsf.build_notsofar_rows(ncfg, cfg, version, max_speakers=4, seed=0)
    assert {r["group"] for r in rows} == {"notsofar:MTG_1:sc_a", "notsofar:MTG_1:sc_b"}
    for r in rows:
        assert (tmp_path / r["recording"]).exists()  # relative to local_dir
    expected = sorted(w for i in range(4) for w in (f"word{i}", f"more{i}."))
    for dev in ("sc_a", "sc_b"):  # every word lands in exactly one chunk of each device
        got = [w for r in rows if r["group"].endswith(dev)
               for _, t in parse_turns(r["target"]) for w in t.split()]  # fmt: skip
        assert sorted(got) == expected


def test_notsofar_config_defaults_match_the_base_config():
    from omegaconf import OmegaConf

    from scripts.speaker_asr.config import CONFIG_DIR

    section = OmegaConf.load(CONFIG_DIR / "config.yaml").notsofar
    for f in fields(nsf.NotsofarConfig):
        assert section[f.name] == f.default, f.name
    assert section.enabled is False


def test_short_notsofar_preset_is_short_plus_notsofar():
    from scripts.speaker_asr.config import load_config

    short = load_config(["+experiment=short"])
    both = load_config(["+experiment=short_notsofar"])
    assert both.notsofar.enabled
    assert both.short_chunks.enabled
    assert both.context.enabled
    assert both.model.model_id == short.model.model_id
    assert both.training.learning_rate == short.training.learning_rate
    assert both.hub_model_id != short.hub_model_id
