"""Tests for scripts.turn_aware.config: presets, PoolConfig mapping, pool signatures."""

from __future__ import annotations

import pytest

from scripts.turn_aware.config import (
    load_config,
    pool_config,
    pool_is_current,
    pool_signature,
    read_signature,
    signature_diff,
    write_signature,
)

MINED = "mazesmazes/turn-end-detection-mined-pauses"


@pytest.mark.parametrize(
    ("preset", "pool_dir", "copies", "strip", "extra", "hub"),
    [
        ("v1", "data/turn_aware", 1, False, None, "mazesmazes/tiny-audio-turn-aware-qwen3-asr"),
        (
            "v2a",
            "data/turn_aware_v2a",
            3,
            False,
            MINED,
            "mazesmazes/tiny-audio-turn-aware-qwen3-asr-v2a",
        ),
        (
            "v2",
            "data/turn_aware_v2",
            3,
            True,
            MINED,
            "mazesmazes/tiny-audio-turn-aware-qwen3-asr-v2",
        ),
        ("eval", "data/turn_aware_eval", 1, False, MINED, None),
        (
            "v3",
            "data/turn_aware_v3",
            3,
            True,
            MINED,
            "mazesmazes/tiny-audio-turn-aware-qwen3-asr-v3",
        ),
        (
            "v4",
            "data/turn_aware_v4",
            3,
            True,
            MINED,
            "mazesmazes/tiny-audio-turn-aware-qwen3-asr-v4",
        ),
    ],
)
def test_presets_resolve_to_their_recipe(preset, pool_dir, copies, strip, extra, hub):
    cfg = load_config([f"+experiment={preset}"])
    pc = pool_config(cfg)
    assert cfg.data.pool_dir == pool_dir
    assert dict(pc.hold_copies)["payload_cut"] == copies
    assert pc.strip_hold_punct is strip
    assert cfg.pool.extra == extra
    assert cfg.hub_model_id == hub


def test_base_config_is_scratch_identity_with_no_push():
    cfg = load_config()
    assert cfg.data.pool_dir == "data/turn_aware_dev"
    assert cfg.hub_model_id is None


def test_pool_config_types_match_dataclass():
    pc = pool_config(
        load_config(["pool.fire_tail_s=[0.4,1.0]", "+pool.hold_copies.trailing_frag=2"])
    )
    assert pc.fire_tail_s == (0.4, 1.0)
    assert dict(pc.hold_copies) == {"payload_cut": 3, "word_cut": 2, "trailing_frag": 2}


def test_signature_is_per_split(tmp_path):
    v1 = pool_signature(load_config(["+experiment=v1"]))
    v2 = pool_signature(load_config(["+experiment=v2"]))
    for split in ("train", "validation"):
        (tmp_path / f"{split}.parquet").write_bytes(b"")
        write_signature(tmp_path, split, v1)
    write_signature(tmp_path, "train", v2)  # train rebuilt under v2 ...
    assert pool_is_current(tmp_path, "train", v2)
    assert not pool_is_current(tmp_path, "validation", v2)  # ... validation is still v1
    assert read_signature(tmp_path, "validation") == v1
    assert not pool_is_current(tmp_path, "test", v1)  # no manifest


def test_signature_diff_names_changed_keys():
    diff = signature_diff(
        pool_signature(load_config(["+experiment=v1"])),
        pool_signature(load_config(["+experiment=v2"])),
    )
    assert "pool.strip_hold_punct: False -> True" in diff
    assert "pool.hold_copies.payload_cut: 1 -> 3" in diff
    assert any(d.startswith("pool.extra: None -> ") for d in diff)


def test_base_config_pool_matches_dataclass_defaults():
    """The YAML is what runs; this keeps PoolConfig's defaults (used by tests) the same recipe."""
    from scripts.turn_aware.data import PoolConfig

    assert pool_config(load_config()) == PoolConfig()


def test_transcript_cache_location_is_not_part_of_the_signature():
    a = pool_signature(load_config(["pool.transcript_cache=/tmp/a", "pool.transcript_repo=x/a"]))
    b = pool_signature(load_config(["pool.transcript_cache=/tmp/b", "pool.transcript_repo=null"]))
    assert a == b


def test_v3_and_eval_filter_mined_pauses_by_label():
    for preset in ("v3", "v4", "eval"):
        assert list(load_config([f"+experiment={preset}"]).pool.extra_labels) == ["incomplete"]
    for preset in ("v1", "v2a", "v2"):
        assert load_config([f"+experiment={preset}"]).pool.extra_labels is None


def test_only_v4_trains_with_context_on_every_example():
    assert pool_config(load_config(["+experiment=v4"])).ctx_prob == 1.0
    for preset in ("v1", "v2", "v3"):
        assert pool_config(load_config([f"+experiment={preset}"])).ctx_prob == 0.5
