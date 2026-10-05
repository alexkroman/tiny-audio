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


def test_v4_borrows_distractor_context_and_context_matched_targets():
    cfg = load_config(["+experiment=v4"])
    assert pool_config(cfg).ctx_distractor_prob == 0.2
    assert cfg.pool.target_with_context is True
    for preset in ("v1", "v2", "v3", "eval"):
        c = load_config([f"+experiment={preset}"])
        assert pool_config(c).ctx_distractor_prob == 0.0
        assert c.pool.target_with_context is False


class TestTrainingArgumentsMac:
    """Qwen3-ASR recipes spawn workers and checkpoint activations on a Mac."""

    @staticmethod
    def _args(monkeypatch, mps: bool, **training):
        import torch
        import transformers.training_args
        from omegaconf import OmegaConf

        from scripts.turn_aware.config import training_arguments

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(transformers.training_args, "is_torch_mps_available", lambda: mps)
        cfg = OmegaConf.create(
            {"training": {"output_dir": "/tmp/ta-test", "dataloader_num_workers": 8, **training}}
        )
        return training_arguments(cfg, remove_unused_columns=False)

    def test_mps_spawns_and_checkpoints(self, monkeypatch):
        args = self._args(monkeypatch, mps=True)
        assert args.dataloader_multiprocessing_context == "spawn"
        assert args.gradient_checkpointing
        assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}

    def test_explicit_context_wins(self, monkeypatch):
        args = self._args(monkeypatch, mps=True, dataloader_multiprocessing_context="forkserver")
        assert args.dataloader_multiprocessing_context == "forkserver"

    def test_no_mps_keeps_default(self, monkeypatch):
        args = self._args(monkeypatch, mps=False)
        assert args.dataloader_multiprocessing_context is None
        assert not args.gradient_checkpointing


class TestCutGradIntoFrozenAudioTower:
    """Backprop stops at the projector input only while the tower is frozen."""

    @staticmethod
    def _model(tower_trainable: bool):
        import torch

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_tower = torch.nn.Linear(4, 4)
                self.multi_modal_projector = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.multi_modal_projector(self.audio_tower(x))

        model = torch.nn.Module()
        model.model = Inner()
        model.model.audio_tower.requires_grad_(tower_trainable)
        return model

    def test_frozen_tower_output_is_detached(self):
        import torch

        from scripts.turn_aware.model import cut_grad_into_frozen_audio_tower

        model = self._model(tower_trainable=False)
        assert cut_grad_into_frozen_audio_tower(model)
        x = torch.randn(2, 4, requires_grad=True)  # what the conv2d1 hook does
        model.model(x).sum().backward()
        assert x.grad is None
        assert model.model.multi_modal_projector.weight.grad is not None

    def test_trainable_tower_is_left_alone(self):
        from scripts.turn_aware.model import cut_grad_into_frozen_audio_tower

        assert not cut_grad_into_frozen_audio_tower(self._model(tower_trainable=True))
