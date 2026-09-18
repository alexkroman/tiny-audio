"""Tests for scripts.eval.datasets.load_eval_dataset (Hub calls mocked)."""

from unittest.mock import MagicMock

import pytest

from scripts.eval import datasets as ds_mod


@pytest.fixture
def fake_load(monkeypatch):
    loaded = MagicMock(name="dataset")
    load = MagicMock(return_value=loaded)
    monkeypatch.setattr(ds_mod, "load_dataset", load)
    return load, loaded


def test_unknown_dataset_lists_available_names():
    with pytest.raises(ValueError, match="Unknown dataset: nope") as excinfo:
        ds_mod.load_eval_dataset("nope", "test")
    for name in ds_mod.DATASET_REGISTRY:
        assert name in str(excinfo.value)


def test_registry_config_is_used_and_audio_is_cast(fake_load):
    load, loaded = fake_load
    cfg = ds_mod.DATASET_REGISTRY["loquacious"]
    out = ds_mod.load_eval_dataset("loquacious", "test")

    load.assert_called_once_with(cfg.path, cfg.config, split="test", streaming=True)
    field, audio = loaded.cast_column.call_args.args
    assert field == cfg.audio_field
    assert audio.sampling_rate == 16000
    assert out is loaded.cast_column.return_value


def test_config_override_wins(fake_load):
    load, _ = fake_load
    ds_mod.load_eval_dataset("loquacious", "dev", config_override="large")
    assert load.call_args.args[1] == "large"
    assert load.call_args.kwargs["split"] == "dev"


def test_dataset_without_config_omits_it(fake_load, monkeypatch):
    load, _ = fake_load
    monkeypatch.setitem(
        ds_mod.DATASET_REGISTRY,
        "plain",
        ds_mod.DatasetConfig(path="org/plain", audio_field="audio"),
    )
    ds_mod.load_eval_dataset("plain", "test")
    load.assert_called_once_with("org/plain", split="test", streaming=True)
