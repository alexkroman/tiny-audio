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
    # Shuffle is applied after the cast, so the returned object is the shuffled
    # view rather than the cast one.
    assert out is loaded.cast_column.return_value.shuffle.return_value


def test_shuffle_is_on_by_default_with_a_fixed_seed(fake_load):
    """Test splits ship grouped by speaker/chapter/meeting, so first-N sampling
    is not corpus-representative: the first 100 rows of LibriSpeech clean/test
    cover 2 of 40 speakers. A fixed-seed shuffle buffer keeps selection
    deterministic while restoring coverage (measured: 2 -> 34 speakers).
    """
    _, loaded = fake_load
    ds_mod.load_eval_dataset("loquacious", "test")
    loaded.cast_column.return_value.shuffle.assert_called_once_with(
        seed=ds_mod.SHUFFLE_SEED, buffer_size=ds_mod.SHUFFLE_BUFFER_SIZE
    )


def test_shuffle_can_be_disabled_to_reproduce_old_numbers(fake_load):
    _, loaded = fake_load
    out = ds_mod.load_eval_dataset("loquacious", "test", shuffle=False)
    loaded.cast_column.return_value.shuffle.assert_not_called()
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
