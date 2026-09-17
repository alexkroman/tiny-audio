"""Tests for DatasetLoader column normalization and duration handling."""

from unittest.mock import patch

import numpy as np
import pytest
from datasets import Audio, Dataset
from omegaconf import OmegaConf

from scripts.train import DatasetLoader


def _make_cfg(datasets, sample_rate=16000, num_proc=1, group_by_length=True):
    return OmegaConf.create(
        {
            "data": {
                "datasets": datasets,
                "sample_rate": sample_rate,
                "dataset_cache_dir": None,
                "num_proc": num_proc,
            },
            "training": {"seed": 42, "group_by_length": group_by_length},
        }
    )


def _fake_dataset(audio_seconds: float, **extra_cols):
    n = int(audio_seconds * 16000)
    rows = {
        "audio": [{"array": np.zeros(n, dtype=np.float32), "sampling_rate": 16000}],
        **{k: [v] for k, v in extra_cols.items()},
    }
    return Dataset.from_dict(rows).cast_column("audio", Audio(sampling_rate=16000))


def _prepare(loader, dataset_cfg, fake):
    with patch("scripts.train.load_dataset", return_value=fake):
        return loader._prepare_split(OmegaConf.create(dataset_cfg), "train")


class TestDurationColumnNormalization:
    def test_duration_column_renamed_when_specified(self):
        fake = _fake_dataset(audio_seconds=1.0, transcript="hello", audio_len_secs=1.0)
        cfg = {
            "path": "fake/dataset",
            "audio_column": "audio",
            "text_column": "transcript",
            "duration_column": "audio_len_secs",
        }
        loader = DatasetLoader(_make_cfg([cfg]))
        ds = _prepare(loader, cfg, fake)

        assert "duration" in ds.column_names
        assert "audio_len_secs" not in ds.column_names
        assert ds[0]["duration"] == pytest.approx(1.0)

    def test_duration_already_named_correctly_kept(self):
        fake = _fake_dataset(audio_seconds=1.0, text="hi", duration=1.0)
        cfg = {"path": "fake/dataset", "audio_column": "audio", "text_column": "text"}
        loader = DatasetLoader(_make_cfg([cfg]))
        ds = _prepare(loader, cfg, fake)

        assert "duration" in ds.column_names
        assert ds[0]["duration"] == pytest.approx(1.0)


class TestEnsureDuration:
    def test_computes_from_audio_when_missing(self):
        fake = _fake_dataset(audio_seconds=2.5, text="hello")
        cfg = {"path": "fake/dataset", "audio_column": "audio", "text_column": "text"}
        loader = DatasetLoader(_make_cfg([cfg]))
        ds = _prepare(loader, cfg, fake)
        assert "duration" not in ds.column_names

        ds = loader._ensure_duration(ds)

        assert "duration" in ds.column_names
        assert ds[0]["duration"] == pytest.approx(2.5, abs=0.01)

    def test_noop_when_duration_present(self):
        fake = _fake_dataset(audio_seconds=2.5, text="hi", duration=999.0)
        cfg = {"path": "fake/dataset", "audio_column": "audio", "text_column": "text"}
        loader = DatasetLoader(_make_cfg([cfg]))
        ds = _prepare(loader, cfg, fake)
        ds = loader._ensure_duration(ds)

        assert ds[0]["duration"] == pytest.approx(999.0)


class TestGroupByLengthDisabled:
    def test_duration_dropped_when_group_by_length_off(self):
        fake = _fake_dataset(audio_seconds=1.0, text="hi", duration=1.0)
        cfg = {"path": "fake/dataset", "audio_column": "audio", "text_column": "text"}
        loader = DatasetLoader(_make_cfg([cfg], group_by_length=False))
        ds = _prepare(loader, cfg, fake)

        assert "duration" not in ds.column_names


class TestTextCaseColumn:
    """`text_case` declares a source's casing policy for _normalize_label.

    It has to survive _prepare_split's column pruning to reach the collator —
    the same plumbing as `_allow_empty_label`.
    """

    @pytest.mark.parametrize("policy", ["mono", "cased"])
    def test_policy_is_attached_to_every_row(self, policy):
        fake = _fake_dataset(audio_seconds=1.0, text="hi")
        cfg = {
            "path": "fake/dataset",
            "audio_column": "audio",
            "text_column": "text",
            "text_case": policy,
        }
        ds = _prepare(DatasetLoader(_make_cfg([cfg], group_by_length=False)), cfg, fake)

        assert "_text_case" in ds.column_names, "pruned before reaching the collator"
        assert ds[0]["_text_case"] == policy

    def test_column_absent_when_undeclared(self):
        """Sources with no policy keep the legacy per-row heuristic."""
        fake = _fake_dataset(audio_seconds=1.0, text="hi")
        cfg = {"path": "fake/dataset", "audio_column": "audio", "text_column": "text"}
        ds = _prepare(DatasetLoader(_make_cfg([cfg], group_by_length=False)), cfg, fake)

        assert "_text_case" not in ds.column_names

    def test_unknown_policy_fails_loudly(self):
        """A typo must not silently fall back to the heuristic it overrides."""
        fake = _fake_dataset(audio_seconds=1.0, text="hi")
        cfg = {
            "path": "fake/dataset",
            "audio_column": "audio",
            "text_column": "text",
            "text_case": "Cased",
        }
        loader = DatasetLoader(_make_cfg([cfg], group_by_length=False))
        with pytest.raises(ValueError, match="text_case must be"):
            _prepare(loader, cfg, fake)
