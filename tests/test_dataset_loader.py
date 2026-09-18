"""Tests for DatasetLoader column normalization."""

from unittest.mock import patch

import numpy as np
import pytest
from datasets import Audio, Dataset
from omegaconf import OmegaConf

from scripts.train import DatasetLoader


def _make_cfg(datasets, sample_rate=16000, num_proc=1):
    return OmegaConf.create(
        {
            "data": {
                "datasets": datasets,
                "sample_rate": sample_rate,
                "dataset_cache_dir": None,
                "num_proc": num_proc,
            },
            "training": {"seed": 42},
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


class TestColumnPruning:
    def test_source_columns_outside_audio_text_are_dropped(self):
        fake = _fake_dataset(audio_seconds=1.0, text="hi", duration=1.0, speaker="spk1")
        cfg = {"path": "fake/dataset", "audio_column": "audio", "text_column": "text"}
        ds = _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)

        assert set(ds.column_names) == {"audio", "text"}


class TestTextCaseColumn:
    """`text_case` declares a source's casing policy for _normalize_label.

    It has to survive _prepare_split's column pruning to reach the collator.
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
        ds = _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)

        assert "_text_case" in ds.column_names, "pruned before reaching the collator"
        assert ds[0]["_text_case"] == policy

    def test_column_absent_when_undeclared(self):
        """Sources with no policy keep the legacy per-row heuristic."""
        fake = _fake_dataset(audio_seconds=1.0, text="hi")
        cfg = {"path": "fake/dataset", "audio_column": "audio", "text_column": "text"}
        ds = _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)

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
        loader = DatasetLoader(_make_cfg([cfg]))
        with pytest.raises(ValueError, match="text_case must be"):
            _prepare(loader, cfg, fake)
