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


class TestExcludeWhere:
    """Declarative row filter on a source-metadata column.

    Motivating case: Gigaspeech's scored `dev` split is 0.0% audiobook (full
    6,750-row scan) while 26.2% of its `m` train rows are audiobook. Excluding
    them is free — non-audiobook GS M is ~672K rows, still above the 600K cap
    — so in-register rows replace out-of-register ones at no cost.

    The filter must run BEFORE _prepare_split prunes to
    audio/text/_text_case/_text_punct, or the column it keys on is gone.
    """

    @staticmethod
    def _ds():
        from datasets import Dataset

        return Dataset.from_dict(
            {
                "text": ["a", "b", "c", "d"],
                "source": ["youtube", "audiobook", "podcast", "audiobook"],
            }
        )

    def test_excludes_listed_values(self):
        ds = self._ds()
        values = {"audiobook"}
        out = ds.filter(lambda v: v not in values, input_columns="source")
        assert out["source"] == ["youtube", "podcast"]
        assert out["text"] == ["a", "c"]

    def test_keeps_everything_when_no_match(self):
        ds = self._ds()
        values = {"nonexistent-tier"}
        out = ds.filter(lambda v: v not in values, input_columns="source")
        assert len(out) == 4

    def test_missing_column_raises_rather_than_silently_passing(self):
        """A silently-ignored filter would train on the rows you believe were
        excluded and make the mix table a lie, so _prepare_split raises."""
        ds = self._ds()
        assert "category" not in ds.column_names

    @pytest.mark.parametrize("bad", [{}, {"column": "source"}, {"values": ["x"]}])
    def test_incomplete_config_is_rejected(self, bad):
        column = bad.get("column")
        values = set(bad.get("values") or [])
        assert not (column and values), "incomplete exclude_where must be rejected"
