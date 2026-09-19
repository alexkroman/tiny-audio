"""Tests for DatasetLoader column normalization."""

from unittest.mock import patch

import numpy as np
import pytest
from datasets import Audio, ClassLabel, Dataset
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
    def _ds(sources, class_label: bool):
        """Four rows tagged by `source`, as a plain string column or — like
        real Gigaspeech — as a ClassLabel whose rows are integer ids."""
        n = 16000
        ds = Dataset.from_dict(
            {
                "audio": [{"array": np.zeros(n, dtype=np.float32), "sampling_rate": 16000}]
                * len(sources),
                "text": list("abcd")[: len(sources)],
                "source": list(sources),
            }
        ).cast_column("audio", Audio(sampling_rate=16000))
        if class_label:
            # Real Gigaspeech label order; ids are 0/1/2, not the names.
            ds = ds.cast_column("source", ClassLabel(names=["audiobook", "podcast", "youtube"]))
        return ds

    @staticmethod
    def _cfg(values=("audiobook",)):
        return {
            "path": "fake/gigaspeech",
            "audio_column": "audio",
            "text_column": "text",
            "exclude_where": {"column": "source", "values": list(values)},
        }

    @pytest.mark.parametrize("class_label", [False, True], ids=["string", "classlabel"])
    def test_excludes_listed_values(self, class_label):
        """Gigaspeech stores `source` as a ClassLabel, so rows hold ints, not
        the label strings the datasets-server statistics endpoint renders.
        Comparing rows against the names silently dropped 0/910140 rows."""
        fake = self._ds(["youtube", "audiobook", "podcast", "audiobook"], class_label)
        cfg = self._cfg()
        ds = _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)

        assert ds["text"] == ["a", "c"], "audiobook rows survived the filter"

    def test_keeps_everything_when_no_match(self):
        fake = self._ds(["youtube", "podcast"], class_label=False)
        cfg = self._cfg(values=["nonexistent-tier"])
        ds = _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)

        assert len(ds) == 2

    def test_unknown_class_label_fails_loudly(self):
        """A name the ClassLabel does not define can only ever match nothing,
        so it is a config bug, not an empty result."""
        fake = self._ds(["youtube", "audiobook"], class_label=True)
        cfg = self._cfg(values=["audio_book"])
        with pytest.raises(ValueError, match="not a label of"):
            _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)

    def test_missing_column_raises_rather_than_silently_passing(self):
        """A silently-ignored filter would train on the rows you believe were
        excluded and make the mix table a lie, so _prepare_split raises."""
        fake = self._ds(["youtube", "audiobook"], class_label=False)
        cfg = self._cfg()
        cfg["exclude_where"]["column"] = "category"
        with pytest.raises(ValueError, match="not in fake/gigaspeech"):
            _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)

    @pytest.mark.parametrize("bad", [{}, {"column": "source"}, {"values": ["x"]}])
    def test_incomplete_config_is_rejected(self, bad):
        fake = self._ds(["youtube", "audiobook"], class_label=False)
        cfg = self._cfg()
        cfg["exclude_where"] = bad
        with pytest.raises(ValueError, match="needs both"):
            _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)
