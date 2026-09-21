"""Tests for DatasetLoader column normalization."""

from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Audio, ClassLabel, Dataset
from omegaconf import OmegaConf

from scripts.train import DatasetLoader


def _make_cfg(datasets, sample_rate=16000, num_proc=1, epoch_expansion=None, epochs=1):
    data = {
        "datasets": datasets,
        "sample_rate": sample_rate,
        "dataset_cache_dir": None,
        "num_proc": num_proc,
    }
    if epoch_expansion is not None:
        data["epoch_expansion"] = epoch_expansion
    return OmegaConf.create({"data": data, "training": {"seed": 42, "num_train_epochs": epochs}})


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
    def _durations(seconds):
        """Fake rows carrying a float `audio_duration` column, as
        mythicinfinity/libriheavy ships. Text is a/b/c/... so assertions can
        name surviving rows after _prepare prunes the bound column away."""
        n = 16000
        return Dataset.from_dict(
            {
                "audio": [{"array": np.zeros(n, dtype=np.float32), "sampling_rate": 16000}]
                * len(seconds),
                "text": list("abcdefghij")[: len(seconds)],
                "audio_duration": list(seconds),
            }
        ).cast_column("audio", Audio(sampling_rate=16000))

    @staticmethod
    def _ds(sources, class_label: bool):
        """Rows tagged by `source`, as a plain string column or — like real
        Gigaspeech — as a ClassLabel whose rows are integer ids.

        `class_label=True` is the case that made the original bug silent: the
        column holds integer codes, not the label strings the datasets-server
        statistics endpoint renders, so comparing the configured strings
        straight against them matched nothing and dropped 0 of 910,140 rows.
        """
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

    _GS_ROWS: ClassVar[list[str]] = ["youtube", "audiobook", "podcast", "audiobook"]

    def test_classlabel_column_holds_ints_not_strings(self):
        """The property that made the original bug silent."""
        ds = self._ds(self._GS_ROWS, class_label=True)
        assert ds["source"] == [2, 0, 1, 0]
        assert ds.features["source"].str2int("audiobook") == 0

    def test_classlabel_values_resolve_before_filtering(self):
        ds = self._ds(self._GS_ROWS, class_label=True)
        feature = ds.features["source"]
        wanted = {feature.str2int(v) for v in ["audiobook"]}
        assert wanted == {0}
        out = ds.filter(lambda v: v not in wanted, input_columns="source")
        assert len(out) == 2
        assert out["text"] == ["a", "c"]

    def test_naive_string_compare_on_classlabel_drops_nothing(self):
        """Regression guard: this is precisely what used to happen."""
        ds = self._ds(self._GS_ROWS, class_label=True)
        out = ds.filter(lambda v: v not in {"audiobook", "podcast"}, input_columns="source")
        assert len(out) == len(ds), "if this passes, the int/str mismatch is real"

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

    def test_no_match_fails_loudly(self):
        """Excluding something that is not there is a config bug, not a no-op.

        Was a warning on this branch; `main` made it raise, which is the
        stronger reading of the same argument -- failing costs seconds, while
        passing means a full training run on the mix you thought you had
        filtered, discovered only at eval.
        """
        fake = self._ds(["youtube", "podcast"], class_label=False)
        cfg = self._cfg(values=["nonexistent-tier"])
        with pytest.raises(ValueError, match="matched 0 of"):
            _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)

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
        with pytest.raises(ValueError, match="needs 'column' plus at least one of"):
            _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)

    def test_numeric_above_bound_excludes_long_rows(self):
        """`above` drops rows EXCEEDING the bound -- an exclusion bound, not a
        keep-ceiling. Added for LibriHeavy, where 19.6% of rows exceed the
        collator's 19.0s cap and were discarded *after* the 600K
        target_samples cap had already been applied, so the source delivered
        482K rows against a nominal 600K."""
        fake = self._durations([0.5, 5.0, 14.0, 19.0, 19.5, 30.0])
        cfg = self._cfg()
        cfg["exclude_where"] = {"column": "audio_duration", "above": 19.0}
        out = _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)
        # Asserted on `text`, not `audio_duration`: _prepare prunes to
        # keep_cols after filtering, so the bound column is gone by then.
        assert out["text"] == ["a", "b", "c", "d"], "19.0 is inclusive; >19.0 drops"

    def test_numeric_below_bound_excludes_short_rows(self):
        fake = self._durations([0.5, 5.0, 14.0])
        cfg = self._cfg()
        cfg["exclude_where"] = {"column": "audio_duration", "below": 0.8}
        out = _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)
        assert out["text"] == ["b", "c"]

    def test_numeric_bound_matching_nothing_still_fails_loudly(self):
        """Same contract as the values path: a filter that drops nothing is a
        config bug, not a legitimate no-op."""
        fake = self._durations([1.0, 2.0])
        cfg = self._cfg()
        cfg["exclude_where"] = {"column": "audio_duration", "above": 99.0}
        with pytest.raises(ValueError, match="matched 0 of 2 rows"):
            _prepare(DatasetLoader(_make_cfg([cfg])), cfg, fake)


class TestEpochExpansion:
    """`epoch_expansion: N` folds N logical epochs into one physical pass so
    that CAPPED sources spend the extra budget on rows they have not seen,
    instead of replaying the same subset N times.

    The bug it fixes: _resample_to_target runs once inside load(), so
    `num_train_epochs: 2` iterates an identical 600K subset twice while the
    rest of the pool is never touched -- measured at ~285K unused LibriHeavy
    rows and ~338K unused CommonVoice rows on the real mix.
    """

    @staticmethod
    def _rows(n, prefix="r"):
        return Dataset.from_dict(
            {
                "audio": [{"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}] * n,
                "text": [f"{prefix}{i}" for i in range(n)],
            }
        ).cast_column("audio", Audio(sampling_rate=16000))

    def _load(self, fake, cfg_entry, expansion, epochs=1):
        cfg = _make_cfg([cfg_entry], epoch_expansion=expansion, epochs=epochs)
        loader = DatasetLoader(cfg)
        with patch("scripts.train.load_dataset", return_value=fake):
            train, _ = loader.load()
        return train

    def test_capped_source_spends_expansion_on_fresh_rows_first(self):
        """Pool 80, cap 50, expansion 2 -> target 100. All 80 unique rows are
        drawn before any repeat-padding; without expansion only 50 are."""
        fake = self._rows(80)
        entry = {
            "path": "fake/big",
            "audio_column": "audio",
            "text_column": "text",
            "target_samples": 50,
            "train_splits": ["train"],
            "eval_splits": [],
        }
        train = self._load(fake, entry, expansion=2)
        assert len(train) == 100, "total rows should equal cap x expansion"
        assert len(set(train["text"])) == 80, "every eligible row should appear"

        baseline = self._load(fake, entry, expansion=1)
        assert len(baseline) == 50
        assert len(set(baseline["text"])) == 50

    def test_uncapped_source_is_repeated_verbatim(self):
        """A source already at natural size has no unused rows, so repeating
        is exactly what a second Trainer epoch would have done."""
        fake = self._rows(7)
        entry = {
            "path": "fake/small",
            "audio_column": "audio",
            "text_column": "text",
            "train_splits": ["train"],
            "eval_splits": [],
        }
        train = self._load(fake, entry, expansion=3)
        assert len(train) == 21
        assert len(set(train["text"])) == 7

    def test_expansion_preserves_relative_mix_share(self):
        """Every source is multiplied by the same factor, so per-step mix
        share is unchanged -- that is what makes this safe to do without
        re-deriving the caps."""
        fake = self._rows(80)
        capped = {
            "path": "fake/big",
            "audio_column": "audio",
            "text_column": "text",
            "target_samples": 50,
            "train_splits": ["train"],
            "eval_splits": [],
        }
        uncapped = {
            "path": "fake/small",
            "audio_column": "audio",
            "text_column": "text",
            "train_splits": ["train"],
            "eval_splits": [],
        }
        sizes = {}
        for exp in (1, 2):
            cfg = _make_cfg([capped, uncapped], epoch_expansion=exp)
            loader = DatasetLoader(cfg)
            with patch("scripts.train.load_dataset", side_effect=[fake, self._rows(20, "s")]):
                train, _ = loader.load()
            sizes[exp] = len(train)
        assert sizes[2] == 2 * sizes[1], "both sources must scale together"

    def test_expansion_with_multiple_epochs_is_rejected(self):
        """They compound: expansion 2 x num_train_epochs 2 is four epochs of
        exposure, which is worse than either intent and silent."""
        fake = self._rows(10)
        entry = {
            "path": "fake/x",
            "audio_column": "audio",
            "text_column": "text",
            "train_splits": ["train"],
            "eval_splits": [],
        }
        with pytest.raises(ValueError, match="compound"):
            self._load(fake, entry, expansion=2, epochs=2)

    def test_default_is_inert(self):
        fake = self._rows(9)
        entry = {
            "path": "fake/x",
            "audio_column": "audio",
            "text_column": "text",
            "train_splits": ["train"],
            "eval_splits": [],
        }
        train = self._load(fake, entry, expansion=None)
        assert len(train) == 9
