"""Tests for DatasetLoader column normalization, especially the duration column."""

from unittest.mock import patch

import pytest
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


class TestDurationColumnNormalization:
    def test_duration_column_renamed_when_specified(self):
        """If duration_column is set, the source-named column is renamed to 'duration'."""
        import numpy as np
        from datasets import Audio, Dataset

        rows = {
            "audio": [{"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}],
            "transcript": ["hello"],
            "audio_len_secs": [1.0],
        }
        fake = Dataset.from_dict(rows).cast_column("audio", Audio(sampling_rate=16000))

        cfg = _make_cfg(
            [
                {
                    "path": "fake/dataset",
                    "audio_column": "audio",
                    "text_column": "transcript",
                    "duration_column": "audio_len_secs",
                    "train_splits": ["train"],
                    "eval_splits": ["validation"],
                }
            ]
        )
        loader = DatasetLoader(cfg)

        with patch("scripts.train.load_dataset", return_value=fake):
            ds = loader._prepare_split(cfg.data.datasets[0], "train")

        assert "duration" in ds.column_names
        assert "audio_len_secs" not in ds.column_names
        assert ds[0]["duration"] == pytest.approx(1.0)

    def test_duration_already_named_correctly_kept(self):
        import numpy as np
        from datasets import Audio, Dataset

        rows = {
            "audio": [{"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}],
            "text": ["hi"],
            "duration": [1.0],
        }
        fake = Dataset.from_dict(rows).cast_column("audio", Audio(sampling_rate=16000))
        cfg = _make_cfg(
            [
                {
                    "path": "fake/dataset",
                    "audio_column": "audio",
                    "text_column": "text",
                    "train_splits": ["train"],
                    "eval_splits": ["validation"],
                }
            ]
        )
        loader = DatasetLoader(cfg)
        with patch("scripts.train.load_dataset", return_value=fake):
            ds = loader._prepare_split(cfg.data.datasets[0], "train")

        assert "duration" in ds.column_names
        assert ds[0]["duration"] == pytest.approx(1.0)

    def test_duration_computed_from_audio_when_missing(self):
        """If neither duration nor duration_column is present, compute from audio array length."""
        import numpy as np
        from datasets import Audio, Dataset

        # Sample is exactly 2.5 seconds at 16kHz
        rows = {
            "audio": [{"array": np.zeros(40000, dtype=np.float32), "sampling_rate": 16000}],
            "text": ["hello"],
        }
        fake = Dataset.from_dict(rows).cast_column("audio", Audio(sampling_rate=16000))
        cfg = _make_cfg(
            [
                {
                    "path": "fake/dataset",
                    "audio_column": "audio",
                    "text_column": "text",
                    "train_splits": ["train"],
                    "eval_splits": ["validation"],
                }
            ]
        )
        loader = DatasetLoader(cfg)
        with patch("scripts.train.load_dataset", return_value=fake):
            ds = loader._prepare_split(cfg.data.datasets[0], "train")

        assert "duration" in ds.column_names
        assert ds[0]["duration"] == pytest.approx(2.5, abs=0.01)
