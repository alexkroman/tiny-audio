"""Tests for RIR augmentation."""

import numpy as np
import pytest
import torch
from datasets import Audio, Dataset

from tiny_audio.augmentation import RIRAugmentation


def _synthetic_rir(length: int = 1600) -> np.ndarray:
    rir = np.zeros(length, dtype=np.float32)
    rir[0] = 1.0
    rir[100:200] = np.linspace(0.5, 0, 100, dtype=np.float32)
    return rir


@pytest.fixture
def fake_rir_dataset():
    """In-memory HF dataset of 3 synthetic RIRs at 16kHz."""
    rirs = [
        {"audio": {"array": _synthetic_rir(), "sampling_rate": 16000, "path": None}}
        for _ in range(3)
    ]
    return Dataset.from_list(rirs).cast_column("audio", Audio(sampling_rate=16000))


@pytest.fixture
def fake_rir_dataset_48k():
    """In-memory HF dataset at 48kHz to exercise the resampling path."""
    rir = _synthetic_rir(length=4800)
    return Dataset.from_list(
        [{"audio": {"array": rir, "sampling_rate": 48000, "path": None}}]
    ).cast_column("audio", Audio(sampling_rate=48000))


class TestRIRAugmentation:
    def test_silent_rirs_are_filtered_and_raise(self):
        # All-zero RIRs have zero norm and should be skipped, leaving none usable.
        silent = Dataset.from_list(
            [
                {
                    "audio": {
                        "array": np.zeros(1600, dtype=np.float32),
                        "sampling_rate": 16000,
                        "path": None,
                    }
                }
            ]
        ).cast_column("audio", Audio(sampling_rate=16000))
        with pytest.raises(ValueError, match="No usable RIRs"):
            RIRAugmentation(dataset=silent)

    def test_output_shape_and_dtype_preserved(self, fake_rir_dataset):
        aug = RIRAugmentation(dataset=fake_rir_dataset, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype

    def test_passthrough_at_prob_zero(self, fake_rir_dataset):
        aug = RIRAugmentation(dataset=fake_rir_dataset, prob=0.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        assert np.array_equal(aug(audio), audio)

    def test_modifies_audio_at_prob_one(self, fake_rir_dataset):
        aug = RIRAugmentation(dataset=fake_rir_dataset, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert not np.allclose(out, audio)

    def test_resamples_rir_to_target_rate(self, fake_rir_dataset_48k):
        aug = RIRAugmentation(dataset=fake_rir_dataset_48k, sample_rate=16000, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape

    def test_amplitude_does_not_clip(self, fake_rir_dataset):
        aug = RIRAugmentation(dataset=fake_rir_dataset, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        out = aug(audio)
        assert np.abs(out).max() <= 1.0

    def test_rirs_are_preloaded(self, fake_rir_dataset):
        aug = RIRAugmentation(dataset=fake_rir_dataset, prob=1.0)
        assert len(aug.rirs) == 3
        assert all(isinstance(r, torch.Tensor) for r in aug.rirs)
