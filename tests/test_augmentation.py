"""Tests for RIR augmentation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from tiny_audio.augmentation import RIRAugmentation


@pytest.fixture
def fake_rir_dir():
    """Directory containing one synthetic RIR (impulse + short decay)."""
    with tempfile.TemporaryDirectory() as d:
        rir = torch.zeros(1, 1600)  # 100ms @ 16kHz
        rir[0, 0] = 1.0
        rir[0, 100:200] = torch.linspace(0.5, 0, 100)
        torchaudio.save(str(Path(d) / "fake.wav"), rir, 16000)
        yield d


class TestRIRAugmentation:
    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as d, pytest.raises(ValueError, match="No .wav files"):
            RIRAugmentation(rir_dir=d)

    def test_output_shape_and_dtype_preserved(self, fake_rir_dir):
        aug = RIRAugmentation(rir_dir=fake_rir_dir, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype

    def test_passthrough_at_prob_zero(self, fake_rir_dir):
        aug = RIRAugmentation(rir_dir=fake_rir_dir, prob=0.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        assert np.array_equal(aug(audio), audio)

    def test_modifies_audio_at_prob_one(self, fake_rir_dir):
        aug = RIRAugmentation(rir_dir=fake_rir_dir, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert not np.allclose(out, audio)

    def test_resamples_rir_to_target_rate(self):
        with tempfile.TemporaryDirectory() as d:
            rir = torch.zeros(1, 4800)  # 100ms @ 48kHz
            rir[0, 0] = 1.0
            torchaudio.save(str(Path(d) / "fake_48k.wav"), rir, 48000)
            aug = RIRAugmentation(rir_dir=d, sample_rate=16000, prob=1.0)
            audio = np.random.randn(16000).astype(np.float32) * 0.1
            out = aug(audio)
            assert out.shape == audio.shape

    def test_amplitude_does_not_clip(self, fake_rir_dir):
        aug = RIRAugmentation(rir_dir=fake_rir_dir, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        out = aug(audio)
        assert np.abs(out).max() <= 1.0
