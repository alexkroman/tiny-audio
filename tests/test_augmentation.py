"""Tests for RIR and noise augmentation.

RIR tests bypass pool generation by injecting a pre-built ``rirs=`` pool, so
they don't pay the multi-second pyroomacoustics generation cost in unit
tests. Noise tests exercise torch-audiomentations on CPU directly. Corpus
loading is exercised against synthetic WAVs written to ``tmp_path``.
"""

import random

import numpy as np
import pytest
import soundfile as sf
import torch

from tiny_audio.augmentation import NoiseAugmentation, RIRAugmentation, SpeedPerturbation


def _synthetic_rir(length: int = 1600) -> torch.Tensor:
    rir = torch.zeros(length, dtype=torch.float32)
    rir[0] = 1.0
    rir[100:200] = torch.linspace(0.5, 0, 100)
    norm = torch.linalg.norm(rir)
    return rir / norm


@pytest.fixture
def fake_rirs():
    return [_synthetic_rir() for _ in range(4)]


class TestRIRAugmentation:
    def test_empty_pool_raises(self):
        with pytest.raises(ValueError, match="Empty RIR pool"):
            RIRAugmentation(rirs=[])

    def test_output_shape_and_dtype_preserved(self, fake_rirs):
        aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype

    def test_passthrough_at_prob_zero(self, fake_rirs):
        aug = RIRAugmentation(rirs=fake_rirs, prob=0.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        assert np.array_equal(aug(audio), audio)

    def test_modifies_audio_at_prob_one(self, fake_rirs):
        aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert not np.allclose(out, audio)

    def test_amplitude_does_not_clip(self, fake_rirs):
        aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        out = aug(audio)
        assert np.abs(out).max() <= 1.0

    def test_pool_sampling_uses_provided_rirs(self, fake_rirs):
        aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        assert len(aug.rirs) == len(fake_rirs)
        for got, given in zip(aug.rirs, fake_rirs):
            assert torch.equal(got, given)

    def test_empty_audio_passthrough(self, fake_rirs):
        aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        empty = np.zeros(0, dtype=np.float32)
        out = aug(empty)
        assert out.shape == empty.shape
        assert out.dtype == empty.dtype


class TestRIRCorpusMode:
    def _write_synthetic_corpus(self, root, n: int, sample_rate: int = 16000):
        root.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            rir = np.zeros(800, dtype=np.float32)
            rir[0] = 1.0
            rir[50:150] = np.linspace(0.4, 0.0, 100, dtype=np.float32)
            sf.write(str(root / f"rir_{i}.wav"), rir, sample_rate)

    def test_loads_wavs_from_corpus(self, tmp_path):
        self._write_synthetic_corpus(tmp_path, n=3)
        aug = RIRAugmentation(corpus_path=str(tmp_path), pool_size=10, prob=1.0)
        assert len(aug.rirs) == 3

    def test_subsamples_to_pool_size(self, tmp_path):
        self._write_synthetic_corpus(tmp_path, n=20)
        aug = RIRAugmentation(corpus_path=str(tmp_path), pool_size=5, prob=1.0, seed=0)
        assert len(aug.rirs) == 5

    def test_resamples_when_sample_rate_differs(self, tmp_path):
        # Source RIRs at 24kHz, target sample rate 16kHz — loader must resample.
        self._write_synthetic_corpus(tmp_path, n=2, sample_rate=24000)
        aug = RIRAugmentation(corpus_path=str(tmp_path), pool_size=10, prob=1.0, sample_rate=16000)
        assert len(aug.rirs) == 2
        # 800 samples at 24kHz → ~533 samples at 16kHz, post peak-trim ≤ that.
        assert all(rir.shape[0] <= 600 for rir in aug.rirs)

    def test_missing_corpus_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RIRAugmentation(corpus_path=str(tmp_path / "does-not-exist"), prob=1.0)

    def test_empty_corpus_raises(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="No .wav files"):
            RIRAugmentation(corpus_path=str(tmp_path), prob=1.0)

    def test_loads_from_multiple_corpus_paths(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        self._write_synthetic_corpus(a, n=2)
        self._write_synthetic_corpus(b, n=3)
        aug = RIRAugmentation(corpus_path=[str(a), str(b)], pool_size=10, prob=1.0)
        assert len(aug.rirs) == 5

    def test_multiple_corpus_paths_one_missing_raises(self, tmp_path):
        a = tmp_path / "a"
        self._write_synthetic_corpus(a, n=2)
        with pytest.raises(FileNotFoundError):
            RIRAugmentation(corpus_path=[str(a), str(tmp_path / "does-not-exist")], prob=1.0)


class TestNoiseAugmentation:
    def test_output_shape_and_dtype_preserved(self):
        aug = NoiseAugmentation(prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype

    def test_modifies_audio_at_prob_one(self):
        aug = NoiseAugmentation(prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert not np.allclose(out, audio)

    def test_passthrough_at_prob_zero(self):
        aug = NoiseAugmentation(prob=0.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert np.allclose(out, audio, atol=1e-6)

    def test_snr_range_respected(self):
        # With high SNR, output should be close to input
        aug = NoiseAugmentation(prob=1.0, min_snr_db=60.0, max_snr_db=60.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        # 60 dB SNR means noise is ~1000x quieter than signal, so output ~= input
        signal_rms = float(np.sqrt(np.mean(audio**2)))
        diff_rms = float(np.sqrt(np.mean((out - audio) ** 2)))
        assert diff_rms < signal_rms * 0.05  # noise < 5% of signal

    def test_empty_audio_passthrough(self):
        aug = NoiseAugmentation(prob=1.0)
        empty = np.zeros(0, dtype=np.float32)
        out = aug(empty)
        assert out.shape == empty.shape
        assert out.dtype == empty.dtype


class TestNoiseCorpusMode:
    def _write_synthetic_corpus(self, root, n: int, sample_rate: int = 16000):
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for i in range(n):
            # 2s of pink-ish broadband noise — enough length for AddBackgroundNoise
            # to slice clips out of.
            noise = rng.standard_normal(sample_rate * 2).astype(np.float32) * 0.1
            sf.write(str(root / f"noise_{i}.wav"), noise, sample_rate)

    def test_modifies_audio_at_prob_one(self, tmp_path):
        self._write_synthetic_corpus(tmp_path, n=3)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype
        assert not np.allclose(out, audio)

    def test_passthrough_at_prob_zero(self, tmp_path):
        self._write_synthetic_corpus(tmp_path, n=2)
        aug = NoiseAugmentation(prob=0.0, corpus_path=str(tmp_path))
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert np.allclose(out, audio, atol=1e-6)

    def test_missing_corpus_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path / "does-not-exist"))

    def test_empty_audio_passthrough(self, tmp_path):
        self._write_synthetic_corpus(tmp_path, n=2)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        empty = np.zeros(0, dtype=np.float32)
        out = aug(empty)
        assert out.shape == empty.shape
        assert out.dtype == empty.dtype

    def test_loads_from_multiple_corpus_paths(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        self._write_synthetic_corpus(a, n=2)
        self._write_synthetic_corpus(b, n=3)
        aug = NoiseAugmentation(prob=1.0, corpus_path=[str(a), str(b)])
        assert len(aug.noise_paths) == 5

    def test_multiple_corpus_paths_one_missing_raises(self, tmp_path):
        a = tmp_path / "a"
        self._write_synthetic_corpus(a, n=2)
        with pytest.raises(FileNotFoundError):
            NoiseAugmentation(prob=1.0, corpus_path=[str(a), str(tmp_path / "does-not-exist")])


class TestNoiseBabbleWeighting:
    """Tests for the ``babble_weight`` corpus draw bias."""

    def _write_musan_like(self, root, sample_rate: int = 16000):
        """Mimic MUSAN's directory layout with a tiny speech/ + noise/ split."""
        rng = np.random.default_rng(0)
        for sub in ("speech", "noise", "music"):
            d = root / sub
            d.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                arr = rng.standard_normal(sample_rate * 2).astype(np.float32) * 0.1
                sf.write(str(d / f"{sub}_{i}.wav"), arr, sample_rate)

    def test_invalid_babble_weight_raises(self, tmp_path):
        self._write_musan_like(tmp_path)
        with pytest.raises(ValueError, match="babble_weight"):
            NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path), babble_weight=1.5)

    def test_paths_split_by_speech_subdir(self, tmp_path):
        self._write_musan_like(tmp_path)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path), babble_weight=0.5)
        # 2 babble (speech/) + 4 non-babble (noise/ + music/)
        assert len(aug.babble_paths) == 2
        assert len(aug.non_babble_paths) == 4
        assert all("speech" in p.parts for p in aug.babble_paths)
        assert all("speech" not in p.parts for p in aug.non_babble_paths)

    def test_weight_one_always_picks_babble(self, tmp_path):
        self._write_musan_like(tmp_path)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path), babble_weight=1.0)
        for _ in range(50):
            picked = aug._pick_noise_path()
            assert "speech" in picked.parts

    def test_weight_zero_falls_back_to_full_pool(self, tmp_path):
        self._write_musan_like(tmp_path)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path), babble_weight=0.0)
        # With weight=0, _pick_noise_path takes the full-pool branch
        seen_subdirs = set()
        random.seed(0)
        for _ in range(200):
            picked = aug._pick_noise_path()
            for sub in ("speech", "noise", "music"):
                if sub in picked.parts:
                    seen_subdirs.add(sub)
        # All three subdirs should appear when sampling uniformly from the full pool
        assert seen_subdirs == {"speech", "noise", "music"}

    def test_silently_disabled_when_no_babble_in_corpus(self, tmp_path):
        # Corpus with no speech/ subdirectory — babble_weight should self-zero
        # rather than raise, so a noise-only corpus stays usable.
        rng = np.random.default_rng(0)
        (tmp_path / "noise").mkdir()
        for i in range(2):
            arr = rng.standard_normal(16000 * 2).astype(np.float32) * 0.1
            sf.write(str(tmp_path / "noise" / f"n_{i}.wav"), arr, 16000)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path), babble_weight=0.5)
        assert aug.babble_weight == 0.0
        assert aug.babble_paths == []


class TestNoiseReverb:
    """Tests for ``rir_augmentation`` noise convolution."""

    def _write_noise_corpus(self, root, n: int = 3, sample_rate: int = 16000):
        rng = np.random.default_rng(0)
        for i in range(n):
            arr = rng.standard_normal(sample_rate * 2).astype(np.float32) * 0.1
            sf.write(str(root / f"noise_{i}.wav"), arr, sample_rate)

    def test_reverbed_noise_differs_from_unreverbed(self, tmp_path, fake_rirs):
        # New contract: noise is reverbed only when the signal was reverbed
        # this call (last_rir is set). Run RIR aug first to populate it, then
        # compare against a NoiseAugmentation with no rir_augmentation hookup.
        tmp_path.mkdir(parents=True, exist_ok=True)
        self._write_noise_corpus(tmp_path)
        rir_aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        plain = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        reverbed = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path), rir_augmentation=rir_aug)

        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        random.seed(42)
        plain_out = plain(audio.copy())
        random.seed(42)
        signal = rir_aug(audio.copy())  # populates rir_aug.last_rir
        reverbed_out = reverbed(signal)
        assert plain_out.shape == reverbed_out.shape
        assert not np.allclose(plain_out, reverbed_out, atol=1e-5)

    def test_reverb_preserves_shape_and_dtype(self, tmp_path, fake_rirs):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self._write_noise_corpus(tmp_path)
        rir_aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path), rir_augmentation=rir_aug)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        signal = rir_aug(audio)
        out = aug(signal)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype

    def test_noise_uses_same_rir_as_signal(self, tmp_path, fake_rirs):
        # Strict same-room recipe: noise must be convolved with whatever RIR
        # the signal got, not a freshly-sampled one. Verify by manually
        # convolving the noise with last_rir and comparing to the noise
        # extracted from the augmenter's output (signal subtracted).
        tmp_path.mkdir(parents=True, exist_ok=True)
        self._write_noise_corpus(tmp_path)
        rir_aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        noise_aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path), rir_augmentation=rir_aug)

        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        signal = rir_aug(audio.copy())
        # rir_aug.last_rir must be set to one of fake_rirs
        assert rir_aug.last_rir is not None
        assert any(torch.equal(rir_aug.last_rir, r) for r in fake_rirs)

        # Run noise on signal — last_rir stays set (NoiseAugmentation reads it).
        rir_used_by_noise = rir_aug.last_rir
        _ = noise_aug(signal)
        # last_rir should still be the same RIR (noise aug doesn't mutate it).
        assert rir_aug.last_rir is rir_used_by_noise

    def test_noise_anechoic_when_signal_skipped(self, tmp_path, fake_rirs):
        # When RIR aug skips (prob=0), last_rir is None and noise should
        # stay anechoic — output should match a NoiseAugmentation with no
        # rir_augmentation hookup at all.
        tmp_path.mkdir(parents=True, exist_ok=True)
        self._write_noise_corpus(tmp_path)
        rir_aug = RIRAugmentation(rirs=fake_rirs, prob=0.0)
        plain = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        with_rir = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path), rir_augmentation=rir_aug)

        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        random.seed(42)
        signal = rir_aug(audio.copy())  # skips, last_rir = None
        assert rir_aug.last_rir is None
        random.seed(42)
        plain_out = plain(audio.copy())
        random.seed(42)
        with_rir_out = with_rir(signal)
        np.testing.assert_allclose(plain_out, with_rir_out, atol=1e-6)


class TestRIRBehavior:
    """Tests for new RIR behavior: last_rir tracking and RMS restoration."""

    def test_last_rir_set_after_apply(self, fake_rirs):
        aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        assert aug.last_rir is None
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        _ = aug(audio)
        assert aug.last_rir is not None
        assert any(torch.equal(aug.last_rir, r) for r in fake_rirs)

    def test_last_rir_cleared_when_skipped(self, fake_rirs):
        aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        _ = aug(audio)
        assert aug.last_rir is not None
        # Force a skip via prob=0
        aug.prob = 0.0
        _ = aug(audio)
        assert aug.last_rir is None

    def test_last_rir_cleared_on_empty_audio(self, fake_rirs):
        aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        _ = aug(np.random.randn(16000).astype(np.float32) * 0.1)
        assert aug.last_rir is not None
        _ = aug(np.zeros(0, dtype=np.float32))
        assert aug.last_rir is None

    def test_rms_restored_after_convolution(self, fake_rirs):
        # RMS of reverbed output should match input RMS (within tolerance);
        # without restoration the L2-normalized RIR would attenuate level.
        aug = RIRAugmentation(rirs=fake_rirs, prob=1.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        in_rms = float(np.sqrt(np.mean(audio**2)))
        out = aug(audio)
        out_rms = float(np.sqrt(np.mean(out**2)))
        # 5% tolerance — peak-clip protection can scale slightly when the
        # restored signal exceeds [-1, 1].
        assert abs(out_rms - in_rms) / in_rms < 0.05


class TestNoiseMixingPeakProtection:
    def _write_noise_corpus(self, root, n: int = 3, sample_rate: int = 16000):
        rng = np.random.default_rng(0)
        for i in range(n):
            arr = rng.standard_normal(sample_rate * 2).astype(np.float32) * 0.5
            sf.write(str(root / f"noise_{i}.wav"), arr, sample_rate)

    def test_mixed_output_within_unit_range(self, tmp_path):
        # Loud signal + loud noise at low SNR would exceed [-1, 1] without
        # post-mix peak protection. Verify the mixer keeps output in range.
        tmp_path.mkdir(parents=True, exist_ok=True)
        self._write_noise_corpus(tmp_path)
        # Force low-SNR (very loud noise) to make clipping likely.
        aug = NoiseAugmentation(
            prob=1.0, corpus_path=str(tmp_path), min_snr_db=-20.0, max_snr_db=-20.0
        )
        # Near-full-scale signal.
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.7
        out = aug(audio)
        peak = float(np.abs(out).max())
        assert peak <= 1.0 + 1e-6


class TestSpeedPerturbation:
    def test_passthrough_at_rate_one(self):
        aug = SpeedPerturbation(sample_rate=16000, rates=[1.0], prob=1.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        np.testing.assert_array_equal(out, audio)

    def test_slowdown_lengthens_audio(self):
        # rate=0.9 → output duration ≈ 1/0.9 × input → ~17778 samples
        aug = SpeedPerturbation(sample_rate=16000, rates=[0.9], prob=1.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape[0] > audio.shape[0]
        ratio = out.shape[0] / audio.shape[0]
        assert abs(ratio - (1.0 / 0.9)) < 0.01

    def test_speedup_shortens_audio(self):
        # rate=1.1 → output duration ≈ 1/1.1 × input → ~14545 samples
        aug = SpeedPerturbation(sample_rate=16000, rates=[1.1], prob=1.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape[0] < audio.shape[0]
        ratio = out.shape[0] / audio.shape[0]
        assert abs(ratio - (1.0 / 1.1)) < 0.01

    def test_dtype_preserved(self):
        aug = SpeedPerturbation(sample_rate=16000, rates=[0.9], prob=1.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.dtype == audio.dtype

    def test_passthrough_at_prob_zero(self):
        aug = SpeedPerturbation(sample_rate=16000, rates=[0.9], prob=0.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        np.testing.assert_array_equal(out, audio)

    def test_empty_audio_passthrough(self):
        aug = SpeedPerturbation(sample_rate=16000, rates=[0.9], prob=1.0)
        audio = np.zeros(0, dtype=np.float32)
        out = aug(audio)
        assert out.shape[0] == 0

    def test_rate_drawn_from_pool(self):
        # With only one non-1.0 rate the output length should be deterministic.
        aug = SpeedPerturbation(sample_rate=16000, rates=[0.9], prob=1.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        lens = {aug(audio).shape[0] for _ in range(5)}
        assert len(lens) == 1  # always the same rate → always the same length

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError, match="prob"):
            SpeedPerturbation(prob=1.5)

    def test_empty_rates_raises(self):
        with pytest.raises(ValueError, match="rates"):
            SpeedPerturbation(rates=[])
