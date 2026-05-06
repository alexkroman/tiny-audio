"""Tests for RIR and noise augmentation.

RIR tests bypass pool generation by injecting a pre-built ``rirs=`` pool, so
they don't pay the multi-second pyroomacoustics generation cost in unit
tests. Noise tests exercise torch-audiomentations on CPU directly. Corpus
loading is exercised against synthetic WAVs written to ``tmp_path``.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from tiny_audio.augmentation import (
    GainPerturbationAugmentation,
    NoiseAugmentation,
    RIRAugmentation,
)


def _synthetic_rir(length: int = 1600) -> torch.Tensor:
    rir = torch.zeros(length, dtype=torch.float32)
    rir[0] = 1.0
    rir[100:200] = torch.linspace(0.5, 0, 100)
    norm = torch.linalg.norm(rir)
    return rir / norm


@pytest.fixture
def fake_rirs():
    return [_synthetic_rir() for _ in range(4)]


class TestGainPerturbationAugmentation:
    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="min_gain_db"):
            GainPerturbationAugmentation(min_gain_db=6.0, max_gain_db=-6.0)

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError, match="prob"):
            GainPerturbationAugmentation(prob=1.5)

    def test_passthrough_at_prob_zero(self):
        aug = GainPerturbationAugmentation(prob=0.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        assert np.array_equal(aug(audio), audio)

    def test_zero_db_range_is_identity(self):
        aug = GainPerturbationAugmentation(min_gain_db=0.0, max_gain_db=0.0, prob=1.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        np.testing.assert_allclose(aug(audio), audio, rtol=1e-6)

    def test_dtype_and_shape_preserved(self):
        aug = GainPerturbationAugmentation(prob=1.0)
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype

    def test_peak_clamped_below_one(self):
        aug = GainPerturbationAugmentation(min_gain_db=20.0, max_gain_db=20.0, prob=1.0)
        audio = np.full(16000, 0.5, dtype=np.float32)
        out = aug(audio)
        assert float(np.abs(out).max()) <= 1.0 + 1e-6

    def test_empty_audio_passthrough(self):
        aug = GainPerturbationAugmentation(prob=1.0)
        empty = np.zeros(0, dtype=np.float32)
        out = aug(empty)
        assert out.shape == empty.shape
        assert out.dtype == empty.dtype


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


class TestSilenceInjection:
    """Tests for silence-injection corpus filtering.

    Silence injection pairs audio with an empty transcript. If the audio
    were drawn from MUSAN/speech, this would teach the model 'real human
    speech → emit nothing' — the opposite of what an ASR system should do.
    ``sample_noise_only`` must skip any path under a ``speech`` subdirectory.
    Noise *mixing* (cocktail-party robustness) should still use the full pool.
    """

    def _write_musan_like(self, root, sample_rate: int = 16000):
        """Mimic MUSAN's directory layout with speech/ + noise/ + music/ subdirs."""
        rng = np.random.default_rng(0)
        for sub in ("speech", "noise", "music"):
            d = root / sub
            d.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                arr = rng.standard_normal(sample_rate * 2).astype(np.float32) * 0.1
                sf.write(str(d / f"{sub}_{i}.wav"), arr, sample_rate)

    def test_sample_noise_only_skips_speech_subdir(self, tmp_path):
        self._write_musan_like(tmp_path)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        sampled_paths: list = []
        original = aug._read_noise_segment

        def probe(path, n_target):
            sampled_paths.append(path)
            return original(path, n_target)

        aug._read_noise_segment = probe  # type: ignore[method-assign]
        for _ in range(100):
            aug.sample_noise_only(8000)
        assert sampled_paths, "no paths were sampled"
        assert all("speech" not in p.parts for p in sampled_paths)

    def test_sample_noise_only_returns_none_when_only_speech(self, tmp_path):
        # If the configured corpus contains only speech/, silence injection
        # has no safe pool — return None so the caller can skip injection
        # rather than emit a speech-paired-with-empty sample.
        rng = np.random.default_rng(0)
        (tmp_path / "speech").mkdir()
        for i in range(2):
            arr = rng.standard_normal(16000 * 2).astype(np.float32) * 0.1
            sf.write(str(tmp_path / "speech" / f"s_{i}.wav"), arr, 16000)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        assert aug.sample_noise_only(16000) is None

    def test_mix_corpus_noise_still_uses_full_pool(self, tmp_path):
        # Noise *mixing* (cocktail-party robustness) should still draw from
        # the full pool, including speech/ — only silence-injection skips it.
        self._write_musan_like(tmp_path)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        sampled_paths: list = []
        original = aug._read_noise_segment

        def probe(path, n_target):
            sampled_paths.append(path)
            return original(path, n_target)

        aug._read_noise_segment = probe  # type: ignore[method-assign]
        rng = np.random.default_rng(0)
        for _ in range(100):
            audio = rng.standard_normal(8000).astype(np.float32) * 0.1
            aug._mix_corpus_noise(audio)
        assert any("speech" in p.parts for p in sampled_paths), (
            "mixing should still draw from speech/ for cocktail-party robustness"
        )


class TestRIRBehavior:
    """Tests for RIR convolution behavior."""

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
