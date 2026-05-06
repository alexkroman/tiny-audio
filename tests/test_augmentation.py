"""Tests for noise augmentation.

Noise tests exercise torch-audiomentations on CPU directly. Corpus loading
is exercised against synthetic WAVs written to ``tmp_path``.
"""

import numpy as np
import pytest
import soundfile as sf

from tiny_audio.augmentation import NoiseAugmentation


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
        aug = NoiseAugmentation(prob=1.0, min_snr_db=60.0, max_snr_db=60.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        signal_rms = float(np.sqrt(np.mean(audio**2)))
        diff_rms = float(np.sqrt(np.mean((out - audio) ** 2)))
        assert diff_rms < signal_rms * 0.05

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
    """Silence injection must not draw from MUSAN/speech subdir."""

    def _write_musan_like(self, root, sample_rate: int = 16000):
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
        rng = np.random.default_rng(0)
        (tmp_path / "speech").mkdir()
        for i in range(2):
            arr = rng.standard_normal(16000 * 2).astype(np.float32) * 0.1
            sf.write(str(tmp_path / "speech" / f"s_{i}.wav"), arr, 16000)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        assert aug.sample_noise_only(16000) is None

    def test_mix_corpus_noise_still_uses_full_pool(self, tmp_path):
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


class TestNoiseMixingPeakProtection:
    def _write_noise_corpus(self, root, n: int = 3, sample_rate: int = 16000):
        rng = np.random.default_rng(0)
        for i in range(n):
            arr = rng.standard_normal(sample_rate * 2).astype(np.float32) * 0.5
            sf.write(str(root / f"noise_{i}.wav"), arr, sample_rate)

    def test_mixed_output_within_unit_range(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self._write_noise_corpus(tmp_path)
        aug = NoiseAugmentation(
            prob=1.0, corpus_path=str(tmp_path), min_snr_db=-20.0, max_snr_db=-20.0
        )
        audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.7
        out = aug(audio)
        peak = float(np.abs(out).max())
        assert peak <= 1.0 + 1e-6
