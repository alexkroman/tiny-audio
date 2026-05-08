"""Tests for RIR and noise augmentation.

Both classes are thin wrappers around audiomentations transforms. Corpus
loading is exercised against synthetic WAVs written to ``tmp_path``.
"""

import numpy as np
import pytest
import soundfile as sf

from tiny_audio.augmentation import NoiseAugmentation, RIRAugmentation


def _write_synthetic_rir_corpus(root, n: int, sample_rate: int = 16000):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rir = np.zeros(800, dtype=np.float32)
        rir[0] = 1.0
        rir[50:150] = np.linspace(0.4, 0.0, 100, dtype=np.float32)
        sf.write(str(root / f"rir_{i}.wav"), rir, sample_rate)


def _write_synthetic_noise_corpus(root, n: int, sample_rate: int = 16000):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        noise = rng.standard_normal(sample_rate * 2).astype(np.float32) * 0.1
        sf.write(str(root / f"noise_{i}.wav"), noise, sample_rate)


class TestRIRAugmentation:
    def test_requires_corpus_path(self):
        with pytest.raises(ValueError, match="corpus_path"):
            RIRAugmentation()

    def test_missing_corpus_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RIRAugmentation(corpus_path=str(tmp_path / "does-not-exist"), prob=1.0)

    def test_output_shape_and_dtype_preserved(self, tmp_path):
        _write_synthetic_rir_corpus(tmp_path, n=3)
        aug = RIRAugmentation(corpus_path=str(tmp_path), prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype

    def test_passthrough_at_prob_zero(self, tmp_path):
        _write_synthetic_rir_corpus(tmp_path, n=3)
        aug = RIRAugmentation(corpus_path=str(tmp_path), prob=0.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        np.testing.assert_array_equal(aug(audio), audio)

    def test_modifies_audio_at_prob_one(self, tmp_path):
        _write_synthetic_rir_corpus(tmp_path, n=3)
        aug = RIRAugmentation(corpus_path=str(tmp_path), prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        assert not np.allclose(aug(audio), audio)

    def test_empty_audio_passthrough(self, tmp_path):
        _write_synthetic_rir_corpus(tmp_path, n=3)
        aug = RIRAugmentation(corpus_path=str(tmp_path), prob=1.0)
        empty = np.zeros(0, dtype=np.float32)
        out = aug(empty)
        assert out.shape == empty.shape
        assert out.dtype == empty.dtype

    def test_loads_from_multiple_corpus_paths(self, tmp_path):
        _write_synthetic_rir_corpus(tmp_path / "a", n=2)
        _write_synthetic_rir_corpus(tmp_path / "b", n=3)
        aug = RIRAugmentation(corpus_path=[str(tmp_path / "a"), str(tmp_path / "b")], prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape


class TestNoiseAugmentation:
    def test_corpus_modifies_audio_at_prob_one(self, tmp_path):
        _write_synthetic_noise_corpus(tmp_path, n=3)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype
        assert not np.allclose(out, audio)

    def test_corpus_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path / "does-not-exist"))

    def test_empty_audio_passthrough(self, tmp_path):
        _write_synthetic_noise_corpus(tmp_path, n=2)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        empty = np.zeros(0, dtype=np.float32)
        out = aug(empty)
        assert out.shape == empty.shape
        assert out.dtype == empty.dtype

    def test_loads_from_multiple_corpus_paths(self, tmp_path):
        _write_synthetic_noise_corpus(tmp_path / "a", n=2)
        _write_synthetic_noise_corpus(tmp_path / "b", n=3)
        aug = NoiseAugmentation(prob=1.0, corpus_path=[str(tmp_path / "a"), str(tmp_path / "b")])
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape

    def test_gaussian_layer_applied_on_top_of_corpus(self, tmp_path):
        _write_synthetic_noise_corpus(tmp_path, n=3)
        aug = NoiseAugmentation(
            prob=1.0,
            corpus_path=str(tmp_path),
            gaussian_min_snr_db=20.0,
            gaussian_max_snr_db=40.0,
        )
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert not np.allclose(out, audio)

    def test_full_chain_composes(self, tmp_path):
        bg = tmp_path / "bg"
        events = tmp_path / "events"
        _write_synthetic_noise_corpus(bg, n=3)
        _write_synthetic_noise_corpus(events, n=3)
        aug = NoiseAugmentation(
            prob=1.0,
            corpus_path=str(bg),
            gaussian_min_snr_db=20.0,
            gaussian_max_snr_db=40.0,
            short_noises_corpus_path=str(events),
            short_noises_prob=1.0,
            eq_prob=1.0,
            clipping_prob=1.0,
            bandlimit_prob=1.0,
        )
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype
        assert not np.allclose(out, audio)

    def test_short_noises_requires_separate_path(self, tmp_path):
        # short_noises_prob > 0 without a path is silently a no-op (skipped),
        # not an error — keeps the path optional from a config standpoint.
        _write_synthetic_noise_corpus(tmp_path, n=3)
        aug = NoiseAugmentation(
            prob=1.0,
            corpus_path=str(tmp_path),
            short_noises_prob=1.0,  # but no short_noises_corpus_path
        )
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        # Verify the chain runs without error and produces output the same
        # shape as input (background mix is what's modifying the audio here).
        assert aug(audio).shape == audio.shape

    def test_eq_only(self):
        aug = NoiseAugmentation(prob=0.0, eq_prob=1.0, eq_min_db=4.0, eq_max_db=4.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert not np.allclose(out, audio)

    def test_bandlimit_modifies_audio(self):
        # OneOf{LPF, BPF} fires; assert audio is modified. Either branch
        # (LPF cutoff 3-7.5 kHz, BPF telephony 200-4000 Hz) attenuates
        # white-noise content enough to be detectable.
        aug = NoiseAugmentation(prob=0.0, bandlimit_prob=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        out = aug(audio)
        assert out.shape == audio.shape
        assert not np.allclose(out, audio)


class TestSilenceInjection:
    def test_sample_noise_only_returns_clip(self, tmp_path):
        _write_synthetic_noise_corpus(tmp_path, n=3)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        clip = aug.sample_noise_only(8000)
        assert clip is not None
        assert clip.shape == (8000,)
        assert clip.dtype == np.float32

    def test_sample_noise_only_skips_speech_subdir(self, tmp_path):
        rng = np.random.default_rng(0)
        for sub in ("speech", "noise", "music"):
            d = tmp_path / sub
            d.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                arr = rng.standard_normal(16000 * 2).astype(np.float32) * 0.1
                sf.write(str(d / f"{sub}_{i}.wav"), arr, 16000)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        sampled_paths: list = []
        original = aug._read_noise_segment

        def probe(path, n_target):
            sampled_paths.append(path)
            return original(path, n_target)

        aug._read_noise_segment = probe  # type: ignore[method-assign]
        for _ in range(50):
            aug.sample_noise_only(8000)
        assert sampled_paths
        assert all("speech" not in p.parts for p in sampled_paths)

    def test_sample_noise_only_returns_none_when_only_speech(self, tmp_path):
        rng = np.random.default_rng(0)
        (tmp_path / "speech").mkdir()
        for i in range(2):
            arr = rng.standard_normal(16000 * 2).astype(np.float32) * 0.1
            sf.write(str(tmp_path / "speech" / f"s_{i}.wav"), arr, 16000)
        aug = NoiseAugmentation(prob=1.0, corpus_path=str(tmp_path))
        assert aug.sample_noise_only(16000) is None

    def test_sample_noise_only_returns_none_without_corpus(self):
        aug = NoiseAugmentation(prob=0.0, eq_prob=1.0)
        assert aug.sample_noise_only(16000) is None
