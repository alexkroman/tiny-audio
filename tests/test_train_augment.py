"""Tests for the URGENT-matched training-batch degradation."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from scripts.train_augment import (
    BatchAugmenter,
    NoiseCorpus,
    bandlimit,
    clip,
    colored_noise,
    mix_at_snr,
)


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.3) -> npt.NDArray[np.float32]:
    t = np.arange(int(seconds * 16000)) / 16000
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rows(n: int) -> tuple[list[npt.NDArray[np.float32]], list[dict[str, Any]]]:
    audios = [_tone(2.0, freq=200.0 + 37 * i) for i in range(n)]
    return audios, [{"text": f"row {i}"} for i in range(n)]


def _high_share(audio: npt.NDArray[Any], above_hz: float) -> float:
    power = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(audio.size, d=1 / 16000)
    return float(power[freqs > above_hz].sum() / power.sum())


class TestMixAtSnr:
    """mix_at_snr puts the noise exactly `snr_db` under the speech."""

    @pytest.mark.parametrize("snr_db", [-5.0, 0.0, 15.0])
    def test_added_signal_sits_at_the_requested_snr(self, snr_db: float) -> None:
        speech = _tone(1.0, amp=0.05)
        noise = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
        added = mix_at_snr(speech, noise, snr_db) - speech
        measured = 10 * np.log10(np.mean(speech**2) / np.mean(added**2))
        assert measured == pytest.approx(snr_db, abs=0.05)

    def test_never_clips(self) -> None:
        loud = np.ones(16000, dtype=np.float32)
        assert np.abs(mix_at_snr(loud, loud, -5.0)).max() <= 1.0


class TestColoredNoise:
    """colored_noise is unit RMS and tilts toward low frequencies as the exponent grows."""

    def test_unit_rms(self) -> None:
        noise = colored_noise(16000, 1.0, np.random.default_rng(0))
        assert float(np.sqrt(np.mean(noise**2))) == pytest.approx(1.0, rel=1e-3)

    def test_brown_has_less_high_frequency_power_than_white(self) -> None:
        def share(exponent: float) -> float:
            return _high_share(colored_noise(16000, exponent, np.random.default_rng(1)), 4000)

        assert share(2.0) < share(0.0) / 10


class TestBandlimit:
    """bandlimit removes everything above the cutoff and keeps the length."""

    @pytest.mark.parametrize("cutoff", [4000.0, 5512.0, 7000.0])
    def test_nothing_left_above_the_cutoff(self, cutoff: float) -> None:
        white = colored_noise(16000, 0.0, np.random.default_rng(0)) * 0.1
        out = bandlimit(white, 16000, cutoff)
        assert out.shape == white.shape
        assert _high_share(out, cutoff + 300) < 0.01
        assert _high_share(white, cutoff + 300) > 0.05

    def test_cutoff_at_or_above_nyquist_is_a_no_op(self) -> None:
        tone = _tone(0.5)
        np.testing.assert_array_equal(bandlimit(tone, 16000, 8000.0), tone)


class TestClip:
    """clip flattens the requested share of samples and keeps the level."""

    @pytest.mark.parametrize("fraction", [0.007, 0.037, 0.09])
    def test_clipped_share_and_level(self, fraction: float) -> None:
        speech = (np.random.default_rng(0).standard_normal(16000) * 0.1).astype(np.float32)
        out = clip(speech, fraction)
        at_peak = np.isclose(np.abs(out), np.abs(out).max(), rtol=1e-4).mean()
        assert at_peak == pytest.approx(fraction, abs=0.003)
        assert np.sqrt(np.mean(out**2)) == pytest.approx(np.sqrt(np.mean(speech**2)), rel=0.02)

    def test_silence_is_left_alone(self) -> None:
        silence = np.zeros(100, dtype=np.float32)
        np.testing.assert_array_equal(clip(silence, 0.05), silence)


class TestNoiseCorpus:
    """NoiseCorpus draws excerpts of the requested length at the requested rate."""

    def test_excerpts_are_resampled_and_fitted(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        (tmp_path / "noise").mkdir()
        sf.write(tmp_path / "noise" / "a.wav", rng.standard_normal(8000) * 0.1, 8000)
        sf.write(tmp_path / "noise" / "b.wav", rng.standard_normal(48000) * 0.1, 16000)
        corpus = NoiseCorpus(str(tmp_path), "noise/*.wav")
        assert len(corpus.files()) == 2
        for _ in range(4):
            assert corpus.excerpt(24000, 16000, rng).shape == (24000,)

    def test_empty_corpus_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no noise files"):
            NoiseCorpus(str(tmp_path)).files()


class TestBatchAugmenter:
    """Rows are degraded independently; targets never change."""

    def test_disabled_returns_the_batch_untouched(self) -> None:
        audios, features = _rows(3)
        aug = BatchAugmenter(16000)
        assert not aug.enabled
        assert aug(audios, features) == (audios, features)

    def test_every_degradation_keeps_targets_and_lengths(self, tmp_path: Path) -> None:
        sf.write(tmp_path / "hum.wav", np.random.default_rng(0).standard_normal(16000) * 0.1, 16000)
        aug = BatchAugmenter(
            16000,
            noise_prob=1.0,
            corpus=NoiseCorpus(str(tmp_path)),
            bandlimit_prob=1.0,
            clip_prob=1.0,
            seed=0,
        )
        audios, features = _rows(4)
        out_audio, out_features = aug(audios, features)
        assert out_features is features
        for before, after in zip(audios, out_audio, strict=True):
            assert after.shape == before.shape
            assert not np.allclose(after, before)
            assert np.abs(after).max() <= 1.0

    @pytest.mark.parametrize("corpus", [False, True])
    def test_noise_sources_cover_every_branch(self, tmp_path: Path, corpus: bool) -> None:
        sf.write(tmp_path / "n.wav", np.random.default_rng(0).standard_normal(4000) * 0.1, 16000)
        aug = BatchAugmenter(
            16000, noise_prob=1.0, corpus=NoiseCorpus(str(tmp_path)) if corpus else None, seed=1
        )
        for _ in range(20):
            assert aug.noise(8000, [_tone(0.3)]).shape == (8000,)
        assert aug.noise(8000, []).shape == (8000,)

    def test_each_process_draws_its_own_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        aug = BatchAugmenter(16000, noise_prob=0.5, seed=0)
        monkeypatch.setattr("os.getpid", lambda: 100)
        first = aug._rng().random(4)
        monkeypatch.setattr("os.getpid", lambda: 101)
        assert not np.allclose(aug._rng().random(4), first)
