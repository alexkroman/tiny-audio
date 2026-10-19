"""Tests for the URGENT-matched training-batch degradation."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from scripts.train_augment import (
    MAX_RIR_SECONDS,
    AudioCorpus,
    BatchAugmenter,
    align_rir,
    bandlimit,
    clip,
    colored_noise,
    mix_at_snr,
    non_silent,
    phone,
    reverberate,
    room_response,
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


class TestNonSilent:
    """ESPnet's detect_non_silence: frames above 1% of the mean frame power are active."""

    def test_burst_is_active_and_silence_is_not(self) -> None:
        x = np.zeros(16000, dtype=np.float32)
        x[4096:8192] = _tone(4096 / 16000, amp=0.3)
        mask = non_silent(x)
        assert mask.shape == x.shape
        assert mask[5000:7000].all()
        assert not mask[:3000].any()
        assert not mask[10000:].any()

    def test_short_or_silent_clips_count_as_all_active(self) -> None:
        assert non_silent(np.ones(500, dtype=np.float32)).all()
        assert non_silent(np.zeros(4000, dtype=np.float32)).all()


class TestUrgentSnr:
    """mix_at_snr measures SNR over non-silent samples, as URGENT's mix_noise does."""

    def test_snr_ignores_pauses_and_noise_gaps(self) -> None:
        speech = np.zeros(32000, dtype=np.float32)
        speech[:16000] = _tone(1.0, amp=0.1)  # a 1 s pause after the speech
        noise = np.random.default_rng(0).standard_normal(32000).astype(np.float32)
        noise[8000:24000] = 0  # a bursty noise
        added = mix_at_snr(speech, noise, 5.0) - speech
        snr = 10 * np.log10(
            np.mean(speech[non_silent(speech)] ** 2) / np.mean(added[non_silent(added)] ** 2)
        )
        assert snr == pytest.approx(5.0, abs=0.05)


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


class TestUrgentBandlimit:
    """The 4 kHz limit is URGENT's resampling to 8 kHz and back, not a brickwall."""

    def test_resampled_limit_cuts_the_top_band_and_keeps_the_passband(self) -> None:
        white = colored_noise(32000, 0.0, np.random.default_rng(0)) * 0.1
        out = bandlimit(white, 16000, 4000.0)
        assert out.shape == white.shape
        assert _high_share(out, 4500.0) < 0.01

        def passband(a: npt.NDArray[np.float32]) -> float:  # ~100 Hz - 3 kHz
            return float((np.abs(np.fft.rfft(a))[200:6000] ** 2).sum())

        assert passband(out) / passband(white) == pytest.approx(1.0, rel=0.05)

    def test_other_cutoffs_stay_brickwall(self) -> None:
        white = colored_noise(32000, 0.0, np.random.default_rng(1)) * 0.1
        assert _high_share(bandlimit(white, 16000, 7000.0), 7000.0 + 1) < 1e-6


class TestClip:
    """clip flattens everything outside two quantiles (URGENT's scheme) and keeps the level."""

    @pytest.mark.parametrize(("low_q", "high_q"), [(0.0, 0.9), (0.05, 0.95), (0.1, 1.0)])
    def test_clipped_shares_and_level(self, low_q: float, high_q: float) -> None:
        speech = (np.random.default_rng(0).standard_normal(16000) * 0.1).astype(np.float32)
        out = clip(speech, low_q, high_q)
        assert np.isclose(out, out.min(), rtol=1e-4).mean() == pytest.approx(low_q, abs=0.003)
        assert np.isclose(out, out.max(), rtol=1e-4).mean() == pytest.approx(1 - high_q, abs=0.003)
        assert np.sqrt(np.mean(out**2)) == pytest.approx(np.sqrt(np.mean(speech**2)), rel=0.02)

    def test_silence_is_left_alone(self) -> None:
        silence = np.zeros(100, dtype=np.float32)
        np.testing.assert_array_equal(clip(silence, 0.05, 0.95), silence)


class TestReverb:
    """The synthetic room decays 60 dB in RT60; reverberated audio keeps length and level."""

    @pytest.mark.parametrize("rt60", [0.3, 0.8])
    def test_tail_falls_60_db_in_rt60(self, rt60: float) -> None:
        rir = room_response(rt60, 16000, np.random.default_rng(0))
        assert rir[0] == 1.0
        tail = rir[1:].astype(np.float64) ** 2
        frame = 400
        level = 10 * np.log10(np.convolve(tail, np.ones(frame) / frame, mode="valid") + 1e-30)
        start = level[0]
        at_rt60 = level[int(rt60 * 16000) - frame]
        assert start - at_rt60 == pytest.approx(60.0, abs=6.0)

    def test_reverberate_keeps_length_and_level(self) -> None:
        tone = _tone(1.0)
        out = reverberate(tone, room_response(0.6, 16000, np.random.default_rng(0)))
        assert out.shape == tone.shape
        assert np.sqrt(np.mean(out**2)) == pytest.approx(np.sqrt(np.mean(tone**2)), rel=0.02)
        assert not np.allclose(out, tone)


class TestPhone:
    """The phone channel cuts the low end and the top band, and applies the gain."""

    def test_highpass_lowpass_and_gain(self) -> None:
        white = colored_noise(32000, 0.0, np.random.default_rng(0)) * 0.05
        out = phone(white, 16000, highpass_hz=150.0, lowpass_hz=6000.0, gain_db=0.0)
        power = np.abs(np.fft.rfft(out)) ** 2
        freqs = np.fft.rfftfreq(out.size, d=1 / 16000)
        mid = power[(freqs > 1000) & (freqs < 4000)].mean()
        assert power[freqs > 7000].mean() < mid / 100  # 8th-order roll-off above 6 kHz
        assert power[freqs < 50].mean() < mid / 5  # high-pass below 150 Hz
        louder = phone(white, 16000, highpass_hz=150.0, lowpass_hz=6000.0, gain_db=6.0)
        ratio = np.sqrt(np.mean(louder**2) / np.mean(out**2))
        assert ratio == pytest.approx(10 ** (6 / 20), rel=0.01)

    def test_lowpass_at_nyquist_is_skipped(self) -> None:
        tone = _tone(0.5, freq=1000.0)
        out = phone(tone, 16000, highpass_hz=50.0, lowpass_hz=8000.0, gain_db=0.0)
        assert out.shape == tone.shape


class TestAudioCorpus:
    """AudioCorpus draws excerpts of the requested length at the requested rate."""

    def test_excerpts_are_resampled_and_fitted(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        (tmp_path / "noise").mkdir()
        sf.write(tmp_path / "noise" / "a.wav", rng.standard_normal(8000) * 0.1, 8000)
        sf.write(tmp_path / "noise" / "b.wav", rng.standard_normal(48000) * 0.1, 16000)
        corpus = AudioCorpus(str(tmp_path), "noise/*.wav")
        assert len(corpus.files()) == 2
        for _ in range(4):
            assert corpus.excerpt(24000, 16000, rng).shape == (24000,)

    def test_empty_corpus_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no audio files"):
            AudioCorpus(str(tmp_path)).files()


class TestMeasuredRooms:
    """Measured responses start at their direct path; the augmenter mixes in synthetic rooms."""

    def test_align_rir_starts_at_the_direct_path(self) -> None:
        rir = np.zeros(48000, dtype=np.float32)
        rir[800] = -0.5  # direct path after 50 ms of pre-delay
        rir[801:] = 0.01 * np.exp(-np.arange(47199) / 3000)
        out = align_rir(rir, 16000)
        assert int(np.argmax(np.abs(out))) == 16  # 1 ms of lead kept
        assert float(np.abs(out).max()) == pytest.approx(1.0)
        assert out.size == int(MAX_RIR_SECONDS * 16000)

    def test_room_draws_measured_and_synthetic(self, tmp_path: Path) -> None:
        marker = np.zeros(4000, dtype=np.float32)
        marker[0], marker[1] = 1.0, 0.77  # a second tap no synthetic room has
        sf.write(tmp_path / "room.wav", marker, 16000)
        aug = BatchAugmenter(
            16000, reverb_prob=1.0, reverb_corpus=AudioCorpus(str(tmp_path)), seed=0
        )
        # 16-bit PCM on disk, so compare loosely.
        measured = sum(np.isclose(aug.room(aug._rng())[1], 0.77, atol=1e-3) for _ in range(200))
        assert 120 < measured < 190  # ~ (1 - SYNTHETIC_REVERB_SHARE) of 200
        no_corpus = BatchAugmenter(16000, reverb_prob=1.0, seed=0)
        assert no_corpus.room(no_corpus._rng())[0] == 1.0


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
            corpus=AudioCorpus(str(tmp_path)),
            bandlimit_prob=0.5,
            clip_prob=0.5,
            reverb_prob=1.0,
            seed=0,
        )
        audios, features = _rows(4)
        out_audio, out_features = aug(audios, features)
        assert out_features is features
        for before, after in zip(audios, out_audio, strict=True):
            assert after.shape == before.shape
            assert not np.allclose(after, before)
            assert np.abs(after).max() <= 1.0

    def test_bandlimit_and_clipping_never_stack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After noise a row gets at most one of the two, as in URGENT."""
        calls: list[str] = []

        def fake_bandlimit(audio: Any, *_: object) -> Any:
            calls.append("bandlimit")
            return audio

        def fake_clip(audio: Any, *_: object) -> Any:
            calls.append("clip")
            return audio

        monkeypatch.setattr("scripts.train_augment.bandlimit", fake_bandlimit)
        monkeypatch.setattr("scripts.train_augment.clip", fake_clip)

        def fake_phone(audio: Any, *_: object) -> Any:
            calls.append("phone")
            return audio

        monkeypatch.setattr("scripts.train_augment.phone", fake_phone)
        aug = BatchAugmenter(16000, bandlimit_prob=0.4, clip_prob=0.4, phone_prob=0.2, seed=0)
        for _ in range(200):
            calls.clear()
            aug.degrade(_tone(0.1), [])
            assert len(calls) == 1
        aug = BatchAugmenter(16000, bandlimit_prob=0.3, clip_prob=0.3, phone_prob=0.1, seed=1)
        counts = {"none": 0, "bandlimit": 0, "clip": 0, "phone": 0}
        for _ in range(2000):
            calls.clear()
            aug.degrade(_tone(0.1), [])
            assert len(calls) <= 1
            counts[calls[0] if calls else "none"] += 1
        assert counts["none"] == pytest.approx(600, abs=90)
        assert counts["phone"] == pytest.approx(200, abs=60)
        assert counts["bandlimit"] == pytest.approx(600, abs=90)

    def test_exclusive_probabilities_over_one_fail_loudly(self) -> None:
        with pytest.raises(ValueError, match="exclusive"):
            BatchAugmenter(16000, bandlimit_prob=0.6, clip_prob=0.5)

    @pytest.mark.parametrize("corpus", [False, True])
    def test_noise_sources_cover_every_branch(self, tmp_path: Path, corpus: bool) -> None:
        sf.write(tmp_path / "n.wav", np.random.default_rng(0).standard_normal(4000) * 0.1, 16000)
        aug = BatchAugmenter(
            16000, noise_prob=1.0, corpus=AudioCorpus(str(tmp_path)) if corpus else None, seed=1
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
