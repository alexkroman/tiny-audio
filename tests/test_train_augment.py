"""Tests for the training-batch babble / noise-only augmentation."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from scripts import train_augment
from scripts.train_augment import BatchAugmenter, colored_noise, mix_at_snr


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.3) -> npt.NDArray[np.float32]:
    t = np.arange(int(seconds * 16000)) / 16000
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rows(n: int) -> tuple[list[npt.NDArray[np.float32]], list[dict[str, Any]]]:
    audios = [_tone(2.0, freq=200.0 + 37 * i) for i in range(n)]
    return audios, [{"text": f"row {i}", "_text_punct": True} for i in range(n)]


class TestMixAtSnr:
    """mix_at_snr puts the second signal exactly `snr_db` under the first."""

    @pytest.mark.parametrize("snr_db", [0.0, 6.0, 15.0])
    def test_added_signal_sits_at_the_requested_snr(self, snr_db: float) -> None:
        speech = _tone(1.0)
        other = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
        added = mix_at_snr(speech, other, snr_db) - speech
        measured = 10 * np.log10(np.mean(speech**2) / np.mean(added**2))
        assert measured == pytest.approx(snr_db, abs=0.05)

    def test_other_is_tiled_to_the_speech_length(self) -> None:
        assert mix_at_snr(_tone(1.5), _tone(0.2), 10.0).shape == (24000,)

    def test_never_clips(self) -> None:
        loud = np.ones(16000, dtype=np.float32)
        assert np.abs(mix_at_snr(loud, loud, 0.0)).max() <= 1.0


class TestColoredNoise:
    """colored_noise is unit RMS and tilts toward low frequencies as the exponent grows."""

    def test_unit_rms(self) -> None:
        noise = colored_noise(16000, 1.0, np.random.default_rng(0))
        assert float(np.sqrt(np.mean(noise**2))) == pytest.approx(1.0, rel=1e-3)

    def test_brown_has_less_high_frequency_power_than_white(self) -> None:
        def high_share(exponent: float) -> float:
            power = (
                np.abs(np.fft.rfft(colored_noise(16000, exponent, np.random.default_rng(1)))) ** 2
            )
            return float(power[len(power) // 2 :].sum() / power.sum())

        assert high_share(2.0) < high_share(0.0) / 10


class TestBatchAugmenter:
    """Noise-only rows get an empty target; babble rows keep theirs."""

    def test_disabled_returns_the_batch_untouched(self) -> None:
        audios, features = _rows(3)
        aug = BatchAugmenter(16000)
        assert not aug.enabled
        assert aug(audios, features) == (audios, features)

    def test_single_row_batch_is_left_alone(self) -> None:
        audios, features = _rows(1)
        assert BatchAugmenter(16000, noise_only_prob=1.0)(audios, features) == (audios, features)

    def test_noise_only_rows_lose_their_target(self) -> None:
        audios, features = _rows(8)
        out_audio, out_features = BatchAugmenter(16000, noise_only_prob=1.0, seed=0)(
            audios, features
        )
        assert [f["text"] for f in out_features] == [""] * 8
        assert all(f["_text_punct"] for f in out_features)
        lo, hi = train_augment.NOISE_SECONDS
        for clip in out_audio:
            assert lo * 16000 <= clip.size <= hi * 16000
            assert np.abs(clip).max() <= 1.0
        assert features[0]["text"] == "row 0", "the input rows must not be mutated"

    def test_babble_rows_keep_their_target_and_length(self) -> None:
        audios, features = _rows(4)
        out_audio, out_features = BatchAugmenter(16000, babble_prob=1.0, seed=0)(audios, features)
        assert [f["text"] for f in out_features] == [f["text"] for f in features]
        for before, after in zip(audios, out_audio, strict=True):
            assert after.shape == before.shape
            assert not np.allclose(after, before)

    def test_each_process_draws_its_own_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        aug = BatchAugmenter(16000, babble_prob=0.5, seed=0)
        monkeypatch.setattr("os.getpid", lambda: 100)
        first = aug._rng().random(4)
        monkeypatch.setattr("os.getpid", lambda: 101)
        assert not np.allclose(aug._rng().random(4), first)
