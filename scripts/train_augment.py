"""Training-batch audio augmentation built from the batch itself: no noise corpus.

Two augmentations, both off by default (`data.babble_prob`, `data.noise_only_prob`):

- babble: mix another row of the same batch under a row's audio at an SNR drawn
  from `data.babble_snr_db`. The target is unchanged: the model learns to
  transcribe the louder talker through competing speech.
- noise only: replace a row's audio with colored noise or the sum of 5-8 other
  rows (many-talker babble, unintelligible), and its target with "". On the
  Open ASR leaderboard's URGENT set the decoder, given audio it cannot make out,
  loops for 100+ words instead of stopping; this shows it that stopping is the
  answer.

Both act on the decoder's input only. The literature on augmenting a frozen
encoder's input leans against it helping recognition; the target here is the
decoder's behaviour when recognition fails.
"""

import os
from typing import Any

import numpy as np
import numpy.typing as npt

# Noise-only clips: duration range (inside the collator's 0.8-19 s window),
# RMS level range in dBFS, and how many rows form many-talker babble.
NOISE_SECONDS = (1.0, 10.0)
NOISE_DBFS = (-45.0, -15.0)
BABBLE_TALKERS = (5, 8)
_EPS = 1e-8


def _rms(audio: npt.NDArray[Any]) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + _EPS))


def _fit(audio: npt.NDArray[Any], size: int) -> npt.NDArray[np.float32]:
    """Tile or crop `audio` to `size` samples."""
    return np.resize(audio.astype(np.float32, copy=False), size)


def _peak_safe(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    peak = float(np.abs(audio).max(initial=0.0))
    return audio / peak if peak > 1.0 else audio


def mix_at_snr(
    speech: npt.NDArray[Any], other: npt.NDArray[Any], snr_db: float
) -> npt.NDArray[np.float32]:
    """`speech` with `other` (tiled or cropped to its length) mixed in `snr_db` below it."""
    other = _fit(other, speech.size)
    scale = _rms(speech) / (_rms(other) * 10 ** (snr_db / 20))
    return _peak_safe(speech.astype(np.float32) + np.float32(scale) * other)


def colored_noise(size: int, exponent: float, rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """Unit-RMS noise with a 1/f**exponent power spectrum (0 white, 1 pink, 2 brown)."""
    spectrum = np.fft.rfft(rng.standard_normal(size))
    freqs = np.fft.rfftfreq(size)
    freqs[0] = freqs[1] if size > 1 else 1.0
    noise = np.fft.irfft(spectrum / freqs ** (exponent / 2), n=size).astype(np.float32)
    return noise / _rms(noise)


class BatchAugmenter:
    """Applies babble and noise-only augmentation to one batch of rows."""

    def __init__(
        self,
        sample_rate: int,
        babble_prob: float = 0.0,
        babble_snr_db: tuple[float, float] = (0.0, 15.0),
        noise_only_prob: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """Keep the rates; the RNG is created per process (see `_rng`)."""
        self.sample_rate = sample_rate
        self.babble_prob = babble_prob
        self.babble_snr_db = babble_snr_db
        self.noise_only_prob = noise_only_prob
        self._seed = seed
        self._pid: int | None = None
        self._generator: np.random.Generator | None = None

    @property
    def enabled(self) -> bool:
        """Whether either augmentation can fire."""
        return self.babble_prob > 0 or self.noise_only_prob > 0

    def _rng(self) -> np.random.Generator:
        # DataLoader workers fork the collator: one generator per process, or
        # every worker would draw the same "random" choices.
        if self._generator is None or self._pid != os.getpid():
            self._pid = os.getpid()
            seed = None if self._seed is None else self._seed + self._pid
            self._generator = np.random.default_rng(seed)
        return self._generator

    def noise_clip(self, others: list[npt.NDArray[Any]]) -> npt.NDArray[np.float32]:
        """Colored noise or many-talker babble from `others`, at a random level."""
        rng = self._rng()
        size = int(rng.uniform(*NOISE_SECONDS) * self.sample_rate)
        if len(others) >= BABBLE_TALKERS[0] and rng.random() < 0.5:
            talkers = int(rng.integers(BABBLE_TALKERS[0], min(len(others), BABBLE_TALKERS[1]) + 1))
            picks = rng.choice(len(others), size=talkers, replace=False)
            clip = sum(
                (_fit(others[i], size) / _rms(others[i]) for i in picks), np.zeros(size, np.float32)
            )
            clip = clip / _rms(clip)
        else:
            clip = colored_noise(size, float(rng.choice([0.0, 1.0, 2.0])), rng)
        level = 10 ** (rng.uniform(*NOISE_DBFS) / 20)
        return _peak_safe((clip * level).astype(np.float32))

    def __call__(
        self, audios: list[npt.NDArray[Any]], features: list[dict[str, Any]]
    ) -> tuple[list[npt.NDArray[Any]], list[dict[str, Any]]]:
        """Augmented copies of `audios` and `features` (noise-only rows get text "")."""
        if not self.enabled or len(audios) < 2:
            return audios, features
        rng = self._rng()
        out_audio, out_features = list(audios), list(features)
        for i in range(len(audios)):
            others = audios[:i] + audios[i + 1 :]
            draw = rng.random()
            if draw < self.noise_only_prob:
                out_audio[i] = self.noise_clip(others)
                out_features[i] = {**features[i], "text": ""}
            elif draw < self.noise_only_prob + self.babble_prob:
                other = others[int(rng.integers(len(others)))]
                out_audio[i] = mix_at_snr(audios[i], other, float(rng.uniform(*self.babble_snr_db)))
        return out_audio, out_features
