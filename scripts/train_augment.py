"""Training-batch audio degradation matched to the Open ASR leaderboard's URGENT 2024 set.

The leaderboard's URGENT test is the challenge's simulated validation set, and
its recipe is public (urgent-challenge/urgent2024_challenge,
conf/simulation_validation.yaml): reverberation on half the clips, noise at an
SNR uniform in -5..20 dB on every clip, then either nothing, a bandwidth limit
or clipping. Measured on the leaderboard pair (noisy and clean configs are the
same 1,000 utterances; granite-qwen-frozen-nc, 2026-10-17), SNR below 0 dB is
66% of the excess errors and clipping and a sub-8 kHz bandwidth each add
several WER points. So each training row independently may get, in this order:

- reverberation (`data.reverb_prob`) from a synthetic room response with an
  RT60 drawn from `data.reverb_rt60` (URGENT's own RIRs are DNS5's, kept out);
- noise (`data.noise_prob`) at an SNR drawn from `data.noise_snr_db`, from a
  noise corpus (`data.noise_corpus`, e.g. MUSAN's noise set, which URGENT did
  not draw from), synthetic white/pink/brown noise, or another row of the batch;
- a bandwidth limit (`data.bandlimit_prob`) at a cutoff drawn from
  `data.bandlimit_hz`;
- clipping (`data.clip_prob`) below a quantile drawn from
  `data.clip_low_quantile` and above one drawn from `data.clip_high_quantile`,
  URGENT's two-sided scheme.

Targets are unchanged. Applied to training batches only.
"""

import io
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf
from huggingface_hub import snapshot_download
from scipy.signal import fftconvolve, resample_poly

# Noise source mix when a row gets noise: corpus, synthetic broadband, batch babble.
# Without a corpus its share goes to synthetic noise.
SOURCE_WEIGHTS = {"corpus": 0.5, "synthetic": 0.3, "babble": 0.2}
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
    speech: npt.NDArray[Any], noise: npt.NDArray[Any], snr_db: float
) -> npt.NDArray[np.float32]:
    """`speech` with `noise` (tiled or cropped to its length) mixed in `snr_db` below it."""
    noise = _fit(noise, speech.size)
    scale = _rms(speech) / (_rms(noise) * 10 ** (snr_db / 20))
    return _peak_safe(speech.astype(np.float32) + np.float32(scale) * noise)


def colored_noise(size: int, exponent: float, rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """Unit-RMS noise with a 1/f**exponent power spectrum (0 white, 1 pink, 2 brown)."""
    spectrum = np.fft.rfft(rng.standard_normal(size))
    freqs = np.fft.rfftfreq(size)
    freqs[0] = freqs[1] if size > 1 else 1.0
    noise = np.fft.irfft(spectrum / freqs ** (exponent / 2), n=size).astype(np.float32)
    return noise / _rms(noise)


def bandlimit(
    audio: npt.NDArray[Any], sample_rate: int, cutoff_hz: float
) -> npt.NDArray[np.float32]:
    """`audio` with every frequency above `cutoff_hz` removed (a brickwall, like
    URGENT's cutoffs: a 30+ dB drop within 500 Hz)."""
    if 2 * cutoff_hz >= sample_rate:
        return audio.astype(np.float32, copy=False)
    spectrum = np.fft.rfft(audio)
    spectrum[np.fft.rfftfreq(audio.size, d=1 / sample_rate) > cutoff_hz] = 0
    return np.fft.irfft(spectrum, n=audio.size).astype(np.float32)


def clip(audio: npt.NDArray[Any], low_q: float, high_q: float) -> npt.NDArray[np.float32]:
    """Clip every sample below the `low_q` quantile or above the `high_q` one
    (URGENT's scheme: up to 10% on each side), keeping the overall level."""
    x = audio.astype(np.float32, copy=False)
    low, high = np.quantile(x, [low_q, high_q])
    if high <= low:
        return x
    clipped = np.clip(x, low, high)
    return _peak_safe(clipped * np.float32(_rms(x) / _rms(clipped)))


def room_response(
    rt60: float, sample_rate: int, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """A synthetic room impulse response: a direct path and an exponentially
    decaying noise tail that falls 60 dB in `rt60` seconds."""
    n = max(2, int(min(1.2 * rt60, 1.5) * sample_rate))
    t = np.arange(n) / sample_rate
    tail = rng.standard_normal(n) * np.exp(-np.log(1000.0) * t / rt60)
    tail[0] = 0.0
    rir = tail / (np.sqrt(np.sum(tail**2)) + _EPS) * 10 ** (-rng.uniform(0.0, 10.0) / 20)
    rir[0] = 1.0  # direct path; the tail sits 0-10 dB below it in energy
    return rir.astype(np.float32)


def reverberate(
    audio: npt.NDArray[Any], rt60: float, sample_rate: int, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """`audio` through a synthetic room with reverberation time `rt60`, same length and level."""
    x = audio.astype(np.float32, copy=False)
    wet = fftconvolve(x, room_response(rt60, sample_rate, rng))[: x.size].astype(np.float32)
    return _peak_safe(wet * np.float32(_rms(x) / _rms(wet)))


class NoiseCorpus:
    """Random excerpts from a folder (or Hub dataset) of noise recordings."""

    def __init__(self, source: str, pattern: str = "**/*.wav") -> None:
        """`source`: a local directory, or a Hub dataset id downloaded on first use."""
        self.source, self.pattern = source, pattern
        self._files: list[Path] | None = None

    def files(self) -> list[Path]:
        """The recordings, resolved (and downloaded) on first call."""
        if self._files is None:
            root = Path(self.source)
            if not root.is_dir():
                root = Path(
                    snapshot_download(
                        self.source, repo_type="dataset", allow_patterns=[self.pattern]
                    )
                )
            self._files = sorted(root.glob(self.pattern))
            if not self._files:
                msg = f"no noise files match {self.pattern!r} under {root}"
                raise FileNotFoundError(msg)
        return self._files

    def excerpt(
        self, size: int, sample_rate: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float32]:
        """`size` samples at `sample_rate` from a random recording at a random offset."""
        path = self.files()[int(rng.integers(len(self.files())))]
        data, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
        noise = data.mean(axis=1)
        if sr != sample_rate:
            ratio = Fraction(sample_rate, sr).limit_denominator(1000)
            noise = resample_poly(noise, ratio.numerator, ratio.denominator).astype(np.float32)
        if noise.size > size:
            start = int(rng.integers(noise.size - size))
            noise = noise[start : start + size]
        return _fit(noise, size)


class BatchAugmenter:
    """Applies reverb, noise, bandwidth limits and clipping to one batch of rows."""

    def __init__(
        self,
        sample_rate: int,
        noise_prob: float = 0.0,
        noise_snr_db: tuple[float, float] = (-5.0, 20.0),
        corpus: NoiseCorpus | None = None,
        bandlimit_prob: float = 0.0,
        bandlimit_hz: tuple[float, ...] = (4000.0,),
        clip_prob: float = 0.0,
        clip_low_quantile: tuple[float, float] = (0.0, 0.1),
        clip_high_quantile: tuple[float, float] = (0.9, 1.0),
        reverb_prob: float = 0.0,
        reverb_rt60: tuple[float, float] = (0.2, 1.0),
        seed: int | None = None,
    ) -> None:
        """Keep the rates; the RNG is created per process (see `_rng`)."""
        self.sample_rate = sample_rate
        self.noise_prob, self.noise_snr_db, self.corpus = noise_prob, noise_snr_db, corpus
        self.bandlimit_prob, self.bandlimit_hz = bandlimit_prob, bandlimit_hz
        self.clip_prob = clip_prob
        self.clip_low_quantile, self.clip_high_quantile = clip_low_quantile, clip_high_quantile
        self.reverb_prob, self.reverb_rt60 = reverb_prob, reverb_rt60
        self._seed = seed
        self._pid: int | None = None
        self._generator: np.random.Generator | None = None

    @property
    def enabled(self) -> bool:
        """Whether any degradation can fire."""
        probs = (self.reverb_prob, self.noise_prob, self.bandlimit_prob, self.clip_prob)
        return any(p > 0 for p in probs)

    def _rng(self) -> np.random.Generator:
        # DataLoader workers fork the collator: one generator per process, or
        # every worker would draw the same "random" choices.
        if self._generator is None or self._pid != os.getpid():
            self._pid = os.getpid()
            seed = None if self._seed is None else self._seed + self._pid
            self._generator = np.random.default_rng(seed)
        return self._generator

    def noise(self, size: int, others: list[npt.NDArray[Any]]) -> npt.NDArray[np.float32]:
        """`size` samples of noise from the corpus, synthetic noise or another row."""
        rng = self._rng()
        weights = dict(SOURCE_WEIGHTS)
        if self.corpus is None:
            weights["synthetic"] += weights.pop("corpus")
        if not others:
            weights["synthetic"] += weights.pop("babble")
        names = list(weights)
        source = names[
            int(rng.choice(len(names), p=np.array(list(weights.values())) / sum(weights.values())))
        ]
        if source == "corpus" and self.corpus is not None:
            return self.corpus.excerpt(size, self.sample_rate, rng)
        if source == "babble":
            return _fit(others[int(rng.integers(len(others)))], size)
        return colored_noise(size, float(rng.choice([0.0, 1.0, 2.0])), rng)

    def degrade(self, audio: npt.NDArray[Any], others: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
        """One row through reverb, noise, a bandwidth limit and clipping, each by chance."""
        rng = self._rng()
        out = audio
        if rng.random() < self.reverb_prob:
            out = reverberate(out, float(rng.uniform(*self.reverb_rt60)), self.sample_rate, rng)
        if rng.random() < self.noise_prob:
            out = mix_at_snr(
                out, self.noise(out.size, others), float(rng.uniform(*self.noise_snr_db))
            )
        if rng.random() < self.bandlimit_prob:
            cutoff = float(self.bandlimit_hz[int(rng.integers(len(self.bandlimit_hz)))])
            out = bandlimit(out, self.sample_rate, cutoff)
        if rng.random() < self.clip_prob:
            low_q = float(rng.uniform(*self.clip_low_quantile))
            out = clip(out, low_q, float(rng.uniform(*self.clip_high_quantile)))
        return out

    def __call__(
        self, audios: list[npt.NDArray[Any]], features: list[dict[str, Any]]
    ) -> tuple[list[npt.NDArray[Any]], list[dict[str, Any]]]:
        """Degraded copies of `audios`; `features` (the targets) pass through unchanged."""
        if not self.enabled:
            return audios, features
        return [
            self.degrade(a, audios[:i] + audios[i + 1 :]) for i, a in enumerate(audios)
        ], features
