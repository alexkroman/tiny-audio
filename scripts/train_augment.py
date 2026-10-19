"""Training-batch audio degradation matched to the Open ASR leaderboard's URGENT 2024 set.

The leaderboard's URGENT test is the challenge's simulated validation set, and
its recipe is public (urgent-challenge/urgent2024_challenge,
conf/simulation_validation.yaml): reverberation on half the clips, noise at an
SNR uniform in -5..20 dB on every clip, then either nothing, a bandwidth limit
or clipping. Measured on the leaderboard pair (noisy and clean configs are the
same 1,000 utterances; granite-qwen-frozen-nc, 2026-10-17), SNR below 0 dB is
66% of the excess errors and clipping and a sub-8 kHz bandwidth each add
several WER points. So each training row may get, in this order:

- reverberation (`data.reverb_prob`) through a measured room response from
  `data.reverb_corpus` (DNS5's, URGENT's own RIR set);
- noise (`data.noise_prob`) at an SNR drawn from `data.noise_snr_db`, from a
  noise corpus (`data.noise_corpus`: DNS5 + WHAM!, URGENT's sources minus what
  its validation drew from), synthetic white/pink/brown noise, or another row
  of the batch;
- then at most ONE of, as in URGENT: a bandwidth limit (`data.bandlimit_prob`)
  at a cutoff drawn from `data.bandlimit_hz`, clipping (`data.clip_prob`)
  below a quantile drawn from `data.clip_low_quantile` and above one drawn
  from `data.clip_high_quantile` (URGENT's two-sided scheme), or a phone
  channel (`data.phone_prob`, see `phone`).

Targets are unchanged. Applied to training batches only.

The first run with this (verbatim-aug, 2026-10-18) degraded ~83% of rows and
let band limits and clipping stack on top of reverb and -5 dB noise: noisy
URGENT clips below 0 dB improved 5.1 WER points, but clean conversational
audio started to hallucinate fluent unrelated text (AMI 13 -> 29 such clips,
GigaSpeech 121 -> 165, URGENT-clean 2 -> 9), wiping out the gain. Hence the
exclusive post-noise step and the lower default rates (~half the rows clean).
"""

import io
import os
import shutil
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HOME
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

# Noise source mix when a row gets noise: corpus, synthetic broadband, batch babble.
# Without a corpus its share goes to synthetic noise.
SOURCE_WEIGHTS = {"corpus": 0.5, "synthetic": 0.3, "babble": 0.2}
# Measured responses are cut to this long after their direct path.
MAX_RIR_SECONDS = 1.5
_EPS = 1e-8


def _rms(audio: npt.NDArray[Any]) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + _EPS))


def _fit(audio: npt.NDArray[Any], size: int) -> npt.NDArray[np.float32]:
    """Tile or crop `audio` to `size` samples."""
    return np.resize(audio.astype(np.float32, copy=False), size)


def _peak_safe(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    peak = float(np.abs(audio).max(initial=0.0))
    return audio / peak if peak > 1.0 else audio


def non_silent(
    x: npt.NDArray[Any], threshold: float = 0.01, frame_length: int = 1024, frame_shift: int = 512
) -> npt.NDArray[np.bool_]:
    """ESPnet's `detect_non_silence` (espnet2/train/preprocessor.py), for one channel.

    A sample is non-silent when its frame's power exceeds `threshold` of the
    clip's mean frame power. URGENT measures both sides of an SNR over these
    samples only, so pauses in the speech or gaps in a bursty noise do not
    shift the mix.
    """
    if x.size < frame_length:
        return np.ones(x.size, dtype=bool)
    nadd = (-(x.size - frame_length) % frame_shift) % frame_length
    padded = np.pad(x.astype(np.float64), (0, nadd))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::frame_shift]
    power = np.mean(frames**2, axis=-1)
    mean_power = power.mean()
    if mean_power == 0:
        return np.ones(x.size, dtype=bool)
    detects = np.repeat(power / mean_power > threshold, frame_shift)[: x.size]
    return np.pad(detects, (0, x.size - detects.size), mode="edge")


def _active_power(x: npt.NDArray[Any]) -> float:
    return float(np.mean(np.square(x[non_silent(x)], dtype=np.float64)))


def mix_at_snr(
    speech: npt.NDArray[Any], noise: npt.NDArray[Any], snr_db: float
) -> npt.NDArray[np.float32]:
    """`speech` with `noise` (tiled or cropped to its length) mixed in `snr_db` below it.

    URGENT's `mix_noise`: speech and noise power are taken over their
    non-silent samples (`non_silent`), with the same 1e-10 floor on the noise.
    """
    noise = _fit(noise, speech.size)
    scale = 10 ** (-snr_db / 20) * np.sqrt(_active_power(speech) / max(_active_power(noise), 1e-10))
    return _peak_safe(speech.astype(np.float32) + np.float32(scale) * noise)


def colored_noise(size: int, exponent: float, rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """Unit-RMS noise with a 1/f**exponent power spectrum (0 white, 1 pink, 2 brown)."""
    spectrum = np.fft.rfft(rng.standard_normal(size))
    freqs = np.fft.rfftfreq(size)
    freqs[0] = freqs[1] if size > 1 else 1.0
    noise = np.fft.irfft(spectrum / freqs ** (exponent / 2), n=size).astype(np.float32)
    return noise / _rms(noise)


# URGENT's bandwidth limitation resamples to one of these lower rates and back.
URGENT_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


def bandlimit(
    audio: npt.NDArray[Any], sample_rate: int, cutoff_hz: float
) -> npt.NDArray[np.float32]:
    """`audio` with every frequency above `cutoff_hz` removed.

    When `2 * cutoff_hz` is one of URGENT's rates (at 16 kHz input only 8 kHz,
    the 4 kHz limit), this is URGENT's own augmentation: resample down and back
    with `resample_poly`, i.e. librosa's "polyphase", one of URGENT's resamplers.
    Other cutoffs mimic band-limited source recordings, whose edges measure
    sharp (a 30+ dB drop within 500 Hz), so they are a brickwall. Either rings
    past a full-scale input's peak, hence the final `_peak_safe`.
    """
    if 2 * cutoff_hz >= sample_rate:
        return audio.astype(np.float32, copy=False)
    if int(2 * cutoff_hz) in URGENT_SAMPLE_RATES:
        ratio = Fraction(int(2 * cutoff_hz), sample_rate).limit_denominator(1000)
        down = resample_poly(audio, ratio.numerator, ratio.denominator)
        out = _fit(resample_poly(down, ratio.denominator, ratio.numerator), audio.size)
    else:
        spectrum = np.fft.rfft(audio)
        spectrum[np.fft.rfftfreq(audio.size, d=1 / sample_rate) > cutoff_hz] = 0
        out = np.fft.irfft(spectrum, n=audio.size)
    return _peak_safe(out.astype(np.float32))


def clip(audio: npt.NDArray[Any], low_q: float, high_q: float) -> npt.NDArray[np.float32]:
    """Clip every sample below the `low_q` quantile or above the `high_q` one
    (URGENT's scheme: up to 10% on each side), keeping the overall level."""
    x = audio.astype(np.float32, copy=False)
    low, high = np.quantile(x, [low_q, high_q])
    if high <= low:
        return x
    clipped = np.clip(x, low, high)
    return _peak_safe(clipped * np.float32(_rms(x) / _rms(clipped)))


def reverberate(audio: npt.NDArray[Any], rir: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """`audio` convolved with room response `rir`, same length and level."""
    x = audio.astype(np.float32, copy=False)
    wet = fftconvolve(x, rir)[: x.size].astype(np.float32)
    return _peak_safe(wet * np.float32(_rms(x) / _rms(wet)))


def align_rir(rir: npt.NDArray[Any], sample_rate: int) -> npt.NDArray[np.float32]:
    """A measured response from 1 ms before its direct path (its peak), cut to
    `MAX_RIR_SECONDS` and scaled so the direct path is 1."""
    start = max(0, int(np.argmax(np.abs(rir))) - sample_rate // 1000)
    out = rir[start : start + int(MAX_RIR_SECONDS * sample_rate)].astype(np.float32)
    return out / (float(np.abs(out).max()) + _EPS)


def phone(
    audio: npt.NDArray[Any], sample_rate: int, highpass_hz: float, lowpass_hz: float, gain_db: float
) -> npt.NDArray[np.float32]:
    """A phone channel, matched to the leaderboard's Monsoon en-IN recordings
    (Android phones, measured 2026-10-18): a 2nd-order high-pass at `highpass_hz`
    (the small-mic roll-off, Monsoon's median ~110 Hz), an 8th-order low-pass at
    `lowpass_hz` (voice codecs roll off gently, unlike URGENT's resampling
    brickwall; Monsoon's band edge median 7.2 kHz), then `gain_db` (its RMS
    spans -35 to -16 dBFS)."""
    x = audio.astype(np.float64, copy=False)
    nyquist = sample_rate / 2
    x = sosfilt(butter(2, highpass_hz / nyquist, btype="highpass", output="sos"), x)
    if lowpass_hz < nyquist:
        x = sosfilt(butter(8, lowpass_hz / nyquist, btype="lowpass", output="sos"), x)
    return _peak_safe((x * 10 ** (gain_db / 20)).astype(np.float32))


class AudioCorpus:
    """Recordings from local folders and Hub datasets: noise excerpts or whole room responses.

    `sources` maps a local directory or Hub dataset id to a glob of its files.
    Audio files are read as they are. Parquet shards (Hub audio datasets: an
    `audio` column of encoded bytes) are unpacked to one file per row under
    `cache_dir` (default `HF_HOME/tiny_audio_corpora`) on first use, so drawing
    a recording reads one file, not a shard. A draw picks a file uniformly over
    all sources.
    """

    def __init__(self, sources: Mapping[str, str], cache_dir: Path | None = None) -> None:
        """`sources`: {local dir or Hub dataset id: glob of audio files or parquet shards}."""
        self.sources = dict(sources)
        self.cache_dir = cache_dir or Path(HF_HOME) / "tiny_audio_corpora"
        self._files: list[Path] | None = None

    def files(self) -> list[Path]:
        """The recordings, resolved (downloaded and unpacked) on first call."""
        if self._files is None:
            files: list[Path] = []
            for source, pattern in self.sources.items():
                found = _resolve(source, pattern, self.cache_dir)
                if not found:
                    msg = f"no audio files match {pattern!r} in {source}"
                    raise FileNotFoundError(msg)
                files += found
            self._files = files
        return self._files

    def recording(self, sample_rate: int, rng: np.random.Generator) -> npt.NDArray[np.float32]:
        """A random whole recording, mono, at `sample_rate`."""
        path = self.files()[int(rng.integers(len(self.files())))]
        data, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
        if sr != sample_rate:
            ratio = Fraction(sample_rate, sr).limit_denominator(1000)
            audio = resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
        return audio

    def excerpt(
        self, size: int, sample_rate: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float32]:
        """`size` samples at `sample_rate` from a random recording at a random offset."""
        noise = self.recording(sample_rate, rng)
        if noise.size > size:
            start = int(rng.integers(noise.size - size))
            noise = noise[start : start + size]
        return _fit(noise, size)


def _resolve(source: str, pattern: str, cache_dir: Path) -> list[Path]:
    """The audio files `pattern` names in `source`, with parquet shards unpacked."""
    root = Path(source)
    if not root.is_dir():
        root = Path(snapshot_download(source, repo_type="dataset", allow_patterns=[pattern]))
    files: list[Path] = []
    for path in sorted(root.glob(pattern)):
        files += (
            _unpack(path, cache_dir / source.replace("/", "--"))
            if path.suffix == ".parquet"
            else [path]
        )
    return files


def _unpack(shard: Path, cache_dir: Path) -> list[Path]:
    """One file per row of a parquet shard's `audio` column, unpacked once.

    Written to a temporary directory and renamed into place, so concurrent
    ranks never see a half-unpacked shard.
    """
    out = cache_dir / shard.stem
    if not out.is_dir():
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(shard, columns=["audio"])
        for i, cell in enumerate(table.column("audio").to_pylist()):
            # Hub `Audio` columns are {bytes, path} structs; some mirrors store raw bytes.
            data = cell["bytes"] if isinstance(cell, dict) else cell
            (tmp / f"{i:06d}").write_bytes(data)
        try:
            tmp.rename(out)
        except OSError:  # another rank got there first
            shutil.rmtree(tmp)
    return sorted(out.iterdir())


class BatchAugmenter:
    """Applies reverb, noise, bandwidth limits and clipping to one batch of rows."""

    def __init__(
        self,
        sample_rate: int,
        noise_prob: float = 0.0,
        noise_snr_db: tuple[float, float] = (-5.0, 20.0),
        corpus: AudioCorpus | None = None,
        bandlimit_prob: float = 0.0,
        bandlimit_hz: tuple[float, ...] = (4000.0,),
        clip_prob: float = 0.0,
        clip_low_quantile: tuple[float, float] = (0.0, 0.1),
        clip_high_quantile: tuple[float, float] = (0.9, 1.0),
        reverb_prob: float = 0.0,
        reverb_corpus: AudioCorpus | None = None,
        phone_prob: float = 0.0,
        phone_highpass_hz: tuple[float, float] = (50.0, 200.0),
        phone_lowpass_hz: tuple[float, ...] = (7000.0,),
        phone_gain_db: tuple[float, float] = (-10.0, 6.0),
        seed: int | None = None,
    ) -> None:
        """Keep the rates; the RNG is created per process (see `_rng`)."""
        if bandlimit_prob + clip_prob + phone_prob > 1.0:
            total = bandlimit_prob + clip_prob + phone_prob
            msg = f"bandlimit_prob + clip_prob + phone_prob is {total}; they are exclusive"
            raise ValueError(msg)
        if reverb_prob > 0 and reverb_corpus is None:
            msg = "reverb_prob needs a reverb_corpus of measured room responses"
            raise ValueError(msg)
        self.sample_rate = sample_rate
        self.noise_prob, self.noise_snr_db, self.corpus = noise_prob, noise_snr_db, corpus
        self.bandlimit_prob, self.bandlimit_hz = bandlimit_prob, bandlimit_hz
        self.clip_prob = clip_prob
        self.clip_low_quantile, self.clip_high_quantile = clip_low_quantile, clip_high_quantile
        self.reverb_prob, self.reverb_corpus = reverb_prob, reverb_corpus
        self.phone_prob, self.phone_lowpass_hz = phone_prob, phone_lowpass_hz
        self.phone_highpass_hz, self.phone_gain_db = phone_highpass_hz, phone_gain_db
        self._seed = seed
        self._pid: int | None = None
        self._generator: np.random.Generator | None = None

    @property
    def enabled(self) -> bool:
        """Whether any degradation can fire."""
        probs = (
            self.reverb_prob,
            self.noise_prob,
            self.bandlimit_prob,
            self.clip_prob,
            self.phone_prob,
        )
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

    def room(self, rng: np.random.Generator) -> npt.NDArray[np.float32]:
        """A random response from the room corpus, aligned to its direct path."""
        if self.reverb_corpus is None:
            msg = "no reverb_corpus"
            raise ValueError(msg)
        return align_rir(self.reverb_corpus.recording(self.sample_rate, rng), self.sample_rate)

    def degrade(self, audio: npt.NDArray[Any], others: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
        """One row through reverb and noise, each by chance, then at most one of
        a bandwidth limit, clipping or a phone channel."""
        rng = self._rng()
        out = audio
        if rng.random() < self.reverb_prob:
            out = reverberate(out, self.room(rng))
        if rng.random() < self.noise_prob:
            out = mix_at_snr(
                out, self.noise(out.size, others), float(rng.uniform(*self.noise_snr_db))
            )
        draw = rng.random()
        if draw < self.bandlimit_prob:
            cutoff = float(self.bandlimit_hz[int(rng.integers(len(self.bandlimit_hz)))])
            out = bandlimit(out, self.sample_rate, cutoff)
        elif draw < self.bandlimit_prob + self.clip_prob:
            low_q = float(rng.uniform(*self.clip_low_quantile))
            out = clip(out, low_q, float(rng.uniform(*self.clip_high_quantile)))
        elif draw < self.bandlimit_prob + self.clip_prob + self.phone_prob:
            lowpass = float(self.phone_lowpass_hz[int(rng.integers(len(self.phone_lowpass_hz)))])
            out = phone(
                out,
                self.sample_rate,
                float(rng.uniform(*self.phone_highpass_hz)),
                lowpass,
                float(rng.uniform(*self.phone_gain_db)),
            )
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
