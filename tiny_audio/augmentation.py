"""Audio data augmentation using audiomentations.

RIRAugmentation models acoustic propagation: convolves with a recorded RIR
(e.g. OpenSLR-28). Recorded RIRs already encode source-to-mic propagation
including air absorption at the recorded distance, so no separate
AirAbsorption stage is added.

NoiseAugmentation models capture-side corruptions: long-stationary
background noise (e.g. MUSAN), sparse short transients, an always-on
Gaussian sensor floor, EQ, clipping, and a band-limit branch (low-pass
or telephony band-pass, exactly one). Order roughly follows the
physical signal chain (mic → preamp → channel).

NoiseAugmentation also exposes ``sample_noise_only`` for silence-injection
training: pulls a noise-only clip from the background corpus (excluding
any ``speech/`` subdir) so the dataloader can pair it with an empty
transcript and teach the model to emit nothing on non-speech.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from audiomentations import (
    AddBackgroundNoise,
    AddColorNoise,
    AddShortNoises,
    ApplyImpulseResponse,
    BandPassFilter,
    ClippingDistortion,
    Compose,
    Gain,
    LowPassFilter,
    Mp3Compression,
    OneOf,
    SevenBandParametricEQ,
)
from torchaudio import functional as taf


def _resolve_corpus_roots(corpus_path: str | Sequence[str], hint: str = "") -> list[str]:
    paths = [corpus_path] if isinstance(corpus_path, str) else list(corpus_path)
    roots: list[str] = []
    for p in paths:
        root = Path(p).expanduser()
        if not root.exists():
            msg = f"Corpus path not found: {root}"
            if hint:
                msg = f"{msg}. {hint}"
            raise FileNotFoundError(msg)
        roots.append(str(root))
    return roots


_SPEECH_DIR = "speech"  # MUSAN convention; pairing speech audio with empty
# transcripts during silence injection would teach
# the model to skip real speech.


def _build_background_noise(
    corpus_path: str | Sequence[str],
    *,
    min_snr_db: float,
    max_snr_db: float,
    prob: float,
    hint: str,
) -> AddBackgroundNoise:
    roots = _resolve_corpus_roots(corpus_path, hint=hint)
    transform = AddBackgroundNoise(
        sounds_path=roots,
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        p=prob,
    )
    if not transform.sound_file_paths:
        raise ValueError(f"No audio files found under {roots}")
    return transform


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        # HF Audio feature returns channel-last (`[time, channels]`).
        arr = arr.mean(axis=-1)
    return arr


class RIRAugmentation:
    """Recorded RIR convolution (room + propagation)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        prob: float = 0.5,
        corpus_path: str | Sequence[str] | None = None,
    ):
        if corpus_path is None:
            raise ValueError("RIRAugmentation requires `corpus_path`")
        roots = _resolve_corpus_roots(
            corpus_path, hint="Run `ta dev download-rirs` to fetch OpenSLR-28."
        )
        self.sample_rate = sample_rate
        self.transform = ApplyImpulseResponse(ir_path=roots, p=prob)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        in_dtype = audio.dtype if hasattr(audio, "dtype") else np.float32
        out = self.transform(samples=_to_mono_float32(audio), sample_rate=self.sample_rate)
        return out.astype(in_dtype)


class NoiseAugmentation:
    """MUSAN background + short transients + always-on Gaussian floor +
    EQ + clipping + OneOf{low-pass, telephony band-pass}."""

    def __init__(
        self,
        sample_rate: int = 16000,
        prob: float = 0.5,
        # Default lower bound is 5 dB. With a frozen Whisper encoder, going
        # below ~5 dB sustained SNR pushes encoder features outside the
        # distribution Whisper was trained on, so the projector ends up
        # training on features that don't correspond to anything realistic
        # at inference. Only drop below 5 if deployment audio actually hits
        # those SNRs sustained (rare). Upper bound of 30 dB lets some
        # samples be effectively clean.
        min_snr_db: float = 5.0,
        max_snr_db: float = 30.0,
        corpus_path: str | Sequence[str] | None = None,
        # Babble (multi-talker) background — second AddBackgroundNoise
        # pointing at MUSAN's `speech/` subdir. Targets meeting / conference
        # / uncontrolled-mic evals where stationary ambient noise alone
        # underspecifies the acoustic scene. SNR floor is intentionally
        # higher than ambient: at low babble SNR the masker speech competes
        # with the target and the label becomes ambiguous, so we keep the
        # background recognisable as background.
        babble_corpus_path: str | Sequence[str] | None = None,
        babble_prob: float = 0.0,
        babble_min_snr_db: float = 10.0,
        babble_max_snr_db: float = 25.0,
        # Always-on additive colored-noise sensor floor (when configured).
        # Models the per-clip background dither that real recordings have
        # whether or not MUSAN ambient noise was layered on. Spectral shape
        # defaults bracket pink (~-3 dB/octave): pink/brown matches real
        # mic + analog-front-end + acoustic-room noise far better than
        # white Gaussian, and Whisper has seen vastly more pink-ish dither
        # in pretraining than flat-spectrum noise. Set both SNR bounds to
        # None to disable.
        color_noise_min_snr_db: float | None = None,
        color_noise_max_snr_db: float | None = None,
        # f_decay in dB/octave: 0 = white, -3 = pink, -6 = brown.
        color_noise_min_f_decay: float = -4.0,
        color_noise_max_f_decay: float = -2.0,
        # Short-duration transient events — must be a corpus of sound events
        # (e.g. FSD50K), NOT MUSAN. MUSAN/noise is stationary, so sampling
        # short windows from it duplicates AddBackgroundNoise's distribution.
        short_noises_corpus_path: str | Sequence[str] | None = None,
        short_noises_prob: float = 0.0,
        # ±4 dB tuned for projector training (frozen Whisper encoder). For
        # training the encoder from scratch, ±6-8 dB is more appropriate.
        eq_prob: float = 0.0,
        eq_min_db: float = -4.0,
        eq_max_db: float = 4.0,
        # `clipping_max_percentile=10` (top 10% of samples clipped) is tuned
        # for projector training. For encoder-from-scratch, push to 20.
        clipping_prob: float = 0.0,
        clipping_max_percentile: int = 10,
        # Band-limit branch: when fired, exactly one of LPF (cheap-mic /
        # consumer-device cutoff) or BPF (telephony 200-4000 Hz) is applied,
        # never both. Prevents incoherent stacks (LPF cutting at 4 kHz then
        # BPF cutting at 200-4000 Hz on the same clip).
        bandlimit_prob: float = 0.0,
        lowpass_min_cutoff: float = 3000.0,
        lowpass_max_cutoff: float = 7500.0,
        bandpass_min_center_freq: float = 2000.0,
        bandpass_max_center_freq: float = 2200.0,
        bandpass_min_bandwidth_fraction: float = 1.7,
        bandpass_max_bandwidth_fraction: float = 1.9,
        # Gain perturbation — applied late in the chain so it shifts the
        # mixed signal level (target + noise together), simulating ADC /
        # input-level variability. Whisper normalizes mels per-clip but
        # not perfectly across very different input levels, so this
        # actually shifts encoder features (unlike speed perturbation,
        # which a frozen encoder largely absorbs).
        gain_prob: float = 0.0,
        gain_min_db: float = -6.0,
        gain_max_db: float = 6.0,
        # MP3 round-trip — applied last so the codec sees the leveled
        # mixed signal, matching the real-world chain (ADC → encoder).
        # Targets web/YouTube-derived training corpora and inference on
        # MP3-compressed audio. Even though Whisper saw lots of MP3 in
        # pretraining, encoder features still drift across bitrates and
        # the projector benefits from explicit exposure.
        mp3_prob: float = 0.0,
        mp3_min_bitrate: int = 32,
        mp3_max_bitrate: int = 128,
        rms_epsilon: float = 1e-4,
        tile_silence_sec: float = 0.25,
    ):
        self.sample_rate = sample_rate
        self.rms_epsilon = float(rms_epsilon)
        self.tile_silence_sec = float(tile_silence_sec)
        self.non_speech_paths: list[Path] = []

        transforms: list = []
        if corpus_path is not None:
            bg = _build_background_noise(
                corpus_path,
                min_snr_db=min_snr_db,
                max_snr_db=max_snr_db,
                prob=prob,
                hint="Run `ta dev download-musan` to fetch MUSAN.",
            )
            # Precompute the non-speech subset for `sample_noise_only`.
            # Reuses audiomentations' walk so we don't re-rglob.
            self.non_speech_paths = [
                Path(p) for p in bg.sound_file_paths if _SPEECH_DIR not in Path(p).parts
            ]
            transforms.append(bg)
        if babble_prob > 0.0 and babble_corpus_path is not None:
            transforms.append(
                _build_background_noise(
                    babble_corpus_path,
                    min_snr_db=babble_min_snr_db,
                    max_snr_db=babble_max_snr_db,
                    prob=babble_prob,
                    hint="Run `ta dev download-musan` to fetch MUSAN (provides musan/speech).",
                )
            )
        if short_noises_prob > 0.0 and short_noises_corpus_path is not None:
            short_roots = _resolve_corpus_roots(
                short_noises_corpus_path,
                hint="Run `ta dev download-fsd50k` to fetch FSD50K sound events.",
            )
            transforms.append(AddShortNoises(sounds_path=short_roots, p=short_noises_prob))
        if color_noise_min_snr_db is not None and color_noise_max_snr_db is not None:
            transforms.append(
                AddColorNoise(
                    min_snr_db=color_noise_min_snr_db,
                    max_snr_db=color_noise_max_snr_db,
                    min_f_decay=color_noise_min_f_decay,
                    max_f_decay=color_noise_max_f_decay,
                    p=1.0,
                )
            )
        if eq_prob > 0.0:
            transforms.append(
                SevenBandParametricEQ(min_gain_db=eq_min_db, max_gain_db=eq_max_db, p=eq_prob)
            )
        if clipping_prob > 0.0:
            transforms.append(
                ClippingDistortion(
                    max_percentile_threshold=clipping_max_percentile, p=clipping_prob
                )
            )
        if bandlimit_prob > 0.0:
            transforms.append(
                OneOf(
                    [
                        LowPassFilter(
                            min_cutoff_freq=lowpass_min_cutoff,
                            max_cutoff_freq=lowpass_max_cutoff,
                            p=1.0,
                        ),
                        BandPassFilter(
                            min_center_freq=bandpass_min_center_freq,
                            max_center_freq=bandpass_max_center_freq,
                            min_bandwidth_fraction=bandpass_min_bandwidth_fraction,
                            max_bandwidth_fraction=bandpass_max_bandwidth_fraction,
                            p=1.0,
                        ),
                    ],
                    p=bandlimit_prob,
                )
            )
        if gain_prob > 0.0:
            transforms.append(Gain(min_gain_db=gain_min_db, max_gain_db=gain_max_db, p=gain_prob))
        if mp3_prob > 0.0:
            transforms.append(
                Mp3Compression(min_bitrate=mp3_min_bitrate, max_bitrate=mp3_max_bitrate, p=mp3_prob)
            )
        self.transform = Compose(transforms)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        in_dtype = audio.dtype if hasattr(audio, "dtype") else np.float32
        out = self.transform(samples=_to_mono_float32(audio), sample_rate=self.sample_rate)
        return out.astype(in_dtype)

    def _read_noise_segment(self, path: Path, n_target: int) -> np.ndarray | None:
        # Partial read: seek to a random window so multi-minute clips don't
        # drag in the full file just to crop most of it away.
        with sf.SoundFile(str(path)) as f:
            if f.samplerate == self.sample_rate:
                if f.frames >= n_target:
                    f.seek(random.randint(0, f.frames - n_target))
                    arr = f.read(n_target, dtype="float32", always_2d=False)
                else:
                    arr = f.read(dtype="float32", always_2d=False)
            else:
                src_needed = int(np.ceil(n_target * f.samplerate / self.sample_rate))
                if f.frames >= src_needed:
                    f.seek(random.randint(0, f.frames - src_needed))
                    arr = f.read(src_needed, dtype="float32", always_2d=False)
                else:
                    arr = f.read(dtype="float32", always_2d=False)
                arr = (
                    taf.resample(
                        torch.from_numpy(np.ascontiguousarray(arr)),
                        f.samplerate,
                        self.sample_rate,
                    )
                    .numpy()
                    .astype(np.float32)
                )
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.size == 0:
            return None
        if arr.shape[-1] < n_target:
            silence_n = int(round(self.sample_rate * self.tile_silence_sec))
            silence = np.zeros(silence_n, dtype=arr.dtype)
            pieces: list[np.ndarray] = []
            total = 0
            while total < n_target:
                pieces.append(arr)
                total += arr.shape[-1]
                if total >= n_target:
                    break
                pieces.append(silence)
                total += silence_n
            arr = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        return arr[:n_target]

    def sample_noise_only(self, num_samples: int) -> np.ndarray | None:
        """Return a noise-only clip of ``num_samples`` length, or None.

        For silence-injection training: pairs a non-speech audio sample with
        an empty transcription so the model learns "no speech → emit nothing"
        instead of backchanneling on silent / non-speech segments — a known
        failure mode for Whisper-based and SALMONN-based speech LLMs.

        Returns None when no corpus is configured, the corpus contains only
        speech (filtered for the same reason), or no usable file is found
        in three attempts.
        """
        if not self.non_speech_paths or num_samples <= 0:
            return None
        for _ in range(3):
            path = random.choice(self.non_speech_paths)
            try:
                noise = self._read_noise_segment(path, num_samples)
            except (RuntimeError, OSError, sf.LibsndfileError):
                continue
            if noise is not None:
                return noise.astype(np.float32)
        return None
