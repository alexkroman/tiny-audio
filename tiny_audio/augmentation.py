"""Audio data augmentation utilities for training.

NoiseAugmentation mixes background noise from a corpus (MUSAN / OpenSLR /
WHAM!) at random SNR. Apply via dataset.with_transform on the train split.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio import functional as taf


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
    return arr


def _resolve_corpus_roots(corpus_path: str | Sequence[str], hint: str = "") -> list[Path]:
    paths = [corpus_path] if isinstance(corpus_path, str) else list(corpus_path)
    roots: list[Path] = []
    for p in paths:
        root = Path(p).expanduser()
        if not root.exists():
            msg = f"Corpus path not found: {root}"
            if hint:
                msg = f"{msg}. {hint}"
            raise FileNotFoundError(msg)
        roots.append(root)
    return roots


class NoiseAugmentation:
    """Mix random background noise into audio.

    Two backends:
      * ``corpus_path`` set — sample a random clip from a directory of audio
        files (e.g. MUSAN), random-crop or tile to the input length, scale to
        a random target SNR, and add. This is the SOTA recipe used by NeMo /
        Canary / Parakeet.
      * Otherwise — ``torch_audiomentations.AddColoredNoise`` synthesizes
        white / pink / blue / violet noise at random SNR. Cheap, no corpus,
        but lacks real-world temporal / spectral structure.

    Accepts/returns numpy arrays so it can run in the dataloader transform
    pipeline.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        prob: float = 0.5,
        min_snr_db: float = 0.0,
        max_snr_db: float = 25.0,
        corpus_path: str | Sequence[str] | None = None,
        gaussian_min_snr_db: float | None = None,
        gaussian_max_snr_db: float | None = None,
        target_db: float = -25.0,
        rms_epsilon: float = 1e-2,
        tile_silence_sec: float = 0.25,
    ):
        self.sample_rate = sample_rate
        self.prob = prob
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        # When set, an additive Gaussian noise floor is layered on top of the
        # corpus mix. SNR is sampled uniformly in [gaussian_min_snr_db,
        # gaussian_max_snr_db] per clip and σ derived from clip RMS — matches
        # the SpeechBrain / NeMo additive-noise convention.
        self.gaussian_min_snr_db = gaussian_min_snr_db
        self.gaussian_max_snr_db = gaussian_max_snr_db
        # NeMo `NoisePerturbationWithNormalization` parameters: both clean and
        # noise are normalized to ``target_db`` (dB-FS) before SNR scaling, so
        # SNR semantics are decoupled from per-clip absolute level. ``rms_epsilon``
        # protects the divide when a clip is effectively silent. ``tile_silence_sec``
        # inserts a short silence between repeats when a noise clip is shorter
        # than the utterance — kills the periodic seam artifact that butt-joined
        # tiling produces.
        self.target_db = float(target_db)
        self.rms_epsilon = float(rms_epsilon)
        self.tile_silence_sec = float(tile_silence_sec)
        self.noise_paths: list[Path] | None = None
        self.babble_paths: list[Path] = []
        self.non_babble_paths: list[Path] = []
        self.augment = None

        if corpus_path is not None:
            roots = _resolve_corpus_roots(
                corpus_path, hint="Run `ta dev download-musan` to fetch MUSAN."
            )
            collected: list[Path] = []
            for root in roots:
                collected.extend(root.rglob("*.wav"))
            self.noise_paths = sorted(collected)
            if not self.noise_paths:
                raise ValueError(f"No .wav files found under {[str(r) for r in roots]}")
            for p in self.noise_paths:
                if any(part == "speech" for part in p.parts):
                    self.babble_paths.append(p)
                else:
                    self.non_babble_paths.append(p)
        else:
            try:
                from torch_audiomentations import AddColoredNoise
            except ImportError as e:
                raise ImportError(
                    "torch-audiomentations is required for synthetic noise. "
                    "Install with: pip install torch-audiomentations"
                ) from e
            self.augment = AddColoredNoise(
                min_snr_in_db=min_snr_db,
                max_snr_in_db=max_snr_db,
                p=prob,
                sample_rate=sample_rate,
                output_type="dict",
            )

    def _read_noise_segment(self, path: Path, n_target: int) -> np.ndarray | None:
        # Partial read: seek to a random window so MUSAN's multi-minute clips
        # don't drag in the full file just to crop most of it away.
        with sf.SoundFile(str(path)) as f:
            if f.samplerate == self.sample_rate:
                if f.frames >= n_target:
                    f.seek(random.randint(0, f.frames - n_target))
                    arr = f.read(n_target, dtype="float32", always_2d=False)
                else:
                    arr = f.read(dtype="float32", always_2d=False)
            else:
                # Read enough source frames to cover n_target after resampling.
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

    def _try_read_random_noise(
        self, num_samples: int, paths: list[Path] | None = None
    ) -> np.ndarray | None:
        """Pick + read a random noise segment, retrying past unreadable files.

        MUSAN occasionally has zero-length or otherwise broken entries; up to
        three picks before giving up. ``paths`` defaults to the full noise
        pool; callers pass a filtered subset (e.g. silence injection passes
        a no-speech subset). Returns None when the candidate pool is empty
        or no usable file is found in three attempts.
        """
        candidates = paths if paths is not None else self.noise_paths
        if not candidates:
            return None
        for _ in range(3):
            path = random.choice(candidates)
            try:
                noise = self._read_noise_segment(path, num_samples)
            except (RuntimeError, OSError, sf.LibsndfileError):
                continue
            if noise is not None:
                return noise
        return None

    def _norm_to_target_db(self, x: np.ndarray) -> np.ndarray:
        """Normalize ``x`` so its RMS matches ``self.target_db`` (dB-FS).

        Mirrors NeMo's ``NoisePerturbationWithNormalization.norm_audio_to_db``:
        clamp near-zero RMS to ``rms_epsilon`` to avoid divide-by-zero on
        effectively silent inputs.
        """
        rms = float(np.sqrt(np.mean(x**2)))
        if rms < self.rms_epsilon:
            rms = self.rms_epsilon
        scalar = (10 ** (self.target_db / 20.0)) / rms
        return x * scalar

    def _mix_corpus_noise(self, audio: np.ndarray) -> np.ndarray:
        noise = self._try_read_random_noise(audio.shape[-1])
        if noise is None:
            return audio
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        # NeMo `snr_mixer`: normalize both clean and noise to a common dB-FS
        # target, then attenuate noise by `-snr_db`. SNR semantics are then
        # independent of per-clip absolute level (close-talk vs. distant mic).
        clean = self._norm_to_target_db(audio.astype(np.float32))
        noise = self._norm_to_target_db(noise.astype(np.float32))
        mixed = clean + noise * (10 ** (-snr_db / 20.0))
        # Float32 won't physically clip, but downstream mel extraction expects
        # in-range samples — renormalize if mixing pushed the peak above 1.0.
        peak = float(np.abs(mixed).max())
        if peak > 1.0:
            mixed = mixed / peak
        return mixed.astype(audio.dtype)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        in_dtype = audio.dtype if hasattr(audio, "dtype") else np.float32

        if self.noise_paths is not None:
            if random.random() > self.prob:
                return audio
            arr = _to_mono_float32(audio)
            arr = self._mix_corpus_noise(arr)
            if self.gaussian_min_snr_db is not None and self.gaussian_max_snr_db is not None:
                signal_rms = float(np.sqrt(np.mean(arr**2)))
                if signal_rms >= 1e-8:
                    snr_db = random.uniform(self.gaussian_min_snr_db, self.gaussian_max_snr_db)
                    target_std = signal_rms / (10 ** (snr_db / 20))
                    arr = arr + (np.random.randn(*arr.shape).astype(np.float32) * target_std)
                    peak = float(np.abs(arr).max())
                    if peak > 1.0:
                        arr = arr / peak
            return arr.astype(in_dtype)

        audio_t = torch.from_numpy(_to_mono_float32(audio)).unsqueeze(0).unsqueeze(0)
        out = self.augment(audio_t).samples
        return out.squeeze(0).squeeze(0).numpy().astype(in_dtype)

    def sample_noise_only(self, num_samples: int) -> np.ndarray | None:
        """Return a noise-only clip of ``num_samples`` length, or None on failure.

        Used for silence-injection training: pairs a non-speech audio sample
        with an empty transcription so the model learns "no speech → emit
        nothing" instead of backchanneling on silent / non-speech segments.

        Filters out paths under any ``speech`` subdirectory (e.g.
        MUSAN/speech) — pairing real human speech with an empty transcript
        would teach the model to skip transcribing legitimate speech, the
        opposite of what we want. The full pool (including speech) is still
        used by ``_mix_corpus_noise`` for cocktail-party robustness; only
        this empty-target path filters it.

        Returns None when no corpus is configured (synthetic noise has no
        "noise-only" mode) or when the corpus contains only speech.
        """
        if self.noise_paths is None or num_samples <= 0:
            return None
        non_speech = [p for p in self.noise_paths if "speech" not in p.parts]
        if not non_speech:
            return None
        noise = self._try_read_random_noise(num_samples, paths=non_speech)
        return None if noise is None else noise.astype(np.float32)
