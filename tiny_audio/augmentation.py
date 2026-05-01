"""Audio data augmentation utilities for training.

RIR (Room Impulse Response) convolution simulates far-field / reverberant
recording conditions to close the meeting / distant-mic WER gap when training
mostly on close-talk audio. Two backends:

  * ``corpus_path`` set — load real recorded RIRs from a directory of WAV
    files (e.g. OpenSLR-28's ``real_rirs_isotropic_noises``). This is the
    SOTA recipe (NeMo / Canary / Parakeet).
  * Otherwise — generate synthetic RIRs on-the-fly via pyroomacoustics'
    image-source method (CPU, pure-Python — no CUDA toolchain). Faster to
    set up, slightly less realistic.

NoiseAugmentation adds synthetic colored noise via torch-audiomentations
and works on CPU.

Apply both via dataset.with_transform on the train split. They're decoupled
so either can be enabled independently. RIR runs first (waveform shaping),
then noise (additive), matching the standard far-field training recipe.

All required packages are listed in pyproject.toml; no extra install steps.
"""

from __future__ import annotations

import random
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


class RIRAugmentation:
    """Convolve audio with a random RIR from a pool.

    Pool source:
      * ``corpus_path`` set — recursively load WAV RIRs from that directory
        (real recorded RIRs, e.g. OpenSLR-28).
      * Otherwise — generate ``pool_size`` synthetic RIRs at init by sampling
        random room geometry / T60 / src+mic positions and running
        pyroomacoustics' image-source method.

    Pool is held on CPU. Per call, picks a random RIR and convolves with input
    audio. Output is trimmed to input length aligned to the RIR direct-path
    peak so frame timing is preserved, then clip-normalized.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        prob: float = 0.5,
        pool_size: int = 2048,
        corpus_path: str | None = None,
        room_x_range: tuple[float, float] = (3.0, 10.0),
        room_y_range: tuple[float, float] = (3.0, 10.0),
        room_z_range: tuple[float, float] = (2.4, 4.0),
        t60_range: tuple[float, float] = (0.1, 1.0),
        source_margin: float = 0.3,
        seed: int | None = None,
        rirs: list[torch.Tensor] | None = None,
    ):
        if rirs is not None:
            self.rirs = list(rirs)
        elif corpus_path is not None:
            self.rirs = self._load_corpus_pool(
                corpus_path=corpus_path,
                sample_rate=sample_rate,
                pool_size=pool_size,
                seed=seed,
            )
        else:
            self.rirs = self._generate_pool(
                pool_size=pool_size,
                sample_rate=sample_rate,
                room_x_range=room_x_range,
                room_y_range=room_y_range,
                room_z_range=room_z_range,
                t60_range=t60_range,
                margin=source_margin,
                seed=seed,
            )
        if not self.rirs:
            raise ValueError("Empty RIR pool")
        self.sample_rate = sample_rate
        self.prob = prob

    @staticmethod
    def _load_corpus_pool(
        corpus_path: str,
        sample_rate: int,
        pool_size: int,
        seed: int | None,
    ) -> list[torch.Tensor]:
        root = Path(corpus_path).expanduser()
        if not root.exists():
            raise FileNotFoundError(
                f"RIR corpus_path not found: {root}. Run `ta dev download-rirs` "
                "to fetch OpenSLR-28."
            )
        wav_paths = sorted(root.rglob("*.wav"))
        if not wav_paths:
            raise ValueError(f"No .wav files found under {root}")

        rng = np.random.default_rng(seed)
        if pool_size and pool_size < len(wav_paths):
            picks = rng.choice(len(wav_paths), size=pool_size, replace=False)
            wav_paths = [wav_paths[i] for i in picks]

        rirs: list[torch.Tensor] = []
        for path in wav_paths:
            try:
                arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
            except (RuntimeError, OSError, sf.LibsndfileError):
                continue
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            wav = torch.from_numpy(np.ascontiguousarray(arr))
            if sr != sample_rate:
                wav = taf.resample(wav, sr, sample_rate)
            peak_idx = int(wav.abs().argmax().item())
            wav = wav[peak_idx:]
            norm = torch.linalg.norm(wav)
            if norm < 1e-8:
                continue
            rirs.append(wav / norm)
        return rirs

    @staticmethod
    def _generate_pool(
        pool_size: int,
        sample_rate: int,
        room_x_range: tuple[float, float],
        room_y_range: tuple[float, float],
        room_z_range: tuple[float, float],
        t60_range: tuple[float, float],
        margin: float,
        seed: int | None,
    ) -> list[torch.Tensor]:
        try:
            import pyroomacoustics as pra
        except ImportError as e:
            raise ImportError(
                "pyroomacoustics is required for on-the-fly RIR generation. "
                "Install with: pip install pyroomacoustics"
            ) from e

        rng = np.random.default_rng(seed)
        rirs: list[torch.Tensor] = []
        for _ in range(pool_size):
            room_sz = [
                float(rng.uniform(*room_x_range)),
                float(rng.uniform(*room_y_range)),
                float(rng.uniform(*room_z_range)),
            ]
            t60 = float(rng.uniform(*t60_range))
            pos_src = [
                float(rng.uniform(margin, room_sz[0] - margin)),
                float(rng.uniform(margin, room_sz[1] - margin)),
                float(rng.uniform(margin, room_sz[2] - margin)),
            ]
            pos_rcv = [
                float(rng.uniform(margin, room_sz[0] - margin)),
                float(rng.uniform(margin, room_sz[1] - margin)),
                float(rng.uniform(margin, room_sz[2] - margin)),
            ]
            try:
                e_absorption, max_order = pra.inverse_sabine(t60, room_sz)
            except ValueError:
                # T60 unreachable for this geometry (room too small / too absorptive).
                continue
            room = pra.ShoeBox(
                room_sz,
                fs=sample_rate,
                materials=pra.Material(e_absorption),
                max_order=max_order,
            )
            room.add_source(pos_src)
            room.add_microphone(pos_rcv)
            room.compute_rir()
            rir_np = np.asarray(room.rir[0][0], dtype=np.float32)
            rir_t = torch.from_numpy(rir_np)
            peak_idx = int(rir_t.abs().argmax().item())
            rir_t = rir_t[peak_idx:]
            norm = torch.linalg.norm(rir_t)
            if norm < 1e-8:
                continue
            rirs.append(rir_t / norm)
        return rirs

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return audio

        in_dtype = audio.dtype if hasattr(audio, "dtype") else np.float32
        audio_t = torch.from_numpy(_to_mono_float32(audio))
        n = audio_t.shape[-1]

        rir = random.choice(self.rirs)
        out = taf.fftconvolve(audio_t, rir, mode="full")[:n]
        peak = float(out.abs().max())
        if peak > 1.0:
            out = out / peak
        return out.numpy().astype(in_dtype)


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

    Accepts/returns numpy arrays so it can be chained with
    :class:`RIRAugmentation` in the dataloader transform pipeline.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        prob: float = 0.5,
        min_snr_db: float = 0.0,
        max_snr_db: float = 25.0,
        corpus_path: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.prob = prob
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.noise_paths: list[Path] | None = None
        self.augment = None

        if corpus_path is not None:
            corpus = Path(corpus_path).expanduser()
            if not corpus.exists():
                raise FileNotFoundError(
                    f"Noise corpus_path not found: {corpus}. "
                    "Run `ta dev download-musan` to fetch MUSAN."
                )
            self.noise_paths = sorted(corpus.rglob("*.wav"))
            if not self.noise_paths:
                raise ValueError(f"No .wav files found under {corpus}")
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
            repeats = (n_target + arr.shape[-1] - 1) // arr.shape[-1]
            arr = np.tile(arr, repeats)
        return arr[:n_target]

    def _mix_corpus_noise(self, audio: np.ndarray) -> np.ndarray:
        n = audio.shape[-1]
        # Retry a few files: MUSAN occasionally has unreadable / zero-length
        # entries; fail soft to clean signal if every attempt fails.
        for _ in range(3):
            path = random.choice(self.noise_paths)
            try:
                noise = self._read_noise_segment(path, n)
            except (RuntimeError, OSError, sf.LibsndfileError):
                continue
            if noise is None:
                continue
            signal_rms = float(np.sqrt(np.mean(audio**2)))
            noise_rms = float(np.sqrt(np.mean(noise**2)))
            if signal_rms < 1e-8 or noise_rms < 1e-8:
                return audio
            snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
            target_noise_rms = signal_rms / (10 ** (snr_db / 20))
            noise = noise * (target_noise_rms / noise_rms)
            return audio + noise.astype(audio.dtype)
        return audio

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        in_dtype = audio.dtype if hasattr(audio, "dtype") else np.float32

        if self.noise_paths is not None:
            if random.random() > self.prob:
                return audio
            arr = _to_mono_float32(audio)
            return self._mix_corpus_noise(arr).astype(in_dtype)

        audio_t = torch.from_numpy(_to_mono_float32(audio)).unsqueeze(0).unsqueeze(0)
        out = self.augment(audio_t).samples
        return out.squeeze(0).squeeze(0).numpy().astype(in_dtype)
