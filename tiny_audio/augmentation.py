"""Audio data augmentation utilities for training.

RIR (Room Impulse Response) convolution simulates far-field / reverberant
recording conditions to close the meeting / distant-mic WER gap when training
mostly on close-talk audio.

RIRs are generated on-the-fly via gpuRIR's image method using randomized room
dimensions, T60, and source/receiver positions — no corpus download required.
A pool of RIRs is materialized once at init and sampled per training step.

gpuRIR requires CUDA; install with:
    pip install https://github.com/DavidDiazGuerra/gpuRIR/zipball/master
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torchaudio import functional as taf


class RIRAugmentation:
    """Convolve audio with a random RIR from an on-the-fly generated pool.

    At init, generates ``pool_size`` RIRs by sampling random room geometry,
    T60, and source / receiver positions, then running gpuRIR's image method.
    Pool is held on CPU. Per call, picks a random RIR and convolves with input
    audio. Output is trimmed to input length aligned to the RIR direct-path
    peak so frame timing is preserved, then clip-normalized.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        prob: float = 0.4,
        pool_size: int = 1024,
        room_x_range: tuple[float, float] = (3.0, 8.0),
        room_y_range: tuple[float, float] = (3.0, 8.0),
        room_z_range: tuple[float, float] = (2.4, 3.5),
        t60_range: tuple[float, float] = (0.2, 0.8),
        source_margin: float = 0.3,
        seed: int | None = None,
        rirs: list[torch.Tensor] | None = None,
    ):
        if rirs is not None:
            self.rirs = list(rirs)
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
            import gpuRIR  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise ImportError(
                "gpuRIR is required for on-the-fly RIR generation. Install with: "
                "pip install https://github.com/DavidDiazGuerra/gpuRIR/zipball/master "
                "(requires CUDA)."
            ) from e

        rng = np.random.default_rng(seed)
        rirs: list[torch.Tensor] = []
        for _ in range(pool_size):
            room_sz = np.array(
                [
                    rng.uniform(*room_x_range),
                    rng.uniform(*room_y_range),
                    rng.uniform(*room_z_range),
                ]
            )
            t60 = float(rng.uniform(*t60_range))
            pos_src = np.array(
                [
                    [
                        rng.uniform(margin, room_sz[0] - margin),
                        rng.uniform(margin, room_sz[1] - margin),
                        rng.uniform(margin, room_sz[2] - margin),
                    ]
                ]
            )
            pos_rcv = np.array(
                [
                    [
                        rng.uniform(margin, room_sz[0] - margin),
                        rng.uniform(margin, room_sz[1] - margin),
                        rng.uniform(margin, room_sz[2] - margin),
                    ]
                ]
            )
            beta = gpuRIR.beta_SabineEstimation(room_sz, t60)
            nb_img = gpuRIR.t2n(t60, room_sz)
            rir = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, t60, sample_rate)
            rir_t = torch.from_numpy(np.asarray(rir[0, 0], dtype=np.float32))
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
        audio_t = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if audio_t.ndim > 1:
            audio_t = audio_t.squeeze()
            if audio_t.ndim > 1:
                audio_t = audio_t.mean(dim=0)
        n = audio_t.shape[-1]

        rir = random.choice(self.rirs)
        out = taf.fftconvolve(audio_t, rir, mode="full")[:n]
        peak = float(out.abs().max())
        if peak > 1.0:
            out = out / peak
        return out.numpy().astype(in_dtype)
