"""Audio data augmentation utilities for training.

RIR (Room Impulse Response) convolution simulates far-field / reverberant
recording conditions. Used to close the meeting / distant-mic WER gap when
training mostly on close-talk audio. Standard RIR corpus is OpenSLR-28
(https://www.openslr.org/28/) — both real (~325) and simulated (~60k) RIRs
work; simulated_rirs gives broader coverage of room geometries.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio import functional as taf


class RIRAugmentation:
    """Convolve audio with a random RIR sampled from a directory of WAV files.

    Output is trimmed to the same length as the input (aligned to the RIR's
    direct-path peak) and clipped-amplitude-normalized.
    """

    def __init__(self, rir_dir: str, sample_rate: int = 16000, prob: float = 0.4):
        rir_paths = sorted(Path(rir_dir).rglob("*.wav"))
        if not rir_paths:
            raise ValueError(f"No .wav files found under {rir_dir}")
        self.rir_paths = rir_paths
        self.sample_rate = sample_rate
        self.prob = prob

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

        rir_path = random.choice(self.rir_paths)
        rir, sr = torchaudio.load(str(rir_path))
        if sr != self.sample_rate:
            rir = taf.resample(rir, sr, self.sample_rate)
        rir = rir.mean(dim=0)
        peak_idx = int(rir.abs().argmax().item())
        rir = rir[peak_idx:]
        rir_norm = torch.linalg.norm(rir)
        if rir_norm < 1e-8:
            return audio
        rir = rir / rir_norm

        out = taf.fftconvolve(audio_t, rir, mode="full")[:n]
        peak = float(out.abs().max())
        if peak > 1.0:
            out = out / peak
        return out.numpy().astype(in_dtype)
