"""Audio data augmentation utilities for training.

RIR (Room Impulse Response) convolution simulates far-field / reverberant
recording conditions to close the meeting / distant-mic WER gap when training
mostly on close-talk audio.

RIRs are loaded from a Hugging Face dataset (auto-downloaded, no manual setup).
Default corpus is the MIT IR Survey (Traer & McDermott 2016) — 271 real
environmental RIRs at 16kHz spanning bedroom, office, classroom, outdoor, etc.
Small enough (~4MB) to preload into memory at init.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from torchaudio import functional as taf

if TYPE_CHECKING:
    from datasets import Dataset


DEFAULT_RIR_DATASET = "benjamin-paine/mit-impulse-response-survey-16khz"


class RIRAugmentation:
    """Convolve audio with a random RIR from a HuggingFace dataset.

    All RIRs are decoded, resampled, energy-normalized, and held in memory at
    init (the default dataset is only a few MB). Output is trimmed to the input
    length aligned to the RIR's direct-path peak so frame timing is preserved.
    """

    def __init__(
        self,
        hf_dataset: str = DEFAULT_RIR_DATASET,
        *,
        config: str | None = None,
        split: str = "train",
        audio_column: str = "audio",
        sample_rate: int = 16000,
        prob: float = 0.4,
        cache_dir: str | None = None,
        dataset: Dataset | None = None,
    ):
        if dataset is None:
            from datasets import load_dataset

            dataset = load_dataset(hf_dataset, name=config, split=split, cache_dir=cache_dir)

        self.rirs = self._extract_rirs(dataset, audio_column, sample_rate)
        if not self.rirs:
            raise ValueError(f"No usable RIRs in {hf_dataset}")
        self.sample_rate = sample_rate
        self.prob = prob

    @staticmethod
    def _extract_rirs(dataset: Dataset, audio_column: str, sample_rate: int) -> list[torch.Tensor]:
        rirs: list[torch.Tensor] = []
        for sample in dataset:
            audio = sample[audio_column]
            rir = torch.from_numpy(np.asarray(audio["array"], dtype=np.float32))
            sr = audio["sampling_rate"]
            if sr != sample_rate:
                rir = taf.resample(rir, sr, sample_rate)
            if rir.ndim > 1:
                rir = rir.mean(dim=0)
            peak_idx = int(rir.abs().argmax().item())
            rir = rir[peak_idx:]
            norm = torch.linalg.norm(rir)
            if norm < 1e-8:
                continue
            rirs.append(rir / norm)
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
