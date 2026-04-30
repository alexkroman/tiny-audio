"""Dump the Whisper-style mel filterbank to swift/Sources/TinyAudio/Mel/MelFilterbank.json.

128 mels, n_fft=400, sr=16000 — matches transformers.WhisperFeatureExtractor defaults
that GLM-ASR-Nano-2512 uses.
"""

import json
from pathlib import Path

import numpy as np
from transformers import WhisperFeatureExtractor

OUT = Path("swift/Sources/TinyAudio/Mel/MelFilterbank.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

fe = WhisperFeatureExtractor.from_pretrained("zai-org/GLM-ASR-Nano-2512")
# HF stores mel_filters as [n_fft/2+1, n_mels]; transpose to [n_mels, n_fft/2+1]
filters = np.asarray(fe.mel_filters).T  # [n_mels, n_fft/2+1]
OUT.write_text(
    json.dumps(
        {
            "n_mels": int(filters.shape[0]),
            "n_fft_bins": int(filters.shape[1]),
            "filters": filters.astype(np.float32).flatten().tolist(),
            "n_fft": int(fe.n_fft),
            "hop_length": int(fe.hop_length),
            "sampling_rate": int(fe.sampling_rate),
        }
    )
)
print(f"Wrote {OUT} ({filters.shape[0]} mels x {filters.shape[1]} bins, {filters.size} floats)")
