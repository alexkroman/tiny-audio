"""Quick MLX RTFx benchmark.

Loads MLXASRModel, warms it up, then times transcribe() on a few audio clips.
Reports RTFx = audio_seconds / wall_clock_seconds. RTFx > 1 means faster
than realtime.
"""

from __future__ import annotations

import time

import librosa
import numpy as np
import soundfile as sf

from tiny_audio.mlx import MLXASRModel

REPO_ID = "mazesmazes/tiny-audio-embedded"


def load_clip() -> np.ndarray:
    """LibriSpeech sample, downsampled to 16kHz mono float32."""
    path = librosa.ex("libri1")
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data.astype(np.float32)


def time_transcribe(model, audio: np.ndarray, *, max_new_tokens: int) -> tuple[float, str]:
    t0 = time.perf_counter()
    text = model.transcribe(audio, max_new_tokens=max_new_tokens)
    return time.perf_counter() - t0, text


def main():
    print("Loading MLXASRModel...")
    t0 = time.perf_counter()
    model = MLXASRModel.from_pretrained(REPO_ID)
    print(f"  load: {time.perf_counter() - t0:.2f}s")

    print("Warming up (compiles MLX kernels)...")
    t0 = time.perf_counter()
    model.warmup(audio_seconds=1.0, max_new_tokens=4)
    print(f"  warmup: {time.perf_counter() - t0:.2f}s")

    full = load_clip()  # ~14.84s at 16kHz
    full_seconds = len(full) / 16000

    # Three durations: short / medium / full
    clips = [
        ("3.0s", full[: 16000 * 3]),
        ("8.0s", full[: 16000 * 8]),
        (f"{full_seconds:.2f}s", full),
    ]

    print(f"\n{'clip':>8}  {'tokens':>6}  {'wall':>7}  {'RTFx':>6}  text")
    print("-" * 80)
    for label, audio in clips:
        # Run 3 times and take the median for stability
        runs = []
        text = ""
        for _ in range(3):
            wall, text = time_transcribe(model, audio, max_new_tokens=128)
            runs.append(wall)
        runs.sort()
        wall = runs[1]
        seconds = len(audio) / 16000
        rtfx = seconds / wall
        # Token count is approx: just split words for a quick read
        tok = len(text.split())
        snippet = (text[:60] + "...") if len(text) > 60 else text
        print(f"{label:>8}  {tok:>6}  {wall:>6.2f}s  {rtfx:>5.2f}x  {snippet!r}")


if __name__ == "__main__":
    main()
