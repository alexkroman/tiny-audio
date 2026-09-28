"""Streaming endpointing with a turn-aware model: decode growing prefixes, commit to the first fire."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .model import SAMPLE_RATE, SILENCE_RMS, transcribe


def trailing_is_silent(
    audio: np.ndarray,
    window_s: float,
    sample_rate: int = SAMPLE_RATE,
    threshold: float = SILENCE_RMS,
) -> bool:
    """True when the last `window_s` of audio is below `threshold` RMS."""
    n = int(window_s * sample_rate)
    if n <= 0:
        return True
    tail = audio[-n:]
    return len(tail) > 0 and float(np.sqrt(np.mean(tail**2))) < threshold


def speech_onset_s(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, threshold: float = SILENCE_RMS
) -> float | None:
    """Start of the first 20 ms frame above `threshold` RMS (None if all silent)."""
    n = int(0.02 * sample_rate)
    frames = len(audio) // n
    if frames == 0:
        return None
    loud = np.flatnonzero(
        np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1)) > threshold
    )
    return float(loud[0] * n / sample_rate) if len(loud) else None


def first_fire_times(
    times: Sequence[float], margins: Sequence[float], taus: Sequence[float]
) -> dict[float, float | None]:
    """First window end whose marker margin exceeds each tau (None = never fired).

    `times` must be chronological. A streaming endpointer commits to its first
    fire, so later windows are irrelevant once every tau has fired.
    """
    first: dict[float, float | None] = dict.fromkeys(taus)
    for t, m in zip(times, margins, strict=True):
        for tau in taus:
            if first[tau] is None and m > tau:  # NaN never fires
                first[tau] = float(t)
    return first


def stream_fire_times(
    model,
    processor,
    audio: np.ndarray,
    taus: Sequence[float] = (0.0,),
    hop_s: float = 0.16,
    gate_s: float = 0.1,
    batch_size: int = 16,
    context: str = "",
    max_new_tokens: int = 160,
) -> dict[float, float | None]:
    """Replay `audio` as a stream; first window end (seconds) that fires, per tau.

    Every `hop_s` the model sees the WHOLE prefix so far, as in training. The
    `gate_s` energy gate skips decodes that could not fire anyway (a fire needs
    observed silence), which keeps replay affordable. One pass serves every
    tau: decoding stops once the largest tau has fired, and every smaller tau
    fired at or before that window. No window is decoded before speech onset. Pad `audio` with trailing zeros to give the
    speaker's last words a chance to be endpointed.
    """
    # Nothing to endpoint before anyone has spoken -- and a model trained with
    # context on every speech example (v4) answers context + pure silence by
    # writing the question itself and firing. Windows start after speech onset.
    onset = speech_onset_s(audio)
    if onset is None:
        return dict.fromkeys(taus)
    times = np.arange(hop_s, len(audio) / SAMPLE_RATE + 1e-6, hop_s)
    candidates = [
        t for t in times if t > onset and trailing_is_silent(audio[: int(t * SAMPLE_RATE)], gate_s)
    ]
    seen_t: list[float] = []
    seen_m: list[float] = []
    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i : i + batch_size]
        prefixes = [audio[: int(t * SAMPLE_RATE)] for t in chunk]
        decoded = transcribe(
            model, processor, prefixes, [context] * len(chunk), max_new_tokens=max_new_tokens
        )
        seen_t += [float(t) for t in chunk]
        seen_m += [d.margin for d in decoded]
        if any(d.margin > max(taus) for d in decoded):
            break
    return first_fire_times(seen_t, seen_m, taus)
