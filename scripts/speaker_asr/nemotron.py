"""Speaker-attributed transcription for the eval: stock Qwen3-ASR words, Nemotron speakers.

    result = transcribe_diarized(qwen3_asr_transcriber(model, processor), audio_16k)
    result.text      # '<SPK_1>...<SPK_2>...' over the whole recording
    result.turns     # [Turn(speaker, text, start, end), ...]

Transcribe-then-assign, not diarize-then-transcribe. The recording is
transcribed in chunks of at most `chunk_s` by stock Qwen3-ASR, every word
gets a time from Qwen3-ForcedAligner, and each word goes to the speaker
Nemotron-3-Diarization finds most active over the word's span. Cutting the audio at Nemotron's turns and transcribing each
one instead would transcribe overlapped speech twice and hand the ASR short,
context-poor pieces. Here WER is exactly the ASR's own, so the attribution
gap (cpWER - WER) measures the diarizer and the word assignment alone.

The diarizer and aligner are tiny_audio's (`NemotronDiarizer`,
`QwenForcedAligner`), the same ones the model's pipeline uses for
`return_speakers=True`; a tiny-audio checkpoint is evaluated through that
pipeline directly (scripts/eval NemotronEvaluator). This module differs only
in keeping words the aligner cannot time, so WER sees the whole transcript.
"""

from __future__ import annotations

import numpy as np

from scripts.speaker_asr.longform import SAMPLE_RATE, LongFormResult, Turn, chunk_bounds
from tiny_audio.alignment import QwenForcedAligner
from tiny_audio.diarization import NemotronDiarizer

FRAME_S = 0.01  # Nemotron emits one activity row per 10 ms


def assign_speakers(
    words: list[tuple[float | None, float | None]],
    activity: np.ndarray,
    min_activity: float = 0.1,
) -> list[int | None]:
    """Speaker column per word: the one with the highest mean activity over its span.

    A word with no time, or whose best speaker never reaches `min_activity`
    (Nemotron heard nobody there), is left None for `fill_gaps`.
    """
    out: list[int | None] = []
    for start, end in words:
        if start is None or end is None:
            out.append(None)
            continue
        lo = min(int(start / FRAME_S), len(activity) - 1)
        hi = max(lo + 1, min(int(np.ceil(end / FRAME_S)), len(activity)))
        mean = activity[lo:hi].mean(axis=0)
        best = int(mean.argmax())
        out.append(best if mean[best] >= min_activity else None)
    return out


def fill_gaps(speakers: list[int | None]) -> list[int]:
    """Unassigned words join the previous speaker (the next one at the start)."""
    known = [s for s in speakers if s is not None]
    if not known:
        return [0] * len(speakers)
    filled, last = [], known[0]
    for s in speakers:
        last = s if s is not None else last
        filled.append(last)
    return filled


def time_all_words(text: str, aligned: list[dict], offset: float) -> list[tuple]:
    """(word, start, end) for EVERY word of `text`; None times where alignment skipped it.

    ForcedAligner drops words with no alignable characters ("25", "--"). They
    still count toward WER, so they stay in the transcript with no time
    instead of being dropped (unlike longform.time_words).
    """
    out, j = [], 0
    for word in text.split():
        if j < len(aligned) and aligned[j]["word"] == word:
            out.append((word, aligned[j]["start"] + offset, aligned[j]["end"] + offset))
            j += 1
        else:
            out.append((word, None, None))
    return out


def group_turns(words: list[tuple], speakers: list[int]) -> list[Turn]:
    """Consecutive words of one speaker become one turn."""
    turns: list[Turn] = []
    for (word, start, end), speaker in zip(words, speakers, strict=True):
        if turns and turns[-1].speaker == speaker:
            turn = turns[-1]
            turn.text += " " + word
            turn.end = end if end is not None else turn.end
        else:
            turns.append(Turn(speaker, word, start, end))
    return turns


def qwen3_asr_transcriber(model, processor, max_new_tokens: int = 320):
    """`audio -> text` for a stock Qwen3-ASR checkpoint (no speaker tokens: plain text)."""
    from scripts.speaker_asr.model import transcribe_speakers

    def transcribe(audio: np.ndarray) -> str:
        (text,) = transcribe_speakers(model, processor, [audio], 0, max_new_tokens)
        return text

    return transcribe


def transcribe_diarized(
    transcribe,
    audio: np.ndarray,
    diarizer=NemotronDiarizer,
    chunk_s: float = 25.0,
    min_chunk_s: float = 10.0,
    min_activity: float = 0.1,
    align=None,
) -> LongFormResult:
    """Speaker-attributed transcript of 16 kHz mono `audio` of any length.

    `transcribe(audio_chunk) -> text` is any plain ASR: speakers come from
    Nemotron alone, so the ASR never needs speaker tokens.
    """
    align = align or QwenForcedAligner.align
    audio = np.asarray(audio, dtype=np.float32)
    activity = diarizer.activity(audio)

    words: list[tuple] = []
    bounds = chunk_bounds(audio, chunk_s, min_chunk_s)
    for s, e in bounds:
        live = audio[s:e]
        text = transcribe(live)
        if text.strip():
            words += time_all_words(text, align(live, text), s / SAMPLE_RATE)

    speakers = fill_gaps(assign_speakers([(a, b) for _, a, b in words], activity, min_activity))
    return LongFormResult(
        group_turns(words, speakers), [(s / SAMPLE_RATE, e / SAMPLE_RATE) for s, e in bounds]
    )
