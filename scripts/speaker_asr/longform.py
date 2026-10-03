"""Speaker-attributed transcription of audio of any length (minutes to hours).

    result = transcribe_long(model, processor, audio_16k)
    result.text      # '<SPK_1>...<SPK_2>...' over the whole recording
    result.turns     # [Turn(speaker, text, start, end), ...] with recording-level ids

The audio is cut at quiet points into chunks of at most `chunk_s`. Each chunk
is decoded with a context prefix (scripts/speaker_asr/context.py): one voice
sample per speaker heard so far plus the previous chunk's tail, both with
labelled transcripts, so the model reuses a known speaker's label and opens a
new one only for a new voice. Positional labels are mapped back to
recording-level speaker ids after every chunk.

Voice samples and the tail are cut from the audio at word times from forced
alignment (tiny_audio.alignment.ForcedAligner), which also gives every turn
its start and end.

A checkpoint without the `<CONTINUE>` token (base Qwen3-ASR, or a model
trained without context) has no way to read a prefix: its chunks are decoded
independently and every chunk's speakers become new ids -- the "no linking"
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scripts.speaker_asr.context import ContextConfig, prompt_speakers
from scripts.speaker_asr.metrics import (
    CONTEXT_END,
    SPEAKER_TOKEN,
    parse_turns,
    serialize_turns,
)

SAMPLE_RATE = 16000


@dataclass
class Turn:
    """One speaker turn on the recording timeline (seconds; None if unaligned)."""

    speaker: int
    text: str
    start: float | None = None
    end: float | None = None


@dataclass
class LongFormResult:
    turns: list[Turn]
    chunks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Serialized `<SPK_n>` transcript, speakers numbered by first appearance."""
        return serialize_turns((t.speaker, t.text) for t in self.turns)


def chunk_bounds(
    audio: np.ndarray, max_s: float, min_s: float, frame_s: float = 0.1
) -> list[tuple[int, int]]:
    """Sample ranges of at most `max_s`, each cut at the quietest frame after `min_s`."""
    frame = int(frame_s * SAMPLE_RATE)
    bounds, start, n = [], 0, len(audio)
    while n - start > max_s * SAMPLE_RATE:
        lo = start + int(min_s * SAMPLE_RATE)
        hi = start + int(max_s * SAMPLE_RATE)
        k = (hi - lo) // frame
        rms = np.sqrt(np.mean(audio[lo : lo + k * frame].reshape(k, frame) ** 2, axis=1))
        cut = lo + int(np.argmin(rms)) * frame + frame // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, n))
    return bounds


def time_words(turns: list[tuple[int, str]], aligned: list[dict]) -> list[list[tuple]]:
    """Per turn, its words as (word, start, end), from one alignment of the chunk.

    `aligned` is ForcedAligner.align output for the chunk's words in order; it
    skips words with no alignable characters ("25", "--"), so it is matched
    back as a subsequence. Unaligned words are left out.
    """
    flat = [(i, w) for i, (_, text) in enumerate(turns) for w in text.split()]
    out: list[list[tuple]] = [[] for _ in turns]
    j = 0
    for item in aligned:
        while j < len(flat) and flat[j][1] != item["word"]:
            j += 1
        if j == len(flat):
            break
        out[flat[j][0]].append((item["word"], item["start"], item["end"]))
        j += 1
    return out


def best_span(words: list[tuple], lo: float, hi: float) -> tuple[float, float, str] | None:
    """Longest run of consecutive words lasting between lo and hi seconds."""
    best = None
    i = 0
    for j in range(len(words)):
        while i <= j and words[j][2] - words[i][1] > hi:
            i += 1
        if i > j:  # this word alone is longer than hi
            continue
        dur = words[j][2] - words[i][1]
        # 1 ms margin: equal-length spans differ by float noise; keep the first.
        if dur >= lo and (best is None or dur > best[1] - best[0] + 1e-3):
            best = (words[i][1], words[j][2], " ".join(w for w, _, _ in words[i : j + 1]))
    return best


class SpeakerMemory:
    """Recording-level speakers: first/last appearance and one voice sample each."""

    def __init__(self, cfg: ContextConfig):
        self.cfg = cfg
        self.first_seen: dict[int, int] = {}
        self.last_seen: dict[int, int] = {}
        self.samples: dict[int, tuple[np.ndarray, str]] = {}
        self.next_id = 1

    def new_speaker(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def seen(self, speaker: int, chunk: int) -> None:
        self.first_seen.setdefault(speaker, len(self.first_seen))
        self.last_seen[speaker] = chunk

    def offer(self, speaker: int, audio: np.ndarray, words: list[tuple]) -> None:
        """Keep the longest in-range span of `words` as this speaker's sample."""
        span = best_span(words, *self.cfg.memory_s)
        if span is None:
            return
        start, end, text = span
        clip = audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        current = self.samples.get(speaker)
        if current is None or len(clip) > len(current[0]):
            self.samples[speaker] = (clip, text)

    def prompt(self) -> list[int]:
        return prompt_speakers(
            self.first_seen, self.last_seen, lambda s: s in self.samples, self.cfg.max_memory
        )


def build_prefix(
    memory: SpeakerMemory, carry: tuple[np.ndarray, list[tuple[int, str]]] | None, cfg
) -> tuple[list[np.ndarray], str, dict[int, int]]:
    """(prefix audio pieces, prefix text ending in <CONTINUE>, speaker id -> label)."""
    speakers = memory.prompt()
    labels = {s: i for i, s in enumerate(speakers, 1)}
    gap = np.zeros(int(cfg.memory_gap_s * SAMPLE_RATE), dtype=np.float32)
    pieces, text = [], ""
    for speaker in speakers:
        clip, words = memory.samples[speaker]
        pieces += [clip, gap]
        text += SPEAKER_TOKEN.format(labels[speaker]) + words
    if carry is not None:
        carry_audio, carry_turns = carry
        kept = [(s, t) for s, t in carry_turns if s in labels]
        if kept:
            pieces.append(carry_audio)
            text += serialize_turns(kept, labels)
    if pieces:
        pieces.append(np.zeros(int(cfg.live_gap_s * SAMPLE_RATE), dtype=np.float32))
    return pieces, text + CONTEXT_END, labels


def transcribe_long(
    model,
    processor,
    audio: np.ndarray,
    cfg: ContextConfig | None = None,
    align=None,
    max_new_tokens: int = 320,
) -> LongFormResult:
    """Speaker-attributed transcript of 16 kHz mono `audio` of any length.

    `align(audio, text) -> [{"word", "start", "end"}]` defaults to
    tiny_audio.alignment.ForcedAligner.align; it is needed for context
    (samples and the tail are cut at word times) and gives turns their times.
    """
    from scripts.speaker_asr.model import n_speaker_tokens, transcribe_speakers

    cfg = cfg or ContextConfig()
    if align is None:
        from tiny_audio.alignment import ForcedAligner

        align = ForcedAligner.align
    n_tokens = n_speaker_tokens(processor)
    use_context = CONTEXT_END in processor.tokenizer.get_vocab()
    audio = np.asarray(audio, dtype=np.float32)
    memory = SpeakerMemory(cfg)
    carry = None
    turns: list[Turn] = []
    bounds = chunk_bounds(audio, cfg.chunk_s, cfg.min_chunk_s)
    for index, (s, e) in enumerate(bounds):
        live = audio[s:e]
        offset = s / SAMPLE_RATE
        if use_context:
            pieces, prefix, labels = build_prefix(memory, carry, cfg)
            (decoded,) = transcribe_speakers(
                model, processor, [np.concatenate([*pieces, live])], n_tokens,
                max_new_tokens, prefixes=[prefix],
            )  # fmt: skip
        else:
            labels = {}
            (decoded,) = transcribe_speakers(model, processor, [live], n_tokens, max_new_tokens)
        by_label = {label: speaker for speaker, label in labels.items()}
        chunk_turns = []
        for label, text in parse_turns(decoded):
            if label == 0 and turns:  # words before any token continue the last speaker
                speaker = turns[-1].speaker
            elif label not in by_label:
                by_label[label] = memory.new_speaker()
                speaker = by_label[label]
            else:
                speaker = by_label[label]
            chunk_turns.append((speaker, text))

        words = time_words(chunk_turns, align(live, " ".join(t for _, t in chunk_turns)))
        tail_from = (e - s) / SAMPLE_RATE - cfg.carry_s
        carry_turns, carry_start = [], None
        for (speaker, text), timed in zip(chunk_turns, words, strict=True):
            memory.seen(speaker, index)
            memory.offer(speaker, live, timed)
            start = timed[0][1] + offset if timed else None
            end = timed[-1][2] + offset if timed else None
            turns.append(Turn(speaker, text, start, end))
            tail = [w for w in timed if w[1] >= tail_from]
            if tail:
                carry_start = tail[0][1] if carry_start is None else carry_start
                carry_turns.append((speaker, " ".join(w for w, _, _ in tail)))
        carry = None
        if carry_turns:
            carry = (live[int(carry_start * SAMPLE_RATE) :], carry_turns)
    return LongFormResult(_merge(turns), [(s / SAMPLE_RATE, e / SAMPLE_RATE) for s, e in bounds])


def _merge(turns: list[Turn]) -> list[Turn]:
    """Join consecutive turns of one speaker (a turn split by a chunk cut)."""
    out: list[Turn] = []
    for t in turns:
        if out and out[-1].speaker == t.speaker:
            prev = out[-1]
            prev.text = f"{prev.text} {t.text}"
            prev.end = t.end if t.end is not None else prev.end
            prev.start = prev.start if prev.start is not None else t.start
        else:
            out.append(Turn(t.speaker, t.text, t.start, t.end))
    return out
