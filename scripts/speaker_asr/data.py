"""Speaker-attributed windows re-mixed from AMI's per-speaker headset segments.

AMI ihm has one row per utterance: one speaker's close-talk audio plus its
begin/end time on the meeting timeline. A window is a run of consecutive
utterances of one meeting, placed at their real relative offsets (gaps longer
than `max_gap_s` squeezed) and summed, so turn-taking, backchannels and
overlaps are the meeting's own. Its target is the serialized transcript

    <SPK_1>So what do we think?<SPK_2>I like it.<SPK_1>Yeah.

speakers numbered by first appearance, turns in start-time order (first-in
first-out under overlap), consecutive utterances of one speaker merged.

The manifest stores each window's parts as (utterance row, offset, duration),
never audio; workers decode and mix on demand from the memory-mapped split.
"""

from __future__ import annotations

import io
import json
import random
import re
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
SPEAKER_TOKEN = "<SPK_{}>"
_SPEAKER_RE = re.compile(r"<SPK_(\d+)>")


def speaker_tokens(n: int) -> list[str]:
    """`<SPK_1>` .. `<SPK_n>`."""
    return [SPEAKER_TOKEN.format(i) for i in range(1, n + 1)]


# --------------------------------------------------------------------- audio


def decode_audio(audio: dict) -> np.ndarray:
    """An undecoded `datasets` Audio cell -> mono float32 at 16 kHz."""
    data = audio.get("bytes")
    wav, sr = sf.read(io.BytesIO(data) if data else audio["path"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav


def load_split(dataset_id: str, data_files: str, split: str):
    """One split's parquet shards only (not every split of the config)."""
    from datasets import load_dataset

    files = {split: data_files.format(split=split)}
    return load_dataset(dataset_id, data_files=files, split=split)


class UtteranceStore:
    """Row index -> waveform over one split, decoded with soundfile.

    `meta` is every non-audio column as a DataFrame, renamed to the canonical
    names (id, group, speaker, start, end, text) plus `row`, the dataset index.
    """

    def __init__(self, dataset_id: str, data_files: str, split: str, columns: dict):
        from datasets import Audio

        ds = load_split(dataset_id, data_files, split)
        rename = {src: dst for dst, src in columns.items()}
        self.meta = ds.select_columns(list(rename)).to_pandas().rename(columns=rename)
        self.meta["row"] = np.arange(len(self.meta))
        self._ds = ds.cast_column("audio", Audio(decode=False)).select_columns(["audio"])

    def get(self, row: int) -> np.ndarray:
        return decode_audio(self._ds[int(row)]["audio"])


def mix_window(parts: list[dict], store: UtteranceStore, tail_s: float = 0.0) -> np.ndarray:
    """Sum each part's audio at its offset; rescale only if the sum clips."""
    clips = [(round(p["offset_s"] * SAMPLE_RATE), store.get(p["row"])) for p in parts]
    end = max((start + len(c) for start, c in clips), default=0)
    out = np.zeros(end + round(tail_s * SAMPLE_RATE), dtype=np.float32)
    for start, clip in clips:
        out[start : start + len(clip)] += clip
    peak = float(np.abs(out).max()) if len(out) else 0.0
    return out / peak * 0.99 if peak > 1.0 else out


# --------------------------------------------------------------------- pool


@dataclass(frozen=True)
class WindowConfig:
    """Window-building knobs (configs/speaker_asr `pool:`). Durations in seconds."""

    window_s: tuple[float, float] = (8.0, 30.0)
    max_audio_s: float = 30.0
    lead_s: tuple[float, float] = (0.0, 0.5)
    tail_s: tuple[float, float] = (0.0, 0.5)
    max_gap_s: float = 1.5
    overlap: bool = True
    min_gap_s: float = 0.1
    max_speakers: int = 4
    min_utt_s: float = 0.2


def plan_windows(utts: list[dict], cfg: WindowConfig, seed: int = 0) -> list[dict]:
    """Cut each group's utterances (sorted by start) into consecutive windows.

    `utts` need row, id, group, speaker, start, end. A window closes before an
    utterance that would push it past its drawn length (or max_audio_s), or
    that brings a speaker beyond `max_speakers`; that utterance opens the next
    window. Offsets keep real gaps up to `max_gap_s` and real overlaps (when
    `overlap`), measured from the latest end so far, so a backchannel inside a
    long turn stays inside it. An utterance longer than max_audio_s alone is
    dropped. Deterministic in `seed`.
    """
    rng = random.Random(seed)
    by_group: dict[str, list[dict]] = {}
    for u in utts:
        if u["end"] - u["start"] >= cfg.min_utt_s:
            by_group.setdefault(u["group"], []).append(u)

    windows = []
    for group in sorted(by_group):
        items = sorted(by_group[group], key=lambda u: (u["start"], u["end"], u["id"]))
        i = 0
        while i < len(items):
            limit = min(rng.uniform(*cfg.window_s), cfg.max_audio_s)
            tail = rng.uniform(*cfg.tail_s)
            lead = rng.uniform(*cfg.lead_s)
            parts, speakers = [], set()
            placed_end = real_end = 0.0
            j = i
            while j < len(items):
                u = items[j]
                dur = u["end"] - u["start"]
                if parts:
                    gap = min(u["start"] - real_end, cfg.max_gap_s)
                    if not cfg.overlap:
                        gap = max(gap, cfg.min_gap_s)
                    offset = max(0.0, placed_end + gap)
                else:
                    offset = lead
                fits = offset + dur + tail <= (limit if parts else cfg.max_audio_s)
                new_speaker = u["speaker"] not in speakers
                if not fits or (new_speaker and len(speakers) == cfg.max_speakers):
                    break
                parts.append(
                    {
                        "row": int(u["row"]),
                        "id": u["id"],
                        "speaker": u["speaker"],
                        "offset_s": round(offset, 3),
                        "dur_s": round(dur, 3),
                    }
                )
                speakers.add(u["speaker"])
                placed_end = max(placed_end, offset + dur)
                real_end = max(real_end, u["end"])
                j += 1
            if not parts:  # a single utterance longer than max_audio_s
                i += 1
                continue
            windows.append(
                {
                    "key": f"{group}:{parts[0]['id']}",
                    "group": group,
                    "parts": parts,
                    "tail_s": round(tail, 3),
                    "duration_s": round(placed_end + tail, 3),
                }
            )
            i = j
    return windows


def format_target(parts: list[dict], texts: dict[str, str]) -> tuple[str, int]:
    """Serialized `<SPK_n>text...` target for a window and its speaker count.

    Parts are ordered by offset (first-in first-out under overlap); a part
    whose transcript is empty contributes nothing, not even a speaker number,
    since the model cannot be asked to label a voice it was given no words for.
    """
    labels: dict[str, int] = {}
    turns: list[list] = []
    for part in sorted(parts, key=lambda p: (p["offset_s"], p["dur_s"])):
        text = texts.get(part["id"], "").strip()
        if not text:
            continue
        label = labels.setdefault(part["speaker"], len(labels) + 1)
        if turns and turns[-1][0] == label:
            turns[-1][1] += " " + text
        else:
            turns.append([label, text])
    target = "".join(SPEAKER_TOKEN.format(label) + text for label, text in turns)
    return target, len(labels)


def build_pool(windows: list[dict], texts: dict[str, str]) -> list[dict]:
    """Manifest rows: targets attached, parts as JSON, windows with no words dropped."""
    rows = []
    for w in windows:
        target, n_speakers = format_target(w["parts"], texts)
        if not target:
            continue
        rows.append(
            {
                **w,
                "parts": json.dumps(w["parts"]),
                "target": target,
                "n_speakers": n_speakers,
                "n_parts": len(w["parts"]),
            }
        )
    return rows


def subset(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Up to `n` rows spread evenly across speaker counts, for a balanced eval."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    by_count: dict[int, list[dict]] = {}
    for row in rows:
        by_count.setdefault(int(row["n_speakers"]), []).append(row)
    rng = random.Random(seed)
    per = max(1, n // len(by_count))
    out = []
    for count in sorted(by_count):
        group = by_count[count]
        rng.shuffle(group)
        out += group[:per]
    return out


# ------------------------------------------------------------------ training


class SpeakerASRDataset(torch.utils.data.Dataset):
    """Manifest rows -> {"audio", "ctx", "target"} (TurnAwareCollator's item shape)."""

    def __init__(self, rows: list[dict], store: UtteranceStore):
        self.rows = rows
        self.store = store

    def __len__(self) -> int:
        return len(self.rows)

    def audio(self, idx: int) -> np.ndarray:
        row = self.rows[idx]
        return mix_window(json.loads(row["parts"]), self.store, row["tail_s"])

    def __getitem__(self, idx: int) -> dict:
        return {"audio": self.audio(idx), "ctx": "", "target": self.rows[idx]["target"]}


# ------------------------------------------------------------------- metrics


def parse_turns(text: str) -> list[tuple[int, str]]:
    """'<SPK_1>a<SPK_2>b' -> [(1, 'a'), (2, 'b')]. Text before any token is speaker 0."""
    pieces = _SPEAKER_RE.split(text)
    turns = [(0, pieces[0].strip())] if pieces[0].strip() else []
    for label, chunk in zip(pieces[1::2], pieces[2::2], strict=True):
        if chunk.strip():
            turns.append((int(label), chunk.strip()))
    return turns


def plain_text(text: str) -> str:
    """The transcript with speaker tokens removed."""
    return " ".join(t for _, t in parse_turns(text))


def word_errors(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def _speaker_words(text: str, normalize) -> dict[int, list[str]]:
    words: dict[int, list[str]] = {}
    for label, chunk in parse_turns(text):
        words.setdefault(label, []).extend(normalize(chunk).split())
    return words


def cp_errors(ref: str, hyp: str, normalize=lambda s: s) -> tuple[int, int]:
    """(errors, reference words) of concatenated minimum-permutation WER.

    Each speaker's words are concatenated, and reference and hypothesis
    speakers are matched one-to-one to minimise total word errors (unmatched
    speakers count against empty), so speaker labels are permutation-free and
    a word given to the wrong voice costs a deletion plus an insertion.
    """
    from scipy.optimize import linear_sum_assignment

    r = list(_speaker_words(ref, normalize).values())
    h = list(_speaker_words(hyp, normalize).values())
    n = max(len(r), len(h), 1)
    r += [[]] * (n - len(r))
    h += [[]] * (n - len(h))
    cost = np.array([[word_errors(a, b) for b in h] for a in r])
    rows, cols = linear_sum_assignment(cost)
    return int(cost[rows, cols].sum()), sum(len(a) for a in r)


def speaker_metrics(refs: list[str], hyps: list[str], normalize=lambda s: s) -> dict:
    """WER (speakers ignored), cpWER, their gap, and speaker-count accuracy.

    `cpwer - wer` is what speaker attribution costs on top of recognition;
    by speaker count (`cpwer_<k>spk`) shows where it breaks down.
    """
    wer_err = cp_err = n_words = count_hits = count_abs = 0
    by_count: dict[int, list[int]] = {}
    for ref, hyp in zip(refs, hyps, strict=True):
        ref_words = normalize(plain_text(ref)).split()
        wer_err += word_errors(ref_words, normalize(plain_text(hyp)).split())
        errors, words = cp_errors(ref, hyp, normalize)
        cp_err += errors
        n_words += words
        n_ref = len({label for label, _ in parse_turns(ref)})
        n_hyp = len({label for label, _ in parse_turns(hyp)})
        count_hits += n_ref == n_hyp
        count_abs += abs(n_ref - n_hyp)
        bucket = by_count.setdefault(n_ref, [0, 0])
        bucket[0] += errors
        bucket[1] += words
    n = max(len(refs), 1)
    out = {
        "wer": wer_err / max(n_words, 1),
        "cpwer": cp_err / max(n_words, 1),
        "speaker_count_acc": count_hits / n,
        "speaker_count_mae": count_abs / n,
        "n": len(refs),
    }
    out["attribution_gap"] = out["cpwer"] - out["wer"]
    for k in sorted(by_count):
        out[f"cpwer_{k}spk"] = by_count[k][0] / max(by_count[k][1], 1)
    return out
