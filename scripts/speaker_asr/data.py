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
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

from scripts.speaker_asr.metrics import CONTEXT_END, serialize_turns
from scripts.turn_aware.data import TurnAwareCollator

SAMPLE_RATE = 16000


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

    def verify(self, rows: list[dict]) -> None:
        """Refuse manifest rows whose parts point at a different utterance.

        Parts address audio by dataset row index, which is only stable while
        the source parquet is unchanged; a manifest pulled from the Hub could
        otherwise silently pair one utterance's audio with another's words.
        """
        ids = self.meta["id"].to_numpy()
        for row in rows:
            for part in json.loads(row["parts"]):
                r = int(part["row"])
                if r >= len(ids) or ids[r] != part["id"]:
                    raise ValueError(
                        f"pool part {part['id']} is row {r} in its manifest, but that row of "
                        "the source dataset is a different utterance; the dataset changed "
                        "since the pool was built. Rebuild with `ta speaker-asr build-pool -f`."
                    )


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
    ordered = sorted(parts, key=lambda p: (p["offset_s"], p["dur_s"]))
    turns = [(p["speaker"], texts.get(p["id"], "")) for p in ordered]
    speakers = {speaker for speaker, text in turns if text.strip()}
    return serialize_turns(turns), len(speakers)


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


def meeting_parts(meta, group: str, max_s: float | None = None) -> list[dict]:
    """Every utterance of one meeting at its REAL time (no gap squeezing).

    `meta` is UtteranceStore.meta. Mixing these gives the whole meeting as
    one long recording -- the long-form evaluation input. With `max_s`, only
    utterances that end within the first `max_s` seconds are kept.
    """
    rows = meta[meta["group"] == group].sort_values(["start", "end", "id"])
    t0 = float(rows["start"].min())
    parts = [
        {
            "row": int(r["row"]),
            "id": r["id"],
            "speaker": r["speaker"],
            "offset_s": round(float(r["start"]) - t0, 3),
            "dur_s": round(float(r["end"]) - float(r["start"]), 3),
        }
        for r in rows.to_dict("records")
    ]
    if max_s:
        parts = [p for p in parts if p["offset_s"] + p["dur_s"] <= max_s]
    return parts


def load_texts(cfg, split: str, meta) -> dict[str, str]:
    """Utterance id -> transcript the pool was built with (self or dataset text).

    Context rows need every utterance's own text, not just window targets:
    self-distilled ones come from the pool's transcripts-{split}.jsonl (pulled
    with the pool from pool.hub_repo).
    """
    from pathlib import Path

    from scripts.turn_aware.transcripts import read_cache

    if cfg.pool.target_text != "self":
        return dict(zip(meta["id"], meta["text"].fillna(""), strict=True))
    cache = Path(cfg.pool.transcript_cache or cfg.data.pool_dir) / f"transcripts-{split}.jsonl"
    texts = read_cache(cache)
    if not texts:
        raise FileNotFoundError(
            f"{cache} is missing or empty; `ta speaker-asr build-pool --split {split}` "
            "writes (or pulls) it"
        )
    return texts


class SpeakerASRCollator(TurnAwareCollator):
    """TurnAwareCollator's chat layout, with an optional context prefix unsupervised.

    The assistant turn is `language English<asr_text>` + prefix + target;
    labels start after the prefix's `<CONTINUE>`, so the prefix -- which
    decoding pre-fills -- is context, never a prediction.
    """

    def __init__(self, processor):
        super().__init__(processor)
        self.context_end_id = processor.tokenizer.get_vocab().get(CONTEXT_END)

    def __call__(self, batch: list[dict]) -> dict:
        batch = [{**item, "target": item.get("prefix", "") + item["target"]} for item in batch]
        enc = super().__call__(batch)
        if self.context_end_id is not None:
            for i, row in enumerate(enc["input_ids"]):
                hits = (row == self.context_end_id).nonzero()
                if len(hits):
                    enc["labels"][i, : int(hits[-1]) + 1] = -100
        return enc


class SpeakerASRDataset(torch.utils.data.Dataset):
    """Manifest rows -> {"audio", "ctx", "prefix", "target"} for SpeakerASRCollator."""

    def __init__(self, rows: list[dict], store: UtteranceStore):
        store.verify(rows)
        self.rows = rows
        self.store = store

    def __len__(self) -> int:
        return len(self.rows)

    def audio(self, idx: int) -> np.ndarray:
        row = self.rows[idx]
        return mix_window(json.loads(row["parts"]), self.store, row["tail_s"])

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            "audio": self.audio(idx),
            "ctx": "",
            "prefix": row.get("prefix") or "",
            "target": row["target"],
        }
