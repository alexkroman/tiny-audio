"""Turn-aware training pool built from mazesmazes/turn-end-detection.

Each dataset observation is a prefix `turns.audio[:at_s]` plus a causal label:
1 if the caller had finished, judged only from what was heard so far. That is
exactly the supervision an end-of-turn token needs, with one missing piece --
the marker should fire only once silence has actually been OBSERVED, and the
TTS clips carry just ~0.1-0.3 s of tail. So every prefix is tail-trimmed to
its last speech frame and then re-extended with a controlled silence:

    schema        audio                            target
    fire_sil      label-1 prefix + 0.3-1.2 s sil   "T<END_OF_TURN>"
    fire_nosil    SAME prefix, no tail             "T"
    fire_long     label-1 prefix + 2-4 s sil       "T<END_OF_TURN>"
    hold_sil      label-0 prefix + 0.3-2.0 s sil   "T"
    silence_only  0.5-4 s of zeros                 ""

fire_sil / fire_nosil share the identical speech, so observed silence is the
only feature separating their labels; hold_sil is the converse, silence that
must NOT fire because the thought is unfinished (payload_cut, trailing_frag,
word_cut). Either pairing alone is learnable by a shortcut; both together
force the model to read semantics AND the tail.

`T` defaults to the BASE model's own transcript of the trimmed prefix
(self-distillation), so the only new behaviour the adapter is asked to learn
is the marker; the dataset's `text` column is another recognizer's output,
errors included, and training on it would pull the transcription toward that
recognizer's style.

The manifest stores (turn_key, speech_end_s, lead_s, tail_s), never audio:
inlining ~120k waveforms would cost ~80 GB, and the arrow-backed turns table
is memory-mapped and fork-safe, so workers slice audio on demand.
"""

from __future__ import annotations

import io
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch

DATASET_ID = "mazesmazes/turn-end-detection"
SAMPLE_RATE = 16000
END_OF_TURN = "<END_OF_TURN>"
LANGUAGE = "English"


# --------------------------------------------------------------------- audio


def trim_tail_silence(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    threshold: float = 0.005,
    frame_ms: float = 20.0,
    keep_ms: float = 80.0,
) -> int:
    """Return the sample index ~keep_ms past the last frame with RMS > threshold.

    80 ms of tail is kept so a trimmed clip does not end mid-release; it is
    well under the 0.3 s floor a fire needs, so `fire_nosil` stays a clean
    negative. All-quiet audio returns 0.
    """
    n = int(sample_rate * frame_ms / 1000)
    frames = len(audio) // n
    if frames == 0:
        return len(audio)
    rms = np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1))
    active = np.flatnonzero(rms > threshold)
    if len(active) == 0:
        return 0
    return min(len(audio), (int(active[-1]) + 1) * n + int(sample_rate * keep_ms / 1000))


def trailing_is_silent(
    audio: np.ndarray, window_s: float, sample_rate: int = SAMPLE_RATE, threshold: float = 0.005
) -> bool:
    """True when the last `window_s` of audio is below `threshold` RMS."""
    n = int(window_s * sample_rate)
    if n <= 0:
        return True
    tail = audio[-n:]
    return len(tail) > 0 and float(np.sqrt(np.mean(tail**2))) < threshold


def assemble_audio(turn_audio: np.ndarray, row: dict, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Leading zeros + the speech prefix + trailing zeros, as float32."""
    speech = turn_audio[: round(row["speech_end_s"] * sample_rate)]
    lead = np.zeros(round(row["lead_s"] * sample_rate), dtype=np.float32)
    tail = np.zeros(round(row["tail_s"] * sample_rate), dtype=np.float32)
    return np.concatenate([lead, speech.astype(np.float32), tail])


def decode_flac(data: bytes) -> np.ndarray:
    """Decode FLAC bytes to mono float32 at the file's rate (16 kHz here)."""
    audio, _ = sf.read(io.BytesIO(data), dtype="float32")
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def load_split(config: str, split: str, dataset_id: str = DATASET_ID):
    """One split of one config, downloading only that split's shards.

    `load_dataset(id, "turns", split="validation")` prepares EVERY split of
    the config first -- all 54 train shards, ~9 GB -- before returning the
    800 MB asked for. Naming the split's files directly avoids that.
    """
    from datasets import load_dataset

    files = {split: f"{config}/{config}-{split}-*.parquet"}
    return load_dataset(dataset_id, data_files=files, split=split)


class TurnAudioStore:
    """turn_key -> waveform over one split of the `turns` config.

    Loaded with `decode=False` so `datasets` never reaches for torchcodec;
    the FLAC bytes are decoded here with soundfile. A small LRU cache pays
    off because several observations (and their silence variants) share a
    turn.
    """

    def __init__(self, split: str, dataset_id: str = DATASET_ID):
        from datasets import Audio

        ds = load_split("turns", split, dataset_id)
        self._ds = ds.cast_column("audio", Audio(decode=False)).select_columns(["audio"])
        self._index = {key: i for i, key in enumerate(ds["turn_key"])}
        self.meta = {
            key: {"agent_turn": agent or "", "audio_duration_s": float(dur)}
            for key, agent, dur in zip(
                ds["turn_key"], ds["agent_turn"], ds["audio_duration_s"], strict=True
            )
        }

    def __contains__(self, turn_key: str) -> bool:
        return turn_key in self._index

    @lru_cache(maxsize=64)  # noqa: B019 -- one store per process, lives for the run
    def get(self, turn_key: str) -> np.ndarray:
        if not turn_key:
            return np.zeros(0, dtype=np.float32)
        return decode_flac(self._ds[self._index[turn_key]]["audio"]["bytes"])


# ---------------------------------------------------------------------- pool


@dataclass(frozen=True)
class PoolConfig:
    """Silence and mixing knobs for `build_pool`. Durations are (low, high) seconds."""

    fire_tail_s: tuple[float, float] = (0.3, 1.2)
    hold_tail_s: tuple[float, float] = (0.3, 2.0)
    long_tail_s: tuple[float, float] = (2.0, 4.0)
    # Share of positives whose silenced copy uses the long tail instead, so the
    # marker survives long pauses rather than being learned as "fires at <1.2 s".
    long_tail_prob: float = 0.1
    # Share of positives that ALSO get the no-tail twin. <1 keeps the pool from
    # being half hard-negatives at identical speech.
    nosil_prob: float = 0.6
    # Streaming windows open mid-silence; leading zeros must not change anything.
    lead_sil_s: tuple[float, float] = (0.5, 2.0)
    lead_sil_prob: float = 0.15
    silence_only_frac: float = 0.02
    silence_only_s: tuple[float, float] = (0.5, 4.0)
    # Share of examples that see the agent's preceding question as system
    # context. Below 1 so the model still endpoints with no context at all.
    ctx_prob: float = 0.5
    max_audio_s: float = 30.0


def _example(obs: dict, schema: str, tail_s: float, fire: bool, lead_s: float, ctx: str) -> dict:
    return {
        "key": f"{obs['key']}#{schema}",
        "turn_key": obs["turn_key"],
        "kind": obs["kind"],
        "schema": schema,
        "speech_end_s": float(obs["speech_end_s"]),
        "lead_s": round(lead_s, 3),
        "tail_s": round(tail_s, 3),
        "text": obs["target"],
        "fire": fire,
        "ctx": ctx,
    }


def expand_observation(obs: dict, cfg: PoolConfig, rng: random.Random) -> list[dict]:
    """Training examples for one observation.

    `obs` needs key, turn_key, kind, label, speech_end_s, target, agent_turn.
    """
    lead = rng.uniform(*cfg.lead_sil_s) if rng.random() < cfg.lead_sil_prob else 0.0
    ctx = obs.get("agent_turn") or ""
    ctx = ctx if ctx and rng.random() < cfg.ctx_prob else ""
    if not obs["label"]:
        return [_example(obs, "hold_sil", rng.uniform(*cfg.hold_tail_s), False, lead, ctx)]
    if rng.random() < cfg.long_tail_prob:
        out = [_example(obs, "fire_long", rng.uniform(*cfg.long_tail_s), True, lead, ctx)]
    else:
        out = [_example(obs, "fire_sil", rng.uniform(*cfg.fire_tail_s), True, lead, ctx)]
    if rng.random() < cfg.nosil_prob:
        out.append(_example(obs, "fire_nosil", 0.0, False, lead, ctx))
    return out


def build_pool(observations: list[dict], cfg: PoolConfig, seed: int = 0) -> list[dict]:
    """Expand observations into the training pool, deterministic in `seed`.

    Observations with no speech after trimming, or whose assembled audio would
    exceed `cfg.max_audio_s`, are dropped. Order is shuffled.
    """
    rng = random.Random(seed)
    pool: list[dict] = []
    for obs in sorted(observations, key=lambda o: o["key"]):
        if obs["speech_end_s"] <= 0:
            continue
        for ex in expand_observation(obs, cfg, rng):
            if ex["lead_s"] + ex["speech_end_s"] + ex["tail_s"] <= cfg.max_audio_s:
                pool.append(ex)
    for i in range(round(cfg.silence_only_frac * len(pool))):
        pool.append(
            {
                "key": f"silence_only_{seed}_{i}",
                "turn_key": "",
                "kind": "silence_only",
                "schema": "silence_only",
                "speech_end_s": 0.0,
                "lead_s": 0.0,
                "tail_s": round(rng.uniform(*cfg.silence_only_s), 3),
                "text": "",
                "fire": False,
                "ctx": "",
            }
        )
    rng.shuffle(pool)
    return pool


def stratified_subset(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Up to `n` rows spread evenly across schemas, for a cheap balanced eval."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    by_schema: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_schema[row["schema"]].append(row)
    rng = random.Random(seed)
    for group in by_schema.values():
        rng.shuffle(group)
    per = max(1, n // len(by_schema))
    return [row for group in by_schema.values() for row in group[:per]]


# ------------------------------------------------------------------ training


def target_text(row: dict) -> str:
    return row["text"] + (END_OF_TURN if row["fire"] else "")


class TurnAwareDataset(torch.utils.data.Dataset):
    """Manifest rows -> {"audio", "ctx", "target"} with audio assembled on demand."""

    def __init__(self, rows: list[dict], store: TurnAudioStore):
        self.rows = rows
        self.store = store

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        audio = assemble_audio(self.store.get(row["turn_key"]), row)
        return {"audio": audio, "ctx": row["ctx"], "target": target_text(row)}


def assistant_labels(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, asr_text_id: int, im_end_id: int
) -> torch.Tensor:
    """Supervise only the transcript: tokens after `<asr_text>` through `<|im_end|>`.

    The `language English<asr_text>` prefix is masked because inference
    prefills it (apply_transcription_request forces the language), and the
    template's trailing newline after `<|im_end|>` is masked because decoding
    stops at `<|im_end|>` and never sees it.
    """
    labels = torch.full_like(input_ids, -100)
    for i, row in enumerate(input_ids):
        starts = (row == asr_text_id).nonzero()
        if len(starts) == 0:
            continue
        start = int(starts[-1]) + 1
        ends = (row[start:] == im_end_id).nonzero()
        end = start + int(ends[0]) + 1 if len(ends) else len(row)
        labels[i, start:end] = row[start:end]
    labels[attention_mask == 0] = -100
    return labels


class TurnAwareCollator:
    """Build chat-formatted Qwen3-ASR batches with transcript-only labels."""

    def __init__(self, processor, language: str = LANGUAGE):
        self.processor = processor
        self.prefix = f"language {language}<asr_text>"
        tokenizer = processor.tokenizer
        self.asr_text_id = tokenizer.convert_tokens_to_ids("<asr_text>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, batch: list[dict]) -> dict:
        conversations = []
        for item in batch:
            messages = []
            if item["ctx"]:
                messages.append(
                    {"role": "system", "content": [{"type": "text", "text": item["ctx"]}]}
                )
            messages.append(
                {"role": "user", "content": [{"type": "audio", "audio": item["audio"]}]}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.prefix + item["target"]}],
                }
            )
            conversations.append(messages)
        enc = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        enc["labels"] = assistant_labels(
            enc["input_ids"], enc["attention_mask"], self.asr_text_id, self.im_end_id
        )
        return dict(enc)


# ------------------------------------------------------------------- metrics


def marker_metrics(
    rows: list[dict], preds: list[tuple[str, bool]], normalize=None, text_wer: bool = True
) -> dict:
    """Fire precision/recall/F1, per-schema and per-kind accuracy, transcript drift.

    `text_wer` is against the row's target transcript. With self-distilled
    targets that is the BASE model's own output, so it measures how far the
    adapter has moved transcription rather than absolute accuracy.
    """
    import jiwer

    tp = sum(r["fire"] and p for r, (_, p) in zip(rows, preds, strict=True))
    fp = sum(not r["fire"] and p for r, (_, p) in zip(rows, preds, strict=True))
    fn = sum(r["fire"] and not p for r, (_, p) in zip(rows, preds, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "n": len(rows),
        "fire_precision": precision,
        "fire_recall": recall,
        "fire_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    for field, prefix in (("schema", "acc_schema"), ("kind", "acc_kind")):
        hits, totals = Counter(), Counter()
        for r, (_, p) in zip(rows, preds, strict=True):
            totals[r[field]] += 1
            hits[r[field]] += int(p == r["fire"])
        out.update({f"{prefix}/{k}": hits[k] / totals[k] for k in sorted(totals)})

    if not text_wer:
        return out
    norm = normalize or (lambda s: " ".join(s.lower().split()))
    # Silence-only rows have no reference; WER is undefined there.
    pairs = [(norm(r["text"]), norm(t)) for r, (t, _) in zip(rows, preds, strict=True)]
    pairs = [(ref, hyp) for ref, hyp in pairs if ref]
    if pairs:
        refs, hyps = zip(*pairs, strict=True)
        out["text_wer"] = jiwer.wer(list(refs), [h or "<empty>" for h in hyps])
    return out


SWEEP_METRICS = (
    "fire_precision",
    "fire_recall",
    "fire_f1",
    "acc_schema/hold_sil",
    "acc_schema/fire_nosil",
    "acc_kind/payload_cut",
    "acc_kind/word_cut",
    "acc_kind/trailing_frag",
)


def threshold_sweep(
    rows: list[dict], margins: list[float], taus: list[float], metrics=SWEEP_METRICS
) -> list[dict]:
    """Fire metrics when a fire requires marker margin > tau, for each tau.

    tau = 0 reproduces greedy decoding. A NaN margin (the decode never ended)
    never fires.
    """
    out = []
    for tau in taus:
        preds = [("", bool(m > tau)) for m in margins]  # NaN > tau is False
        scores = marker_metrics(rows, preds, text_wer=False)
        out.append({"tau": tau, **{k: scores[k] for k in metrics if k in scores}})
    return out


def first_fire_times(
    times: list[float], margins: list[float], taus: list[float]
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


def endpoint_summary(fire_s, duration_s) -> dict:
    """Streaming endpoint quality, with the definitions of the `endpoints` table.

    `cut_early` is a first fire before the audio ends (the caller is talked
    over), `missed` is no fire at all, and latency is measured from the end of
    the audio for the fires that were not early. NaN/None in `fire_s` = miss.
    """
    fire = np.asarray(fire_s, dtype=float)
    duration = np.asarray(duration_s, dtype=float)
    fired = ~np.isnan(fire)
    early = fired & (fire < duration)
    on_time = fired & ~early
    latency = fire[on_time] - duration[on_time]
    n = len(fire)
    return {
        "n": n,
        "cut_early": float(early.sum() / n) if n else 0.0,
        "missed": float((~fired).sum() / n) if n else 0.0,
        "latency_p50_s": float(np.median(latency)) if len(latency) else float("nan"),
        "latency_p90_s": float(np.percentile(latency, 90)) if len(latency) else float("nan"),
    }
