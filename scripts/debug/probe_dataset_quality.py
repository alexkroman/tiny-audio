#!/usr/bin/env python3
"""Broader data-quality probe across every multiasr dataset.

Beyond the regex check (probe_label_regex.py), this looks at:
  - audio decode failures / NaN / Inf
  - audio multi-channel rate (squashed via mean — silent data loss
    indicator if a dataset ships stereo by default)
  - duration distribution + DataCollator filter loss (<0.8s, >30s)
  - label length distribution (chars, words)
  - words-per-second ratio (label/audio-drift detector — extreme
    values suggest misaligned segments)
  - character-class outliers (CJK / emoji / control / mojibake-shaped)
  - balanced-quote / balanced-paren sanity
  - leading/trailing-punctuation oddities
  - repeated-substring labels ("yeah yeah yeah ..." style)
  - blank labels post-normalization

Streams a small N per dataset (default 30) so we don't download full
shards. Audio decode cost dominates runtime; 30 samples per dataset is
usually ~3-5 min total.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import Audio, load_dataset

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
from train import _normalize_label  # noqa: E402

SAMPLE_RATE = 16_000
MIN_AUDIO_S = 0.8
MAX_AUDIO_S = 30.0

# Character-class predicates
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
ARABIC_RE = re.compile(r"[؀-ۿ]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MOJIBAKE_RE = re.compile(r"(â€|Ã©|Ã¨|Ã¢|Â£|Â§|Ã±|Ã¶)")
DIGIT_ONLY_RE = re.compile(r"^[\d\s.,$%]+$")  # numeric-only labels
HTML_ENT_RE = re.compile(r"&(amp|lt|gt|quot|apos|#\d+);")


def char_class_anomalies(text: str) -> list[str]:
    out = []
    if CJK_RE.search(text):
        out.append("CJK")
    if ARABIC_RE.search(text):
        out.append("ARABIC")
    if CYRILLIC_RE.search(text):
        out.append("CYRILLIC")
    if EMOJI_RE.search(text):
        out.append("EMOJI")
    if CONTROL_RE.search(text):
        out.append("CONTROL_CHAR")
    if MOJIBAKE_RE.search(text):
        out.append("MOJIBAKE_SHAPE")
    if HTML_ENT_RE.search(text):
        out.append("HTML_ENTITY_SURVIVED")
    # any non-ASCII letter that isn't standard Latin-1 punct
    non_ascii = [c for c in text if ord(c) > 127]
    if non_ascii and not (
        CJK_RE.search(text)
        or ARABIC_RE.search(text)
        or CYRILLIC_RE.search(text)
        or EMOJI_RE.search(text)
        or MOJIBAKE_RE.search(text)
    ):
        # Check if non-ASCII is something more than just latin accents
        weird = [c for c in non_ascii if unicodedata.category(c)[0] not in ("L", "P", "Z")]
        if weird:
            out.append(f"WEIRD_UNICODE: {weird[:3]!r}")
    return out


def balance_issues(text: str) -> list[str]:
    out = []
    if text.count('"') % 2 != 0:
        out.append("UNBALANCED_DOUBLE_QUOTE")
    if text.count("(") != text.count(")"):
        out.append("UNBALANCED_PAREN")
    if text.count("[") != text.count("]"):
        out.append("UNBALANCED_SQUARE")
    if text.count("{") != text.count("}"):
        out.append("UNBALANCED_CURLY")
    return out


def repeated_subseq_score(words: list[str]) -> float:
    """Fraction of words that are immediate repeats of the previous one.
    High score → 'yeah yeah yeah ...' degenerate labels."""
    if len(words) < 3:
        return 0.0
    repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    return repeats / (len(words) - 1)


def analyze_sample(raw_text: str, audio_array: np.ndarray | None, sample_rate: int):
    out = {
        "drop_reason": None,
        "issues": [],
        "label_chars": 0,
        "label_words": 0,
        "audio_seconds": None,
        "audio_channels": None,
        "words_per_second": None,
    }

    # Audio side
    if audio_array is None:
        out["drop_reason"] = "AUDIO_NONE"
    else:
        a = audio_array
        if hasattr(a, "numpy"):
            a = a.numpy()
        channels = 1 if a.ndim == 1 else a.shape[0] if a.ndim == 2 else None
        out["audio_channels"] = channels
        if a.ndim > 1:
            a = a.mean(axis=0)
        if a.size == 0:
            out["drop_reason"] = "AUDIO_EMPTY"
        elif not np.isfinite(a).all():
            out["drop_reason"] = "AUDIO_NONFINITE"
        else:
            duration = a.size / sample_rate
            out["audio_seconds"] = duration
            if duration > MAX_AUDIO_S:
                out["drop_reason"] = "AUDIO_TOO_LONG"
            elif duration < MIN_AUDIO_S:
                out["drop_reason"] = "AUDIO_TOO_SHORT"

    # Text side
    norm = _normalize_label(raw_text or "")
    out["label_norm"] = norm
    out["label_chars"] = len(norm)
    out["label_words"] = len(norm.split())
    if not norm and out["drop_reason"] is None:
        out["drop_reason"] = "LABEL_EMPTY_AFTER_NORM"

    # Quality issues (collected regardless of drop status)
    if raw_text:
        out["issues"].extend(char_class_anomalies(raw_text))
        out["issues"].extend(balance_issues(raw_text))
        if DIGIT_ONLY_RE.match(raw_text):
            out["issues"].append("DIGIT_ONLY_LABEL")
        words = raw_text.split()
        if len(words) >= 3:
            r = repeated_subseq_score(words)
            if r > 0.4:
                out["issues"].append(f"REPEATED_SUBSEQ:{r:.2f}")

    # Words-per-second sanity: typical conversational speech is 2–3 wps,
    # read speech 2.5–4 wps. <0.5 or >6 suggests label/audio drift.
    if out["audio_seconds"] and out["audio_seconds"] > 0 and out["label_words"]:
        wps = out["label_words"] / out["audio_seconds"]
        out["words_per_second"] = wps
        if wps < 0.3:
            out["issues"].append(f"LOW_WPS:{wps:.2f}")
        elif wps > 8:
            out["issues"].append(f"HIGH_WPS:{wps:.2f}")

    return out


def load_multiasr_config() -> list[dict]:
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "data" / "multiasr.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg["datasets"]


def sample_dataset(d_cfg: dict, n: int):
    path = d_cfg["path"]
    name = d_cfg.get("name")
    text_col = d_cfg["text_column"]
    audio_col = d_cfg.get("audio_column", "audio")
    splits = d_cfg.get("train_splits") or d_cfg.get("eval_splits") or ["train"]
    split = splits[0] if splits else "train"

    ds = load_dataset(path, name=name, split=split, streaming=True, trust_remote_code=True)
    # Force audio resampling to our target rate so size/seconds math is consistent
    ds = ds.cast_column(audio_col, Audio(sampling_rate=SAMPLE_RATE))
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        try:
            arr = row[audio_col]["array"]
        except (KeyError, TypeError):
            arr = None
        out.append({"text": row.get(text_col, ""), "audio": arr})
    return out


def report(label: str, samples: list[dict[str, Any]]):
    print(f"\n=== {label} ===")
    if not samples:
        print("  (no samples loaded)")
        return

    drop_reasons: Counter = Counter()
    issue_counts: Counter = Counter()
    issue_examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    durations = []
    wps_vals = []
    char_lengths = []
    word_lengths = []
    channels_counter: Counter = Counter()

    for s in samples:
        a = analyze_sample(s["text"], s["audio"], SAMPLE_RATE)
        if a["drop_reason"]:
            drop_reasons[a["drop_reason"]] += 1
        for issue in a["issues"]:
            key = issue.split(":")[0]
            issue_counts[key] += 1
            if len(issue_examples[key]) < 2:
                issue_examples[key].append((s["text"][:160], a.get("label_norm", "")[:160]))
        if a["audio_seconds"] is not None:
            durations.append(a["audio_seconds"])
        if a["words_per_second"] is not None:
            wps_vals.append(a["words_per_second"])
        if a["audio_channels"] is not None:
            channels_counter[a["audio_channels"]] += 1
        char_lengths.append(a["label_chars"])
        word_lengths.append(a["label_words"])

    n = len(samples)
    print(f"  samples: {n}")
    if drop_reasons:
        print("  drops by DataCollator filter:")
        for k, c in drop_reasons.most_common():
            print(f"    {k}: {c} ({100 * c / n:.1f}%)")
    else:
        print("  drops: none")

    if channels_counter:
        ch_str = ", ".join(f"{k}ch×{v}" for k, v in channels_counter.most_common())
        print(f"  channels: {ch_str}")
    if durations:
        print(
            f"  duration: min={min(durations):.2f}s  median={statistics.median(durations):.2f}s  max={max(durations):.2f}s"
        )
    if wps_vals:
        print(
            f"  words/sec: min={min(wps_vals):.2f}  median={statistics.median(wps_vals):.2f}  max={max(wps_vals):.2f}"
        )
    if word_lengths:
        print(
            f"  label words: min={min(word_lengths)}  median={int(statistics.median(word_lengths))}  max={max(word_lengths)}"
        )

    if issue_counts:
        print("  text-quality flags:")
        for k, c in issue_counts.most_common():
            print(f"    {k}: {c}")
            for raw, norm in issue_examples[k][:1]:
                print(f"      raw : {raw!r}")
                print(f"      norm: {norm!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--per-dataset", type=int, default=30)
    args = ap.parse_args()

    datasets_cfg = load_multiasr_config()
    for d_cfg in datasets_cfg:
        splits = d_cfg.get("train_splits") or d_cfg.get("eval_splits") or ["train"]
        split = splits[0] if splits else "train"
        label = f"{d_cfg['path']}::{d_cfg.get('name', '-')}::{split}"
        try:
            samples = sample_dataset(d_cfg, args.per_dataset)
        except Exception as e:
            print(f"\n=== {label} ===\n  !! LOAD FAILED: {type(e).__name__}: {e}")
            continue
        report(label, samples)


if __name__ == "__main__":
    main()
