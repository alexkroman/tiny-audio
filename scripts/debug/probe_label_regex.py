#!/usr/bin/env python3
"""Stream a few samples from every dataset in configs/data/multiasr.yaml,
run each text through scripts/train.py's _normalize_label pipeline, and
flag potential regex / cleanup bugs.

Checks per sample:
  - residual <angle> tags after normalization
  - residual [square] brackets after normalization
  - residual { } parens or HTML entities
  - empty output (entire label was stripped)
  - whitespace anomalies (leading/trailing/double)
  - changes in non-ASCII characters (mojibake-shaped, smart quotes, etc.)
  - truecase decision (fired vs skipped) + visible damage
  - percent edge cases
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import typer
from datasets import load_dataset

from scripts.train import (
    _GIGASPEECH_PUNCT_RE,
    _RESIDUAL_ANGLE_TAG_RE,
    _TEDLIUM_BRACKET_RE,
    _needs_truecase,
    _normalize_label,
)

# Re-use train.py's normalization
from scripts.utils import load_data_config

# Heuristic "leftover" detectors. These are intentionally broad — anything they
# match is something we should manually eyeball, even if some matches are
# legitimate (e.g. `<2>` references in earnings transcripts).
ANGLE_TAG_RE = re.compile(r"<[^>]+>")
SQUARE_BRACKET_RE = re.compile(r"\[[^\]]*\]")
CURLY_BRACE_RE = re.compile(r"\{[^}]*\}")
HTML_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);")
DOUBLE_SPACE_RE = re.compile(r"  +")
SMART_QUOTE_RE = re.compile(r"[‘’“”]")
MOJIBAKE_HINT_RE = re.compile(r"(â€|Ã©|Ã¨|Ã¢|Â£|Â§)")
NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
PCT_RE = re.compile(r"%")
PERCENT_WORD_RE = re.compile(r"\bper ?cent\b", re.IGNORECASE)


def analyze(raw: str, norm: str) -> dict[str, Any]:
    issues: list[str] = []

    if raw and not norm:
        issues.append("EMPTY_AFTER_NORM")

    # leftovers after normalization
    if m := ANGLE_TAG_RE.search(norm):
        issues.append(f"RESIDUAL_ANGLE_TAG: {m.group(0)!r}")
    if m := SQUARE_BRACKET_RE.search(norm):
        issues.append(f"RESIDUAL_SQUARE_BRACKET: {m.group(0)!r}")
    if m := CURLY_BRACE_RE.search(norm):
        issues.append(f"RESIDUAL_CURLY_BRACE: {m.group(0)!r}")
    if m := HTML_ENTITY_RE.search(norm):
        issues.append(f"RESIDUAL_HTML_ENTITY: {m.group(0)!r}")
    if m := DOUBLE_SPACE_RE.search(norm):
        issues.append("DOUBLE_SPACE")
    if norm and (norm != norm.strip()):
        issues.append("UNTRIMMED")

    # mojibake survival
    if MOJIBAKE_HINT_RE.search(norm):
        issues.append("MOJIBAKE_SHAPE")
    if SMART_QUOTE_RE.search(norm):
        issues.append("SMART_QUOTE_SURVIVED")
    if NON_PRINTABLE_RE.search(norm):
        issues.append("NON_PRINTABLE")

    # percent
    if PCT_RE.search(norm):
        issues.append("LITERAL_PERCENT_SURVIVED")
    # If `per cent` survived as literal text after norm, flag it.
    if PERCENT_WORD_RE.search(norm) and "per cent" in raw.lower() and "per cent" in norm.lower():
        issues.append("PER_CENT_SPACE_SURVIVED")

    # input had tags but we did not strip them? Detect by checking tags in raw
    raw_tags = ANGLE_TAG_RE.findall(raw)
    if raw_tags:
        unknown = []
        for t in raw_tags:
            inner = t.strip("<>").lower()
            # is it a gigaspeech punct, a known marker, or stripped by general angle clean?
            if _GIGASPEECH_PUNCT_RE.fullmatch("<" + inner.upper() + ">"):
                continue
            if _RESIDUAL_ANGLE_TAG_RE.fullmatch("<" + inner + ">"):
                continue
            # case-insensitive recheck for gigaspeech punct (residual stripper is case-agnostic)
            if re.fullmatch(_GIGASPEECH_PUNCT_RE.pattern, t, re.IGNORECASE):
                continue
            unknown.append(t)
        if unknown:
            issues.append(f"UNKNOWN_RAW_TAGS: {unknown!r}")

    raw_brackets = SQUARE_BRACKET_RE.findall(raw)
    if raw_brackets:
        # We strip these only for TEDLIUM via the regex, but the regex isn't
        # dataset-gated; it removes brackets unconditionally. So this is a
        # informational signal, not necessarily a bug.
        issues.append(f"INPUT_HAD_BRACKETS: {raw_brackets!r}")

    needs_tc = _needs_truecase(norm or "")
    return {
        "raw": raw,
        "norm": norm,
        "issues": issues,
        "needs_truecase": needs_tc,
    }


def _pre_truecase(raw_text: str) -> str:
    """Replicate _normalize_label up to (but not including) the truecase step
    so we can detect when truecase fired."""
    import ftfy

    text = (raw_text or "").strip()
    if not text:
        return ""
    text = ftfy.fix_text(text, normalization="NFKC")
    text = _GIGASPEECH_PUNCT_RE.sub(
        lambda m: {"COMMA": ",", "PERIOD": ".", "QUESTIONMARK": "?", "EXCLAMATIONPOINT": "!"}[
            m.group(1).upper()
        ],
        text,
    )
    text = _RESIDUAL_ANGLE_TAG_RE.sub(" ", text)
    text = _TEDLIUM_BRACKET_RE.sub(" ", text)
    text = text.replace("%", " percent")
    text = re.sub(r"\bper ?cent\b", "percent", text)
    return re.sub(r"\s+", " ", text).strip()


def sample_dataset(d_cfg: dict, n: int) -> list[dict]:
    """Stream n samples from a dataset, returning just the text column."""
    path = d_cfg["path"]
    name = d_cfg.get("name")
    text_col = d_cfg["text_column"]
    # Prefer train_splits[0] if available, else eval_splits[0]
    splits = d_cfg.get("train_splits") or d_cfg.get("eval_splits") or ["train"]
    split = splits[0] if splits else "train"

    ds = load_dataset(
        path,
        name=name,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        out.append({"text": row.get(text_col, "")})
    return out


def run(per_dataset: int, verbose: bool):
    datasets_cfg = load_data_config()
    summary: dict[str, dict] = {}

    for d_cfg in datasets_cfg:
        path = d_cfg["path"]
        name = d_cfg.get("name", "-")
        splits = d_cfg.get("train_splits") or d_cfg.get("eval_splits") or ["train"]
        split = splits[0] if splits else "train"
        label = f"{path}::{name}::{split}"
        print(f"\n=== {label} (text_column={d_cfg['text_column']}) ===")
        try:
            samples = sample_dataset(d_cfg, per_dataset)
        except Exception as e:
            print(f"  !! LOAD FAILED: {type(e).__name__}: {e}")
            summary[label] = {"error": str(e)}
            continue

        per_issue: Counter = Counter()
        truecase_fired = 0
        truecase_skipped = 0
        examples: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for s in samples:
            raw = s["text"] or ""
            norm = _normalize_label(raw)
            res = analyze(raw, norm)

            pre = _pre_truecase(raw)
            if norm != pre:
                truecase_fired += 1
            else:
                truecase_skipped += 1

            for issue in res["issues"]:
                key = issue.split(":")[0]
                per_issue[key] += 1
                if len(examples[key]) < 3:
                    examples[key].append((raw, norm))

            if verbose:
                tc_tag = " [TRUECASED]" if norm != pre else ""
                print(f"  RAW : {raw!r}")
                print(f"  NORM: {norm!r}{tc_tag}")
                if res["issues"]:
                    print(f"  ISSUES: {res['issues']}")
                print()

        print(
            f"  samples: {len(samples)}  truecase fired/skipped: {truecase_fired}/{truecase_skipped}"
        )
        if per_issue:
            print("  issues:")
            for k, c in per_issue.most_common():
                print(f"    {k}: {c}")
                for raw, norm in examples[k][:2]:
                    print(f"      raw : {raw!r}")
                    print(f"      norm: {norm!r}")
        else:
            print("  no issues flagged")

        summary[label] = {
            "samples": len(samples),
            "truecase_fired": truecase_fired,
            "issues": dict(per_issue),
        }

    print("\n=== SUMMARY ===")
    for label, info in summary.items():
        if "error" in info:
            print(f"  {label}: ERROR {info['error']!r}")
            continue
        issues_str = ", ".join(f"{k}={v}" for k, v in info["issues"].items()) or "clean"
        print(f"  {label}: n={info['samples']} tc={info['truecase_fired']} | {issues_str}")


def main(
    per_dataset: int = typer.Option(20, "-n", "--per-dataset"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Check which label-cleaning regexes fire on freshly sampled rows."""
    run(per_dataset, verbose)


if __name__ == "__main__":
    typer.run(main)
