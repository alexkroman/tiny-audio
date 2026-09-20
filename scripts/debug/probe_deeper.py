#!/usr/bin/env python3
"""Deeper data-quality probe: skip past the previously-audited row range
to get a different random slice, and check additional issue categories.

Categories beyond the earlier probes:
  - consecutive duplicate words ("the the", "you know you know")
  - mid-sentence ALL-CAPS tokens (truecase missed; could be acronym
    or could be label noise)
  - digit-letter glue (`25g`, `2nd` already-fine vs `5x` ambiguous)
  - apostrophe spacing artifacts beyond `n't` (regression check)
  - mid-sentence period spacing (regression check from earlier fix)
  - multi-punct runs (`...`, `--`, `!!`, `??`) — preserved or mangled?
  - trailing dangling punctuation (orphan quote / paren / comma)
  - hyphenation oddities (`state - of - the - art` spaces)
  - any surviving `<>` / `[]` / `{}` post-normalize (regression check)
  - mojibake / smart-quote survival
  - very-long labels (>80 words — context-overflow risk)
  - mid-word ALL-CAPS (truecase artifact like `gonNA` we already fixed,
    plus any new ones)

Skips the first 500 rows per dataset to dodge the head-of-shard samples
my earlier probes already covered.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import typer
from datasets import load_dataset

from scripts.train import _normalize_label
from scripts.utils import load_data_config

# Detectors
DUPLICATE_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
TRIPLE_DUPLICATE_RE = re.compile(r"\b(\w+)\s+\1\s+\1\b", re.IGNORECASE)
MID_PERIOD_SPACE_RE = re.compile(r"\s+[.,!?]")  # post-fix: should be 0
ORPHAN_NT_RE = re.compile(r"\b\w+\s+'[A-Za-z]\b")  # post-fix: should be ~0
ANY_ANGLE_RE = re.compile(r"<[^>]+>")
ANY_BRACKET_RE = re.compile(r"\[[^\]]*\]")
ANY_CURLY_RE = re.compile(r"\{[^}]*\}")
MOJIBAKE_RE = re.compile(r"(â€|Ã©|Ã¨|Â£|Â§)")
SMART_QUOTE_RE = re.compile(r"[‘’“”]")
MULTI_PUNCT_RE = re.compile(r"([.,!?]){2,}|--+")
HYPHEN_SPACE_RE = re.compile(r"\s-\s")  # `state - of - the - art` pattern
MIDWORD_CAPS_RE = re.compile(r"[a-z][A-Z]+[a-z]")  # camelCase or `gonNA`-style
INTERIOR_CAPS_WORD_RE = re.compile(r"\b[a-z]+[A-Z]+\w*\b")  # word starting lower, has caps inside
ALLCAPS_TOKEN_IN_LOWERCASE_RE = re.compile(r"(?<=\s)([A-Z]{2,})(?=\s|[.,!?]|$)")
DIGIT_LETTER_RE = re.compile(r"\b\d+[a-zA-Z]+\b|\b[a-zA-Z]+\d+\b")
TRAILING_OPEN_QUOTE_RE = re.compile(r'"\s*$')
URL_RE = re.compile(r"https?://|www\.|\.com|\.org|\.net", re.IGNORECASE)


def analyze(raw: str, norm: str) -> list[str]:
    issues = []

    # Regression checks (these should be zero now)
    if MID_PERIOD_SPACE_RE.search(norm):
        issues.append("REGRESSION_SPACE_BEFORE_PUNCT")
    if ORPHAN_NT_RE.search(norm):
        issues.append("REGRESSION_ORPHAN_APOS")
    if ANY_ANGLE_RE.search(norm):
        issues.append("REGRESSION_ANGLE_TAG")
    if ANY_BRACKET_RE.search(norm):
        issues.append("REGRESSION_SQUARE_BRACKET")

    # New issue categories
    if m := DUPLICATE_WORD_RE.search(norm):
        # Filter known-legitimate dupes ("that that", "had had", "is is" in copula constructions)
        w = m.group(1).lower()
        if w not in {"that", "had", "is", "have", "will", "would", "could", "should"}:
            issues.append(f"DUPLICATE_WORD:{w}")
    if m := TRIPLE_DUPLICATE_RE.search(norm):
        issues.append(f"TRIPLE_DUPLICATE_WORD:{m.group(1).lower()}")

    if SMART_QUOTE_RE.search(norm):
        issues.append("SMART_QUOTE_SURVIVED")
    if MOJIBAKE_RE.search(norm):
        issues.append("MOJIBAKE_SURVIVED")
    if ANY_CURLY_RE.search(norm):
        issues.append("CURLY_BRACE_SURVIVED")

    if HYPHEN_SPACE_RE.search(norm):
        issues.append("SPACED_HYPHEN")
    if "--" in norm and " -- " not in norm and norm != "--":
        # `--` without surrounding spaces (post-fix should always have spaces if both sides
        # are alphanumeric). But word-internal `--` is uncommon; still worth flagging.
        issues.append("UNSPACED_EMDASH")

    for m in INTERIOR_CAPS_WORD_RE.finditer(norm):
        word = m.group(0)
        # Skip well-known proper-noun-style patterns
        if word.lower() in {"mcdonalds", "iphone", "ipad", "ebay", "ebooks", "imac"}:
            continue
        # Skip `'s` / `'d` etc with capital after (e.g. McClarnon's after truecase)
        # camelCase / `gonNA`-style midword caps
        issues.append(f"MIDWORD_CAPS:{word}")

    # all-caps acronyms standalone — sometimes real (USA, FBI), sometimes label noise (YEAH).
    # Only flag if all-caps token appears in an otherwise lowercase context AND is suspicious.
    surrounding = norm
    allcaps_tokens = [
        t
        for t in ALLCAPS_TOKEN_IN_LOWERCASE_RE.findall(surrounding)
        if len(t) >= 2 and t not in {"USA", "UK", "EU", "US", "TV", "AI", "OK", "PR", "IT", "ID"}
    ]
    if allcaps_tokens:
        # Are they truly anomalous? heuristic: if the whole text is mono-case (truecase
        # should have run), an all-caps token is suspicious.
        letters = [c for c in norm if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) < 0.3:
            for t in allcaps_tokens[:3]:
                issues.append(f"ALLCAPS_IN_LOWERCASE:{t}")

    if DIGIT_LETTER_RE.search(norm):
        for m in DIGIT_LETTER_RE.finditer(norm):
            tok = m.group(0).lower()
            if tok in {
                "1st",
                "2nd",
                "3rd",
                "4th",
                "5th",
                "6th",
                "7th",
                "8th",
                "9th",
                "10th",
                "11th",
                "12th",
                "13th",
                "14th",
                "15th",
                "16th",
                "17th",
                "18th",
                "19th",
                "20th",
                "21st",
                "22nd",
                "23rd",
                "30th",
                "100th",
            }:
                continue
            if re.match(r"^[12]\d{3}[a-z]*$", tok):  # years
                continue
            issues.append(f"DIGIT_LETTER:{tok}")

    if MULTI_PUNCT_RE.search(norm):
        for m in MULTI_PUNCT_RE.finditer(norm):
            issues.append(f"MULTI_PUNCT:{m.group(0)!r}")

    # Very long label
    word_count = len(norm.split())
    if word_count > 100:
        issues.append(f"VERY_LONG_LABEL:{word_count}w")

    # Trailing open quote (LibriHeavy splits dialogue across segments)
    if TRAILING_OPEN_QUOTE_RE.search(norm) and norm.count('"') % 2 != 0:
        issues.append("DANGLING_OPEN_QUOTE")

    # URL-like patterns (uncommon in ASR, but if present indicates source mismatch)
    if URL_RE.search(norm):
        issues.append("URL_LIKE")

    return issues


def sample(d_cfg, skip_n, take_n):
    path = d_cfg["path"]
    name = d_cfg.get("name")
    text_col = d_cfg["text_column"]
    splits = d_cfg.get("train_splits") or d_cfg.get("eval_splits") or ["train"]
    split = splits[0] if splits else "train"
    ds = load_dataset(path, name=name, split=split, streaming=True, trust_remote_code=True)
    ds = ds.skip(skip_n).take(take_n)
    rows = []
    for row in ds:
        rows.append(row.get(text_col, ""))
    return rows


def main(
    skip: int = typer.Option(500, help="rows to skip (head was audited earlier)"),
    take: int = typer.Option(200, help="rows to analyze per dataset"),
):
    """Audit label normalization on rows past the already-audited head."""
    for d_cfg in load_data_config():
        splits = d_cfg.get("train_splits") or d_cfg.get("eval_splits") or ["train"]
        split = splits[0] if splits else "train"
        label = f"{d_cfg['path']}::{d_cfg.get('name', '-')}::{split}"
        print(f"\n=== {label} (skip={skip}, take={take}) ===", flush=True)
        try:
            rows = sample(d_cfg, skip, take)
        except Exception as e:
            print(f"  !! LOAD FAILED: {type(e).__name__}: {e}")
            continue

        per_issue: Counter = Counter()
        examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for raw in rows:
            norm = _normalize_label(raw)
            for issue in analyze(raw, norm):
                key = issue.split(":")[0]
                per_issue[key] += 1
                if len(examples[key]) < 2:
                    examples[key].append((raw[:200], norm[:200]))

        print(f"  rows analyzed: {len(rows)}")
        if per_issue:
            for k, c in per_issue.most_common():
                print(f"  {k}: {c}")
                for raw, norm in examples[k][:1]:
                    print(f"    raw : {raw!r}")
                    print(f"    norm: {norm!r}")
        else:
            print("  clean")


if __name__ == "__main__":
    typer.run(main)
