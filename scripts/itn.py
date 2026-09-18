#!/usr/bin/env python3
"""Pattern-based ITN (Inverse Text Normalization) scoring.

Why patterns and not spaCy NER: the OntoNotes label set only marks seven
numeric types (CARDINAL, DATE, TIME, MONEY, PERCENT, ORDINAL, QUANTITY), and
most real ITN classes -- phone numbers, emails, URLs, alphanumeric model codes,
fractions, scores, thousands separators -- are never labelled at all. Regexes
over the raw reference find them directly and need no model download.

Why raw text: `results.txt` also stores Whisper-`EnglishTextNormalizer` output,
and that normalizer *performs* ITN ("twenty five dollars" -> "$25") while
destroying other classes ("3:00 p.m." -> "3 0 p m"). Scoring formatting on
normalized text therefore measures word recognition, not formatting. Every
function here expects the raw, un-normalized strings.
"""

import re

# Ordered most-specific-first. A match is skipped when it overlaps a span
# already claimed by an earlier class, so "$1,250.00" counts once as `currency`
# rather than three times across currency/thousands_sep/decimal.
ITN_CLASSES: list[tuple[str, str]] = [
    ("email", r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    ("url", r"https?://\S+|www\.[\w.-]+|\b[\w-]{2,}\.(?:com|org|net|gov|edu|io|ai|co\.uk)\b"),
    ("phone", r"\(\d{3}\)\s?\d{3}[-.–]\d{4}|\b\d{3}[-.–]\d{3}[-.–]\d{4}\b"),
    ("version", r"\bv\d+(?:\.\d+)+\b|\b\d+\.\d+\.\d+\b"),
    ("clock_time", r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]\.?m\.?)?"),
    ("currency", r"[$£€¥]\s?\d[\d,]*(?:\.\d+)?"),
    ("percent", r"\b\d+(?:\.\d+)?\s?%"),
    ("temperature", r"\b\d+(?:\.\d+)?\s?°[CF]?"),
    (
        "unit",
        (
            r"\b\d+(?:\.\d+)?[-\s]?"
            r"(?:kg|km|cm|mm|nm|mph|kph|ft|lb|lbs|oz|gb|mb|kb|tb|hz|khz|mhz|ghz|ml|mg|"
            r"inch|inches|mile|miles)\b"
        ),
    ),
    ("thousands_sep", r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),
    ("fraction", r"\b\d+\s?/\s?\d+\b"),
    ("range_score", r"\b\d+\s?[-–]\s?\d+\b"),
    ("decimal", r"\b\d+\.\d+\b"),
    ("ordinal", r"\b\d+(?:st|nd|rd|th)\b"),
    ("acronym_dotted", r"\b(?:[A-Za-z]\.){2,}"),
    ("title_abbrev", r"\b(?:Mr|Mrs|Ms|Dr|Prof|Rev|St|Ave|Blvd|Rd|Jr|Sr|Inc|Ltd|Co|vs|etc)\."),
    ("alphanum_id", r"\b(?=[\w-]*[A-Za-z])(?=[\w-]*\d)[A-Za-z][\w-]*\b"),
    ("year", r"\b(?:1[6-9]|20)\d{2}\b"),
    ("integer", r"\b\d+\b"),
]

# Classes whose regex is case-sensitive by design: matching case-insensitively
# would erase the very signal they detect (an acronym is only an acronym while
# it is capitalized, and "b737" is not an alphanumeric model code).
_CASE_SENSITIVE = {"acronym_dotted", "title_abbrev", "alphanum_id"}

_COMPILED = [
    (name, re.compile(pat, 0 if name in _CASE_SENSITIVE else re.IGNORECASE))
    for name, pat in ITN_CLASSES
]


def find_itn_spans(text: str) -> list[tuple[int, int, str, str]]:
    """Return non-overlapping (start, end, class_name, matched_text) spans.

    Classes are applied in `ITN_CLASSES` order; a later class cannot claim
    characters an earlier one already took.
    """
    claimed: list[tuple[int, int]] = []
    spans: list[tuple[int, int, str, str]] = []

    for name, rx in _COMPILED:
        for m in rx.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in claimed):
                continue
            claimed.append((m.start(), m.end()))
            spans.append((m.start(), m.end(), name, m.group(0)))

    spans.sort(key=lambda s: s[0])
    return spans


def _loose_tokens(text: str) -> list[str]:
    """Tokenize with every non-alphanumeric character stripped inside tokens.

    "$1,250." -> ["1250"], so a prediction that got the value right but the
    formatting wrong still matches at the loose level while failing `exact`.
    """
    return [t for t in (re.sub(r"[^a-z0-9]", "", w) for w in text.lower().split()) if t]


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return True
    return False


def score_sample(raw_reference: str, raw_prediction: str) -> dict[str, dict[str, int]]:
    """Score one sample's ITN spans against the raw prediction.

    Returns `{class_name: {"total": n, "exact": n, "loose": n}}` where:
      - `total` counts occurrences of the class in the reference,
      - `exact` counts those reproduced byte-for-byte (case-insensitively) in
        the prediction -- correct value *and* correct formatting,
      - `loose` counts those present with any formatting ("1250" for "1,250").

    `loose - exact` is the formatting-only error count; `total - loose` is the
    recognition error count. Separating the two is the whole point: the old
    NER-based metric conflated them.
    """
    pred_lower = raw_prediction.lower()
    pred_tokens = _loose_tokens(raw_prediction)
    out: dict[str, dict[str, int]] = {}

    for _, _, name, matched in find_itn_spans(raw_reference):
        stats = out.setdefault(name, {"total": 0, "exact": 0, "loose": 0})
        stats["total"] += 1

        exact = matched.lower() in pred_lower
        if exact:
            stats["exact"] += 1
            stats["loose"] += 1
        elif _contains_subsequence(pred_tokens, _loose_tokens(matched)):
            stats["loose"] += 1

    return out


def merge_scores(
    target: dict[str, dict[str, int]], addition: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Accumulate `addition` into `target` in place and return it."""
    for name, stats in addition.items():
        dst = target.setdefault(name, {"total": 0, "exact": 0, "loose": 0})
        for key, value in stats.items():
            dst[key] += value
    return target
