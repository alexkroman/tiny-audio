"""Speaker-attributed transcripts: the `<SPK_n>` format and cpWER.

Kept free of torch/datasets so `ta eval` can score any system's
speaker-labelled output (AssemblyAI's utterances, the tiny-audio pipeline's
`return_speakers` turns) without loading the training stack.
"""

from __future__ import annotations

import re

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

SPEAKER_TOKEN = "<SPK_{}>"
_SPEAKER_RE = re.compile(r"<SPK_(\d+)>")
# Ends a labelled context prefix in a transcript; `parse_turns` treats it as a
# word boundary so a hypothesis that still carries it scores cleanly.
CONTEXT_END = "<CONTINUE>"


def has_speakers(text: str) -> bool:
    """True when `text` carries at least one speaker token."""
    return bool(_SPEAKER_RE.search(text or ""))


def serialize_turns(turns, labels: dict | None = None) -> str:
    """[(speaker, text), ...] in time order -> '<SPK_1>text<SPK_2>text'.

    Speakers (any hashable label, e.g. AssemblyAI's "A"/"B") are renumbered
    by first appearance and consecutive turns of one speaker merged -- the
    same convention the training targets use. `labels` pins speakers that
    already have a number (a context prefix's); others continue after the
    largest pinned number. It is not modified.
    """
    labels = dict(labels or {})
    first_new = max(labels.values(), default=0) + 1
    merged: list[list] = []
    for speaker, raw in turns:
        text = (raw or "").strip()
        if not text:
            continue
        if speaker not in labels:
            labels[speaker] = first_new
            first_new += 1
        label = labels[speaker]
        if merged and merged[-1][0] == label:
            merged[-1][1] += " " + text
        else:
            merged.append([label, text])
    return "".join(SPEAKER_TOKEN.format(label) + text for label, text in merged)


def parse_turns(text: str) -> list[tuple[int, str]]:
    """'<SPK_1>a<SPK_2>b' -> [(1, 'a'), (2, 'b')]. Text before any token is speaker 0."""
    pieces = _SPEAKER_RE.split(text.replace(CONTEXT_END, " "))
    turns = [(0, pieces[0].strip())] if pieces[0].strip() else []
    for label, chunk in zip(pieces[1::2], pieces[2::2], strict=True):
        if chunk.strip():
            turns.append((int(label), chunk.strip()))
    return turns


def plain_text(text: str) -> str:
    """The transcript with speaker tokens removed."""
    return " ".join(t for _, t in parse_turns(text))


def word_errors(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance (substitutions + deletions + insertions).

    rapidfuzz (C++, already a jiwer dependency): a whole meeting is ~1-2k
    words per speaker, and cpWER compares every speaker pair.
    """
    return Levenshtein.distance(ref, hyp)


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
