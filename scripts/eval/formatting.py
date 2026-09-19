"""Casing and punctuation scoring — the axis standard WER cannot see.

WHY THIS EXISTS. Every WER number in this repo is computed after Whisper's
`EnglishTextNormalizer`, which lowercases and strips punctuation from BOTH the
reference and the hypothesis. That makes WER blind to casing and punctuation —
which is a problem, because producing them is the stated reason to put an LLM
behind the encoder at all (the frozen Granite CTC head emits neither, and
scores within 0.6 WER of the full stack).

It is also actively masking a measured defect. 42% of the training mix declares
`text_case: mono` and is lifted by the `truecase` library. Measured over 20,860
non-initial words of genuinely-cased eval references, round-tripped through the
real `_normalize_label(..., "mono")`: **2.40% receive a spurious capital**,
0.65% lose a legitimate one, and **23.2% of utterances carry at least one casing
error** (`"and subscriptions revenues"` -> `"and Subscriptions revenues"`,
`"on our IR website"` -> `"on our Ir Website"`, `NASA` -> `Nasa`). None of that
is visible in any existing metric.

WHAT IS SCORED. Three things, all on RAW text with only whitespace collapsed:

  orthographic_wer  WER with case and punctuation preserved. The literature
                    reports this alongside normalized WER; the gap between them
                    is the formatting cost. ESB (arXiv 2210.13352) measures
                    macro-avg 10.6 -> 7.4 for Whisper across that gap.
  case_error_rate   Of word pairs that match case-INSENSITIVELY (i.e. the word
                    was recognised correctly), the fraction whose casing
                    differs. This isolates casing from recognition: a word we
                    got wrong cannot also be a casing error.
  punct_f1          Token-level F1 over punctuation marks attached to aligned
                    words, so it measures placement rather than raw counts.

SCORABILITY. Most eval references are not cased or not punctuated —
LibriSpeech is ALL-CAPS, AMI/TEDLIUM/Peoples/Loquacious are mono-case and
unpunctuated. Scoring casing against an ALL-CAPS reference is meaningless, so
each sample is gated independently and the counts are reported. A metric is
omitted entirely rather than computed over zero scorable samples.
"""

from __future__ import annotations

import re
import unicodedata

import jiwer

# A metric computed over a handful of samples is noise dressed as a number.
# Observed without this floor: AMI-SDM reported punct_f1 85.71 for us and 0.00
# for the baseline, both on a single scorable sample.
MIN_SCORABLE_SAMPLES = 10

# Marks we treat as sentence/clause punctuation. Apostrophes and hyphens are
# excluded deliberately: they are lexical (don't, well-known), not formatting,
# and the normalizer's contraction expansion already churns them.
PUNCT_CHARS = ".,?!;:"
_PUNCT_RE = re.compile(f"[{re.escape(PUNCT_CHARS)}]")
_WS_RE = re.compile(r"\s+")
_DETACHED_PUNCT_RE = re.compile(f"\\s+([{re.escape(PUNCT_CHARS)}]+)")


def light_normalize(text: str) -> str:
    """Unicode-fold, reattach detached punctuation, collapse whitespace.

    Case and punctuation are preserved. Detached marks are reattached because
    some corpora tokenize punctuation as a standalone word -- GigaSpeech dev
    ships `"yeah ."` -- and without this every such mark reads as a missing
    trailing punctuation plus a spurious extra token, which drove GigaSpeech's
    punctuation F1 to 4.6 purely as a tokenization artifact.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _DETACHED_PUNCT_RE.sub(r"\1", text)
    return _WS_RE.sub(" ", text).strip()


def is_orthographic_reference(reference: str) -> bool:
    """True when the reference is a genuine cased AND punctuated target.

    Gating `orthographic_wer` on this is load-bearing. An ALL-CAPS reference
    scores ~98% orthographic WER against correctly-cased output -- that is a
    transcript-convention artifact, not a model error -- so pooling ungated
    would report ~48% for a system whose real orthographic WER on scorable
    corpora is ~11%.
    """
    return is_case_scorable(reference) and is_punct_scorable(reference)


def is_case_scorable(reference: str) -> bool:
    """True when the reference genuinely carries case information.

    Requires BOTH an uppercase and a lowercase letter. An ALL-CAPS reference
    (LibriSpeech, AMI) and an all-lowercase one (Peoples, TEDLIUM) each fail,
    because neither tells us what the correct casing of a given word is.
    """
    ref = reference or ""
    return any(c.isupper() for c in ref) and any(c.islower() for c in ref)


def is_punct_scorable(reference: str) -> bool:
    """True when the reference carries sentence punctuation at all."""
    return bool(_PUNCT_RE.search(reference or ""))


def _strip_punct(word: str) -> str:
    return _PUNCT_RE.sub("", word)


def _trailing_punct(word: str) -> str:
    """Punctuation attached to the end of a token, e.g. 'world.' -> '.'."""
    m = re.search(f"[{re.escape(PUNCT_CHARS)}]+$", word)
    return m.group(0) if m else ""


def _aligned_word_pairs(reference: str, hypothesis: str) -> list[tuple[str, str]]:
    """Word pairs the alignment considers the same position.

    Alignment runs on punctuation-stripped, lowercased tokens so that a casing
    or punctuation difference cannot itself perturb the alignment — otherwise
    the metric would partly measure its own noise.
    """
    ref_w = reference.split()
    hyp_w = hypothesis.split()
    if not ref_w or not hyp_w:
        return []
    ref_key = [_strip_punct(w).lower() for w in ref_w]
    hyp_key = [_strip_punct(w).lower() for w in hyp_w]
    out = jiwer.process_words([" ".join(ref_key)], [" ".join(hyp_key)])
    pairs: list[tuple[str, str]] = []
    for chunk in out.alignments[0]:
        if chunk.type not in ("equal", "substitute"):
            continue
        for i, j in zip(
            range(chunk.ref_start_idx, chunk.ref_end_idx),
            range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            strict=False,
        ):
            pairs.append((ref_w[i], hyp_w[j]))
    return pairs


def score_case(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (case_errors, comparable_words).

    Only word pairs that are equal case-insensitively count, so recognition
    errors are excluded and this measures casing alone.
    """
    errors = comparable = 0
    for r, h in _aligned_word_pairs(reference, hypothesis):
        rs, hs = _strip_punct(r), _strip_punct(h)
        if not rs or rs.lower() != hs.lower():
            continue
        comparable += 1
        if rs != hs:
            errors += 1
    return errors, comparable


def score_punct(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return (true_positives, predicted, actual) for trailing punctuation."""
    tp = pred = act = 0
    for r, h in _aligned_word_pairs(reference, hypothesis):
        rp, hp = _trailing_punct(r), _trailing_punct(h)
        act += bool(rp)
        pred += bool(hp)
        tp += bool(rp) and rp == hp
    return tp, pred, act


def compute_formatting_metrics(pairs: list[tuple[str, str]]) -> dict:
    """Aggregate formatting metrics over (raw_reference, raw_hypothesis) pairs.

    Keys are omitted when nothing in the corpus is scorable for that axis, so a
    mono-case corpus reports no casing number rather than a misleading 0.0.
    """
    metrics: dict[str, float | int] = {}

    ortho = [
        (light_normalize(r), light_normalize(h)) for r, h in pairs if is_orthographic_reference(r)
    ]
    ortho = [(r, h) for r, h in ortho if r]
    if len(ortho) >= MIN_SCORABLE_SAMPLES:
        refs, hyps = zip(*ortho, strict=True)
        metrics["orthographic_wer"] = jiwer.wer(list(refs), list(hyps)) * 100
        metrics["orthographic_scored_samples"] = len(ortho)

    case_err = case_tot = case_samples = 0
    for r, h in pairs:
        if not is_case_scorable(r):
            continue
        case_samples += 1
        e, t = score_case(light_normalize(r), light_normalize(h))
        case_err += e
        case_tot += t
    if case_tot and case_samples >= MIN_SCORABLE_SAMPLES:
        metrics["case_error_rate"] = 100 * case_err / case_tot
        metrics["case_scored_words"] = case_tot
        metrics["case_scored_samples"] = case_samples

    tp = pred = act = punct_samples = 0
    for r, h in pairs:
        if not is_punct_scorable(r):
            continue
        punct_samples += 1
        a, b, c = score_punct(light_normalize(r), light_normalize(h))
        tp += a
        pred += b
        act += c
    if act and punct_samples >= MIN_SCORABLE_SAMPLES:
        precision = tp / pred if pred else 0.0
        recall = tp / act
        metrics["punct_precision"] = 100 * precision
        metrics["punct_recall"] = 100 * recall
        metrics["punct_f1"] = (
            100 * 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
        metrics["punct_scored_samples"] = punct_samples

    return metrics
