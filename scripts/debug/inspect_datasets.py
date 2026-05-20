"""Stream 100 samples from every dataset in multiasr.yaml and report
data-quality issues that train.py's normalization may not handle.

Streams via HF datasets `streaming=True` + shuffle (buffer_size=1000)
so each run samples a fresh slice of the dataset rather than the head.
Override the seed via SHUFFLE_SEED env var for reproducible re-runs.

Run: poetry run python scripts/debug/inspect_datasets.py
     SHUFFLE_SEED=2024 poetry run python scripts/debug/inspect_datasets.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

# Import the production normalizer so we can show before/after for each
# sample — verifies my recent changes (Gigaspeech tag remap, truecase,
# ftfy/NFKC) behave as intended on freshly-sampled data.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import _normalize_label  # noqa: E402  pyright: ignore[reportMissingImports]

# Mirror train.py's regexes so we can flag what's NOT already handled.
_CORPUS_MARKER_RE = re.compile(
    r"\s*<("
    r"comma|period|exclamationpoint|questionmark|"
    r"sil|music|noise|other|unk|"
    r"overlap|laugh|dtmf|foreign|no-speech|lipsmack|"
    r"clear_throat|inaudible|crosstalk"
    r")>",
    re.IGNORECASE,
)
_TEDLIUM_BRACKET_RE = re.compile(r"\s*\[[^\]]*\]")

# Patterns we want to detect that may NOT be handled by train.py.
_ANGLE_TAG_RE = re.compile(r"<[^>]+>")  # any <foo> — superset of _CORPUS_MARKER_RE
_SQUARE_BRACKET_RE = re.compile(r"\[[^\]]*\]")  # any [foo]
_PAREN_RE = re.compile(r"\([^)]*\)")  # any (foo)
_CURLY_RE = re.compile(r"\{[^}]*\}")  # any {foo}
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_ENT_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_SPEAKER_TAG_RE = re.compile(r"^(speaker[\s_-]?\d+|[A-Z]{2,}\s*:)", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s{3,}")  # 3+ consecutive spaces
_NUMERIC_NORM_RE = re.compile(r"\d")
_PUNCT_RE = re.compile(r"[.!?,;:]")

DATASETS = [
    # (path, name-or-None, split, text_column, audio_column, label)
    ("sanchit-gandhi/tedlium-data", None, "train", "text", "audio", "TEDLIUM-3 (train)"),
    ("mythicinfinity/libriheavy", "medium", "train", "text_original", "audio", "LibriHeavy medium"),
    ("MLCommons/peoples_speech", "clean_sa", "train", "text", "audio", "Peoples clean_sa"),
    ("MLCommons/peoples_speech", "dirty_sa", "train", "text", "audio", "Peoples dirty_sa"),
    ("MLCommons/peoples_speech", "validation", "validation", "text", "audio", "Peoples validation"),
    ("fixie-ai/common_voice_17_0", "en", "train", "sentence", "audio", "CommonVoice 17 en"),
    ("speechcolab/gigaspeech", "m", "train", "text", "audio", "Gigaspeech M"),
    ("speechcolab/gigaspeech", "dev", "validation", "text", "audio", "Gigaspeech dev"),
    ("kensho/spgispeech", "M", "train", "transcript", "audio", "SPGISpeech M"),
    ("facebook/voxpopuli", "en", "train", "raw_text", "audio", "VoxPopuli en"),
    ("hhoangphuoc/switchboard", None, "train", "transcript", "audio", "Switchboard"),
    ("edinburghcstr/ami", "ihm", "train", "text", "audio", "AMI IHM"),
    ("edinburghcstr/ami", "sdm", "train", "text", "audio", "AMI SDM"),
]

N_SAMPLES = 100
SHUFFLE_SEED = int(os.environ.get("SHUFFLE_SEED", "2024"))
SHUFFLE_BUFFER = 1000  # rows the streaming shuffler holds in memory


def fmt_classify(text: str) -> str:
    """Coarse format flag: uppercase-no-punct vs cased-with-punct vs lowercase."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "no-letters"
    upper_frac = sum(c.isupper() for c in letters) / len(letters)
    has_punct = bool(_PUNCT_RE.search(text))
    if upper_frac > 0.9:
        return f"UPPER ({'+' if has_punct else 'no '}punct)"
    if upper_frac < 0.05:
        return f"lower ({'+' if has_punct else 'no '}punct)"
    return f"Cased ({'+' if has_punct else 'no '}punct)"


def inspect_one(label: str, path: str, name: str | None, split: str, text_col: str) -> dict:
    print(f"\n=== {label} ===")
    print(f"  path={path}  name={name}  split={split}  text_col={text_col}")
    try:
        ds = load_dataset(path, name=name, split=split, streaming=True, trust_remote_code=True)
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return {"label": label, "error": str(e)}

    samples: list[str] = []
    other_cols: set[str] = set()
    try:
        for ex in islice(ds, N_SAMPLES):
            t = ex.get(text_col)
            if not isinstance(t, str):
                t = "" if t is None else str(t)
            samples.append(t)
            if not other_cols:
                other_cols = set(ex.keys()) - {text_col}
    except Exception as e:
        print(f"  STREAM-FAIL after {len(samples)} samples: {type(e).__name__}: {e}")
        if not samples:
            return {"label": label, "error": str(e)}

    n = len(samples)
    print(f"  streamed: {n} samples")
    print(f"  other cols: {sorted(other_cols)[:8]}{'...' if len(other_cols) > 8 else ''}")

    # Counts of pattern hits (samples containing at least one match).
    angle_hits = sum(1 for s in samples if _ANGLE_TAG_RE.search(s))
    bracket_hits = sum(1 for s in samples if _SQUARE_BRACKET_RE.search(s))
    paren_hits = sum(1 for s in samples if _PAREN_RE.search(s))
    curly_hits = sum(1 for s in samples if _CURLY_RE.search(s))
    control_hits = sum(1 for s in samples if _CONTROL_CHAR_RE.search(s))
    html_hits = sum(1 for s in samples if _HTML_ENT_RE.search(s))
    speaker_hits = sum(1 for s in samples if _SPEAKER_TAG_RE.search(s))
    multispace_hits = sum(1 for s in samples if _MULTISPACE_RE.search(s))
    empty = sum(1 for s in samples if not s.strip())
    very_short = sum(1 for s in samples if 0 < len(s.strip()) < 3)
    numeric_hits = sum(1 for s in samples if _NUMERIC_NORM_RE.search(s))

    # Coverage of train.py's regex.
    train_handled = sum(
        1 for s in samples if _CORPUS_MARKER_RE.search(s) or _TEDLIUM_BRACKET_RE.search(s)
    )

    # Distinct angle-tag / bracket / paren contents (top-10).
    angle_tokens = Counter(m.group(0) for s in samples for m in _ANGLE_TAG_RE.finditer(s))
    bracket_tokens = Counter(m.group(0) for s in samples for m in _SQUARE_BRACKET_RE.finditer(s))
    paren_tokens = Counter(m.group(0) for s in samples for m in _PAREN_RE.finditer(s))

    # Format classification (majority of non-empty samples).
    fmt = Counter(fmt_classify(s) for s in samples if s.strip()).most_common(1)
    fmt_label = fmt[0][0] if fmt else "n/a"

    # Non-ascii: catch mojibake / unicode oddities.
    non_ascii_hits = sum(1 for s in samples if any(ord(c) > 127 for c in s))

    print(f"  format: {fmt_label}")
    print(f"  empty: {empty}  very-short: {very_short}  numeric: {numeric_hits}")
    print(
        f"  angle <...>: {angle_hits:>3}  bracket [...]: {bracket_hits:>3}  paren (...): {paren_hits:>3}  curly {{...}}: {curly_hits:>3}"
    )
    print(
        f"  control-chars: {control_hits}  html-ent: {html_hits}  speaker-tag: {speaker_hits}  multispace: {multispace_hits}  non-ascii: {non_ascii_hits}"
    )
    print(f"  covered-by-train.py-regex: {train_handled}/{n}")
    if angle_tokens:
        print(f"  angle tokens (top): {angle_tokens.most_common(5)}")
    if bracket_tokens:
        print(f"  bracket tokens (top): {bracket_tokens.most_common(5)}")
    if paren_tokens:
        print(f"  paren tokens (top): {paren_tokens.most_common(5)}")
    # How often does _normalize_label collapse to empty (would drop sample)?
    normalized_empty = sum(1 for s in samples if not _normalize_label(s))
    print(f"  normalized-to-empty (sample-drop rate): {normalized_empty}/{n}")
    print("  sample transcripts (raw → _normalize_label):")
    for s in samples[:3]:
        raw = s if len(s) <= 160 else s[:160] + "…"
        norm = _normalize_label(s)
        norm_disp = norm if len(norm) <= 160 else norm[:160] + "…"
        print(f"    RAW : {raw!r}")
        print(f"    NORM: {norm_disp!r}")
    return {
        "label": label,
        "n": n,
        "format": fmt_label,
        "empty": empty,
        "very_short": very_short,
        "angle_hits": angle_hits,
        "bracket_hits": bracket_hits,
        "paren_hits": paren_hits,
        "curly_hits": curly_hits,
        "control_hits": control_hits,
        "html_hits": html_hits,
        "speaker_hits": speaker_hits,
        "non_ascii_hits": non_ascii_hits,
        "train_handled": train_handled,
        "normalized_empty": normalized_empty,
        "angle_tokens": angle_tokens.most_common(10),
        "bracket_tokens": bracket_tokens.most_common(10),
        "paren_tokens": paren_tokens.most_common(10),
    }


def main():
    print(
        f"Streaming {N_SAMPLES} samples per dataset from multiasr.yaml inventory "
        f"(shuffle seed={SHUFFLE_SEED}, buffer={SHUFFLE_BUFFER}).\n"
    )

    results = []
    for path, name, split, text_col, _audio_col, label in DATASETS:
        del _audio_col  # listed for completeness; only text inspected here
        try:
            r = inspect_one(label, path, name, split, text_col)
            results.append(r)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\n  UNEXPECTED FAILURE: {type(e).__name__}: {e}")
            results.append({"label": label, "error": str(e)})

    print("\n\n=== SUMMARY ===")
    print(
        f"{'Dataset':<24} {'Fmt':<22} {'Ang':>4} {'Brk':>4} {'Par':>4} {'Cur':>4} {'Ctl':>4} {'HTM':>4} {'Spk':>4} {'NAS':>4} {'Emp':>4} {'NEm':>4} {'Cov':>6}"
    )
    for r in results:
        if "error" in r:
            print(f"{r['label']:<24} ERROR: {r['error'][:80]}")
            continue
        print(
            f"{r['label']:<24} {r['format']:<22} "
            f"{r['angle_hits']:>4} {r['bracket_hits']:>4} {r['paren_hits']:>4} {r['curly_hits']:>4} "
            f"{r['control_hits']:>4} {r['html_hits']:>4} {r['speaker_hits']:>4} {r['non_ascii_hits']:>4} "
            f"{r['empty']:>4} {r['normalized_empty']:>4} {r['train_handled']:>4}/{r['n']}"
        )
    print(
        "\nLegend: Ang=<...>, Brk=[...], Par=(...), Cur={...}, Ctl=ctrl-char, HTM=html-ent, "
        "Spk=speaker-tag, NAS=non-ascii, Emp=raw-empty, NEm=normalized-empty (post-_normalize_label), "
        "Cov=train.py-regex-handled/total"
    )


if __name__ == "__main__":
    sys.exit(main())
