"""Training-label normalization: canonicalize raw dataset transcripts.

Maps corpus-specific annotation (Gigaspeech punctuation tags, TEDLIUM <unk>
and editorial brackets, EdAcc/Earnings22 event markers) to clean cased text
with punctuation, and flags rows whose edge tags would supervise onset
truncation. Used by the training collator in scripts/train.py.
"""

import functools
import os
import re
from collections.abc import Callable
from typing import cast

import ftfy
import truecase

# Gigaspeech ships inline punctuation as angle-bracket tags so we restore
# them to real punctuation before any other normalization. Pattern follows
# the Ultravox text_proc.format_asr_text recipe.
_GIGASPEECH_PUNCT_MAP = {
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTIONMARK": "?",
    "EXCLAMATIONPOINT": "!",
}
_GIGASPEECH_PUNCT_RE = re.compile(
    r"\s*<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>",
    re.IGNORECASE,
)
# Non-punct annotation markers worth stripping (but keep the rest of the
# label). Gigaspeech ships <SIL>/<NOISE>/<MUSIC>/<OTHER> for non-speech
# segments; TEDLIUM ships <unk> in ~92% of train rows; Switchboard ships
# <laugh>; EdAcc ships <overlap>/<dtmf>/<foreign>/<no-speech>/<lipsmack>;
# Earnings22 ships <clear_throat>/<inaudible>/<crosstalk>. These mark
# intra-utterance events that the eval refs do NOT include, so stripping
# is safe.
#
# Strip-don't-drop is deliberate: a previous revision tried Ultravox's
# whole-sample-drop pattern for the four Gigaspeech non-speech tags and
# broke eval — small eval batches that happened to draw samples with
# those tags came back fully empty and crashed the collator. Stripping
# preserves partial speech transcripts (audio may have speech around the
# tagged non-speech moment) and the empty-label filter at the collator
# still catches the edge case where the entire label was just a tag.
# After the Gigaspeech punct map converts <COMMA>/<PERIOD>/etc. to real
# punctuation, any remaining `<...>` token is a non-speech annotation
# marker (Gigaspeech <MUSIC>/<NOISE>/<SIL>/<OTHER>, TEDLIUM <unk>,
# Switchboard <LAUGH>, EdAcc <overlap>/<dtmf>/<foreign>/<no-speech>/
# <lipsmack>, Earnings22 <clear_throat>/<inaudible>/<crosstalk>, plus the
# long tail of bodily-noise tags like <inhale>/<sigh>/<cough> that vary
# across corpora). ASR transcripts never legitimately contain `<word>`
# tokens, so a generic strip is safer than a whitelist (whitelists
# silently leak whichever marker variant a new corpus happens to use,
# training the decoder to emit it as a literal token). Substitution
# uses a single space so adjacent-no-whitespace forms (`word<sigh>word`)
# don't collapse to a concatenated string before the whitespace pass.
_RESIDUAL_ANGLE_TAG_RE = re.compile(r"<[^>]+>")
# TEDLIUM occasionally inlines editorial commentary in square brackets
# ([ medicine ], [ multi-word stage direction ]) — ~0.25% of train rows;
# zero in dev/test. Same single-space substitution rationale as above.
_TEDLIUM_BRACKET_RE = re.compile(r"\[[^\]]*\]")
# TEDLIUM writes audience events as bare words at segment edges ("laughter so
# i had to add handcuffs", "... united hatzalah applause"): 53 edge vs 2
# mid-segment hits in 15,000 train rows, the mid ones split between an event
# and real speech ("i felt applause on the vest"), so only edges are
# stripped. The TEDLIUM eval refs carry none. ~0.4% of rows, but frozen-2's
# plain prompt amplified it into a leading "Laughter" on 15% of CommonVoice.
# Applied only to all-lowercase mono labels (TEDLIUM's raw form), so cased
# sources and ALL-CAPS Gigaspeech/AMI keep a real edge "laughter".
_EDGE_EVENT_RE = re.compile(r"^(?:(?:laughter|applause)\b\s*)+|(?:\s*\b(?:laughter|applause))+$")
# Word-bounded `per cent` → `percent` to avoid false positives on
# `per centage` / `per centimeter` / etc.; the prior `text.replace("per cent", "percent")`
# silently mangled those. `\bper ?cent\b` also harmlessly matches an
# already-collapsed `percent` (the replacement is identical, so it's a no-op).
_PER_CENT_RE = re.compile(r"\bper ?cent\b")
# TEDLIUM occasionally tokenizes negation contractions with the apostrophe-t
# split off the verb stem — `didn 't` instead of `didn't` or `did n't`. Probe
# of 500 TEDLIUM train rows showed this in ~10%, dominated by
# `don 't` / `didn 't` / `wouldn 't` / `can 't` / `wasn 't`. Truecase
# tokenizes the orphan `'t` as a standalone token and uppercases it, so
# labels arrive as `didn 'T embrace` and train the decoder to emit broken
# contractions. Pre-collapse the orphan before truecase runs. The `\w+n`
# anchor means we only target the negation-contraction shape, so we never
# touch space-before-apostrophe forms like `she 'd` / `it 's` / `friends '`
# that truecase already handles correctly via its contraction vocabulary.
_ORPHAN_NT_RE = re.compile(r"\b(\w+n)\s+'t\b")
_WHITESPACE_RE = re.compile(r"\s+")

# Annotation tags that stand in for SPOKEN LEXICAL CONTENT the audio still
# contains, as opposed to non-speech events. `_RESIDUAL_ANGLE_TAG_RE` deletes
# every tag, which is correct for <noise>/<music>/<sil>/<laugh>/<breath> (no
# word was uttered) but wrong for these three:
#   <unk>     TEDLIUM — a word the transcriber could not identify. The word
#             IS in the audio; only the transcription is missing.
#   <foreign> EdAcc  — speech in another language, present in the audio.
#   <overlap> EdAcc  — overlapping speech, present in the audio.
# When such a tag sits at the START or END of a label, stripping it produces a
# target that omits the first or last spoken word while the audio keeps it —
# i.e. it directly supervises onset/offset truncation.
#
# Measured on 800 streamed `sanchit-gandhi/tedlium-data` train rows
# (2026-09-18): 60.8% of rows contain <unk>, 30.4% START with one, 21.9% END
# with one. TEDLIUM is 8.7% of the multiasr mix, so ~2.6% of ALL training rows
# were teaching leading-word deletion and ~1.9% trailing-word deletion.
#
# This compounds a second, independent source of the same prior: Peoples
# Speech `clean_sa` ships fixed ~15s windows (95.7% of rows in [14.0, 15.1]s)
# whose labels lose words at the chunk seams. Eval symptom: the model dropped
# >=1 leading reference word on 52/100 Peoples samples (92 words = 5.59 WER
# points) versus 22/45 for a commercial baseline, and removing leading and
# trailing deletion runs made the two systems tie. A frozen-encoder CTC
# control — which never saw this training data — beat the full stack by 3.52
# WER on Peoples, confirming the prior is decoder-learned rather than acoustic.
#
# We drop EDGE occurrences only, not medial ones. Medial <unk> is also a
# lexically-incomplete target, but dropping every <unk> row would remove 60.8%
# of TEDLIUM, and TEDLIUM is the single dataset where the decoder measurably
# earns its keep over the frozen encoder (+6.18 WER). Edge position is also
# what the eval evidence actually implicates: a positional prior is learnable,
# a scattered mid-sentence omission is closer to label noise.
# LEADING ONLY as of 2026-09-20. The trailing alternation was dropped after
# measuring what the two halves actually cost: on the 25,550-row TEDLIUM index,
# 27.3% of rows start with a stripped tag and 20.3% end with one, union 41.2%
# (the prior "roughly half" estimate summed the two and double-counted the
# 1,652 rows that do both). Leading-only drops 27.3%, recovering ~13.9% of
# TEDLIUM, about +37,300 rows.
#
# The evidence base only ever supported the leading half. It came from the
# Peoples eval -- a *leading*-word deletion prior, measured as >=1 dropped
# leading reference word on 52/100 samples. Trailing truncation was never
# measured separately, and Peoples has since left the training mix, so the
# trailing half rested on argument-by-analogy to a corpus that is no longer
# there. Meanwhile TEDLIUM is the one dataset where the decoder measurably
# beats the frozen encoder (+6.18 WER), and this filter was cutting it 41%.
#
# To re-justify the trailing half, measure leading/trailing deletion runs
# on the TEDLIUM eval's results.txt; if trailing runs
# are elevated over baseline, restore the `|<(?:unk|foreign|overlap)>\s*$`
# alternation.
_EDGE_CONTENT_TAG_RE = re.compile(r"^\s*<(?:unk|foreign|overlap)>", re.IGNORECASE)


def _has_edge_content_tag(raw_text: str | None) -> bool:
    """True when a label starts or ends with a content-bearing annotation tag.

    Such rows supervise onset/offset truncation once the tag is stripped, so
    the collator drops them rather than training on a label that is known to
    be missing its first or last spoken word.
    """
    return bool(_EDGE_CONTENT_TAG_RE.search(raw_text or ""))


# Post-truecase cleanup. Truecase's NLTK-backed tokenizer reformats text
# in three ways that survive into training labels:
#   1. Sentence-final periods get split off as standalone tokens, then
#      re-joined with a leading space (`rate . But` instead of `rate. But`).
#      Found in ~25% of Gigaspeech rows (the multi-sentence ones).
#   2. Truecase fails to capitalize the next sentence after a mid-sentence
#      period (`E T. the Video game.` instead of `E T. The Video game.`).
#   3. Em-dash spaces get eaten (`for -- we` → `for--we`); seen in SPGI.
#   4. Informal `gonna`/`wanna` get mangled to `gonNA`/`wanNA` regardless
#      of input casing; seen across AMI / Switchboard / Gigaspeech.
# These post-fixes run only when truecase actually fired (already-cased
# sources skip truecase and don't need this cleanup).
_SPACE_BEFORE_SENT_PUNCT_RE = re.compile(r"\s+([.,!?])")
_SENT_START_LOWERCASE_RE = re.compile(r"([.!?])\s+([a-z])")
# A period that closes a run of spelled-out letters is not a sentence boundary.
# AMI writes acronyms inline as "S. S. H." / "X. M. L.", and AMI has no real
# sentence punctuation at all, so capitalizing after one is always wrong there
# (measured: 13 of 300 rows contain a period, and all 13 are spelled letters).
# Matched against the text preceding the period, so it fires on the SECOND and
# later members of a run — the discriminator against Gigaspeech's tag-derived
# boundaries, which look like "e t. the video game." where the letter before
# the period carries no period of its own.
_SPELLED_LETTER_RUN_RE = re.compile(r"\b[A-Za-z]\.\s+[A-Za-z]$")
_EM_DASH_RE = re.compile(r"\s*--\s*")
_GONNA_ARTIFACT_RE = re.compile(r"\bgonNA\b")
_WANNA_ARTIFACT_RE = re.compile(r"\bwanNA\b")
_GOTTA_ARTIFACT_RE = re.compile(r"\bgotTA\b")


def _capitalize_sentence_starts(text: str) -> str:
    """Uppercase the first letter after sentence-final punctuation.

    Truecase fails to capitalize the next sentence after a mid-string period
    ("E T. the Video game." instead of "E T. The Video game."), so this fixes
    it up — except after a spelled-letter run, where the period is part of an
    acronym rather than a boundary.
    """

    def repl(match: re.Match[str]) -> str:
        if _SPELLED_LETTER_RUN_RE.search(text[: match.start()]):
            whole: str = match.group(0)
            return whole
        return f"{match.group(1)} {match.group(2).upper()}"

    return _SENT_START_LOWERCASE_RE.sub(repl, text)


def _post_truecase_cleanup(text: str) -> str:
    """Undo the tokenizer artifacts truecase leaves behind (listed above)."""
    text = _SPACE_BEFORE_SENT_PUNCT_RE.sub(r"\1", text)
    text = _capitalize_sentence_starts(text)
    text = _EM_DASH_RE.sub(" -- ", text)
    text = _GONNA_ARTIFACT_RE.sub("gonna", text)
    text = _WANNA_ARTIFACT_RE.sub("wanna", text)
    return _GOTTA_ARTIFACT_RE.sub("gotta", text)


# Unicode cleanup (`ftfy`, imported above): ftfy fixes mojibake (â€™ → '),
# unescapes HTML entities (&amp; → &), and folds smart quotes (' " → ' ");
# NFKC further normalizes composed/decomposed forms (café vs cafe + ◌́) and
# width variants (full-width Latin → half-width). Applied first in
# _normalize_label so downstream regexes see canonical ASCII-leaning text.
#
# Truecase (`truecase`, imported above): NLTK-backed statistical recasing for
# transcripts that arrive in mono-case form (all-upper or zero-caps). The
# package ships without annotations; `get_true_case` maps a str to a str.
_get_true_case = cast(
    Callable[[str], str],
    truecase.get_true_case,  # pyright: ignore[reportUnknownMemberType]
)
# LOCAL_RANK=0 guard mirrors Ultravox — avoids multiple workers racing on the
# punkt download.
if int(os.environ.get("LOCAL_RANK", "0")) == 0:
    try:
        _get_true_case("test")
    except LookupError:
        import nltk

        # NLTK 3.9+ requires `punkt_tab`; older NLTKs use `punkt`. Download
        # both so this works on either base image. Quiet=True suppresses
        # progress bars; the fetch is ~13 MB and usually completes in
        # seconds.
        # nltk leaves `download`'s parameters unannotated.
        download = cast(
            Callable[..., bool],
            nltk.download,  # pyright: ignore[reportUnknownMemberType]
        )
        download("punkt_tab", quiet=True)
        download("punkt", quiet=True)


# Per-source casing policy, set via a dataset config's `text_case` field and
# carried to the collator on the `_text_case` column. Declaring it beats the
# per-row heuristic below because the answer is a property of the SOURCE, not
# of the row — see _needs_truecase's own docstring, which names the sources it
# is trying to re-derive from characters.
TEXT_CASE_MONO = "mono"  # ALL-CAPS or zero-cap source; recase it
TEXT_CASE_CASED = "cased"  # ships case + proper nouns; never touch
# Below this many letters the statistical truecaser has too little context to
# be reliable — it promotes backchannels to proper nouns. Short mono-case text
# gets a deterministic recase instead.
_MIN_TRUECASE_LETTERS = 5


def _needs_truecase(text: str) -> bool:
    """Heuristic fallback for sources with no declared `text_case`.

    Apply truecase only to mono-case text. Already-cased sources (LibriHeavy
    text_original, CV, VoxPopuli raw_text, SPGISpeech) carry proper-noun
    casing that the statistical truecaser would damage (e.g. "McClarnon" ->
    "Mcclarnon"). Heuristic: text with any internal capitalization beyond what
    truecase would produce is already cased.

    Prefer declaring `text_case` on the dataset. This heuristic misclassifies
    in both directions and cannot do better from a single row: a lowercase
    FRAGMENT of a cased source (SPGISpeech's sliding window emits these for
    13% of rows) is character-identical to a row from a genuinely uncased
    source, and punctuation does not separate them either.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < _MIN_TRUECASE_LETTERS:
        # Too short to recase meaningfully ("yeah", "OH"). Leave alone. Note
        # this is only safe when the source is already cased; a declared
        # `mono` source routes to _recase_monocase_text instead, which handles
        # short text deterministically rather than passing it through.
        return False
    upper_count = sum(c.isupper() for c in letters)
    upper_frac = upper_count / len(letters)
    if upper_frac > 0.9:
        return True  # ALL-CAPS source (Gigaspeech post-restoration, AMI)
    # zero-cap (TEDLIUM, Peoples, Switchboard) → truecase;
    # otherwise already cased (LibriHeavy, CV, SPGI, VoxPopuli) → skip.
    return upper_count == 0


def _capitalize_first_letter(text: str) -> str:
    """Uppercase the first alphabetic character, leaving the rest untouched."""
    for i, char in enumerate(text):
        if char.isalpha():
            return f"{text[:i]}{char.upper()}{text[i + 1 :]}"
    return text


def _recase_monocase_text(text: str) -> str:
    """Recase a row from a source declared `text_case: mono`.

    Long text goes to the statistical truecaser. Short text does not: the
    truecaser needs context, and without it the old code simply passed the row
    through unchanged — which on an ALL-CAPS source means shipping "YEAH" /
    "OKAY" / "HMM" as training labels. Measured at 21% of AMI rows. A
    deterministic lowercase-then-capitalize is all these actually need and it
    cannot invent proper nouns.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= _MIN_TRUECASE_LETTERS:
        return _post_truecase_cleanup(_get_true_case(text))
    return _capitalize_first_letter(text.lower())


# Pure function of its input, and the collator normalizes each row twice: once
# to test for an empty label and once to build the sample. Cache sized well
# above the largest training batch so the second call is always a hit.
@functools.lru_cache(maxsize=4096)
def _normalize_label(raw_text: str | None, text_case: str | None = None) -> str:
    """Canonicalize a training transcript label to cased+punct form.

    Pipeline (in order):
    1. ftfy + NFKC unicode cleanup: fix mojibake (â€™ → '), unescape HTML
       entities, fold smart quotes to straight, normalize composed /
       decomposed forms and width variants. Defensive — our 100-sample-
       per-dataset audit found zero non-ASCII in current sources, but
       tail samples (especially OCR-derived audiobook text in LibriHeavy)
       may carry curly quotes / Unicode oddities. Idempotent on clean
       text; ~10us per call.
    2. Map Gigaspeech inline-punct tags (<COMMA>/<PERIOD>/etc.) to real
       punctuation. Done before the residual-marker strip so the tags
       become punct rather than getting stripped to nothing.
    3. Strip non-punct annotation markers (<unk>, <LAUGH>, <inaudible>,
       Gigaspeech <MUSIC>/<NOISE>/<SIL>/<OTHER>, etc.) and TEDLIUM
       editorial brackets ([ ... ]). For Gigaspeech non-speech tags the
       audio segment may still contain speech around the tagged moment;
       strip-not-drop preserves the partial transcript. The collator's
       empty-label filter catches the entire-label-was-just-a-tag case.
    4. Collapse the `per cent` spelling variant to `percent`. The literal
       `%` character is PRESERVED — see the note below.
    5. Collapse whitespace.
    6. Recase according to `text_case`, the source's declared casing policy
       (set per dataset in the data config, carried on the `_text_case`
       column). `mono` lifts ALL-CAPS sources (Gigaspeech, AMI) and zero-cap
       sources (TEDLIUM, Peoples, Switchboard) to proper-cased form; `cased`
       leaves already-cased sources (LibriHeavy, CV, SPGI, VoxPopuli)
       untouched. When a source declares nothing, fall back to the per-row
       _needs_truecase heuristic — which is what every source used to get,
       and which misclassifies lowercase fragments of cased sources.

    A prior revision of step 4 also ran `text.replace("%", " percent")`, to
    mirror an eval-side analysis rule. That was removed
    (2026-09-18) because it destroyed the `%` character in 100% of training
    targets: only Earnings22 and SPGISpeech ship `%` natively (~6,188 rows of
    the ~3.09M mix) and both were rewritten, so the decoder emitted 0 `%` in
    6,055 sampled eval predictions and scored 0% on the `percent` class of
    the raw-text ITN metric against 92.9% for a commercial baseline — ~93% of
    a measured 66%-vs-94% ITN gap, from this one line.

    Two facts make the removal safe rather than a trade:
      - It is WER-neutral by construction. Whisper's EnglishTextNormalizer,
        which the eval applies symmetrically to reference and hypothesis,
        maps "105 percent" and "105%" to the identical string. WER cannot
        see this change in either direction.
      - The capability was never missing. `$` was not stripped and the
        decoder reproduces it correctly — including spontaneously, e.g.
        "five thousand dollars" -> "$400,000" — on ~240 training rows, 26x
        less supervision than `%` would have had. So the cause was the
        rewrite, not the data volume.

    An eval-side copy of that rule is fine: applied to both sides at scoring
    time, it is canonicalization rather than label destruction.

    Output target format is cased text with punctuation where available —
    aligning the dominant training label distribution to the Qwen3
    decoder's native output format. WER scoring uses Whisper's
    EnglishTextNormalizer which lowercases + strips punct on both
    prediction and reference, so the format choice does not affect WER
    comparability across runs.
    """
    text = (raw_text or "").strip()
    if not text:
        return ""
    text = ftfy.fix_text(text, normalization="NFKC")
    text = _GIGASPEECH_PUNCT_RE.sub(lambda m: _GIGASPEECH_PUNCT_MAP[m.group(1).upper()], text)
    text = _RESIDUAL_ANGLE_TAG_RE.sub(" ", text)
    text = _TEDLIUM_BRACKET_RE.sub(" ", text)
    text = _PER_CENT_RE.sub("percent", text)
    text = _ORPHAN_NT_RE.sub(r"\1't", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if text_case == TEXT_CASE_MONO and text.islower():
        text = _EDGE_EVENT_RE.sub("", text).strip()
    if not text:
        return ""
    if text_case == TEXT_CASE_CASED:
        return text
    if text_case == TEXT_CASE_MONO:
        return _recase_monocase_text(text)
    if _needs_truecase(text):
        text = _get_true_case(text)
        text = _post_truecase_cleanup(text)
    return text
