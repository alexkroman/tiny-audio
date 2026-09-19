#!/usr/bin/env python3
"""Training script for ASR models using Hydra configuration."""

# ruff: noqa: E402
# The trl env-var must be set, and the noisy-logger silencer must run,
# *before* their respective modules are imported below — so non-import
# statements precede some imports here. Suppress E402 file-wide rather
# than per-line.

import contextlib
import functools
import logging
import math
import os
import re
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub.file_download"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

import hydra
import numpy as np
import torch
import wandb
from datasets import (
    Audio,
    Dataset,
    concatenate_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import get_total_norm
from tqdm.auto import tqdm
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl.experimental.utils import DataCollatorForChatML  # pyright: ignore[reportMissingImports]

from tiny_audio.asr_config import (
    DEFAULT_ENCODER_CONV_LAYERS,
    ASRConfig,
    compute_encoder_output_length,
)
from tiny_audio.asr_modeling import ASRModel

TRANSCRIBE_PROMPT = "Transcribe the speech to text"
# Used for sources whose transcripts natively carry punctuation, selected per
# row via the `text_punct` dataset field. Granite Speech 4.1 documents exactly
# this mechanism -- its model card says punctuation and truecasing are chosen
# "with a simple prompt change", and its usage example is literally
# "<|audio|>transcribe the speech with proper punctuation and capitalization."
# Qwen3-ASR does the equivalent through a system turn plus an assistant prefill.
#
# Without the split, the ~25% of multiasr that is truecased-but-unpunctuated
# (TEDLIUM, Peoples, AMI, Switchboard) trains the model to SUPPRESS punctuation
# under the same prompt the punctuated ~75% uses to produce it. Identical
# conditioning, contradictory targets: the model can only learn a hedge, and
# every dropped mark scores as an error against punctuated references.
TRANSCRIBE_PROMPT_PUNCT = "Transcribe the speech with proper punctuation and capitalization"

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
_EDGE_CONTENT_TAG_RE = re.compile(
    r"^\s*<(?:unk|foreign|overlap)>|<(?:unk|foreign|overlap)>\s*$", re.IGNORECASE
)


def _has_edge_content_tag(raw_text: str) -> bool:
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

    def repl(match: re.Match) -> str:
        if _SPELLED_LETTER_RUN_RE.search(text[: match.start()]):
            return match.group(0)
        return f"{match.group(1)} {match.group(2).upper()}"

    return _SENT_START_LOWERCASE_RE.sub(repl, text)


def _post_truecase_cleanup(text: str) -> str:
    text = _SPACE_BEFORE_SENT_PUNCT_RE.sub(r"\1", text)
    text = _capitalize_sentence_starts(text)
    text = _EM_DASH_RE.sub(" -- ", text)
    text = _GONNA_ARTIFACT_RE.sub("gonna", text)
    text = _WANNA_ARTIFACT_RE.sub("wanna", text)
    return _GOTTA_ARTIFACT_RE.sub("gotta", text)


# Unicode cleanup: ftfy fixes mojibake (â€™ → '), unescapes HTML entities
# (&amp; → &), and folds smart quotes (' " → ' "); NFKC further normalizes
# composed/decomposed forms (café vs cafe + ◌́) and width variants
# (full-width Latin → half-width). Applied first in _normalize_label so
# downstream regexes see canonical ASCII-leaning text.
import ftfy

# Truecase: NLTK-backed statistical recasing for transcripts that arrive
# in mono-case form (all-upper or zero-caps). LOCAL_RANK=0 guard mirrors
# Ultravox — avoids multiple workers racing on the punkt download.
import truecase

if int(os.environ.get("LOCAL_RANK", "0")) == 0:
    try:
        truecase.get_true_case("test")
    except LookupError:
        import nltk

        # NLTK 3.9+ requires `punkt_tab`; older NLTKs use `punkt`. Download
        # both so this works on either base image. Quiet=True suppresses
        # progress bars; the fetch is ~13 MB and usually completes in
        # seconds.
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)


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
        return _post_truecase_cleanup(truecase.get_true_case(text))
    return _capitalize_first_letter(text.lower())


# Pure function of its input, and the collator normalizes each row twice: once
# to test for an empty label and once to build the sample. Cache sized well
# above the largest training batch so the second call is always a hit.
@functools.lru_cache(maxsize=4096)
def _normalize_label(raw_text: str, text_case: str | None = None) -> str:
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
    mirror an eval-side rule in scripts/analysis.py. That was removed
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

    The eval-side copy in scripts/analysis.py is correct and should stay: it
    is applied to both sides at scoring time, which is canonicalization
    rather than label destruction.

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
    if not text:
        return ""
    if text_case == TEXT_CASE_CASED:
        return text
    if text_case == TEXT_CASE_MONO:
        return _recase_monocase_text(text)
    if _needs_truecase(text):
        text = truecase.get_true_case(text)
        text = _post_truecase_cleanup(text)
    return text


class DatasetLoader:
    """Loads and prepares datasets for training.

    Downloads each train/eval split fully via HuggingFace's Arrow cache,
    then concatenates and shuffles.
    """

    def __init__(self, config: DictConfig):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir
        self.seed = config.training.get("seed", 42)
        self.num_proc = self.config.get("num_proc", 16)

    def _prepare_split(self, dataset_cfg: DictConfig, split: str) -> Dataset:
        dataset_path = dataset_cfg.get("path")
        if not dataset_path:
            raise ValueError("Dataset path is required")

        ds = load_dataset(
            dataset_path,
            name=dataset_cfg.get("name"),
            split=split,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc,
            trust_remote_code=True,
        )

        # CommonVoice strict-validated filter: Mozilla's `train` split is
        # already up-vote validated (up_votes >= 2 AND up_votes > down_votes),
        # but still admits clips with non-zero down_votes. Filtering to
        # down_votes == 0 cuts the small tail of community-flagged
        # audio/transcript mismatches. Applied to all CV splits (train +
        # eval) for consistency with the TEDLIUM marker-filter pattern
        # below. Guarded on column presence in case a future mirror strips
        # the voting metadata.
        if "common_voice" in dataset_path.lower() and "down_votes" in ds.column_names:
            ds = ds.filter(
                lambda dv: dv == 0,
                num_proc=self.num_proc,
                input_columns="down_votes",
            )

        # Declarative row filter on a source-metadata column, e.g.
        #   exclude_where: {column: source, values: [audiobook]}
        # It must run HERE, before the keep_cols pruning below drops every
        # column that is not audio/text/_text_case/_text_punct -- by then the
        # column you want to filter on no longer exists.
        #
        # Motivating case (Gigaspeech): the `dev` split we score contains
        # ZERO audiobook rows (full 6,750-row scan: 55.3% youtube, 44.7%
        # podcast), while 26.2% of Gigaspeech `m` train rows ARE audiobook --
        # a register that is 0% of the eval, on top of the 600K LibriHeavy
        # audiobook rows the mix already carries. Excluding it is free:
        # non-audiobook GS M is ~672K rows, still above the 600K
        # target_samples cap, so row count, mix share and download are all
        # unchanged. Rows are swapped, not lost.
        exclude_where = dataset_cfg.get("exclude_where")
        if exclude_where:
            column = exclude_where.get("column")
            values = set(exclude_where.get("values") or [])
            if not column or not values:
                raise ValueError(
                    f"exclude_where needs both 'column' and non-empty 'values', "
                    f"got {exclude_where!r} for {dataset_path}"
                )
            if column not in ds.column_names:
                # Fail loudly: a silently-ignored filter would train on the
                # rows you believe you excluded, and the mix table would lie.
                raise ValueError(
                    f"exclude_where column {column!r} not in {dataset_path} "
                    f"(available: {sorted(ds.column_names)})"
                )
            before = len(ds)
            ds = ds.filter(
                lambda v: v not in values,
                num_proc=self.num_proc,
                input_columns=column,
            )
            logger.info(
                "exclude_where on %s: dropped %d/%d rows where %s in %s",
                dataset_path,
                before - len(ds),
                before,
                column,
                sorted(values),
            )

        col_map = {
            "text": dataset_cfg.get("text_column", "text"),
            "audio": dataset_cfg.get("audio_column", "audio"),
        }
        for target, source in col_map.items():
            if source != target and source in ds.column_names:
                if target in ds.column_names:
                    ds = ds.remove_columns([target])
                ds = ds.rename_column(source, target)

        # text_case: declares whether this source's transcripts already carry
        # case ("cased") or arrive mono-case and need recasing ("mono").
        # Stored per row so _normalize_label does not have to re-derive a
        # source property from a single row's characters.
        # Omit it to keep the legacy per-row heuristic.
        text_case = dataset_cfg.get("text_case")
        if text_case is not None:
            if text_case not in (TEXT_CASE_MONO, TEXT_CASE_CASED):
                raise ValueError(
                    f"text_case must be {TEXT_CASE_MONO!r} or {TEXT_CASE_CASED!r}, "
                    f"got {text_case!r} for {dataset_path}"
                )
            ds = ds.add_column("_text_case", [text_case] * len(ds))

        # text_punct: declares whether this source's transcripts carry
        # punctuation. Deliberately separate from text_case -- they are not the
        # same axis, and conflating them gets Gigaspeech wrong, which is
        # ALL-CAPS (text_case: mono) yet natively punctuated. Omit it and the
        # row gets the plain prompt, i.e. today's behaviour.
        text_punct = dataset_cfg.get("text_punct")
        if text_punct is not None:
            if not isinstance(text_punct, bool):
                raise ValueError(
                    f"text_punct must be a bool, got {text_punct!r} for {dataset_path}"
                )
            ds = ds.add_column("_text_punct", [text_punct] * len(ds))

        ds = ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        keep_cols = {"audio", "text"}
        # Preserve the declared casing policy so _normalize_label can use it.
        if "_text_case" in ds.column_names:
            keep_cols = keep_cols | {"_text_case"}
        # Preserve the declared punctuation policy so _build_sample can pick
        # the matching prompt.
        if "_text_punct" in ds.column_names:
            keep_cols = keep_cols | {"_text_punct"}
        extra_cols = [c for c in (ds.column_names or []) if c not in keep_cols]

        if extra_cols:
            ds = ds.remove_columns(extra_cols)

        # Filter `ignore_time_segment_in_scoring` placeholder labels. TEDLIUM
        # uses them to mark unscored regions; EdAcc reuses the same convention
        # in its validation transcripts. Both ship rows where the entire label
        # IS that string — training on them teaches the model to emit it.
        # Case-insensitive: TEDLIUM ships lowercase, EdAcc ships uppercase.
        # Duration filtering happens in DataCollator to avoid loading all audio upfront.
        if "tedlium" in dataset_path.lower() or "edacc" in dataset_path.lower():

            def filter_ignore_marker(text):
                return text.strip().lower() != "ignore_time_segment_in_scoring"

            ds = ds.filter(filter_ignore_marker, num_proc=self.num_proc, input_columns="text")

        return ds

    def _resample_to_target(self, ds: Dataset, target: int) -> Dataset:
        """Cap (downsample) or repeat-pad (upsample) to ``target`` samples.

        When downsampling, shuffle deterministically before subsetting so
        the cap is a representative sample rather than the first N rows
        in the dataset's natural order. Several HF datasets ship with
        non-random ordering (LibriHeavy by chapter/speaker, CV by
        validation date, etc.); taking `range(target)` directly would
        introduce selection bias on top of the intended volume cap. Seed
        pinned to `self.seed` for reproducibility across runs with the
        same config.
        """
        current = len(ds)
        if current == target:
            return ds
        if current > target:
            return ds.shuffle(seed=self.seed).select(range(target))
        repeats = (target // current) + 1
        indices = list(range(current)) * repeats
        return ds.select(indices[:target])

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []

        for d_cfg in tqdm(self.config.datasets, desc="Loading datasets"):
            train_splits = d_cfg.get("train_splits", ["train"])
            val_splits = d_cfg.get("eval_splits", ["validation"])
            target_samples = d_cfg.get("target_samples")

            for train_split in train_splits:
                ds = self._prepare_split(d_cfg, train_split)
                if target_samples:
                    ds = self._resample_to_target(ds, target_samples)
                train_datasets.append(ds)

            # Per-dataset eval cap applied here (pre-concat) so each eval
            # source contributes a balanced slice. Prior behavior — cap-
            # then-concat-then-truncate — silently dropped late-list eval
            # splits (e.g. AMI, Switchboard) because the global
            # max_eval_samples cap filled up on early-list splits (TEDLIUM
            # + head of Peoples val) before reaching them.
            eval_cap_per_dataset = self.config.get("max_eval_samples_per_dataset")
            for val_split in val_splits:
                ds = self._prepare_split(d_cfg, val_split)
                if eval_cap_per_dataset:
                    ds = ds.select(range(min(len(ds), eval_cap_per_dataset)))
                val_datasets.append(ds)

        train_ds = (
            concatenate_datasets(train_datasets).shuffle(seed=self.seed) if train_datasets else None
        )
        val_ds = concatenate_datasets(val_datasets) if val_datasets else None

        # Global cap still applied last as a backstop. With per-dataset
        # cap set, this is usually a no-op (per-dataset × num-eval-sets
        # comes in under the global limit).
        if val_ds and self.config.get("max_eval_samples"):
            n_samples = min(len(val_ds), self.config.max_eval_samples)
            val_ds = val_ds.select(range(n_samples))

        return train_ds, val_ds


class DataCollator:
    """Collates audio and text data for training."""

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        projector: Any = None,
        encoder_conv_layers: list | None = None,
        audio_token: str = "<audio>",
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.projector = projector
        self.encoder_conv_layers = encoder_conv_layers or DEFAULT_ENCODER_CONV_LAYERS
        # Must match ASRModel.audio_token -- the collator emits this string and
        # forward() locates the scatter positions by its token id.
        self.audio_token = audio_token
        # Whisper's encoder requires a fixed 3000 mel frames; other encoders
        # (GLM-ASR) accept variable-length input, so only pad to longest.
        self._audio_padding = (
            "max_length"
            if type(feature_extractor).__name__ == "WhisperFeatureExtractor"
            else "longest"
        )
        # 4096 tokens accommodates the long-tail of audio (up to 30s ≈ 187
        # audio tokens) + user prompt + assistant transcript
        # (dense speech can produce 1000-1500 transcript tokens). At 2048 the
        # longest TEDLIUM / Earnings22 samples silently truncated the
        # assistant turn — model trained on partial labels. Qwen3-0.6B
        # supports 32K context so 4096 is well within capacity.
        self.text_collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=4096)

    # Whisper's feature extractor pads/truncates to a fixed 30s window. Audio
    # longer than this is silently truncated while the label is kept whole,
    # training the model to transcribe content it never sees. Drop those rows.
    # Lowered from 30s to 19s to reduce batch-memory pressure: with
    # group_by_length disabled, a single long sample forces the whole batch
    # to its length. 19s sits just under the ~20s production-norm cap for
    # ASR fine-tunes and drops the long-form tail of TEDLIUM / Earnings22 /
    # Peoples / VoxPopuli (roughly 3-8% of rows in those sources). In
    # exchange, mel-spec peak memory drops ~37% vs the 30s default, freeing
    # headroom for auto_find_batch_size (observed batch=70 at max=30s →
    # expected ~100+ at max=19s for the same mix without WHAM).
    _MAX_AUDIO_SECONDS = 19.0
    # Sub-0.8s clips are dominated by boundary-cut segments and isolated
    # backchannels ("yeah", "ok", "umhum") where the audio span and the
    # reference transcript don't actually line up — eval-side analysis on
    # Peoples / CV / Switchboard / AMI showed these as the bulk of >=50%
    # WER samples, with model output reflecting adjacent content rather
    # than the labeled token.
    _MIN_AUDIO_SECONDS = 0.8

    def _extract_audio_arrays(self, features):
        audio_arrays = []
        valid_features = []
        for f in features:
            try:
                audio = f["audio"]["array"]
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                # Drop samples that would poison the gradient or break the
                # encoder: empty / NaN audio, labels that normalize to empty
                # (entire label was an annotation marker like <noise>), audio
                # longer than Whisper's 30s window (label/audio mismatch via
                # silent truncation), or sub-floor backchannels (label/audio
                # don't actually line up — boundary-cut segments dominate the
                # >50% WER tail). One bad sample is enough to NaN the
                # optimizer state. Applied uniformly to train and eval — the
                # filter is correctness, not policy, and the per-dataset eval
                # cap (max_eval_samples_per_dataset) keeps any single dataset
                # cluster from saturating an eval batch.
                if audio.size == 0:
                    continue
                if not np.isfinite(audio).all():
                    continue
                # Drop rows whose entire text was an annotation marker
                # (e.g. Gigaspeech <NOISE>-only segments).
                raw_text = f.get("text") or ""
                if not _normalize_label(raw_text, f.get("_text_case")):
                    continue
                # Drop rows whose label starts or ends with a content-bearing
                # tag (<unk>/<foreign>/<overlap>). Stripping those yields a
                # target missing its first or last spoken word while the audio
                # retains it, which supervises onset/offset truncation — the
                # measured root cause of this recipe's Peoples regression.
                # See _EDGE_CONTENT_TAG_RE for the rates and the evidence.
                if _has_edge_content_tag(raw_text):
                    continue
                duration_s = audio.size / self.sample_rate
                if duration_s > self._MAX_AUDIO_SECONDS:
                    continue
                if duration_s < self._MIN_AUDIO_SECONDS:
                    continue
                audio_arrays.append(audio)
                valid_features.append(f)
            except (KeyError, TypeError, AttributeError, ValueError, OSError) as e:
                # Narrow exception set covers genuine per-row decode/access
                # failures: missing audio dict keys, audio==None, shape
                # mismatch on squeeze, soundfile decode errors. Everything
                # else (LookupError from NLTK punkt_tab, ImportError,
                # RuntimeError from a CUDA path, AssertionError on broken
                # invariants) MUST propagate — silently swallowing them
                # masks real bugs and silently drops samples from training.
                # The prior `except Exception: continue` was hiding an
                # NLTK punkt_tab LookupError that was silently dropping
                # ~48% of training samples (every mono-case row from
                # Gigaspeech / AMI / Peoples / TEDLIUM / Switchboard).
                logger.debug("Skipping row in DataCollator: %s: %s", type(e).__name__, e)
                continue
            finally:
                f["audio"] = None
        if not audio_arrays:
            raise ValueError("No valid audio samples in batch")
        return audio_arrays, valid_features

    def _build_sample(self, feature: dict, num_audio_tokens: int) -> dict:
        """Build a single chat sample."""
        text = _normalize_label(feature.get("text") or "", feature.get("_text_case"))
        # Prompt carries the label convention, so the punctuated and
        # unpunctuated halves of the mix stop competing for the same
        # conditioning. Undeclared sources keep the plain prompt.
        prompt = TRANSCRIBE_PROMPT_PUNCT if feature.get("_text_punct") else TRANSCRIBE_PROMPT
        return self._make_messages(num_audio_tokens, prompt, text)

    def _make_messages(self, num_audio_tokens: int, prompt: str, response: str) -> dict:
        user_content = (self.audio_token * num_audio_tokens) + " " + prompt
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
        return {"messages": messages}

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_arrays, valid_features = self._extract_audio_arrays(features)

        audio_out = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding=self._audio_padding,
            return_attention_mask=True,
            return_tensors="pt",
        )

        mel_lengths = audio_out.attention_mask.sum(dim=-1)
        encoder_lengths = compute_encoder_output_length(mel_lengths, self.encoder_conv_layers)
        token_counts_tensor = self.projector.get_output_length(encoder_lengths).to(torch.long)
        audio_token_counts = token_counts_tensor.tolist()

        text_features = [
            self._build_sample(f, n) for f, n in zip(valid_features, audio_token_counts)
        ]

        batch = self.text_collator(text_features)
        batch["input_features"] = audio_out.input_features
        batch["audio_attention_mask"] = audio_out.attention_mask
        batch["audio_token_counts"] = token_counts_tensor
        return batch


class ASRTrainer(Trainer):
    """Trainer subclass for ASR models."""

    def __init__(
        self,
        *args,
        decoder_learning_rate: float | None = None,
        projector_weight_decay: float | None = None,
        encoder_learning_rate: float | None = None,
        encoder_weight_decay: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.decoder_learning_rate = decoder_learning_rate
        self.projector_weight_decay = projector_weight_decay
        self.encoder_learning_rate = encoder_learning_rate
        self.encoder_weight_decay = encoder_weight_decay

    def create_optimizer(self):
        """Optimizer with separate LR / weight decay per component.

        Mirrors HF Trainer.create_optimizer's decay/no-decay split, but adds a
        second axis: parameters under `audio_tower.` get `encoder_learning_rate`
        / `encoder_weight_decay`; parameters under `language_model.` get
        `decoder_learning_rate`; everything else
        (projector) gets `projector_weight_decay` (when set). Each falls back
        to `args.learning_rate` / `args.weight_decay`.

        The encoder LR override is only meaningful when
        `config.freeze_audio_encoder=False` — frozen encoder parameters have
        `requires_grad=False` and never enter the optimizer regardless.

        The no-decay set is wider than HF's: biases, every `*Norm` gain, and
        all `nn.Embedding` tables (see the inline notes for why each).
        """
        overrides = (
            self.decoder_learning_rate is not None
            or self.projector_weight_decay is not None
            or self.encoder_learning_rate is not None
            or self.encoder_weight_decay is not None
        )
        if self.optimizer is not None or not overrides:
            return super().create_optimizer()

        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        # ALL_LAYERNORM_LAYERS only contains torch.nn.LayerNorm, but every
        # decoder here normalizes with an RMSNorm subclass instead, whose gain
        # weights would silently land in the decay group and be pulled toward
        # zero — destabilizing the residual-stream scale the projector's
        # _NORM_INIT was tuned to. This used to be a two-entry allowlist
        # (Qwen3RMSNorm, LlamaRMSNorm), which quietly excluded every other
        # family — an unfrozen Gemma 4 E2B would have decayed all 247 of its
        # Gemma4RMSNorm gain tensors across 9 distinct sites. Match
        # structurally on the class name so a new decoder is covered on
        # arrival rather than needing an import added here.
        #
        # Substring, not endswith: Qwen3.5's gated-delta-net layers normalize
        # with `Qwen3_5RMSNormGated`, which does NOT end in "Norm" and so
        # escaped an endswith() match entirely — putting all 18
        # `linear_attn.norm.weight` gains (ones-init, so decay pulls them
        # toward zero) into the decay group, the exact failure this block
        # exists to prevent.
        opt_model = self.model
        norm_modules = [type(m) for m in opt_model.modules() if "Norm" in type(m).__name__]
        forbidden = list(ALL_LAYERNORM_LAYERS) + norm_modules
        decay_parameters = set(get_parameter_names(opt_model, forbidden))
        decay_parameters = {n for n in decay_parameters if "bias" not in n}

        # State-space / gated-delta-rule tensors are excluded by convention in
        # every Mamba-family recipe: `A_log` sets each head's memory horizon
        # (decaying it homogenizes the decay spectrum toward A=1) and
        # `conv1d.weight` is the short causal convolution that carries local
        # token order — which matters more than usual here, since Qwen3.5 is
        # 75% NoPE and 18 of 24 layers have no RoPE at all. `dt_bias` and `D`
        # are already caught by the "bias" filter and the norm match, but are
        # listed for completeness.
        ssm_no_decay = ("A_log", "conv1d.weight", "dt_bias", ".D")
        decay_parameters = {
            n for n in decay_parameters if not any(tag in n for tag in ssm_no_decay)
        }

        # Embedding tables are excluded from weight decay on top of the norm
        # exclusion above. Under narrow ASR fine-tuning most of a 248k-row
        # vocab never appears in any batch, so those rows receive no task
        # gradient and WD is the *only* force acting on them: they shrink
        # monotonically toward zero. With tie_word_embeddings=True that same
        # tensor backs lm_head, so the damage lands on the output projection
        # and degrades rare-token prediction at decode time. (This is a
        # fine-tuning-regime argument, not a universal one — under pretraining
        # every token is seen and decaying embeddings is the usual choice.)
        #
        # Matched by tensor identity rather than by name. Tying means
        # lm_head.weight IS embed_tokens.weight, and get_parameter_names walks
        # the module tree so it yields BOTH names, while named_parameters()
        # below deduplicates and yields only whichever the traversal reaches
        # first. A name-based exclusion would therefore work on Qwen (where
        # model.embed_tokens precedes lm_head) and silently fail on any
        # architecture that registers its output head first. Identity holds
        # regardless of which name wins.
        no_decay_param_ids = {
            id(p)
            for module in opt_model.modules()
            if isinstance(module, torch.nn.Embedding)
            for p in module.parameters(recurse=False)
        }

        # Three-way component split. Names are checked against fixed prefixes
        # so the routing matches the freeze flags exactly: `audio_tower.*`,
        # `language_model.*`, and everything else (projector + auxiliary).
        groups: dict[tuple[str, bool], list] = {
            ("encoder", True): [],
            ("encoder", False): [],
            ("decoder", True): [],
            ("decoder", False): [],
            ("other", True): [],
            ("other", False): [],
        }
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("audio_tower."):
                component = "encoder"
            elif name.startswith("language_model."):
                component = "decoder"
            else:
                component = "other"
            decay = name in decay_parameters and id(param) not in no_decay_param_ids
            groups[(component, decay)].append(param)

        base_wd = self.args.weight_decay
        base_lr = self.args.learning_rate
        dec_lr = self.decoder_learning_rate if self.decoder_learning_rate is not None else base_lr
        dec_wd = base_wd
        proj_wd = (
            self.projector_weight_decay if self.projector_weight_decay is not None else base_wd
        )
        enc_lr = self.encoder_learning_rate if self.encoder_learning_rate is not None else base_lr
        enc_wd = self.encoder_weight_decay if self.encoder_weight_decay is not None else base_wd

        # `name` is carried purely so `_trust_ratios` can attribute each group;
        # torch ignores keys it does not recognize in a param group.
        optimizer_grouped_parameters = [
            {
                "name": "projector",
                "params": groups[("other", True)],
                "weight_decay": proj_wd,
                "lr": base_lr,
            },
            {
                "name": "projector",
                "params": groups[("other", False)],
                "weight_decay": 0.0,
                "lr": base_lr,
            },
            {
                "name": "decoder",
                "params": groups[("decoder", True)],
                "weight_decay": dec_wd,
                "lr": dec_lr,
            },
            {
                "name": "decoder",
                "params": groups[("decoder", False)],
                "weight_decay": 0.0,
                "lr": dec_lr,
            },
            {
                "name": "encoder",
                "params": groups[("encoder", True)],
                "weight_decay": enc_wd,
                "lr": enc_lr,
            },
            {
                "name": "encoder",
                "params": groups[("encoder", False)],
                "weight_decay": 0.0,
                "lr": enc_lr,
            },
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _clip_grad_norm(self, model):
        """Clip exactly as the base Trainer does, but log per-group norms first.

        Global-norm clipping scales every parameter by one factor
        `min(1, max_grad_norm / ||g||_global)`. With a fresh projector at
        ||grad|| ~= 9 alongside a decoder at ~1.2, the projector dominates that
        global norm and any clip scales the decoder's update down with it.
        Nothing in this repo logged the two groups separately, so that concern
        could never be checked against a number — HF reports one scalar, which
        by construction cannot separate them.

        Instrumentation only: the clipping below is byte-identical to the base
        implementation, and `_get_grad_norm` still receives the global pre-clip
        norm so `grad_norm` stays comparable across runs. Split the clip only
        if these logs show projector >> decoder persistently past warmup —
        per-group clipping makes the combined update no longer a scalar
        multiple of the true gradient, which every comparable published recipe
        avoids.

        `on_pre_optimizer_step` cannot do this: Trainer fires it after
        `_clip_grad_norm`, so a callback only ever sees post-clip gradients.
        """
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            groups: dict[str, list] = {"projector": [], "decoder": [], "encoder": []}
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                clean = name.removeprefix("_orig_mod.").removeprefix("module.")
                if clean.startswith("audio_tower."):
                    groups["encoder"].append(param.grad)
                elif clean.startswith("language_model."):
                    groups["decoder"].append(param.grad)
                else:
                    groups["projector"].append(param.grad)

            metrics = {}
            for group, grads in groups.items():
                if grads:
                    # Global L2 over the group's gradients (`foreach` fast path).
                    metrics[f"grad_norm/{group}"] = get_total_norm(grads, norm_type=2.0).item()
            if metrics:
                total = math.sqrt(sum(v * v for v in metrics.values()))
                if self.args.max_grad_norm > 0 and total > 0:
                    metrics["grad_norm/clip_factor"] = min(1.0, self.args.max_grad_norm / total)
                metrics.update(self._trust_ratios())
                metrics.update(self._projector_output_rms())
                self.log(metrics)

        return super()._clip_grad_norm(model)

    def _trust_ratios(self) -> dict:
        """Per-group ||update|| / ||w||, the scale-free "is this LR sane" metric.

        Grad-norm ratios cannot answer that question: ||g|| = r * sqrt(N), so
        they are dominated by parameter count (the decoder has 109x the
        projector's params, so its norm is ~10x larger at equal per-parameter
        gradient). AdamW also divides gradient magnitude out entirely -- the
        step is lr * m_hat/(sqrt(v_hat)+eps), i.e. ~lr per parameter regardless
        of ||g||. What actually governs learning is the step relative to the
        weight, and the healthy fine-tuning band is roughly 1e-3 to 1e-2.

        Read from the optimizer's own Adam state, so this reflects the update
        actually applied on the previous step rather than a theoretical bound.
        """
        opt = self.optimizer
        if opt is None or not getattr(opt, "param_groups", None):
            return {}

        totals: dict[str, list[float]] = {}
        for pg in opt.param_groups:
            name = pg.get("name") or pg.get("component") or "other"
            lr, eps = pg.get("lr", 0.0), pg.get("eps", 1e-8)
            b1, b2 = pg.get("betas", (0.9, 0.999))
            # Accumulate on-device and sync once per device, rather than
            # blocking on .item() twice for every parameter in the group.
            upd_terms: dict[torch.device, torch.Tensor] = {}
            w_terms: dict[torch.device, torch.Tensor] = {}
            for p in pg["params"]:
                st = opt.state.get(p)
                if not st or "exp_avg" not in st:
                    continue
                t = int(
                    st.get("step", 0)
                    if not torch.is_tensor(st.get("step", 0))
                    else st["step"].item()
                )
                if t < 1:
                    continue
                m = st["exp_avg"].to(torch.float32) / (1 - b1**t)
                v = st["exp_avg_sq"].to(torch.float32) / (1 - b2**t)
                upd = (lr * m / (v.sqrt() + eps)).pow(2).sum()
                w = torch.linalg.vector_norm(p.detach(), 2, dtype=torch.float32).pow(2)
                dev = p.device
                upd_terms[dev] = upd_terms[dev] + upd if dev in upd_terms else upd
                w_terms[dev] = w_terms[dev] + w if dev in w_terms else w
            upd_sq = sum(x.item() for x in upd_terms.values())
            w_sq = sum(x.item() for x in w_terms.values())
            if w_sq > 0:
                totals.setdefault(name, [0.0, 0.0])
                totals[name][0] += upd_sq
                totals[name][1] += w_sq

        return {
            f"trust_ratio/{name}": math.sqrt(u) / math.sqrt(w)
            for name, (u, w) in totals.items()
            if w > 0
        }

    def _projector_output_rms(self) -> dict:
        """Projector output RMS against the decoder's embedding RMS.

        The whole point of `projector_output_rms` is that audio tokens should
        enter the residual stream at the same magnitude as text tokens. Nothing
        else in the run reports whether that holds once training starts moving
        the weights, and the repo has already measured one small-init attempt
        drifting back up by ~35x. This is the number that says whether it stuck.
        """
        model = self.model
        projector = getattr(model, "projector", None)
        if projector is None or not hasattr(projector, "measure_output_rms"):
            return {}
        try:
            was_training = projector.training
            projector.eval()
            out_rms = projector.measure_output_rms()
            with torch.no_grad():
                emb = model.language_model.get_input_embeddings().weight
                emb_rms = (
                    torch.linalg.vector_norm(emb.detach(), 2, dtype=torch.float32)
                    / math.sqrt(emb.numel())
                ).item()
        except Exception:  # diagnostics must never take the run down
            return {}
        finally:
            projector.train(was_training)

        metrics = {"projector/output_rms": out_rms}
        if emb_rms > 0:
            metrics["projector/output_rms_over_embed"] = out_rms / emb_rms

        # dL/d(log c) for a hypothetical output-scale multiplier c, accumulated
        # by the projector's backward hook. This is the metric that decides
        # whether the observed drift toward ~47x embed RMS is the loss pursuing
        # a larger injection magnitude (persistently negative) or just Adam
        # dragging the scale along as the weights grow (hovering near zero).
        # Drained here so it does not accumulate across logging intervals.
        # Averaged over the micro-batches accumulated since the last drain, so
        # the value does not scale with logging_steps or grad accumulation.
        scale_grad = getattr(projector, "scale_grad", None)
        count = getattr(projector, "scale_grad_count", 0)
        if scale_grad is not None and count:
            metrics["projector/dloss_dlogscale"] = float(scale_grad.detach().item()) / count
            projector.scale_grad = None
            projector.scale_grad_count = 0
        return metrics


class PushToHubCallback(TrainerCallback):
    """Pushes model to Hub on every save."""

    def on_save(self, args, state, control, **kwargs):
        if not (args.push_to_hub and args.hub_model_id):
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        with contextlib.suppress(Exception):
            model.push_to_hub(
                repo_id=args.hub_model_id,
                commit_message=f"Training in progress - step {state.global_step}",
                private=args.hub_private_repo,
            )

        return control


def get_valid_training_args(config: dict) -> dict:
    """Filter config to only valid TrainingArguments fields."""
    valid_fields = {f.name for f in fields(TrainingArguments)}
    return {k: v for k, v in config.items() if k in valid_fields}


def _git_state() -> tuple[str | None, bool]:
    """Return (commit_sha, is_dirty) for the repo containing this script.

    Returns (None, False) if git is unavailable or this isn't a checkout
    (e.g. shipped wheel, pip install). Run from the script's directory so
    Hydra's cwd change doesn't push us outside the repo.
    """
    cwd = Path(__file__).resolve().parent
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False
    return sha, dirty


TRAINING_MODEL_PARAMS = [
    "attn_implementation",
    "use_lora",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "freeze_projector",
    "freeze_language_model",
    "freeze_text_embed_tokens",
    "freeze_audio_encoder",
    "encoder_trainable_top_layers",
]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    push_to_hub = cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id")
    if push_to_hub and not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN environment variable is required when push_to_hub is enabled. "
            "Set it with: export HF_TOKEN=your_token"
        )

    if cfg.training.get("report_to") == "wandb":
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(wandb_config, dict)
        git_commit, git_dirty = _git_state()
        if git_commit:
            # Surface the commit in the run config so it's queryable/filterable
            # in the wandb UI alongside the run's hyperparameters. Wandb does
            # capture git metadata on its own, but it lives in a separate panel
            # and can't be used to group/filter runs.
            wandb_config["git_commit"] = git_commit
            wandb_config["git_dirty"] = git_dirty
        wandb.init(
            project=cfg.training.get("wandb_project", "tiny-audio"),
            config=wandb_config,
        )
        if git_commit:
            wandb.run.summary["git_commit"] = git_commit
            wandb.run.summary["git_dirty"] = git_dirty

    # Patch the decoder's transformers module with liger fused kernels before
    # the LM class is instantiated. The big win is fused linear cross-entropy:
    # instead of materializing the (B, T, V) fp32 log-softmax tensor that HF's
    # standard CE / LabelSmoother path requires (~15GB at B=50, V=151k on
    # Qwen3-0.6B), liger fuses lm_head @ hidden_states + softmax + CE into a
    # single kernel with peak memory O(B·T·D). Label smoothing flows through
    # this kernel via the loss_function's **kwargs path (see ASRModel.forward)
    # — so set HF Trainer's label_smoothing_factor=0 in configs to bypass the
    # LabelSmoother and rely on model.config.label_smoothing instead.
    #
    # The patcher is per-architecture, so it must track text_model_id. Getting
    # this wrong is not a crash but an OOM: Gemma 4's vocab is 262,144, so an
    # unfused (B, T, V) logits tensor is ~17GB at B=32/T=512 before the
    # log_softmax copy. First match wins, so longer keys are listed first.
    if cfg.training.get("use_liger", True):
        liger_patchers = (
            ("gemma-4", "apply_liger_kernel_to_gemma4"),
            ("qwen3.5", "apply_liger_kernel_to_qwen3_5"),
            ("qwen3", "apply_liger_kernel_to_qwen3"),
        )
        text_model_id = str(cfg.model.get("text_model_id", "")).lower()
        patcher_name = next((fn for key, fn in liger_patchers if key in text_model_id), None)
        if patcher_name is None:
            logger.warning(
                "No liger patcher mapped for text_model_id=%r — training with stock "
                "kernels and unfused cross-entropy. Add an entry to liger_patchers "
                "if this decoder has liger support.",
                cfg.model.get("text_model_id"),
            )
        else:
            try:
                import liger_kernel.transformers as liger

                getattr(liger, patcher_name)()
                logger.info("Applied liger kernels via %s()", patcher_name)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    "liger-kernel unavailable or missing %s (%s) — falling back to "
                    "stock kernels. Install with `poetry install` on Linux and pin a "
                    "version that exports it to enable fused linear CE.",
                    patcher_name,
                    e,
                )

    model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_config_dict, dict), "model config must be a dict"
    for param in TRAINING_MODEL_PARAMS:
        val = cfg.training.get(param)
        if val is None:
            continue
        # Strip OmegaConf wrappers so list/dict params (e.g. lora_target_modules)
        # land in ASRConfig as plain Python types — otherwise config.save_pretrained
        # hits a TypeError when json.dumps walks a ListConfig at checkpoint time.
        if OmegaConf.is_config(val):
            val = OmegaConf.to_container(val, resolve=True)
        model_config_dict[param] = val
    asr_config = ASRConfig(**model_config_dict)

    model = ASRModel(asr_config)

    # Disable the KV cache for training on the decoder's own config, NOT on the
    # ASRConfig. ASRConfig.use_cache is an inference setting: __init__ copies it
    # into generation_config, and save_pretrained serializes it, so writing
    # False here baked `use_cache: false` into every checkpoint and every model
    # pushed to the Hub. Generation then ran without a cache, re-encoding the
    # whole prompt at each step -- quadratic decode on the reload path.
    model.language_model.config.use_cache = False

    if hub_model_id := cfg.training.get("hub_model_id"):
        model.config.pretrained_model_path = hub_model_id

    # Workaround: TRL's DataCollatorForChatML doesn't pass enable_thinking=False to Qwen3.
    # See https://github.com/huggingface/trl/issues/3387
    if model.tokenizer.chat_template and "enable_thinking" in model.tokenizer.chat_template:
        model.tokenizer.chat_template = model.tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "true",
        )

    train_dataset, val_dataset = DatasetLoader(cfg).load()

    data_collator = DataCollator(
        tokenizer=model.tokenizer,
        feature_extractor=model.feature_extractor,
        sample_rate=cfg.data.sample_rate,
        projector=model.projector,
        encoder_conv_layers=model.config.encoder_conv_layers,
        audio_token=model.audio_token,
    )

    callbacks = []
    if push_to_hub:
        callbacks.append(PushToHubCallback())

    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_config, dict)
    decoder_learning_rate = training_config.pop("decoder_learning_rate", None)
    projector_weight_decay = training_config.pop("projector_weight_decay", None)
    encoder_learning_rate = training_config.pop("encoder_learning_rate", None)
    encoder_weight_decay = training_config.pop("encoder_weight_decay", None)
    # Dynamo flags set unconditionally — applies whether the user enables
    # torch.compile via TrainingArguments or whether some upstream dep
    # (liger / transformers) invokes dynamo internally. cache_size_limit
    # defaults to 8, which audio batches blow past quickly because
    # group_by_length=false + variable seq lengths produce dozens of
    # distinct shapes; without bumping it dynamo gives up and falls back
    # to eager mid-run (you see "torch._dynamo hit config.recompile_limit"
    # warnings). capture_scalar_outputs lets dynamo capture .item() /
    # scalar-tensor outputs into the graph instead of graph-breaking on
    # the first scalar-producing op (e.g. token_counts.max().item() in
    # _gather_audio_embeds).
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.capture_scalar_outputs = True
    trainer = ASRTrainer(
        model=model,
        args=TrainingArguments(**get_valid_training_args(training_config)),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=model.tokenizer,
        callbacks=callbacks,
        decoder_learning_rate=decoder_learning_rate,
        projector_weight_decay=projector_weight_decay,
        encoder_learning_rate=encoder_learning_rate,
        encoder_weight_decay=encoder_weight_decay,
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    trainer.save_model()

    if push_to_hub:
        trainer.model.push_to_hub(
            cfg.training.hub_model_id,
            commit_message="Training complete - final model",
            private=cfg.training.get("hub_private_repo", False),
        )


if __name__ == "__main__":
    main()
