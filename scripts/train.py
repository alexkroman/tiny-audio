#!/usr/bin/env python3
"""Training script for ASR models using Hydra configuration."""

import contextlib
import logging
import os
import subprocess
import warnings
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import hydra
import numpy as np
import torch
import wandb
from datasets import (
    Audio,
    ClassLabel,
    Dataset,
    concatenate_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from trl.import_utils import TRLExperimentalWarning

from scripts.labels import (
    TEXT_CASE_CASED,
    TEXT_CASE_MONO,
    _has_edge_content_tag,
    _normalize_label,
)
from scripts.train_config import register_configs
from tiny_audio.asr_config import (
    DEFAULT_ENCODER_CONV_LAYERS,
    ASRConfig,
    compute_encoder_output_length,
)
from tiny_audio.asr_modeling import ASRModel

# trl.experimental warns (TRLExperimentalWarning) the first time it is
# imported; DataCollatorForChatML is the only thing used from it.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", TRLExperimentalWarning)
    from trl.experimental.utils import DataCollatorForChatML

# liger is a linux-only optional dependency (see pyproject.toml); without it
# training falls back to stock kernels and unfused cross-entropy.
try:
    from liger_kernel import transformers as liger_transformers
except ImportError as exc:
    LIGER_TRANSFORMERS: ModuleType | None = None
    _LIGER_IMPORT_ERROR: ImportError | None = exc
else:
    LIGER_TRANSFORMERS = liger_transformers
    _LIGER_IMPORT_ERROR = None

for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub.file_download"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Register the `base_config` structured-config schema that configs/config.yaml
# lists first in its defaults (scripts/train_config.py). Must happen before
# @hydra.main composes the config.
register_configs()

TRANSCRIBE_PROMPT = "Transcribe the speech to text"
# Used for sources whose transcripts natively carry punctuation, selected per
# row via the `text_punct` dataset field. Granite Speech 4.1 documents exactly
# this mechanism -- its model card says punctuation and truecasing are chosen
# "with a simple prompt change", and its usage example is literally
# "<|audio|>transcribe the speech with proper punctuation and capitalization."
# Qwen3-ASR does the equivalent through a system turn plus an assistant prefill.
#
# Without the split, the unpunctuated share of multiasr trains the model to
# SUPPRESS punctuation under the same prompt the punctuated share uses to
# produce it. Identical conditioning, contradictory targets: the model can
# only learn a hedge, and every dropped mark scores as an error against
# punctuated references.
#
# Share re-measured 2026-09-20: 12.1% plain-prompt (TEDLIUM ~194,900 + AMI
# 147,504) against 87.9% punct-prompt, NOT the ~25%/~75% this comment used to
# claim. Two of the four sources it named -- Peoples and Switchboard -- have
# left the mix entirely, and the TEDLIUM leading-<unk> filter shrank a third.
# The split still earns its keep at 12.1%, but the real format heterogeneity
# now lives INSIDE the punct-prompt majority: 39.5% of SPGISpeech rows start
# lowercase and 41.5% end without terminal punctuation (mid-stream 5-15s
# window cuts), i.e. ~6% of the whole mix teaching "begin mid-sentence, no
# final period" under the punctuation prompt. That is invisible to WER -- the
# eval normalizer strips case and punctuation from both sides -- and shows up
# only in orthographic_wer, the same blind spot that hid the %-stripping bug.
# Measure before acting.
TRANSCRIBE_PROMPT_PUNCT = "Transcribe the speech with proper punctuation and capitalization"


def _resolve_transcribe_prompt(configured: str | None, datasets: list) -> str | None:
    """Pick the inference prompt a checkpoint is saved with.

    `transcribe_prompt` never reaches training -- `_build_sample` routes each
    row by `text_punct` -- so it only decides which trained convention
    inference asks for. Left unset, ASRModel falls back to the plain prompt,
    which is the minority unpunctuated bucket (TEDLIUM/AMI/Peoples) whenever
    any source is punctuated. frozen-2 shipped that way: 0% terminal
    punctuation, "Laughter" prefixed to 15% of CommonVoice rows, CV 7.38 ->
    9.50 WER on identical weights. So an unset prompt resolves to the
    punctuated one when a punctuated source trains, and is written into the
    config so the checkpoint decodes under it.
    """
    if configured is not None:
        return configured
    if any(d.get("text_punct") and d.get("train_splits", ["train"]) for d in datasets):
        logger.info(
            "transcribe_prompt unset; resolving to %r (a text_punct source trains)",
            TRANSCRIBE_PROMPT_PUNCT,
        )
        return TRANSCRIBE_PROMPT_PUNCT
    return None


# `Dataset.add_column` annotates `new_fingerprint: str` as a required argument,
# but its @fingerprint_transform wrapper computes the fingerprint whenever the
# caller leaves it out, which is how it is meant to be called. This is that
# runtime signature, called exactly as `ds.add_column(name, column)` would be.
_add_column = cast(Callable[[Dataset, str, list[Any]], Dataset], Dataset.add_column)


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
        self.num_train_epochs = config.training.get("num_train_epochs", 1)
        # See _expand_epochs / load() for what this does and why it exists.
        self.epoch_expansion = int(self.config.get("epoch_expansion", 1) or 1)

    def _prepare_split(self, dataset_cfg: DictConfig, split: str) -> Dataset:
        dataset_path = dataset_cfg.get("path")
        if not dataset_path:
            msg = "Dataset path is required"
            raise ValueError(msg)

        ds = load_dataset(
            dataset_path,
            name=dataset_cfg.get("name"),
            split=split,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc,
            trust_remote_code=True,
        )
        # A concrete `split` without streaming always yields a single Dataset
        # (not a DatasetDict / IterableDataset).
        assert isinstance(ds, Dataset)

        # Constant per-source provenance columns. These MUST be added here,
        # before any filter() below, and not down next to the other column
        # surgery. Dataset.add_column calls flatten_indices() whenever the
        # dataset carries an indices mapping, and every filter() attaches one --
        # so adding them post-filter rewrites the entire source, audio bytes
        # included, through a single-process "Flattening the indices" map.
        # Pre-filter the table has no indices mapping, so add_column is a
        # zero-copy horizontal concat and the filters below simply carry the
        # new columns along. Same rows, same values, nothing written.
        # text_case: declares whether this source's transcripts already carry
        # case ("cased") or arrive mono-case and need recasing ("mono").
        # Stored per row so _normalize_label does not have to re-derive a
        # source property from a single row's characters.
        # Omit it to keep the legacy per-row heuristic.
        text_case = dataset_cfg.get("text_case")
        if text_case is not None:
            if text_case not in (TEXT_CASE_MONO, TEXT_CASE_CASED):
                msg = (
                    f"text_case must be {TEXT_CASE_MONO!r} or {TEXT_CASE_CASED!r}, "
                    f"got {text_case!r} for {dataset_path}"
                )
                raise ValueError(msg)
            ds = _add_column(ds, "_text_case", [text_case] * len(ds))

        # text_punct: declares whether this source's transcripts carry
        # punctuation. Deliberately separate from text_case -- they are not the
        # same axis, and conflating them gets Gigaspeech wrong, which is
        # ALL-CAPS (text_case: mono) yet natively punctuated. Omit it and the
        # row gets the plain prompt, i.e. today's behaviour.
        text_punct = dataset_cfg.get("text_punct")
        if text_punct is not None:
            if not isinstance(text_punct, bool):
                msg = f"text_punct must be a bool, got {text_punct!r} for {dataset_path}"
                raise ValueError(msg)
            ds = _add_column(ds, "_text_punct", [text_punct] * len(ds))

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
        #   exclude_where: {column: audio_duration, above: 19.0}
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
        if exclude_where is not None:
            column = exclude_where.get("column")
            names = list(exclude_where.get("values") or [])
            # Numeric bounds, added 2026-09-20 for LibriHeavy. Semantics follow
            # the key's name: this EXCLUDES rows, so `above: 19.0` drops rows
            # whose value exceeds 19.0 (it is not a keep-ceiling).
            above = exclude_where.get("above")
            below = exclude_where.get("below")
            if not column or (not names and above is None and below is None):
                msg = (
                    f"exclude_where needs 'column' plus at least one of "
                    f"'values' / 'above' / 'below', got {exclude_where!r} "
                    f"for {dataset_path}"
                )
                raise ValueError(msg)
            if column not in ds.column_names:
                # Fail loudly: a silently-ignored filter would train on the
                # rows you believe you excluded, and the mix table would lie.
                msg = (
                    f"exclude_where column {column!r} not in {dataset_path} "
                    f"(available: {sorted(ds.column_names)})"
                )
                raise ValueError(msg)
            # Gigaspeech's `source` is a ClassLabel, so its rows hold ints
            # (0=audiobook, 1=podcast, 2=youtube), NOT the label strings the
            # datasets-server `statistics` endpoint renders. Comparing rows
            # against the human-readable names matches nothing and silently
            # dropped 0 of 910,140 rows. Resolve names -> ids so the config
            # stays readable, and reject a name the column does not define.
            feature = (ds.features or {}).get(column)
            wanted: set | None = None
            if names:
                if isinstance(feature, ClassLabel):
                    # Report every bad name at once rather than dying on the first.
                    unknown = sorted(n for n in names if n not in feature.names)
                    if unknown:
                        msg = (
                            f"exclude_where value {unknown} not a label of {column!r} in "
                            f"{dataset_path} (defined: {feature.names})"
                        )
                        raise ValueError(msg)
                    wanted = {feature.str2int(n) for n in names}
                else:
                    wanted = set(names)

            def _keep(v, _wanted=wanted, _above=above, _below=below):
                excluded = (
                    (_wanted is not None and v in _wanted)
                    or (v is not None and _above is not None and v > _above)
                    or (v is not None and _below is not None and v < _below)
                )
                return not excluded

            before = len(ds)
            # `input_columns` keeps this from materialising the audio column --
            # it matters for a duration filter over ~1.1M rows, which would
            # otherwise decode every clip to answer a float comparison.
            ds = ds.filter(
                _keep,
                num_proc=self.num_proc,
                input_columns=column,
            )
            dropped = before - len(ds)
            logger.info(
                "exclude_where on %s: dropped %d/%d rows (%s in %s, above=%s, below=%s)",
                dataset_path,
                dropped,
                before,
                column,
                names or "-",
                above,
                below,
            )
            # A filter that matches nothing is a configuration bug, not a
            # legitimate no-op: you asked to exclude something that is not
            # there. Failing here costs seconds; not failing means training a
            # full run on the mix you thought you had excluded, and only
            # finding out from the eval.
            if dropped == 0:
                msg = (
                    f"exclude_where on {dataset_path} matched 0 of {before} rows "
                    f"({column}: values={sorted(names)} above={above} below={below}). "
                    f"Check the column's value type and spelling -- feature is {feature!r}."
                )
                raise ValueError(msg)

        col_map = {
            "text": dataset_cfg.get("text_column", "text"),
            "audio": dataset_cfg.get("audio_column", "audio"),
        }
        for target, source in col_map.items():
            if source != target and source in ds.column_names:
                if target in ds.column_names:
                    ds = ds.remove_columns([target])
                ds = ds.rename_column(source, target)

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
        # Upsampling repeats rows verbatim, so the extra "samples" carry no
        # new signal. That is intended for small sources, but it is also what
        # happens when a filter (e.g. exclude_where) cuts a large source below
        # its cap -- silently, since the row count still reads 600K. Say so.
        logger.warning(
            "target_samples %d exceeds the %d available rows; repeat-padding "
            "%.2fx (no new signal in the duplicated rows)",
            target,
            current,
            target / current,
        )
        repeats = (target // current) + 1
        indices = list(range(current)) * repeats
        return ds.select(indices[:target])

    @staticmethod
    def _expand_epochs(ds: Dataset, times: int) -> Dataset:
        """Repeat an uncapped source `times` over, verbatim.

        Only for sources already at their natural size: there are no unused
        rows to draw, so repeating is exactly what a second Trainer epoch
        would have done. Capped sources go through _resample_to_target with a
        multiplied target instead, which spends the multiplier on FRESH rows
        first and only repeat-pads what the pool cannot cover.

        Built with concatenate_datasets rather than the equivalent
        `ds.select(list(range(len(ds))) * times)`. The two yield the identical
        row sequence; they do not carry the identical disk cost. `select`
        attaches an indices mapping, and the concatenate_datasets in load()
        flattens any dataset carrying one -- materializing a full verbatim
        copy, embedded audio bytes and all. At epoch_expansion=2 that wrote
        roughly 620 GB of duplicate audio for the uncapped sources and is what
        exhausted the network volume mid-run. Concatenating builds a
        ConcatenationTable over the same memory-mapped blocks instead: same
        rows, same order, nothing written. (A source that already has an
        indices mapping from a _prepare_split filter still flattens once here,
        but once rather than `times` over.)
        """
        if times <= 1:
            return ds
        return concatenate_datasets([ds] * times)

    def load(self) -> tuple[Dataset | None, Dataset | None]:
        train_datasets, val_datasets = [], []

        # epoch_expansion: build ONE physical epoch that is worth N logical
        # ones, so that capped sources contribute fresh rows instead of
        # replaying the same subset.
        #
        # The problem it fixes: _resample_to_target runs once, here in load(),
        # so `num_train_epochs: 2` iterates the IDENTICAL 600K subset twice
        # while the rest of the pool is never touched. Measured on the current
        # mix, that leaves ~285K eligible LibriHeavy rows and ~338K eligible
        # CommonVoice rows unseen while their siblings are shown twice.
        #
        # Why this shape rather than a per-epoch sampler: the uncapped sources
        # (SPGI, TEDLIUM, VoxPopuli, AMI) are already at natural size, so the
        # only place extra unique data can come from is the capped ones --
        # and raising their caps alone would change per-step mix share, which
        # is the one thing the caps exist to control. Multiplying EVERY
        # source by N holds per-step share exactly where it was while letting
        # the capped sources spend their larger budget on unseen rows. What
        # the model sees per step is unchanged; what it sees over the run is
        # strictly more diverse.
        #
        # Mutually exclusive with num_train_epochs > 1 -- the two multiply,
        # and silently training 4 epochs' worth would be worse than either.
        expansion = self.epoch_expansion
        if expansion > 1 and self.num_train_epochs > 1:
            msg = (
                f"epoch_expansion={expansion} and num_train_epochs="
                f"{self.num_train_epochs} would compound to "
                f"{expansion * self.num_train_epochs} epochs of exposure. "
                f"epoch_expansion already folds the repeats into one physical "
                f"epoch, so set num_train_epochs: 1 when using it."
            )
            raise ValueError(msg)
        if expansion > 1:
            logger.info(
                "epoch_expansion=%d: building one physical epoch worth %d "
                "logical epochs; capped sources draw fresh rows up to their "
                "pool before any repeat-padding",
                expansion,
                expansion,
            )

        for d_cfg in tqdm(self.config.datasets, desc="Loading datasets"):
            train_splits = d_cfg.get("train_splits", ["train"])
            val_splits = d_cfg.get("eval_splits", ["validation"])
            target_samples = d_cfg.get("target_samples")

            for train_split in train_splits:
                ds = self._prepare_split(d_cfg, train_split)
                if target_samples:
                    # Multiply the cap, not the dataset: _resample_to_target
                    # shuffles then takes the first target*N, so the extra
                    # budget is spent on unseen rows first and only
                    # repeat-pads (with a warning) once the pool runs out.
                    ds = self._resample_to_target(ds, target_samples * expansion)
                else:
                    ds = self._expand_epochs(ds, expansion)
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
        # cap set, this is usually a no-op (per-dataset x num-eval-sets
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
    # to its length. 19s sits just under the ~20s production-norm cap
    # for ASR fine-tunes.
    #
    # Per-source loss re-measured 2026-09-20 -- the old "TEDLIUM / Earnings22
    # / Peoples / VoxPopuli, roughly 3-8%" was wrong in every particular:
    # LibriHeavy 19.6% (the source it hits hardest was not even named, and is
    # now pre-filtered at prep time via exclude_where so the 600K cap
    # delivers a true 600K), VoxPopuli 13.7%, TEDLIUM 0.07%, SPGISpeech 0.0%;
    # Earnings22 and Peoples have left the mix. Because these drops run at
    # COLLATE time -- after target_samples -- a capped source silently
    # delivers fewer rows than its cap, which is how the mix table came to
    # overstate the corpus by 235K rows. In
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
                # See scripts/labels.py _EDGE_CONTENT_TAG_RE for the rates and the evidence.
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
            msg = "No valid audio samples in batch"
            raise ValueError(msg)
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
            self._build_sample(f, n)
            for f, n in zip(valid_features, audio_token_counts, strict=True)
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

    def create_optimizer(self, model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
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
            return super().create_optimizer(model)

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
        # Same model resolution as Trainer.create_optimizer, which train() calls
        # with the accelerator-prepared model when optimizer creation is delayed.
        opt_model = self.model if model is None else model
        if opt_model is None:
            msg = "ASRTrainer.create_optimizer needs a model"
            raise ValueError(msg)
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

        optimizer_grouped_parameters = [
            {
                "params": groups[("other", True)],
                "weight_decay": proj_wd,
                "lr": base_lr,
            },
            {
                "params": groups[("other", False)],
                "weight_decay": 0.0,
                "lr": base_lr,
            },
            {
                "params": groups[("decoder", True)],
                "weight_decay": dec_wd,
                "lr": dec_lr,
            },
            {
                "params": groups[("decoder", False)],
                "weight_decay": 0.0,
                "lr": dec_lr,
            },
            {
                "params": groups[("encoder", True)],
                "weight_decay": enc_wd,
                "lr": enc_lr,
            },
            {
                "params": groups[("encoder", False)],
                "weight_decay": 0.0,
                "lr": enc_lr,
            },
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args, opt_model if isinstance(opt_model, PreTrainedModel) else None
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


class PushToHubCallback(TrainerCallback):
    """Pushes model to Hub on every save."""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # Returning None leaves `control` as is: CallbackHandler only replaces
        # it when a callback returns a new one.
        del control
        if not (args.push_to_hub and args.hub_model_id):
            return

        model = kwargs.get("model")
        if model is None:
            return

        with contextlib.suppress(Exception):
            model.push_to_hub(
                repo_id=args.hub_model_id,
                commit_message=f"Training in progress - step {state.global_step}",
                private=args.hub_private_repo,
            )


def get_valid_training_args(config: dict) -> dict:
    """Filter config to only valid, set TrainingArguments fields.

    None means "unset" in the structured-config schema (scripts/train_config.py),
    so those keys are dropped and TrainingArguments applies its own default.
    """
    valid_fields = {f.name for f in fields(TrainingArguments)}
    return {k: v for k, v in config.items() if k in valid_fields and v is not None}


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
    "lora_rank_pattern",
    "lora_alpha_pattern",
    "freeze_projector",
    "freeze_language_model",
    "freeze_text_embed_tokens",
    "freeze_audio_encoder",
    "encoder_trainable_top_layers",
    "encoder_trainable_post_projections",
]


def _require_fused_cross_entropy(model, cfg) -> None:
    """Fail at startup when fused CE is unavailable on a GPU run.

    Without liger's fused linear cross-entropy, every labelled forward
    materializes a (batch, seq, vocab) logits tensor plus its fp32 upcast and
    its gradient. On Qwen3.5-2B (vocab 248,320) at batch 48 / seq 330 that is
    ~25 GiB -- most of an 80 GB card, spent on a tensor nothing in this repo
    reads. It is also what killed several granite_qwen_lora launches with an
    OOM in backward.

    This used to be two `logger.warning` calls (one here, one in
    `ASRModel.__init__`). Both fired correctly and both were missed, because a
    warning scrolls past during model load and the run then trains normally --
    just 25 GiB heavier and at permanent OOM risk. The failure has no symptom
    until the batch that does not fit.

    Raised only where the memory actually costs something: CUDA, a large
    vocabulary, and liger requested. CPU/MPS smoke runs on a mac cannot have
    liger at all (it is a linux-only dependency) and must stay runnable.
    Set `training.allow_unfused_ce: true` to proceed anyway.
    """
    if getattr(model, "_lm_accepts_skip_logits", False):
        return
    if not cfg.training.get("use_liger", True):
        return  # deliberately off; the planner already accounts for it
    if cfg.training.get("allow_unfused_ce", False):
        logger.warning(
            "Fused cross-entropy is NOT active and allow_unfused_ce is set -- "
            "continuing with the unfused path. Expect materially higher VRAM."
        )
        return
    if not torch.cuda.is_available():
        return  # mac / CPU smoke runs: liger is linux-only, nothing to fix

    vocab = getattr(model.language_model.config, "vocab_size", 0) or 0
    if vocab < 100_000:
        return  # small vocab: the logits tensor is not the dominant term

    batch = cfg.training.get("per_device_train_batch_size", 1)
    est_gib = batch * 330 * vocab * 4 * 2 / 2**30
    msg = (
        f"liger's fused linear cross-entropy is NOT active for "
        f"{type(model.language_model).__name__} (vocab {vocab:,}). Every "
        f"training step would materialize a (batch, seq, {vocab:,}) logits "
        f"tensor -- roughly {est_gib:.0f} GiB at batch {batch}, seq 330 -- and "
        f"risk an OOM in backward.\n"
        f"Fix: `poetry install` on this machine (liger-kernel >=0.8.0 is "
        f"required for qwen3.5 and is pinned in pyproject.toml), then check "
        f"the log for 'Applied liger kernels via ...'.\n"
        f"Override with `training.allow_unfused_ce=true` if this is "
        f"deliberate."
    )
    raise RuntimeError(msg)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    push_to_hub = cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id")
    if push_to_hub and not os.environ.get("HF_TOKEN"):
        msg = (
            "HF_TOKEN environment variable is required when push_to_hub is enabled. "
            "Set it with: export HF_TOKEN=your_token"
        )
        raise ValueError(msg)

    if cfg.training.get("report_to") == "wandb":
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_container, dict)
        # The root config's keys are the group names (model/data/training).
        wandb_config = {str(k): v for k, v in cfg_container.items()}
        git_commit, git_dirty = _git_state()
        if git_commit:
            # Surface the commit in the run config so it's queryable/filterable
            # in the wandb UI alongside the run's hyperparameters. Wandb does
            # capture git metadata on its own, but it lives in a separate panel
            # and can't be used to group/filter runs.
            wandb_config["git_commit"] = git_commit
            wandb_config["git_dirty"] = git_dirty
        run = wandb.init(
            project=cfg.training.get("wandb_project", "tiny-audio"),
            config=wandb_config,
        )
        if git_commit:
            run.summary["git_commit"] = git_commit
            run.summary["git_dirty"] = git_dirty

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
                if LIGER_TRANSFORMERS is None:
                    assert _LIGER_IMPORT_ERROR is not None
                    raise _LIGER_IMPORT_ERROR
                getattr(LIGER_TRANSFORMERS, patcher_name)()
                logger.info("Applied liger kernels via %s()", patcher_name)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    "liger-kernel unavailable or missing %s (%s) — falling back to "
                    "stock kernels. Install with `poetry install` on Linux and pin a "
                    "version that exports it to enable fused linear CE.",
                    patcher_name,
                    e,
                )

    model_container = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_container, dict), "model config must be a dict"
    # Keys are ModelConfig field names (scripts/train_config.py), i.e. strings.
    model_config_dict = {str(k): v for k, v in model_container.items()}
    for param in TRAINING_MODEL_PARAMS:
        val = cfg.training.get(param)
        if val is None:
            continue
        # Warn when both blocks set the same key. `training:` silently wins, so
        # a `model:`-block value is dead config -- which is exactly how
        # granite_qwen.yaml's `attn_implementation: sdpa` was overridden by
        # production.yaml's flash_attention_2 for a full 33k-step run, with 15
        # lines of FLOP arithmetic above it describing a setting the run never
        # used. Loud rather than silent; the merge itself is unchanged.
        model_val = model_config_dict.get(param)
        if model_val is not None and model_val != val:
            logger.warning(
                "Config conflict on %r: model=%r is overridden by training=%r. "
                "`training:` wins the TRAINING_MODEL_PARAMS merge -- set the "
                "value you want under `training:`, and keep `model:` in sync or "
                "remove it.",
                param,
                model_val,
                val,
            )
        # Strip OmegaConf wrappers so list/dict params (e.g. lora_target_modules)
        # land in ASRConfig as plain Python types — otherwise config.save_pretrained
        # hits a TypeError when json.dumps walks a ListConfig at checkpoint time.
        if OmegaConf.is_config(val):
            val = OmegaConf.to_container(val, resolve=True)
        model_config_dict[param] = val
    model_config_dict["transcribe_prompt"] = _resolve_transcribe_prompt(
        model_config_dict.get("transcribe_prompt"), cfg.data.get("datasets") or []
    )
    # None marks a schema field the configs left unset; drop it so ASRConfig's
    # own default applies (see ModelConfig in scripts/train_config.py).
    asr_config = ASRConfig(**{k: v for k, v in model_config_dict.items() if v is not None})

    model = ASRModel(asr_config)

    _require_fused_cross_entropy(model, cfg)

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
    # `_internal_call=True` suppresses Trainer's own hub push, which
    # `upload_folder`s the entire output_dir. The explicit push below is the
    # one that matters: it runs through `ASRModel.push_to_hub`, which sets
    # `base_model_name_or_path` in adapter_config.json so the HF pipeline can
    # load the repo. Letting both fire uploads the same multi-GB checkpoint
    # twice. Only suppress it when we are the ones pushing -- a config with
    # `push_to_hub: true` but no `hub_model_id` still gets Trainer's push to
    # its output_dir-derived repo.
    trainer.save_model(_internal_call=bool(push_to_hub))

    if push_to_hub:
        # `model` is the object Trainer holds as `trainer.model` (no
        # model_init, no FSDP re-wrapping here), typed as the ASRModel it is.
        model.push_to_hub(
            cfg.training.hub_model_id,
            commit_message="Training complete - final model",
            private=cfg.training.get("hub_private_repo", False),
        )


if __name__ == "__main__":
    main()
