#!/usr/bin/env python3
"""Training script for ASR models using Hydra configuration."""

# ruff: noqa: E402
# The trl env-var must be set, and the noisy-logger silencer must run,
# *before* their respective modules are imported below — so non-import
# statements precede some imports here. Suppress E402 file-wide rather
# than per-line.

import contextlib
import logging
import os
import random
import re
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub.file_download"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import hydra
import numpy as np
import torch
import wandb
from datasets import (
    Audio,
    Dataset,
    Value,
    concatenate_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import (
    EarlyStoppingCallback,
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
from tiny_audio.augmentation import NoiseAugmentation, RIRAugmentation

TRANSCRIBE_PROMPTS = ["Transcribe the speech to text"]
DESCRIBE_PROMPTS = ["Describe all the information you can hear"]

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
_CORPUS_MARKER_RE = re.compile(
    r"\s*<("
    r"sil|music|noise|other|unk|"
    r"overlap|laugh|dtmf|foreign|no-speech|lipsmack|"
    r"clear_throat|inaudible|crosstalk"
    r")>",
    re.IGNORECASE,
)
# TEDLIUM occasionally inlines editorial commentary in square brackets
# ([ medicine ], [ multi-word stage direction ]) — ~0.25% of train rows;
# zero in dev/test. Strip the entire bracketed block, including any
# preceding whitespace, to avoid leaving a double-space behind.
_TEDLIUM_BRACKET_RE = re.compile(r"\s*\[[^\]]*\]")
_WHITESPACE_RE = re.compile(r"\s+")

# Unicode cleanup: ftfy fixes mojibake (â€™ → '), unescapes HTML entities
# (&amp; → &), and folds smart quotes (' " → ' "); NFKC further normalizes
# composed/decomposed forms (café vs cafe + ◌́) and width variants
# (full-width Latin → half-width). Applied first in _normalize_label so
# downstream regexes see canonical ASCII-leaning text.
import ftfy  # noqa: E402  pyright: ignore[reportMissingImports]

# Truecase: NLTK-backed statistical recasing for transcripts that arrive
# in mono-case form (all-upper or zero-caps). LOCAL_RANK=0 guard mirrors
# Ultravox — avoids multiple workers racing on the punkt download.
import truecase  # noqa: E402  pyright: ignore[reportMissingImports]

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    try:
        truecase.get_true_case("test")
    except LookupError:
        import nltk  # noqa: E402  pyright: ignore[reportMissingImports]

        # NLTK 3.9+ requires `punkt_tab`; older NLTKs use `punkt`. Download
        # both so this works on either base image. Quiet=True suppresses
        # progress bars; the fetch is ~13 MB and usually completes in
        # seconds.
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)


def _needs_truecase(text: str) -> bool:
    """Apply truecase only to mono-case text. Already-cased sources
    (LibriHeavy text_original, CV, VoxPopuli raw_text, SPGISpeech) carry
    proper-noun casing that the statistical truecaser would damage
    (e.g. "McClarnon" -> "Mcclarnon"). Heuristic: text with any internal
    capitalization beyond what truecase would produce is already cased.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 5:
        # Too short to recase meaningfully ("yeah", "OH"). Leave alone.
        return False
    upper_count = sum(c.isupper() for c in letters)
    upper_frac = upper_count / len(letters)
    if upper_frac > 0.9:
        return True  # ALL-CAPS source (Gigaspeech post-restoration, AMI)
    # zero-cap (TEDLIUM, Peoples, Switchboard) → truecase;
    # otherwise already cased (LibriHeavy, CV, SPGI, VoxPopuli) → skip.
    return upper_count == 0


def _normalize_label(raw_text: str) -> str:
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
    4. Canonicalize percent — mirrors scripts/analysis.py:normalize_text
       so the train-time label matches eval-time WER canonicalization.
    5. Collapse whitespace.
    6. Apply truecase only to mono-case text (see _needs_truecase). This
       lifts ALL-CAPS sources (Gigaspeech, AMI) and zero-cap sources
       (TEDLIUM, Peoples, Switchboard) to proper-cased form without
       damaging already-cased sources (LibriHeavy, CV, SPGI, VoxPopuli).

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
    text = _CORPUS_MARKER_RE.sub("", text)
    text = _TEDLIUM_BRACKET_RE.sub("", text)
    text = text.replace("%", " percent").replace("per cent", "percent")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return ""
    if _needs_truecase(text):
        text = truecase.get_true_case(text)
    return text


class DatasetLoader:
    """Loads and prepares datasets for training.

    Downloads each train/eval split fully via HuggingFace's Arrow cache,
    then concatenates and shuffles. ``group_by_length`` is supported via
    an optional duration column.
    """

    def __init__(self, config: DictConfig, multitask_enabled: bool = False):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir
        self.seed = config.training.get("seed", 42)
        self.num_proc = self.config.get("num_proc", 16)
        # `duration` is only consumed by Trainer's group_by_length sampler; skip
        # any duration prep when group_by_length is off. Avoids a slow column
        # cast (and potentially a full-decode map) on every train run.
        self.needs_duration = bool(config.training.get("group_by_length", False))
        self.multitask_enabled = multitask_enabled

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

        col_map = {
            "text": dataset_cfg.get("text_column", "text"),
            "audio": dataset_cfg.get("audio_column", "audio"),
        }
        if duration_source := dataset_cfg.get("duration_column"):
            col_map["duration"] = duration_source

        for target, source in col_map.items():
            if source != target and source in ds.column_names:
                if target in ds.column_names:
                    ds = ds.remove_columns([target])
                ds = ds.rename_column(source, target)

        ds = ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        if self.multitask_enabled:
            task = dataset_cfg.get("task", "transcribe")
            ds = ds.add_column("task", [task] * len(ds))
            keep_cols = {"audio", "text", "sift_response", "task"}
        else:
            keep_cols = {"audio", "text"}
        if self.needs_duration:
            keep_cols = keep_cols | {"duration"}
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
        """Cap (downsample) or repeat-pad (upsample) to ``target`` samples."""
        current = len(ds)
        if current == target:
            return ds
        if current > target:
            return ds.select(range(target))
        repeats = (target // current) + 1
        indices = list(range(current)) * repeats
        return ds.select(indices[:target])

    def _ensure_duration(self, ds: Dataset) -> Dataset:
        # `group_by_length` requires a `duration` column. Compute from audio
        # length when no source field provides one. Run AFTER resampling so we
        # don't decode samples that get trimmed. Cast to float32 only when the
        # source ships a different dtype (sources mix float32/float64/null and
        # concatenate_datasets rejects the mismatch); skipping a no-op cast
        # avoids a multi-minute column rewrite on large streams.
        if "duration" not in ds.column_names:
            sr = self.sample_rate

            def _add_duration(batch):
                return {"duration": [a["array"].shape[0] / sr for a in batch["audio"]]}

            ds = ds.map(_add_duration, batched=True, num_proc=self.num_proc)
        if ds.features["duration"].dtype != "float32":
            ds = ds.cast_column("duration", Value("float32"))
        return ds

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []

        for d_cfg in tqdm(self.config.datasets, desc="Loading datasets"):
            train_splits = d_cfg.get("train_splits", [d_cfg.get("train_split", "train")])
            val_splits = d_cfg.get("eval_splits", [d_cfg.get("eval_split", "validation")])
            target_samples = d_cfg.get("target_samples")

            for train_split in train_splits:
                ds = self._prepare_split(d_cfg, train_split)
                if target_samples:
                    ds = self._resample_to_target(ds, target_samples)
                if self.needs_duration:
                    ds = self._ensure_duration(ds)
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
                if self.needs_duration:
                    ds = self._ensure_duration(ds)
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
        system_prompt: str = None,
        projector: Any = None,
        encoder_conv_layers: list = None,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.system_prompt = system_prompt
        self.projector = projector
        self.encoder_conv_layers = encoder_conv_layers or DEFAULT_ENCODER_CONV_LAYERS
        # Whisper's encoder requires a fixed 3000 mel frames; other encoders
        # (GLM-ASR) accept variable-length input, so only pad to longest.
        self._audio_padding = (
            "max_length"
            if type(feature_extractor).__name__ == "WhisperFeatureExtractor"
            else "longest"
        )
        # 4096 tokens accommodates the long-tail of audio (up to 30s ≈ 187
        # audio tokens) + system prompt + user prompt + assistant transcript
        # (dense speech can produce 1000-1500 transcript tokens). At 2048 the
        # longest TEDLIUM / Earnings22 samples silently truncated the
        # assistant turn — model trained on partial labels. Qwen3-0.6B
        # supports 32K context so 4096 is well within capacity.
        self.text_collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=4096)

    # Whisper's feature extractor pads/truncates to a fixed 30s window. Audio
    # longer than this is silently truncated while the label is kept whole,
    # training the model to transcribe content it never sees. Drop those rows.
    _MAX_AUDIO_SECONDS = 30.0
    # Sub-0.8s clips are dominated by boundary-cut segments and isolated
    # backchannels ("yeah", "ok", "umhum") where the audio span and the
    # reference transcript don't actually line up — eval-side analysis on
    # Peoples / CV / Switchboard / AMI showed these as the bulk of >=50%
    # WER samples, with model output reflecting adjacent content rather
    # than the labeled token. Sub-100ms clips also break augmentations
    # (Mp3Compression's fast-mp3-augment backend rejects <100ms input);
    # raising the floor to 0.8s subsumes that constraint.
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
                if not _normalize_label(f.get("text") or ""):
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
                logging.debug("Skipping row in DataCollator: %s: %s", type(e).__name__, e)
                continue
            finally:
                f["audio"] = None
        if not audio_arrays:
            raise ValueError("No valid audio samples in batch")
        return audio_arrays, valid_features

    def _build_sample(self, feature: dict, num_audio_tokens: int) -> dict:
        """Build a single chat sample. Subclasses can override for task-specific prompts."""
        text = _normalize_label(feature.get("text") or "")
        return self._make_messages(num_audio_tokens, random.choice(TRANSCRIBE_PROMPTS), text)

    def _make_messages(self, num_audio_tokens: int, prompt: str, response: str) -> dict:
        user_content = ("<audio>" * num_audio_tokens) + " " + prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": response})
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


class MultiTaskDataCollator(DataCollator):
    """Collates audio and text data for multi-task ASR + SIFT training."""

    def __init__(self, *args, **kwargs):
        kwargs["system_prompt"] = ""
        super().__init__(*args, **kwargs)

    def _build_sample(self, feature: dict, num_audio_tokens: int) -> dict:
        if feature.get("task") == "sift":
            response = (feature.get("sift_response") or feature.get("text") or "").strip()
            prompt = random.choice(DESCRIBE_PROMPTS)
        else:
            response = (feature.get("text") or "").strip().lower()
            prompt = random.choice(TRANSCRIBE_PROMPTS)
        return self._make_messages(num_audio_tokens, prompt, response)


class ASRTrainer(Trainer):
    """Trainer subclass for ASR models."""

    def __init__(
        self,
        *args,
        decoder_learning_rate: float | None = None,
        decoder_weight_decay: float | None = None,
        projector_weight_decay: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.decoder_learning_rate = decoder_learning_rate
        self.decoder_weight_decay = decoder_weight_decay
        self.projector_weight_decay = projector_weight_decay

    def create_optimizer(self):
        """Optimizer with separate LR / weight decay for projector and language model.

        Mirrors HF Trainer.create_optimizer's decay/no-decay split, but adds a
        second axis: parameters under `language_model.` get `decoder_learning_rate`
        and `decoder_weight_decay`; projector params get `projector_weight_decay`
        (when set). Each falls back to `args.learning_rate` / `args.weight_decay`.
        """
        overrides = (
            self.decoder_learning_rate is not None
            or self.decoder_weight_decay is not None
            or self.projector_weight_decay is not None
        )
        if self.optimizer is not None or not overrides:
            return super().create_optimizer()

        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        opt_model = self.model
        decay_parameters = set(get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS))
        decay_parameters = {n for n in decay_parameters if "bias" not in n}

        groups: dict[tuple[bool, bool], list] = {
            (True, True): [],  # decoder, decay
            (True, False): [],  # decoder, no decay
            (False, True): [],  # other, decay
            (False, False): [],  # other, no decay
        }
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            is_decoder = name.startswith("language_model.")
            decay = name in decay_parameters
            groups[(is_decoder, decay)].append(param)

        base_wd = self.args.weight_decay
        base_lr = self.args.learning_rate
        dec_lr = self.decoder_learning_rate if self.decoder_learning_rate is not None else base_lr
        dec_wd = self.decoder_weight_decay if self.decoder_weight_decay is not None else base_wd
        proj_wd = (
            self.projector_weight_decay if self.projector_weight_decay is not None else base_wd
        )
        optimizer_grouped_parameters = [
            {"params": groups[(False, True)], "weight_decay": proj_wd, "lr": base_lr},
            {"params": groups[(False, False)], "weight_decay": 0.0, "lr": base_lr},
            {"params": groups[(True, True)], "weight_decay": dec_wd, "lr": dec_lr},
            {"params": groups[(True, False)], "weight_decay": 0.0, "lr": dec_lr},
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


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
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


TRAINING_MODEL_PARAMS = [
    "attn_implementation",
    "use_lora",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "freeze_projector",
    "freeze_language_model",
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

    if cfg.model.get("pretrained_model_path"):
        model = ASRModel.from_pretrained(cfg.model.pretrained_model_path, config=asr_config)
    else:
        model = ASRModel(asr_config)

    model.config.use_cache = False

    if hub_model_id := cfg.training.get("hub_model_id"):
        model.config.pretrained_model_path = hub_model_id

    # Workaround: TRL's DataCollatorForChatML doesn't pass enable_thinking=False to Qwen3.
    # See https://github.com/huggingface/trl/issues/3387
    if model.tokenizer.chat_template and "enable_thinking" in model.tokenizer.chat_template:
        model.tokenizer.chat_template = model.tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "true",
        )

    multitask_enabled = cfg.get("multitask", {}).get("enabled", False)

    train_dataset, val_dataset = DatasetLoader(cfg, multitask_enabled=multitask_enabled).load()

    augmentations: list = []

    def _aug_kwargs(cfg_block) -> dict:
        # Forward every yaml key (minus `enabled`) as a kwarg — defaults
        # live on the augmentation class signatures, not duplicated here.
        d = OmegaConf.to_container(cfg_block, resolve=True)
        d.pop("enabled", None)
        return d

    rir_aug: RIRAugmentation | None = None
    rir_cfg = cfg.training.get("rir_augmentation") or {}
    if rir_cfg.get("enabled"):
        rir_aug = RIRAugmentation(sample_rate=cfg.data.sample_rate, **_aug_kwargs(rir_cfg))
        augmentations.append(rir_aug)

    noise_aug: NoiseAugmentation | None = None
    noise_cfg = cfg.training.get("noise_augmentation") or {}
    if noise_cfg.get("enabled"):
        noise_aug = NoiseAugmentation(sample_rate=cfg.data.sample_rate, **_aug_kwargs(noise_cfg))
        augmentations.append(noise_aug)

    silence_injection_prob = float(cfg.training.get("silence_injection_prob", 0.0))
    if silence_injection_prob > 0.0 and noise_aug is None:
        raise ValueError(
            "silence_injection_prob > 0 requires noise_augmentation.enabled "
            "(the noise corpus is the source of noise-only samples)."
        )

    if augmentations or silence_injection_prob > 0.0:
        # Skip augmentation for clips below the DataCollator's drop threshold:
        # the row will be filtered out of the batch downstream, and some
        # audiomentations stages (Mp3Compression's fast-mp3-augment backend)
        # error on sub-100ms input rather than no-op'ing.
        _aug_min_samples = int(cfg.data.sample_rate * DataCollator._MIN_AUDIO_SECONDS)

        def _apply_aug(batch):
            audios = batch.get("audio") or []
            texts = batch.get("text")
            n_texts = len(texts) if texts is not None else 0
            for i, a in enumerate(audios):
                if not a or "array" not in a:
                    continue
                arr = a["array"]
                if arr.shape[-1] < _aug_min_samples:
                    continue
                # Silence injection targets backchannel hallucinations
                # ("yeah" / "huh" on empty GT) — a documented Whisper /
                # SALMONN failure mode — by pairing noise-only audio with
                # an empty transcript so the model learns "no speech → EOS".
                if (
                    silence_injection_prob > 0.0
                    and noise_aug is not None
                    and i < n_texts
                    and random.random() < silence_injection_prob
                ):
                    noise = noise_aug.sample_noise_only(arr.shape[-1])
                    if noise is not None:
                        arr = noise.astype(arr.dtype)
                        texts[i] = ""
                for aug in augmentations:
                    arr = aug(arr)
                a["array"] = arr
            return batch

        train_dataset = train_dataset.with_transform(_apply_aug)

    if multitask_enabled:
        data_collator = MultiTaskDataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=cfg.data.sample_rate,
            projector=model.projector,
            encoder_conv_layers=model.config.encoder_conv_layers,
        )
    else:
        data_collator = DataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=cfg.data.sample_rate,
            system_prompt=cfg.model.system_prompt,
            projector=model.projector,
            encoder_conv_layers=model.config.encoder_conv_layers,
        )

    callbacks = []
    if cfg.early_stopping.patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping.patience,
                early_stopping_threshold=cfg.early_stopping.threshold,
            )
        )
    if push_to_hub:
        callbacks.append(PushToHubCallback())

    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_config, dict)
    decoder_learning_rate = training_config.pop("decoder_learning_rate", None)
    decoder_weight_decay = training_config.pop("decoder_weight_decay", None)
    projector_weight_decay = training_config.pop("projector_weight_decay", None)
    if compile_config := training_config.pop("torch_compile_config", None):
        torch._dynamo.config.cache_size_limit = compile_config.get("cache_size_limit", 64)
        torch._dynamo.config.capture_scalar_outputs = compile_config.get(
            "capture_scalar_outputs", True
        )
        torch._inductor.config.compile_threads = compile_config.get("compile_threads", 4)

    trainer = ASRTrainer(
        model=model,
        args=TrainingArguments(**get_valid_training_args(training_config)),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=model.tokenizer,
        callbacks=callbacks,
        decoder_learning_rate=decoder_learning_rate,
        decoder_weight_decay=decoder_weight_decay,
        projector_weight_decay=projector_weight_decay,
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
