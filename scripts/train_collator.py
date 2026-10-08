"""Batch collation for ASR training: audio features plus chat-formatted labels.

Split out of scripts/train.py; `DataCollator` turns dataset rows into the
`input_features` / `audio_attention_mask` / `audio_token_counts` / chat-token
batch ASRModel.forward consumes.
"""

import logging
import warnings
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor
from trl.import_utils import TRLExperimentalWarning

from scripts.labels import has_edge_content_tag, normalize_label
from tiny_audio.asr_config import (
    DEFAULT_ENCODER_CONV_LAYERS,
    ConvLayerSpec,
    compute_encoder_output_length,
)
from tiny_audio.asr_types import AudioFeatureExtractor, OutputLengthProjector

# trl.experimental warns (TRLExperimentalWarning) the first time it is
# imported; DataCollatorForChatML is the only thing used from it.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", TRLExperimentalWarning)
    from trl.experimental.utils import DataCollatorForChatML

logger = logging.getLogger(__name__)

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


ChatSample = dict[str, list[dict[str, str]]]


class DataCollator:
    """Collates audio and text data for training."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        feature_extractor: SequenceFeatureExtractor,
        sample_rate: int,
        projector: OutputLengthProjector | None = None,
        encoder_conv_layers: Sequence[ConvLayerSpec] | None = None,
        audio_token: str = "<audio>",
    ) -> None:
        self.tokenizer = tokenizer
        # Every concrete SequenceFeatureExtractor (Whisper, GLM-ASR) is callable.
        self.feature_extractor = cast(AudioFeatureExtractor, feature_extractor)
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

    def _extract_audio_arrays(
        self, features: list[dict[str, Any]]
    ) -> tuple[list[npt.NDArray[Any]], list[dict[str, Any]]]:
        audio_arrays: list[npt.NDArray[Any]] = []
        valid_features: list[dict[str, Any]] = []
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
                if not normalize_label(raw_text, f.get("_text_case")):
                    continue
                # Drop rows whose label starts or ends with a content-bearing
                # tag (<unk>/<foreign>/<overlap>). Stripping those yields a
                # target missing its first or last spoken word while the audio
                # retains it, which supervises onset/offset truncation — the
                # measured root cause of this recipe's Peoples regression.
                # See scripts/labels.py _EDGE_CONTENT_TAG_RE for the rates and the evidence.
                if has_edge_content_tag(raw_text):
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

    def _build_sample(self, feature: dict[str, Any], num_audio_tokens: int) -> ChatSample:
        """Build a single chat sample."""
        text = normalize_label(feature.get("text") or "", feature.get("_text_case"))
        # Prompt carries the label convention, so the punctuated and
        # unpunctuated halves of the mix stop competing for the same
        # conditioning. Undeclared sources keep the plain prompt.
        prompt = TRANSCRIBE_PROMPT_PUNCT if feature.get("_text_punct") else TRANSCRIBE_PROMPT
        return self._make_messages(num_audio_tokens, prompt, text)

    def _make_messages(self, num_audio_tokens: int, prompt: str, response: str) -> ChatSample:
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

        audio_attention_mask: torch.Tensor = audio_out["attention_mask"]
        input_features: torch.Tensor = audio_out["input_features"]
        mel_lengths = audio_attention_mask.sum(dim=-1)
        encoder_lengths = compute_encoder_output_length(mel_lengths, self.encoder_conv_layers)
        assert self.projector is not None, "DataCollator needs a projector to count audio tokens"
        token_counts_tensor = self.projector.get_output_length(encoder_lengths).to(torch.long)
        # torch annotates `Tensor.tolist` with a bare `list`.
        audio_token_counts = cast(
            list[int],
            token_counts_tensor.tolist(),  # pyright: ignore[reportUnknownMemberType]
        )

        text_features = [
            self._build_sample(f, n)
            for f, n in zip(valid_features, audio_token_counts, strict=True)
        ]

        batch = self.text_collator(text_features)
        batch["input_features"] = input_features
        batch["audio_attention_mask"] = audio_attention_mask
        batch["audio_token_counts"] = token_counts_tensor
        return batch
