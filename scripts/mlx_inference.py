"""MLX inference on Apple silicon for Granite Speech 5.0 + Qwen3 checkpoints (`ta eval --mlx`).

The same transcription the PyTorch pipeline produces, without torch on the hot
path. Each piece is an existing, verified MLX implementation or a direct port:

- encoder: mlx-audio's port of Granite Speech 5.0 (`granite_speech5_ctc`),
  loaded from the checkpoint's `audio_model_id`. Its features match
  transformers' extractor to 3e-5 and, run in float32, its hidden states match
  `GraniteSpeech5Encoder` to 4e-6 (2026-10-21); it runs in bfloat16 here, as
  our CUDA inference does.
- projector: `MLPAudioProjector` (frame stack, RMSNorm, linear, exact GELU,
  linear) with the checkpoint's `projector.*` weights, kept in float32 as
  training holds them.
- decoder: mlx-lm's Qwen3 with the checkpoint's own decoder weights, built
  from its embedding table's shape (training pads the vocabulary for the
  added audio token, so the base config's vocab size is wrong).
- everything else is the PyTorch pipeline's own code: chunking at the quietest
  point, the silence gate, the inference lead-in, the chat prompt with one
  placeholder per audio frame, greedy decoding with the 12-gram loop guard,
  and the same text cleanup.

Only fully fine-tuned decoders are supported: their weights are in the
checkpoint. A LoRA checkpoint stores adapters against a base model and is
rejected with a clear error rather than decoded with the adapters missing.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download, snapshot_download
from mlx import nn
from mlx_audio.stt.models.granite_speech5_ctc.granite_speech5 import compute_features
from mlx_audio.stt.utils import load_model as load_stt_model
from mlx_lm.generate import generate_step
from mlx_lm.models import qwen3
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tiny_audio.asr_processing import (
    CHUNK_MAX_S,
    audible_chunks,
    chunk_bounds,
    is_silent,
    prepend_lead_in,
    render_audio_prompt,
)
from tiny_audio.asr_text import truncate_repetitions

SAMPLE_RATE = 16000
# Qwen3 sometimes opens with an empty think block despite enable_thinking=False;
# the PyTorch pipeline strips it the same way.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_DECODER_PREFIX = "language_model."
_PROJECTOR_PREFIX = "projector."


class Projector(nn.Module):
    """MLX twin of `tiny_audio.projectors.MLPAudioProjector`, same parameter names."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, stride: int) -> None:
        """Layers named as in the checkpoint's `projector.*` keys."""
        super().__init__()
        self.stride = stride
        self.input_norm = nn.RMSNorm(in_dim * stride, eps=1e-6)
        self.linear_1 = nn.Linear(in_dim * stride, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """`(batch, frames, in_dim)` encoder states to `(batch, tokens, out_dim)`."""
        batch, frames, dim = x.shape
        tokens = (frames - self.stride) // self.stride + 1
        x = x[:, : tokens * self.stride].reshape(batch, tokens, dim * self.stride)
        x = self.input_norm(x.astype(mx.float32))
        return self.linear_2(nn.gelu(self.linear_1(x)))


def banned_next_tokens(tokens: list[int], n: int) -> list[int]:
    """Ids that would complete an n-gram already in `tokens` (HF's no_repeat_ngram rule)."""
    if n <= 0 or len(tokens) < n:
        return []
    prefix = tuple(tokens[len(tokens) - n + 1 :])
    return [
        tokens[i + n - 1]
        for i in range(len(tokens) - n + 1)
        if tuple(tokens[i : i + n - 1]) == prefix
    ]


def no_repeat_ngram_processor(n: int) -> Callable[[mx.array, mx.array], mx.array]:
    """An mlx-lm logits processor banning repeated n-grams, as generation_config asks."""

    def process(tokens: mx.array, logits: mx.array) -> mx.array:
        banned = banned_next_tokens(tokens.tolist(), n)
        if banned:
            logits[:, mx.array(banned)] = -float("inf")
        return logits

    return process


def decoder_weights(checkpoint: dict[str, Any], tie_word_embeddings: bool) -> dict[str, Any]:
    """The checkpoint's `language_model.*` tensors under mlx-lm Qwen3's parameter names."""
    weights = {
        k.removeprefix(_DECODER_PREFIX): v
        for k, v in checkpoint.items()
        if k.startswith(_DECODER_PREFIX)
    }
    if tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights


def projector_weights(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint's `projector.*` tensors, unprefixed."""
    return {
        k.removeprefix(_PROJECTOR_PREFIX): v
        for k, v in checkpoint.items()
        if k.startswith(_PROJECTOR_PREFIX)
    }


def clean_text(text: str) -> str:
    """The PyTorch pipeline's text cleanup: strip, drop think blocks, cut runaway loops."""
    text = text.strip()
    if "<think>" in text:
        text = _THINK_TAG_RE.sub("", text).strip()
    return truncate_repetitions(text)


@dataclass
class GenerationSettings:
    """What the checkpoint's generation_config.json asks for (greedy, always)."""

    max_new_tokens: int
    no_repeat_ngram_size: int
    eos_token_ids: set[int]


def _checkpoint_dir(model_id: str) -> Path:
    """A local checkpoint directory as-is, or the Hub repo's snapshot."""
    local = Path(model_id)
    return local if local.is_dir() else Path(snapshot_download(model_id))


def _cache_file(directory: Path) -> Path | None:
    """Where a Hub snapshot's converted weights live, one file per repo revision.

    Snapshot directories are `.../models--org--name/snapshots/<commit>`, so the
    commit names the revision: a new push gets a new file, a rerun reuses it.
    A plain local directory has no revision to key on and is converted in
    memory every time.
    """
    if directory.parent.name != "snapshots":
        return None
    repo = directory.parent.parent.name.removeprefix("models--")
    root = Path(os.environ.get("TINY_AUDIO_MLX_CACHE") or Path.home() / ".cache/tiny-audio/mlx")
    return root / repo / f"{directory.name}.safetensors"


def converted_weights(directory: Path, tie_word_embeddings: bool) -> dict[str, mx.array]:
    """The checkpoint as MLX weights (bf16 decoder, fp32 projector), converted once per revision.

    Keys are `decoder.<mlx-lm name>` and `projector.<name>`. Converting reads
    the float32 training checkpoint (2.4 GB for Qwen3-0.6B); the cached file
    is half that and needs no renaming or casting on load.
    """
    cache = _cache_file(directory)
    if cache is not None and cache.exists():
        return mx.load(str(cache))
    checkpoint = mx.load(str(directory / "model.safetensors"))
    weights = {
        **{
            f"decoder.{k}": v.astype(mx.bfloat16)
            for k, v in decoder_weights(checkpoint, tie_word_embeddings).items()
        },
        **{
            f"projector.{k}": v.astype(mx.float32) for k, v in projector_weights(checkpoint).items()
        },
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_suffix(".partial.safetensors")
        mx.save_safetensors(str(partial), weights)
        partial.replace(cache)  # atomic: an interrupted conversion never looks finished
    return weights


def _subset(weights: dict[str, mx.array], prefix: str) -> list[tuple[str, mx.array]]:
    """`weights` under `prefix`, unprefixed, as `load_weights` takes them."""
    return [(k.removeprefix(prefix), v) for k, v in weights.items() if k.startswith(prefix)]


def _generation_settings(directory: Path, config: dict[str, Any]) -> GenerationSettings:
    """Decode settings from generation_config.json, falling back to config.json."""
    path = directory / "generation_config.json"
    gen: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    eos = gen.get("eos_token_id", [])
    return GenerationSettings(
        max_new_tokens=int(gen.get("max_new_tokens") or config.get("max_new_tokens") or 128),
        no_repeat_ngram_size=int(
            gen.get("no_repeat_ngram_size") or config.get("no_repeat_ngram_size") or 0
        ),
        eos_token_ids=set(eos if isinstance(eos, list) else [eos]),
    )


class MLXTranscriber:
    """A Granite Speech 5.0 + Qwen3 checkpoint, transcribing on Apple silicon."""

    def __init__(
        self,
        encoder: Callable[[mx.array], mx.array],
        projector: Projector,
        decoder: qwen3.Model,
        tokenizer: PreTrainedTokenizerBase,
        config: dict[str, Any],
        generation: GenerationSettings,
    ) -> None:
        """Hold the loaded parts; use `from_pretrained` to build one."""
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.config = config
        self.generation = generation
        self.audio_token: str = config.get("audio_token") or "<audio>"
        audio_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        if not isinstance(audio_id, int):
            msg = f"audio token {self.audio_token!r} has no single id in the checkpoint tokenizer"
            raise TypeError(msg)
        self.audio_token_id: int = audio_id
        self.prompt: str | None = config.get("transcribe_prompt")
        self.lead_in_seconds = float(config.get("inference_lead_in_seconds") or 0.0)
        self.chunk_max_seconds = float(config.get("inference_chunk_seconds") or CHUNK_MAX_S)
        # The Hub commit transcribed with, when loaded from a snapshot.
        self.revision: str | None = None

    @classmethod
    def from_pretrained(cls, model_id: str) -> MLXTranscriber:
        """Load a Hub repo or local checkpoint directory saved by `ASRModel`."""
        directory = _checkpoint_dir(model_id)
        config: dict[str, Any] = json.loads((directory / "config.json").read_text())
        audio_id = str(config.get("audio_model_id", ""))
        if "granite-speech-5" not in audio_id.lower():
            msg = f"MLX inference supports Granite Speech 5.0 encoders, not {audio_id}"
            raise ValueError(msg)
        if config.get("use_lora"):
            msg = (
                "This checkpoint stores LoRA adapters, not decoder weights; MLX inference supports "
                "fully fine-tuned decoders only."
            )
            raise ValueError(msg)

        text_cfg: dict[str, Any] = json.loads(
            Path(hf_hub_download(str(config["text_model_id"]), "config.json")).read_text()
        )
        decoder_type = text_cfg.get("model_type")
        if decoder_type != "qwen3":
            msg = f"MLX inference supports Qwen3 decoders; this checkpoint uses {decoder_type}"
            raise ValueError(msg)
        weights = converted_weights(directory, bool(text_cfg.get("tie_word_embeddings")))
        if "decoder.model.embed_tokens.weight" not in weights:
            msg = (
                "The checkpoint holds no decoder weights (a frozen decoder); MLX inference "
                "supports fully fine-tuned decoders only."
            )
            raise ValueError(msg)
        text_cfg["vocab_size"] = int(weights["decoder.model.embed_tokens.weight"].shape[0])
        decoder = qwen3.Model(qwen3.ModelArgs.from_dict(text_cfg))
        decoder.load_weights(_subset(weights, "decoder."), strict=True)

        stride = int(config.get("projector_pool_stride") or 4)
        hidden_dim, in_dim = weights["projector.linear_1.weight"].shape
        out_dim = int(weights["projector.linear_2.weight"].shape[0])
        projector = Projector(int(in_dim) // stride, int(hidden_dim), out_dim, stride)
        projector.load_weights(_subset(weights, "projector."), strict=True)

        encoder = load_stt_model(audio_id, strict=True).encoder
        mx.eval(decoder.parameters(), projector.parameters(), encoder.parameters())
        tokenizer = AutoTokenizer.from_pretrained(str(directory))
        transcriber = cls(
            encoder, projector, decoder, tokenizer, config, _generation_settings(directory, config)
        )
        if directory.parent.name == "snapshots":
            transcriber.revision = directory.name
        return transcriber

    def _audio_embeddings(self, chunk: npt.NDArray[np.float32]) -> mx.array:
        """One chunk's audio as `(tokens, hidden)` decoder-input embeddings."""
        audio = prepend_lead_in(chunk, SAMPLE_RATE, self.lead_in_seconds)
        features = compute_features(mx.array(np.asarray(audio, dtype=np.float32)))
        states = self.encoder(features[None])
        return self.projector(states)[0].astype(mx.bfloat16)

    def _prompt_embeddings(self, audio: mx.array) -> tuple[mx.array, mx.array]:
        """Prompt token ids and their embeddings, the audio spliced over the placeholders."""
        n_audio = int(audio.shape[0])
        ids: list[int] = render_audio_prompt(
            self.tokenizer, self.audio_token, n_audio, self.prompt
        ).tolist()
        start = ids.index(self.audio_token_id)
        embeds = self.decoder.model.embed_tokens(mx.array(ids))
        spliced = mx.concatenate([embeds[:start], audio, embeds[start + n_audio :]], axis=0)
        return mx.array(ids), spliced

    def _decode(self, ids: mx.array, embeddings: mx.array) -> Iterator[int]:
        """Greedy token ids until an end-of-turn id or the token budget."""
        processors = (
            [no_repeat_ngram_processor(self.generation.no_repeat_ngram_size)]
            if self.generation.no_repeat_ngram_size
            else []
        )
        steps = generate_step(
            ids,
            self.decoder,
            max_tokens=self.generation.max_new_tokens,
            logits_processors=processors,
            input_embeddings=embeddings,
        )
        for token, _ in steps:
            value = int(token)
            if value in self.generation.eos_token_ids:
                return
            yield value

    def transcribe_chunk(self, chunk: npt.NDArray[np.float32]) -> str:
        """One chunk (at most `chunk_max_seconds`) to text; digital silence is ""."""
        # The PyTorch pipeline's gate (`_transcribe_chunk`): an all-zero chunk
        # never reaches the decoder, which would otherwise make something up.
        if is_silent(chunk):
            return ""
        ids, embeddings = self._prompt_embeddings(self._audio_embeddings(chunk))
        tokens = list(self._decode(ids, embeddings))
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return clean_text(str(decoded))

    def transcribe(self, audio: npt.NDArray[np.float32]) -> str:
        """16 kHz mono audio of any length to text, chunked as the PyTorch pipeline does."""
        bounds = chunk_bounds(audio, SAMPLE_RATE, self.chunk_max_seconds)
        texts = [self.transcribe_chunk(c) for c in audible_chunks(audio, bounds, SAMPLE_RATE)]
        return " ".join(t for t in texts if t)
