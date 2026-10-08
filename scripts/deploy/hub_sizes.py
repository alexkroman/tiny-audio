"""Exact parameter and download sizes read from Hub metadata for the deploy planner.

Nothing here downloads weights: parameter counts come from safetensors headers
(HTTP range requests) and byte totals from the Hub's file metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from huggingface_hub import dataset_info, get_safetensors_metadata, model_info


class HubConfig(Protocol):
    def get_text_config(self) -> object: ...


class ConfigLoader(Protocol):
    def from_pretrained(self, pretrained_model_name_or_path: str, /) -> HubConfig: ...


def safetensors_params(repo_id: str, exclude_prefixes: tuple[str, ...] = ()) -> tuple[int, str]:
    """Exact parameter count from the safetensors header (no weight download).

    `exclude_prefixes` drops towers that live in the checkpoint but that the
    loader never instantiates. Pass `NON_LM_TOWER_PREFIXES` for a decoder:
    `meta.parameter_count` is a whole-repo total, so without it Qwen3.5-2B is
    charged 2.2741B where AutoModelForCausalLM builds 1.8818B -- the 0.3314B
    `visual.` tower and the 0.0608B `mtp.` head are counted but never loaded.
    That 0.3922B overstatement propagated into weights, gradients and
    optimizer state, i.e. it was charged four times over.

    Do NOT pass it for an audio encoder: `audio_tower.` is one of the
    prefixes, and for a checkpoint that IS the audio tower that would zero out
    the thing being measured.
    """
    meta = get_safetensors_metadata(repo_id)
    counts: dict[str, int]
    if not exclude_prefixes:
        counts = dict(meta.parameter_count.items())
    else:
        # parameter_count is pre-aggregated by dtype, so filtering by name
        # means re-deriving it from the per-tensor headers.
        counts = {}
        for f in meta.files_metadata.values():
            for name, info in f.tensors.items():
                if any(name.startswith(p) or f".{p}" in name for p in exclude_prefixes):
                    continue
                numel = 1
                for dim in info.shape:
                    numel *= dim
                counts[info.dtype] = counts.get(info.dtype, 0) + numel
    # Ignore integer buffers (rotary caches, position ids); they aren't params.
    total = sum(n for dtype, n in counts.items() if not dtype.startswith("I"))
    dominant = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "?"
    return total, dominant


# Towers that live in a multimodal checkpoint but that AutoModelForCausalLM
# does not load, so neither LoRA nor the memory model ever sees them.
# Qwen3.5-2B ships an `mtp.` multi-token-prediction head (0.0608B) beside a
# `visual.` tower (0.3314B); counting the former added a phantom 25th layer
# and overstated a rank-64 estimate by 2.55M, and counting both overstated the
# decoder's parameter count by 0.3922B.
#
# Applies to the DECODER only -- `audio_tower.` is in the list, so filtering an
# audio-encoder repo with it would discard the encoder itself.
NON_LM_TOWER_PREFIXES = ("mtp.", "visual.", "audio_tower.", "vision_tower.")


def lora_trainable_params(
    repo_id: str, rank: int, target_modules: str | Sequence[str] | None
) -> int:
    """Exact LoRA parameter count from safetensors headers (no weight download).

    A rank-r adapter on a linear of shape (out, in) adds r*(in+out). Reading the
    real shapes beats deriving them from the config: Qwen3.5 is hybrid, so its
    24 layers carry five differently-shaped linear-attention projections plus
    MLP, and only 6 of them have q/k/v/o at all.

    `target_modules` follows peft: the string "all-linear" means every 2-D
    weight in the transformer body except the embedding and the output head;
    a list matches against the module-name suffix.

    Validated against peft 0.20.0 on the real checkpoint: r=64 / "all-linear"
    on Qwen3.5-2B gives 67.28M over 186 matrices here and 67.28M there.
    """
    meta = get_safetensors_metadata(repo_id)
    shapes = {
        name: info.shape for f in meta.files_metadata.values() for name, info in f.tensors.items()
    }
    all_linear = isinstance(target_modules, str) and target_modules == "all-linear"
    names = list(target_modules or []) if not all_linear else []

    total = 0
    for name, shape in shapes.items():
        if len(shape) != 2 or ".layers." not in name:
            continue
        if any(name.startswith(p) or f".{p}" in name for p in NON_LM_TOWER_PREFIXES):
            continue
        # lm_head and the embedding table are never adapted by "all-linear",
        # and adapting lm_head would be wrong here anyway -- it is tied to the
        # frozen embed_tokens.
        if "embed" in name or "lm_head" in name:
            continue
        if not all_linear and not any(f".{n}.weight" == name[-len(n) - 8 :] for n in names):
            continue
        out_f, in_f = shape
        total += rank * (in_f + out_f)
    return total


def vocab_table_params(text_cfg: object) -> dict[str, int]:
    """Per-token lookup-table sizes for a decoder, computed from its config.

    These can be frozen independently of the rest of the decoder
    (`freeze_text_embed_tokens`), and on Gemma 4 they are the majority of the
    checkpoint -- so counting them as trainable overstates AdamW state badly.
    Empty entries are omitted, so decoders without a per-layer table simply
    don't report one.
    """
    tables: dict[str, int] = {}
    vocab = int(getattr(text_cfg, "vocab_size", 0) or 0)
    hidden = int(getattr(text_cfg, "hidden_size", 0) or 0)
    if vocab and hidden:
        tables["embed_tokens"] = vocab * hidden
    ple_vocab = int(getattr(text_cfg, "vocab_size_per_layer_input", 0) or 0)
    ple_hidden = int(getattr(text_cfg, "hidden_size_per_layer_input", 0) or 0)
    layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    if ple_vocab and ple_hidden and layers:
        tables["embed_tokens_per_layer"] = ple_vocab * layers * ple_hidden
    return tables


def repo_weight_bytes(repo_id: str, repo_type: str = "model", name: str | None = None) -> int:
    """Total bytes the Hub will hand us for a repo, optionally one config only.

    Multi-config dataset repos (libriheavy ships small/medium/large) would be
    wildly overcounted by summing every shard, so when a config `name` is given
    we keep only files whose path mentions it. Falls back to the full repo when
    nothing matches, since a silent zero would be worse than an overestimate.
    """
    info = (model_info if repo_type == "model" else dataset_info)(repo_id, files_metadata=True)
    siblings = [s for s in (info.siblings or []) if (s.size or 0) > 0]
    if name:
        scoped = [s for s in siblings if name.lower() in s.rfilename.lower()]
        if scoped:
            siblings = scoped
    return sum(s.size or 0 for s in siblings)


def hidden_dim(cfg_obj: object) -> int | None:
    for attr in ("hidden_size", "d_model"):
        if getattr(cfg_obj, attr, None):
            dim: int = getattr(cfg_obj, attr)
            return dim
    return None
