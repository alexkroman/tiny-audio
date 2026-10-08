"""Attention-implementation selection for ASRModel's encoder and decoder loaders.

Decides which `attn_implementation` each sub-model is loaded with: FA2 only
where CUDA and a working flash-attn are present and every head fits its
kernels, eager on MPS for models that build a sliding-window mask, and the
requested implementation otherwise.
"""

import functools
import importlib
import logging
from typing import TYPE_CHECKING, cast

import torch
from transformers import AutoConfig, PretrainedConfig

if TYPE_CHECKING:
    from .asr_config import text_config_of
else:
    try:
        from .asr_config import text_config_of
    except ImportError:  # flat layout on the Hub: sibling modules, no package
        from asr_config import text_config_of

logger = logging.getLogger(__name__)

# FlashAttention's kernels are compiled for head dimensions up to 256.
FLASH_ATTENTION_MAX_HEAD_DIM = 256


def _max_attention_head_dim(text_config: object) -> int | None:
    """Largest attention head dim across layers, or None if undeterminable.

    Has to cope with heterogeneous configs: Gemma 4 varies head_dim per layer,
    and reading `config.head_dim` on one of those raises
    AmbiguousGlobalPerLayerAttributeError instead of returning a number, so the
    per-layer list must be consulted explicitly.
    """
    per_layer = getattr(text_config, "per_layer_config", None)
    dims: list[object] = []
    if per_layer:
        dims = [getattr(layer, "head_dim", None) for layer in per_layer]
    else:
        try:
            dims = [getattr(text_config, "head_dim", None)]
        except Exception:
            # Raised by the heterogeneity guard; treated as "unknown".
            dims = []
    usable = [d for d in dims if isinstance(d, int) and d > 0]
    return max(usable) if usable else None


@functools.lru_cache(maxsize=16)
def _has_sliding_window_attention(model_id: str) -> bool:
    """Whether `model_id`'s text stack has any sliding-window attention layer.

    Read from the config rather than the loaded module so the answer is
    available before `from_pretrained` picks an attn implementation.

    Two independent spellings, because transformers has both. Gemma 4 declares
    `sliding_window: 512` AND `layer_types: [full_attention,
    sliding_attention]` (verified against google/gemma-4-E2B-it); Qwen2-style
    configs carry a non-null `sliding_window` that is inert unless
    `use_sliding_window` is true, so that flag has to veto the window.

    Returns True when the config can't be read, so an unknown architecture
    keeps the conservative eager path on MPS.
    """
    try:
        probe = cast(
            PretrainedConfig,
            AutoConfig.from_pretrained(model_id, trust_remote_code=True),
        )
        text_config = text_config_of(probe) if hasattr(probe, "get_text_config") else probe
    except Exception:
        logger.warning(
            "Could not read the config for %s to check for sliding-window "
            "attention; assuming it has some and using eager attention on MPS.",
            model_id,
        )
        return True

    layer_types: list[object] = getattr(text_config, "layer_types", None) or []
    if any("sliding" in str(layer_type) for layer_type in layer_types):
        return True
    if getattr(text_config, "use_sliding_window", None) is False:
        return False
    return bool(getattr(text_config, "sliding_window", None))


def resolve_attn_implementation(requested: str | None, model_id: str | None = None) -> str | None:
    """Coerce flash_attention_2 to sdpa when CUDA isn't available, and avoid sdpa on MPS.

    FA2 is CUDA-only. On MPS/CPU, requesting it either errors at load or
    silently falls back to a slower path; either way the user pays the FA2
    install + import cost for no win. Coerce here so a saved config that
    pins flash_attention_2 still loads on Mac / CPU-only Linux boxes.

    MPS needs eager for SOME models. PyTorch's Metal sdpa kernel returns wrong
    results for cached single-token decode against a sliding-window mask, which
    is exactly Gemma 4's layout (`sliding_window=512`, four sliding layers per
    full-attention layer). A full-sequence forward is fine, so the damage shows
    up only during generation: greedy decode with no cache transcribed
    correctly while the identical cached decode produced "Mr. and a".

    The mask is the trigger, so the coercion is scoped to models that actually
    build one. `model_id` opts into that check; without it every MPS load falls
    back to eager, which is what this function used to do unconditionally. That
    blanket version taxed the models it was never protecting: Qwen3.5-2B
    declares `sliding_window: None` and cannot hit the bug, and eager cost it
    41% of decode throughput on an M4 Max (19.9 -> 28.1 tok/s, bf16, 64 tokens
    from a 200-token prompt). Audio encoders are likewise unaffected -- they run
    full-sequence forwards with no cache at all, and the Granite branch below
    has been pinning sdpa on MPS for every granite_qwen run to date.

    SECOND, SEPARATE MPS sdpa HAZARD, and the reason `generate` carries a guard:
    Metal's sdpa returns NaN for a row containing any fully-masked position, so
    a LEFT-PADDED batch comes back as NaN logits, argmax 0, and a transcript of
    "!!!!!!". One pad token is enough. Measured on bare transformers with no
    tiny_audio code involved (Qwen3.5-2B, bf16): the identical forward is exact
    to 0.000 on CPU/fp32, and eager is correct on MPS at any padding. Batch 1
    pads nothing, which is why the eval harness is safe under sdpa -- but this
    is what the blanket eager was accidentally also protecting, so anything
    that batches ragged audio on a Mac must ask for eager explicitly. See
    `_assert_sdpa_safe_on_mps`.
    """
    if (
        torch.backends.mps.is_available()
        and requested in (None, "sdpa", "flash_attention_2")
        and (model_id is None or _has_sliding_window_attention(model_id))
    ):
        return "eager"
    # Otherwise fall through: on MPS, sdpa and None (let transformers pick,
    # which is sdpa where available) are both safe for a model with no
    # sliding-window layer, and a config pinning flash_attention_2 is still
    # downgraded to sdpa below.
    if requested != "flash_attention_2":
        return requested
    if not torch.cuda.is_available():
        return "sdpa"
    # CUDA alone isn't enough -- the flash_attn package must actually be
    # importable, and new enough for transformers. It is a source build that
    # routinely fails on RunPod images (its metadata hook imports torch, so any
    # broken torch install takes it down with it). Without this check a CUDA
    # box with no flash-attn raises at from_pretrained instead of quietly
    # using sdpa, which is the same numerics at lower throughput.
    #
    # Looked up on `transformers.utils` at call time rather than bound at
    # import, so patching it there (as the tests do) reaches this probe.
    if not importlib.import_module("transformers.utils").is_flash_attn_2_available():
        logger.warning(
            "flash_attention_2 requested but flash_attn is not installed or too old; "
            "falling back to sdpa."
        )
        return "sdpa"
    return requested


def resolve_decoder_attn_implementation(requested: str | None, model_id: str) -> str | None:
    """`resolve_attn_implementation` for the decoder, plus FlashAttention's head-dim limit.

    FlashAttention only has kernels for head_dim <= 256, and it fails at
    the first forward rather than at load. Gemma 4 E2B trips this: 7 of
    its 35 layers use head_dim=512 (the other 28 use 256), so FA2 can
    never run this architecture regardless of flash-attn version. Detect
    it from the config and fall back instead of dying a step into
    training.
    """
    attn_implementation = resolve_attn_implementation(requested, model_id)
    if attn_implementation == "flash_attention_2":
        probe = cast(
            PretrainedConfig,
            AutoConfig.from_pretrained(model_id, trust_remote_code=True),
        )
        text_probe = text_config_of(probe) if hasattr(probe, "get_text_config") else probe
        head_dim = _max_attention_head_dim(text_probe)
        if head_dim is not None and head_dim > FLASH_ATTENTION_MAX_HEAD_DIM:
            logger.warning(
                "%s has max head_dim=%d, above FlashAttention's limit of %d; "
                "using sdpa for the decoder.",
                model_id,
                head_dim,
                FLASH_ATTENTION_MAX_HEAD_DIM,
            )
            attn_implementation = "sdpa"
    return attn_implementation
