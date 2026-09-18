#!/usr/bin/env python3
"""Map tiny-audio decoder tensor names onto their base-LM counterparts.

Shared by analyze_weights and compare_to_base so the two cannot disagree about
where a base checkpoint keeps its text stack.
"""

import torch

TRAINED_LM_PREFIX = "language_model."


def base_key_candidates(trained_key: str) -> tuple[str, ...]:
    """Base-LM names a tiny-audio decoder tensor could be stored under.

    tiny-audio saves the decoder as ``language_model.model.X``. Where ``X``
    lives in the base checkpoint depends on the base architecture:

    - plain CausalLM (Qwen3-0.6B, SmolLM3-3B) -> ``model.X``
    - multimodal wrapper -> ``model.language_model.X``. Qwen/Qwen3.5-2B ships
      as ``Qwen3_5ForConditionalGeneration`` with ``visual.*`` and ``mtp.*``
      towers beside the text stack, which pushes the decoder down one level.

    Both forms are returned, plain-CausalLM first, so callers take the first
    hit and stay architecture-agnostic. Checking only ``model.X`` matched
    nothing on Qwen3.5 and made every layer report 0.0000 drift -- a result
    indistinguishable from a decoder that never trained.

    Returns an empty tuple for non-decoder tensors (``projector.*``), which
    lets callers tell "not an LM tensor" apart from "LM tensor the base
    lacks".
    """
    if not trained_key.startswith(TRAINED_LM_PREFIX):
        return ()
    stripped = trained_key[len(TRAINED_LM_PREFIX) :]
    if stripped.startswith("model."):
        return (stripped, "model.language_model." + stripped[len("model.") :])
    return (stripped,)


def resolve_base_key(trained_key: str, base_weights) -> str | None:
    """First candidate name actually present in `base_weights`, else None."""
    if not base_weights:
        return None
    for key in base_key_candidates(trained_key):
        if key in base_weights:
            return key
    return None


def resolve_base_tensor(trained_key: str, base_weights) -> torch.Tensor | None:
    """The base tensor matching `trained_key`, or None when there is no match."""
    key = resolve_base_key(trained_key, base_weights)
    return None if key is None else base_weights[key]
