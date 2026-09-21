#!/usr/bin/env python3
"""Map tiny-audio tensor names onto their base-checkpoint counterparts.

Shared by analyze_weights and compare_to_base so the two cannot disagree about
where a base checkpoint keeps its text stack -- or, since encoder support was
added, its audio stack.
"""

import torch

TRAINED_LM_PREFIX = "language_model."
TRAINED_ENCODER_PREFIX = "audio_tower."

# Where a base checkpoint keeps the audio encoder that tiny-audio saves flat
# under ``audio_tower.``. Verified against the shipped headers:
#   ibm-granite/granite-speech-5.0-470m-turboctc -> ``encoder.`` (1,100
#     tensors, all under that one prefix; GraniteSpeech5Encoder.from_pretrained
#     pulls the encoder out of the ForCTC checkpoint and drops the CTC head)
#   openai/whisper-*                             -> ``model.encoder.``
#   GLM-ASR                                      -> ``audio_tower.`` flat, or
#     ``model.audio_tower.`` under the 5.x composite layout
# The empty string covers a base published as a bare encoder. Ordered most
# specific first so a checkpoint carrying both ``encoder.`` and a bare copy
# resolves to the qualified name.
_ENCODER_BASE_PREFIXES = (
    "encoder.",
    "model.encoder.",
    "audio_tower.",
    "model.audio_tower.",
    "",
)


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


def encoder_base_key_candidates(trained_key: str) -> tuple[str, ...]:
    """Base names a tiny-audio ``audio_tower.*`` tensor could be stored under.

    Mirrors `base_key_candidates` for the encoder half. Without this, every
    encoder tensor fell through `compare_to_base`'s
    ``if not base_key_candidates(tk): continue`` guard and the tool reported
    "Matched tensors: 0" rather than failing -- so the encoder-drift check
    that granite_qwen_full prescribes had no instrument behind it.
    """
    if not trained_key.startswith(TRAINED_ENCODER_PREFIX):
        return ()
    stripped = trained_key[len(TRAINED_ENCODER_PREFIX) :]
    return tuple(prefix + stripped for prefix in _ENCODER_BASE_PREFIXES)


def all_base_key_candidates(trained_key: str) -> tuple[str, ...]:
    """Decoder and encoder candidates together.

    The two are disjoint by construction -- each keys off a different trained
    prefix -- so callers that handle both can use this and callers that only
    ever see one half are unaffected.
    """
    return base_key_candidates(trained_key) or encoder_base_key_candidates(trained_key)


def resolve_base_key(trained_key: str, base_weights) -> str | None:
    """First candidate name actually present in `base_weights`, else None."""
    if not base_weights:
        return None
    for key in all_base_key_candidates(trained_key):
        if key in base_weights:
            return key
    return None


def resolve_base_tensor(trained_key: str, base_weights) -> torch.Tensor | None:
    """The base tensor matching `trained_key`, or None when there is no match."""
    key = resolve_base_key(trained_key, base_weights)
    return None if key is None else base_weights[key]
