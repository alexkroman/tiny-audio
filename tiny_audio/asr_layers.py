"""Module-surgery helpers for ASRModel: encoder unfreezing.

`find_encoder_layer_stack` / `unfreeze_encoder_top_layers` locate an encoder's
transformer-block list and switch its top blocks back to trainable.
"""

import torch.nn as nn


def find_encoder_layer_stack(encoder: nn.Module) -> tuple[str, nn.ModuleList] | None:
    """Locate the encoder's ordered transformer-block list.

    Returns `(attribute_path, ModuleList)` or None. Encoders disagree on the
    attribute name -- Granite Speech uses `layers`, Whisper `encoder.layers`,
    others `blocks` or `encoder_layers` -- so probe the known names rather
    than hardcoding one and silently unfreezing nothing on a new encoder.
    """
    candidates = ("layers", "blocks", "encoder_layers", "encoder.layers", "encoder.blocks")
    for path in candidates:
        try:
            obj = encoder.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return path, obj
    return None


def unfreeze_encoder_top_layers(
    encoder: nn.Module, top_n: int, include_post_projections: bool = True
) -> list[str]:
    """Unfreeze the top `top_n` encoder blocks, optionally with the output projections.

    Assumes the caller has already frozen the whole encoder. Returns the sorted
    names of every parameter switched back on, so callers (and tests) can
    assert on exactly what moved rather than trusting a count.

    The post-stack projections (Granite's `out` / `out_mid`) are included by
    default: they sit between the top block and the projector, so leaving them
    frozen forces the newly-trainable blocks to adapt through a fixed output
    map. That is a reasonable prior, but on this stack it is not what the
    gradient says. Probing the step-2000 granite_qwen_lora checkpoint on real
    audio, `out`/`out_mid` carry 33.57M params -- 23% of a top-4 budget -- for
    an RMS-per-param gradient of 1.14e-05 against 1.4-1.7e-04 in the top
    blocks, i.e. **0.27% of the selection's squared gradient**. Dropping them
    moved the aggregate norm from 1.5840 to 1.5827. Pass False to reclaim the
    parameters; the default stays True so existing recipes are unchanged.

    The pre-stack `input_linear` stays frozen either way -- it is the feature
    front-end, the part most expensive to damage and least specialised to the
    encoder's own CTC head.

    Raises if the layer stack cannot be found or `top_n` exceeds its depth:
    silently unfreezing nothing would produce a run that looks like a
    partial-unfreeze experiment and is actually the frozen baseline.
    """
    if top_n <= 0:
        return []
    found = find_encoder_layer_stack(encoder)
    if found is None:
        msg = (
            f"Could not locate a transformer-block ModuleList on "
            f"{type(encoder).__name__}; cannot unfreeze its top {top_n} layers. "
            f"Children: {[n for n, _ in encoder.named_children()]}"
        )
        raise ValueError(msg)
    path, stack = found
    depth = len(stack)
    if top_n > depth:
        msg = f"encoder_trainable_top_layers={top_n} exceeds the encoder's {depth} blocks"
        raise ValueError(msg)

    unfrozen: list[str] = []
    for idx in range(depth - top_n, depth):
        for name, param in stack[idx].named_parameters():
            param.requires_grad_(True)
            unfrozen.append(f"{path}.{idx}.{name}")
    # Post-stack projections, identified positionally: direct children that are
    # not the stack itself and not the pre-stack input projection.
    for name, child in encoder.named_children() if include_post_projections else ():
        if name == path.split(".")[0] or name.startswith("input"):
            continue
        for pname, param in child.named_parameters():
            param.requires_grad_(True)
            unfrozen.append(f"{name}.{pname}")
    return sorted(unfrozen)
