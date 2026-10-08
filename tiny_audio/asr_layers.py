"""Module-surgery helpers for ASRModel: MPS-safe embeddings and encoder unfreezing.

Two self-contained groups of `nn.Module` utilities that ASRModel applies to its
loaded sub-models:

- `mps_unsafe_parameters` / `ChunkedEmbedding` / `chunk_oversized_embeddings`
  work around PyTorch's 32-bit indexing limit in its Metal kernels.
- `find_encoder_layer_stack` / `unfreeze_encoder_top_layers` locate an
  encoder's transformer-block list and switch its top blocks back to trainable.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional

# PyTorch's Metal kernels index tensor storage with 32-bit offsets, so a gather
# into a tensor holding more than INT32_MAX elements wraps around and silently
# returns the wrong rows -- no error, no warning, just corrupt values.
MPS_MAX_TENSOR_ELEMENTS = 2**31 - 1


def mps_unsafe_parameters(model: nn.Module) -> list[tuple[str, int]]:
    """Parameters too large for MPS to index correctly, as (name, numel) pairs.

    Gemma 4 E2B trips this on `embed_tokens_per_layer`: its per-layer embedding
    table is 262144 x (35*256) = 2,348,810,240 elements, 201M past INT32_MAX.
    On MPS the lookup returns garbage, which Gemma then adds into all 35 layers
    at every position. The failure is entirely silent -- the model loads, runs,
    and emits fluent, confident, wrong transcripts. Measured on this stack:
    teacher-forced loss 0.14 on CPU versus 3.80 on MPS from identical weights.

    Returns an empty list for models that are safe, so callers can treat a
    non-empty result as "this model needs `chunk_oversized_embeddings`".
    """
    return [
        (name, param.numel())
        for name, param in model.named_parameters()
        if param.numel() > MPS_MAX_TENSOR_ELEMENTS
    ]


class ChunkedEmbedding(nn.Module):
    """Drop-in `nn.Embedding` that splits its table to stay within 32-bit indexing.

    The overflow lives in the kernel's arithmetic over the weight's *storage*,
    so slicing is not enough -- views share that storage and stay broken. Each
    chunk therefore has to be its own contiguous allocation. Gathering per
    chunk and concatenating on the feature axis reproduces the full gather
    exactly: verified bit-identical to CPU on MPS, where the single-tensor
    lookup was off by up to 3.16.

    Wraps rather than replaces the semantics of `Gemma4TextScaledWordEmbedding`,
    whose forward multiplies the lookup by `embed_scale` -- dropping that would
    shrink every per-layer embedding by sqrt(hidden_size_per_layer_input).
    """

    def __init__(self, embedding: nn.Embedding) -> None:
        """Split `embedding`'s table into MPS-safe column chunks, keeping its metadata."""
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        # Gemma 4's scaled embedding carries this; a plain nn.Embedding does not.
        self.embed_scale = getattr(embedding, "embed_scale", None)

        num_chunks = math.ceil(embedding.weight.numel() / MPS_MAX_TENSOR_ELEMENTS)
        device = embedding.weight.device
        # Chunk from a CPU copy: carving contiguous pieces out of an
        # already-oversized MPS tensor would run the same overflowing kernel
        # this class exists to avoid.
        source = embedding.weight.detach().cpu()
        self.chunks = nn.ParameterList(
            [
                nn.Parameter(
                    chunk.contiguous().to(device), requires_grad=embedding.weight.requires_grad
                )
                for chunk in torch.chunk(source, num_chunks, dim=1)
            ]
        )
        del source

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up `input_ids` in every chunk and concatenate along the feature dim."""
        out = torch.cat([functional.embedding(input_ids, chunk) for chunk in self.chunks], dim=-1)
        if self.embed_scale is not None:
            out = out * self.embed_scale.to(out.dtype)
        return out

    @property
    def weight(self) -> torch.Tensor:
        """Reassembled table, for callers that introspect `.weight`.

        Materializes the full tensor, so this is a debugging/serialization
        convenience rather than something to call on a hot path.
        """
        return torch.cat(list(self.chunks), dim=1)

    def extra_repr(self) -> str:
        """Describe the table like `nn.Embedding` does, plus the chunk count."""
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"chunks={len(self.chunks)} (MPS int32-indexing workaround)"
        )


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


def chunk_oversized_embeddings(root: nn.Module) -> list[str]:
    """Replace every embedding MPS cannot index with a `ChunkedEmbedding`.

    Returns the qualified names that were replaced, empty when nothing needed
    it. Idempotent: `ChunkedEmbedding` is not an `nn.Embedding`, so a second
    pass finds nothing.
    """
    replaced: list[str] = []
    for module_name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Embedding) and child.weight.numel() > MPS_MAX_TENSOR_ELEMENTS:
                setattr(module, child_name, ChunkedEmbedding(child))
                replaced.append(f"{module_name}.{child_name}" if module_name else child_name)
    return replaced
