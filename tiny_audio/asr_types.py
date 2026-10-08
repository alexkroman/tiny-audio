"""Static types shared by ASRModel's loaders: loader kwargs and decoder protocols.

Pure typing scaffolding with no behaviour of its own: the `TypedDict`s describe
the keyword arguments ASRModel forwards to `from_pretrained` / `cached_file`,
and the protocol classes describe the decoder surfaces it calls.
"""

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerateOutput, GenerationMixin

if TYPE_CHECKING:

    class GenerativeDecoder(PreTrainedModel, GenerationMixin):  # type: ignore[no-untyped-call]
        """Static view of the decoder: a PreTrainedModel that can generate.

        transformers annotates GenerationMixin's `self` with a protocol
        (`GenerativePreTrainedModel`) that no PreTrainedModel satisfies -- it
        asks for a mutable `device` where the model has a read-only property --
        so neither checker can bind `generate` on a real model. This restates
        the two generation entry points ASRModel calls without that bound.
        Type-checking only; the runtime object is whatever the loader returned.
        """

        def generate(self, *args: Any, **kwargs: Any) -> GenerateOutput | torch.LongTensor:
            """`GenerationMixin.generate`."""
            ...

        def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            """`GenerationMixin.prepare_inputs_for_generation`."""
            ...


class LoadKwargs(TypedDict):
    """Loader arguments shared by the encoder and decoder `from_pretrained` calls."""

    attn_implementation: str | None
    low_cpu_mem_usage: bool
    dtype: torch.dtype


class DecoderLoadKwargs(LoadKwargs):
    """Decoder loader arguments: the shared ones plus remote-code opt-in."""

    trust_remote_code: bool


class HubFileKwargs(TypedDict, total=False):
    """Hub location arguments forwarded to `cached_file` and PEFT's loader."""

    subfolder: str
    revision: str


class PerLayerInputsTextModel(Protocol):
    """A Gemma 4 style text model, which builds its per-layer inputs (PLE) itself."""

    config: PretrainedConfig

    def get_per_layer_inputs(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor | None
    ) -> torch.Tensor:
        """Per-layer embeddings for `input_ids` (or recovered from `inputs_embeds`)."""
        ...
