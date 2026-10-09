"""Static types shared by the asr_* modules: audio aliases, loader kwargs, protocols.

Typing scaffolding with no behaviour of its own: the `TypedDict`s describe
the keyword arguments ASRModel forwards to `from_pretrained` / `cached_file`,
and the protocol classes describe surfaces with several implementations (or
test fakes) and no common library base that declares them.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeVar

import numpy.typing as npt
import torch
from transformers import BatchFeature, PreTrainedModel
from transformers.generation.utils import GenerateOutput, GenerationMixin

# One waveform (array, tensor or list of samples), or a batch of them.
Waveform = npt.ArrayLike | torch.Tensor
AudioInput = Waveform | Sequence[Waveform]

# A `state_dict(destination=...)` mapping, returned as the caller's own type.
StateDictT = TypeVar("StateDictT", bound=dict[str, Any])


class PreparedChunk(TypedDict):
    """One chunk's encoder inputs, as `ASRPipeline.prepare_chunk` builds them (batch of 1)."""

    input_features: torch.Tensor
    attention_mask: torch.Tensor


if TYPE_CHECKING:

    class GenerativeDecoder(PreTrainedModel, GenerationMixin):
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


class AudioFeatureExtractor(Protocol):
    """A concrete audio feature extractor's `__call__` (e.g. Whisper's).

    `SequenceFeatureExtractor` declares no `__call__`, though every concrete
    extractor has one; cast an extractor to this to call it.
    """

    def __call__(
        self,
        raw_speech: AudioInput,
        /,
        *,
        sampling_rate: int,
        return_attention_mask: bool,
        return_tensors: str,
        padding: bool | str = ...,
        **kwargs: Any,
    ) -> BatchFeature:
        """Featurize raw audio into `input_features` (plus `attention_mask`)."""
        ...


class OutputLengthProjector(Protocol):
    """The projector surface the token-count check needs."""

    def get_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Projector output length for encoder output length `input_length`."""
        ...


class LoadStateDictResult(Protocol):
    """What `load_state_dict` reports back (torch's `_IncompatibleKeys`)."""

    @property
    def missing_keys(self) -> Sequence[str]:
        """Model keys the state dict did not provide."""
        ...

    @property
    def unexpected_keys(self) -> Sequence[str]:
        """State-dict keys the model has no slot for."""
        ...
