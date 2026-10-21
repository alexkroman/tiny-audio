"""Qwen3-ASR's audio encoder behind ASRModel's encoder interface.

`Qwen/Qwen3-ASR-1.7B-hf` (Apache-2.0) is #9 on the Open ASR leaderboard (mean
of 9 4.936, 2026-10-20), the best system whose encoder loads natively in our
transformers, and its encoder was trained to feed a Qwen3 decoder.

transformers' `Qwen3ASREncoder` owns its preprocessing contract, which is why
it is used as-is rather than reimplemented:

- its convolutions run on independent 1 s chunks (100 mel frames, 13 tokens
  each) and attention is limited to 8 s windows, both inside `forward`;
- it takes the feature extractor's mel mask and drops padding itself.

Two things differ from what ASRModel expects, and this module bridges them:

- the encoder returns the valid frames of the whole batch packed into one
  `(sum(tokens), d_model)` tensor, which is split back into a padded
  `(batch, tokens, d_model)` tensor here;
- it lives inside the full ASR checkpoint, so only its `model.audio_tower.*`
  tensors are read.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from torch import nn
from transformers.models.qwen3_asr import Qwen3ASRConfig, Qwen3ASREncoder, Qwen3ASREncoderConfig
from transformers.utils.generic import ModelOutput

try:
    from .asr_config import (
        QWEN3_ASR_CHUNK_FRAMES,
        QWEN3_ASR_CONV_LAYERS,
        compute_encoder_output_length,
    )
except ImportError:  # flat layout on the Hub: sibling modules, no package
    from asr_config import (
        QWEN3_ASR_CHUNK_FRAMES,
        QWEN3_ASR_CONV_LAYERS,
        compute_encoder_output_length,
    )

# The encoder's tensors inside the ASR checkpoint.
_WEIGHT_PREFIX = "model.audio_tower."


@dataclass
class Qwen3ASRAudioOutput(ModelOutput):
    """Padded encoder states and the mask of their valid frames."""

    last_hidden_state: torch.Tensor | None = None
    # (batch, tokens), True on valid frames. ASRModel checks the prompt's audio
    # token counts against it, as it does for Granite's own returned mask.
    attention_mask: torch.Tensor | None = None


class Qwen3ASRAudioEncoder(nn.Module):
    """`Qwen3ASREncoder` with its packed output split back into a padded batch."""

    def __init__(self, encoder: Qwen3ASREncoder) -> None:
        """Wrap a loaded encoder; `config` is exposed for encoder_dim detection."""
        super().__init__()
        self.encoder: Qwen3ASREncoder = encoder
        self.config: Qwen3ASREncoderConfig = encoder.config

    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> Qwen3ASRAudioOutput:
        """Encode `(batch, n_mels, frames)` mel features to `(batch, tokens, d_model)`.

        `frames` must be a multiple of 100, which Qwen3-ASR's feature extractor
        guarantees. Without `attention_mask` every frame counts as audio.
        """
        batch = input_features.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones(
                batch, input_features.shape[-1], dtype=torch.long, device=input_features.device
            )
        mask = attention_mask.to(device=input_features.device, dtype=torch.long)
        packed = self.encoder(input_features=input_features, input_features_mask=mask)
        flat = packed.last_hidden_state

        lengths = compute_encoder_output_length(
            mask.sum(dim=-1), QWEN3_ASR_CONV_LAYERS, QWEN3_ASR_CHUNK_FRAMES
        )
        if int(lengths.sum()) != flat.shape[0]:
            msg = (
                f"Qwen3-ASR encoder packed {flat.shape[0]} frames but the chunked length "
                f"formula expects {int(lengths.sum())} ({lengths.tolist()} per clip)."
            )
            raise ValueError(msg)
        rows = list(flat.split(lengths.tolist()))
        hidden = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        valid = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
        return Qwen3ASRAudioOutput(last_hidden_state=hidden, attention_mask=valid)

    @classmethod
    def from_pretrained(
        cls, repo_id: str, dtype: torch.dtype, attn_implementation: str | None
    ) -> "Qwen3ASRAudioEncoder":
        """Build the encoder from `repo_id`'s audio config and load only its tensors.

        Built on CPU, not the meta device: the sinusoidal position table is a
        non-persistent buffer that is absent from the checkpoint, so a meta
        build would leave it without values.
        """
        audio_config = Qwen3ASRConfig.from_pretrained(repo_id).audio_config
        if not isinstance(audio_config, Qwen3ASREncoderConfig):
            msg = f"{repo_id} has no Qwen3-ASR audio_config (got {type(audio_config).__name__})"
            raise TypeError(msg)
        encoder = Qwen3ASREncoder(audio_config)

        state: dict[str, torch.Tensor] = {}
        for path, names in _encoder_shards(repo_id).items():
            with safe_open(path, framework="pt") as f:
                for name in names:
                    state[name.removeprefix(_WEIGHT_PREFIX)] = f.get_tensor(name)
        # strict: a renamed or missing tensor must fail here, not train on random weights.
        encoder.load_state_dict(state, strict=True)
        # PreTrainedModel.to is functools.wraps'd, which pyright cannot bind as a
        # method; it casts in place.
        cast(nn.Module, encoder).to(dtype=dtype)
        # After the cast: transformers warns when FA2 is set on an fp32 model.
        # Its other warning, "not initialized on GPU", is expected here; the
        # Trainer moves the whole ASRModel to the GPU before the first step.
        if attn_implementation:
            encoder.set_attn_implementation(attn_implementation)
        return cls(encoder)


def _encoder_shards(repo_id: str) -> dict[str, list[str]]:
    """Local path of each checkpoint file holding encoder tensors -> those tensor names."""
    try:
        index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    except EntryNotFoundError:  # a single-file checkpoint (the 1.7B and 0.6B are) has no index
        path = hf_hub_download(repo_id, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            keys = cast("list[str]", f.keys())
            names = [k for k in keys if k.startswith(_WEIGHT_PREFIX)]
        return {path: names}
    weight_map: dict[str, str] = json.loads(Path(index_path).read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith(_WEIGHT_PREFIX):
            shards.setdefault(shard, []).append(name)
    return {hf_hub_download(repo_id, shard): names for shard, names in shards.items()}
