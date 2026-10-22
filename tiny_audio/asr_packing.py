"""Padding-free packing of a labelled decoder batch for FlashAttention's varlen kernels.

Training batches are padded to their longest row, and the clips run 0.1-30 s,
so a large share of every step's decoder compute (attention AND MLP) went to
padding. Packing keeps only the real tokens: the batch's rows are concatenated
into one `(1, total_tokens)` sequence, and FlashAttention's variable-length
kernels keep each row's attention inside its own segment via the cumulative
sequence lengths passed alongside.

The batch's rows, their order and the per-token loss are unchanged:

- each row's positions restart at 0. Training batches are left-padded, so
  padded rows used to start at an offset; RoPE scores depend only on relative
  position, and 0-based positions are what `generate()` uses at inference;
- the first token of every row gets label -100, so no row is trained to
  predict its first token from the previous row's last one. Every row starts
  with a prompt token, which is already unlabelled, so this drops nothing; in
  the padded layout such a target would have been predicted from a fully
  masked pad position anyway.

Only valid under flash_attention_2: SDPA and eager would see one unmasked
sequence and let rows attend to each other. ASRModel checks before packing.
"""

from dataclasses import dataclass

import torch


@dataclass
class PackedBatch:
    """One flattened sequence and the varlen metadata FlashAttention needs."""

    inputs_embeds: torch.Tensor  # (1, total_tokens, hidden)
    labels: torch.Tensor  # (1, total_tokens)
    position_ids: torch.Tensor  # (1, total_tokens), restarting at 0 per row
    cu_seq_lens: torch.Tensor  # (rows + 1,) int32 row boundaries
    max_length: int


def pack_rows(
    inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
) -> PackedBatch:
    """Concatenate the real tokens of a padded `(rows, length)` batch into one sequence."""
    valid = attention_mask.bool()
    lengths = valid.sum(dim=-1)
    position_ids = (valid.long().cumsum(dim=-1) - 1)[valid].unsqueeze(0)
    packed_labels = labels[valid].clone()
    starts = torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0)[:-1]])
    packed_labels[starts[lengths > 0]] = -100
    cu_seq_lens = torch.nn.functional.pad(lengths.cumsum(dim=0), (1, 0)).to(torch.int32)
    return PackedBatch(
        inputs_embeds=inputs_embeds[valid].unsqueeze(0),
        labels=packed_labels.unsqueeze(0),
        position_ids=position_ids,
        cu_seq_lens=cu_seq_lens,
        max_length=int(lengths.max()),
    )
