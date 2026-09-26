"""Training-only model setup: the marker row's init and LoRA.

Loading, decoding and thresholds are runtime concerns and live in
`tiny_audio.turns`. Qwen3-ASR's embeddings are TIED, which is why one
trainable row (PEFT `trainable_token_indices` on `embed_tokens`) is enough:
PEFT wraps `lm_head` with the same delta, so the marker's input embedding and
output logit train together.
"""

from __future__ import annotations

import torch

from tiny_audio.turns import END_OF_TURN

# Decoder attention only -- the audio tower also names its projections
# q_proj/k_proj/..., so the pattern is anchored on `language_model`.
LORA_TARGET_PATTERN = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"


def add_end_of_turn_token(
    model, processor, *, init_row: bool, init_std: float = 0.02, seed: int = 0
) -> int:
    """Return the marker's id, initialising its embedding row when `init_row`.

    Pass `register_end_of_turn(processor)`'s result: True for a base model
    (new token), False when continuing from a turn-aware checkpoint, whose
    TRAINED row must survive. Deliberately no default -- guessing True would
    silently wipe a trained marker.

    The tokenizer's next free id (151705) already falls inside the checkpoint's
    151,936 reserved rows, so no resize is needed. That id is also the
    config's `timestamp_token_id`, which only the forced-aligner checkpoint's
    post-processing reads; plain ASR generation never uses it. The row starts
    at the vocabulary mean plus small noise: a reserved row's arbitrary
    contents would otherwise set the marker's initial logit.
    """
    token_id = processor.tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if not init_row:
        return token_id
    embed = model.get_input_embeddings().weight
    if token_id >= embed.shape[0]:
        model.resize_token_embeddings(len(processor.tokenizer))
        embed = model.get_input_embeddings().weight
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mean = embed.float().mean(dim=0)
        noise = torch.randn(mean.shape, generator=generator) * init_std
        embed[token_id] = (mean + noise.to(mean.device)).to(embed.dtype)
    return token_id


def apply_lora(model, token_id: int, rank: int, alpha: int, dropout: float):
    """LoRA on decoder attention + a trainable marker row; everything else frozen."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_PATTERN,
        trainable_token_indices={"embed_tokens": [token_id]},
        bias="none",
    )
    return get_peft_model(model, config)
