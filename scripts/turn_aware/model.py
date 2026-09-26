"""Qwen3-ASR loading, the end-of-turn token, LoRA, and batched greedy decode.

Uses the transformers-native port (`Qwen/Qwen3-ASR-0.6B-hf`), so no extra
package is needed. Its embeddings are TIED, which is why one trainable row
(PEFT `trainable_token_indices` on `embed_tokens`) is enough: PEFT wraps
`lm_head` with the same delta, so the marker's input embedding and output
logit train together.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from scripts.turn_aware.data import END_OF_TURN, LANGUAGE

MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"

# Decoder attention only -- the audio tower also names its projections
# q_proj/k_proj/..., so the pattern is anchored on `language_model`.
LORA_TARGET_PATTERN = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_processor(model_id: str = MODEL_ID):
    """Processor with `<END_OF_TURN>` registered (a no-op on a trained checkpoint)."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    if END_OF_TURN not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [END_OF_TURN]})
    return processor


def load_model(model_id: str = MODEL_ID, dtype: torch.dtype = torch.bfloat16, device=None):
    """Load Qwen3-ASR for training or decode.

    On MPS attention falls back to eager: Metal sdpa returns NaN for fully
    masked rows, and every left-padded batch has them.
    """
    from transformers import Qwen3ASRForConditionalGeneration

    device = device or pick_device()
    attn = "eager" if device.type == "mps" else "sdpa"
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, attn_implementation=attn
    )
    return model.to(device)


def add_end_of_turn_token(model, processor, init_std: float = 0.02, seed: int = 0) -> int:
    """Initialise the marker's embedding row and return its id.

    The tokenizer's next free id (151705) already falls inside the checkpoint's
    151,936 reserved rows, so no resize is needed. That id is also the
    config's `timestamp_token_id`, which only the forced-aligner checkpoint's
    post-processing reads; plain ASR generation never uses it. The row starts
    at the vocabulary mean plus small noise: a reserved row's arbitrary
    contents would otherwise set the marker's initial logit.
    """
    token_id = processor.tokenizer.convert_tokens_to_ids(END_OF_TURN)
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


class _PairLogitRecorder(LogitsProcessor):
    """LogitsProcessor that copies out two token columns at every decode step.

    Keeping only (end_of_turn, im_end) per step avoids `output_logits=True`,
    which would hold steps x batch x 151,936 floats (~2.5 GB at batch 32).
    """

    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids
        self.steps: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.steps.append(scores[:, self.token_ids].detach().float())
        return scores


def marker_margins(
    new_ids: torch.Tensor, pair_logits: torch.Tensor, eot_id: int, im_end_id: int
) -> torch.Tensor:
    """logit(<END_OF_TURN>) - logit(<|im_end|>) at the step where the transcript ends.

    That step is the first one that emitted either token: greedy decoding fires
    exactly when this margin is > 0, so thresholding it at tau > 0 trades recall
    for precision without re-decoding. NaN when neither token was emitted (the
    decode ran out of `max_new_tokens`).

    Args:
        new_ids: (B, T) generated ids.
        pair_logits: (B, T, 2) recorded [eot, im_end] logits per step.
    """
    hit = (new_ids == eot_id) | (new_ids == im_end_id)
    first = hit.int().argmax(dim=1)
    rows = torch.arange(new_ids.shape[0], device=new_ids.device)
    at = pair_logits[rows, first]
    margin = at[:, 0] - at[:, 1]
    return torch.where(hit.any(dim=1), margin, torch.full_like(margin, float("nan")))


@torch.inference_mode()
def decode_batch(
    model,
    processor,
    audios: list[np.ndarray],
    contexts: list[str] | None = None,
    max_new_tokens: int = 128,
    language: str = LANGUAGE,
    return_margin: bool = False,
    tau: float | None = None,
) -> list[tuple]:
    """Greedy-decode a batch; return (transcript, fired[, margin]) per clip.

    `fired` is whether `<END_OF_TURN>` appears anywhere in the output, or,
    when `tau` is given, whether the marker margin exceeds it (tau=0 matches
    greedy). The transcript has every special token stripped. With
    `return_margin`, each tuple also carries `marker_margins` for the clip.
    """
    tokenizer = processor.tokenizer
    token_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    prompts = contexts if contexts and any(contexts) else None
    inputs = processor.apply_transcription_request(audios, language=language, prompt=prompts)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {
        k: v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device)
        for k, v in inputs.items()
    }
    need_margin = return_margin or tau is not None
    recorder = _PairLogitRecorder([token_id, im_end_id]) if need_margin else None
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        logits_processor=LogitsProcessorList([recorder]) if recorder else None,
    )
    new = out[:, inputs["input_ids"].shape[1] :]
    fired = (new == token_id).any(dim=1).tolist()
    texts = [t.strip() for t in processor.batch_decode(new, skip_special_tokens=True)]
    if not recorder:
        return [(t, bool(f)) for t, f in zip(texts, fired, strict=True)]
    margins = marker_margins(new, torch.stack(recorder.steps, dim=1), token_id, im_end_id)
    if tau is not None:
        fired = (margins > tau).tolist()  # NaN > tau is False
    if not return_margin:
        return [(t, bool(f)) for t, f in zip(texts, fired, strict=True)]
    return [(t, bool(f), float(m)) for t, f, m in zip(texts, fired, margins.tolist(), strict=True)]
