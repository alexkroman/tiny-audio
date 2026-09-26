"""Load a turn-aware Qwen3-ASR checkpoint and decode transcript + end-of-turn.

A turn-aware checkpoint is a stock `Qwen3ASRForConditionalGeneration` whose
tokenizer carries one extra special token, `<END_OF_TURN>`. The model
transcribes as usual and appends it once the speaker has finished AND enough
trailing silence has been heard; holding the turn is its absence. Nothing
here is needed to *run* such a checkpoint -- plain `generate()` works -- but
`transcribe` also reports the marker's confidence margin, which thresholds
and streaming endpointing are built on.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

END_OF_TURN = "<END_OF_TURN>"
LANGUAGE = "English"
SAMPLE_RATE = 16000
MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"


class Decoded(NamedTuple):
    """One clip's decode: transcript (special tokens stripped), fired, margin."""

    text: str
    fired: bool
    # logit(<END_OF_TURN>) - logit(<|im_end|>) where the transcript ended; NaN
    # if the decode ran out of tokens first. See `marker_margins`.
    margin: float


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_processor(model_id: str = MODEL_ID):
    """Processor with `<END_OF_TURN>` registered (a no-op on a trained checkpoint).

    `processor.end_of_turn_added` records whether the token was new, so
    training that continues from a turn-aware checkpoint leaves its trained
    marker row alone.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    processor.end_of_turn_added = END_OF_TURN not in processor.tokenizer.get_vocab()
    if processor.end_of_turn_added:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [END_OF_TURN]})
    return processor


def load_model(model_id: str = MODEL_ID, dtype: torch.dtype = torch.bfloat16, device=None):
    """Load Qwen3-ASR (base or turn-aware) onto `device` (default: best available).

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


def set_end_of_turn_threshold(generation_config, token_id: int, tau: float) -> None:
    """Make plain `generate()` fire only when the marker margin exceeds `tau`.

    Implemented as `sequence_bias` = -tau on `<END_OF_TURN>`, which
    generation_config.json serialises and `generate()` applies with stock
    transformers -- the threshold ships with the checkpoint, no custom decode
    loop needed. At the step where a transcript ends the competitors are
    `<|im_end|>` and the marker, so greedy then fires iff
    logit(marker) - logit(<|im_end|>) > tau. tau=0 removes the bias.

    Margins reported by `transcribe` are read AFTER built-in processors, so
    on a biased checkpoint they are relative to its shipped threshold.
    """
    keep = [e for e in (generation_config.sequence_bias or []) if list(e[0]) != [token_id]]
    if tau:
        keep.append([[token_id], -float(tau)])
    generation_config.sequence_bias = keep or None


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
def transcribe(
    model,
    processor,
    audios: list[np.ndarray],
    contexts: list[str] | None = None,
    tau: float | None = None,
    max_new_tokens: int = 128,
    language: str = LANGUAGE,
) -> list[Decoded]:
    """Greedy-decode 16 kHz mono clips; one `Decoded` per clip.

    `fired` is whether `<END_OF_TURN>` appeared in the output, or, when `tau`
    is given, whether the marker margin exceeds it (tau=0 matches greedy).
    `contexts` are optional system prompts (e.g. the agent's last question).
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
    recorder = _PairLogitRecorder([token_id, im_end_id])
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        logits_processor=LogitsProcessorList([recorder]),
    )
    new = out[:, inputs["input_ids"].shape[1] :]
    margins = marker_margins(new, torch.stack(recorder.steps, dim=1), token_id, im_end_id)
    fired = (margins > tau) if tau is not None else (new == token_id).any(dim=1)  # NaN never fires
    texts = processor.batch_decode(new, skip_special_tokens=True)
    return [
        Decoded(t.strip(), bool(f), float(m))
        for t, f, m in zip(texts, fired.tolist(), margins.tolist(), strict=True)
    ]
