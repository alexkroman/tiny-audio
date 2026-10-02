"""Speaker tokens, LoRA, and a decode that keeps the speaker tokens.

Qwen3-ASR's embeddings are TIED, so one trainable row per speaker token (PEFT
`trainable_token_indices` on `embed_tokens`) trains its input embedding and
its output logit together.
"""

from __future__ import annotations

import numpy as np
import torch

from scripts.speaker_asr.data import SPEAKER_TOKEN, speaker_tokens

LANGUAGE = "English"


def register_speaker_tokens(processor, n: int) -> bool:
    """Add `<SPK_1>`..`<SPK_n>` to the tokenizer if missing; True when any was new.

    `add_tokens(special_tokens=True)`, not `add_special_tokens`: the latter
    rewrites `additional_special_tokens`, which already holds Qwen3-ASR's own
    `<asr_text>` and friends.
    """
    vocab = processor.tokenizer.get_vocab()
    missing = [t for t in speaker_tokens(n) if t not in vocab]
    if missing:
        processor.tokenizer.add_tokens(missing, special_tokens=True)
    return bool(missing)


def speaker_token_ids(processor, n: int) -> list[int]:
    return processor.tokenizer.convert_tokens_to_ids(speaker_tokens(n))


def init_speaker_rows(model, token_ids: list[int], init_std: float = 0.02, seed: int = 0):
    """Start each new speaker row at the vocabulary mean plus small, distinct noise.

    The tokenizer's next free ids (151705+) fall inside the checkpoint's
    151,936 reserved rows, whose arbitrary contents would otherwise set the
    tokens' initial logits; distinct noise keeps the rows from being
    interchangeable at step 0.
    """
    embed = model.get_input_embeddings().weight
    if max(token_ids) >= embed.shape[0]:
        model.resize_token_embeddings(max(token_ids) + 1)
        embed = model.get_input_embeddings().weight
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mean = embed.float().mean(dim=0)
        for token_id in token_ids:
            noise = torch.randn(mean.shape, generator=generator) * init_std
            embed[token_id] = (mean + noise.to(mean.device)).to(embed.dtype)


def apply_lora(model, token_ids: list[int], target_modules: str, rank: int, alpha: int, dropout):
    """LoRA on `target_modules` (a regex) + trainable speaker rows; all else frozen."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        trainable_token_indices={"embed_tokens": list(token_ids)},
        bias="none",
    )
    return get_peft_model(model, config)


def split_at_speakers(ids: list[int], speaker_ids: dict[int, int], decode) -> str:
    """Generated ids -> '<SPK_1>text<SPK_2>text', other special tokens stripped.

    Speaker tokens are special (so the tokenizer never splits them), which
    means `skip_special_tokens` would also drop them; instead each run of
    text between speaker tokens is decoded on its own.
    """
    out, chunk = [], []
    for token in ids:
        if token in speaker_ids:
            out.append(decode(chunk).strip())
            out.append(SPEAKER_TOKEN.format(speaker_ids[token]))
            chunk = []
        else:
            chunk.append(token)
    out.append(decode(chunk).strip())
    return "".join(out)


@torch.inference_mode()
def transcribe_speakers(
    model,
    processor,
    audios: list[np.ndarray],
    n_speakers: int,
    max_new_tokens: int = 320,
    language: str = LANGUAGE,
) -> list[str]:
    """Greedy-decode 16 kHz mono windows into serialized speaker transcripts."""
    inputs = processor.apply_transcription_request(audios, language=language)
    inputs = {
        k: (
            v.to(device=model.device, dtype=model.dtype)
            if v.is_floating_point()
            else v.to(model.device)
        )
        for k, v in inputs.items()
    }
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    new = out[:, inputs["input_ids"].shape[1] :].tolist()
    ids = {tid: i for i, tid in enumerate(speaker_token_ids(processor, n_speakers), 1)}

    def decode(chunk: list[int]) -> str:
        return processor.tokenizer.decode(chunk, skip_special_tokens=True)

    return [split_at_speakers(row, ids, decode) for row in new]


def predict_rows(
    model,
    processor,
    dataset,
    n_speakers: int,
    batch_size: int,
    max_new_tokens: int = 320,
    progress: bool = False,
) -> list[str]:
    """Decode every row of a SpeakerASRDataset, longest-first so batches pad evenly."""
    order = sorted(range(len(dataset)), key=lambda i: -dataset.rows[i]["duration_s"])
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if progress:
        from rich.progress import track

        batches = track(batches, description="decoding")
    preds: list[str] = [""] * len(order)
    for idx in batches:
        audios = [dataset.audio(i) for i in idx]
        texts = transcribe_speakers(model, processor, audios, n_speakers, max_new_tokens)
        for i, text in zip(idx, texts, strict=True):
            preds[i] = text
    return preds
