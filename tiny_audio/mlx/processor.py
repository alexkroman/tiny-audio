"""Audio-token counting + Qwen3 chat-template prompt construction for MLX inference."""

from __future__ import annotations

import numpy as np

from tiny_audio.asr_config import compute_encoder_output_length

AUDIO_TOKEN = "<audio>"
TRANSCRIBE_PROMPT = "Transcribe the speech to text"


def compute_num_audio_tokens(
    audio_len: int,
    encoder_conv_layers: list[tuple[int, int, int]],
    projector,
    hop_length: int = 160,
) -> int:
    """Compute how many <audio> placeholder tokens the prompt should contain.

    Pipeline: audio_len -> mel frames (audio_len // hop_length) -> encoder frames
    (conv formula) -> projector output length.
    """
    mel_len = audio_len // hop_length
    encoder_len = compute_encoder_output_length(mel_len, encoder_conv_layers)
    return projector.get_output_length(encoder_len)


def build_prompt_input_ids(
    tokenizer,
    num_audio_tokens: int,
    system_prompt: str = "",
) -> np.ndarray:
    """Render the chat-template prompt with N <audio> placeholders.

    Returns input_ids as a numpy [1, T] int64 array - same format as the PT
    inference pipeline (see asr_modeling.py:580-590).

    Qwen3 ships a chat template that supports `enable_thinking=False` to suppress
    `<think>...</think>` reasoning blocks. ASR responses must use enable_thinking=False
    to produce plain transcripts.
    """
    audio_placeholder = AUDIO_TOKEN * num_audio_tokens
    user_content = audio_placeholder
    if TRANSCRIBE_PROMPT:
        user_content += " " + TRANSCRIBE_PROMPT

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="np",
        enable_thinking=False,
    )
    # apply_chat_template can return either a raw array or a BatchEncoding
    # (dict-like). Normalize to a numpy [1, T] int64 array.
    if isinstance(out, np.ndarray):
        ids = out
    elif hasattr(out, "input_ids"):
        ids = np.asarray(out["input_ids"])
    else:
        ids = np.asarray(out)
    if ids.ndim == 1:
        ids = ids[None, :]
    return ids.astype(np.int64)
