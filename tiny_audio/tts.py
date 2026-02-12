"""LLASA-style TTS: a single LLM with expanded vocabulary for text-to-speech.

Architecture:
  Text tokens + xcodec2 speech tokens → SmolLM3-3B (all params trainable)
  → next-token prediction, loss on speech tokens only

The LLM's vocabulary is expanded with 65536 xcodec2 speech tokens (<|s_0|>..<|s_65535|>)
plus 8 special tokens for sequence structure. Training uses standard causal LM loss
with labels masked to -100 for the text portion.

Dataset: minato-ryan/emilia-en-xcodec2 (17.3M samples with pre-computed xcodec2 codes).
"""

import importlib
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_FLASH_ATTN_AVAILABLE = importlib.util.find_spec("flash_attn") is not None

logger = logging.getLogger(__name__)

# xcodec2 constants
XCODEC2_VOCAB_SIZE = 65536
XCODEC2_SAMPLE_RATE = 16000

# Special tokens for sequence structure
SPECIAL_TOKENS = [
    "<|TEXT_UNDERSTANDING_START|>",
    "<|TEXT_UNDERSTANDING_END|>",
    "<|SPEECH_GENERATION_START|>",
    "<|SPEECH_GENERATION_END|>",
    "<|RESERVE_0|>",
    "<|RESERVE_1|>",
    "<|RESERVE_2|>",
    "<|RESERVE_3|>",
]

# Speech tokens: <|s_0|> through <|s_65535|>
SPEECH_TOKENS = [f"<|s_{i}|>" for i in range(XCODEC2_VOCAB_SIZE)]


def setup_tts_model(
    model_id: str = "HuggingFaceTB/SmolLM3-3B",
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: Optional[str] = None,
) -> tuple:
    """Load an LLM and expand its vocabulary with xcodec2 speech tokens.

    Args:
        model_id: HuggingFace model ID for the base LLM.
        dtype: Model dtype.
        checkpoint_path: If provided, load model+tokenizer from this checkpoint
            (already has expanded vocab from Stage1 training). Skips vocab expansion.

    Returns:
        (model, tokenizer, token_ids) where token_ids is a dict mapping
        special token names to their IDs.
    """
    if checkpoint_path:
        # Load from a previous training checkpoint (e.g. Stage1 → Stage2)
        # Tokenizer already has expanded vocab, model already has resized embeddings
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=dtype,
            attn_implementation="flash_attention_2" if _FLASH_ATTN_AVAILABLE else "eager",
            trust_remote_code=True,
        )
        logger.info(f"Loaded TTS model from checkpoint: {checkpoint_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Add special tokens + speech tokens
        num_added = tokenizer.add_tokens(SPECIAL_TOKENS + SPEECH_TOKENS)
        logger.info(
            f"Added {num_added} tokens to tokenizer (8 special + {XCODEC2_VOCAB_SIZE} speech)"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation="flash_attention_2" if _FLASH_ATTN_AVAILABLE else "eager",
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))

    # Build token_ids dict for quick lookup
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS}
    # Speech token offset: ID of <|s_0|>
    token_ids["speech_token_offset"] = tokenizer.convert_tokens_to_ids("<|s_0|>")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"TTS model: {total_params:,} params (all trainable)")

    return model, tokenizer, token_ids


def generate_speech(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    token_ids: dict,
    max_new_tokens: int = 2000,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Optional[list[int]]:
    """Generate xcodec2 speech codes from text.

    Args:
        model: The TTS model with expanded vocabulary.
        tokenizer: Tokenizer with speech tokens added.
        text: Input text to synthesize.
        token_ids: Dict from setup_tts_model with special token IDs.
        max_new_tokens: Maximum speech tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        List of xcodec2 codec codes (0-65535), or None if generation failed.
    """
    text_start = token_ids["<|TEXT_UNDERSTANDING_START|>"]
    text_end = token_ids["<|TEXT_UNDERSTANDING_END|>"]
    speech_start = token_ids["<|SPEECH_GENERATION_START|>"]
    speech_end = token_ids["<|SPEECH_GENERATION_END|>"]
    speech_offset = token_ids["speech_token_offset"]

    # Build input: <|TEXT_START|> text_tokens <|TEXT_END|> <|SPEECH_START|>
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = [text_start] + text_tokens + [text_end, speech_start]
    input_ids = torch.tensor([input_ids], device=model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            eos_token_id=speech_end,
        )

    # Extract generated tokens (skip input)
    generated = output[0, input_ids.shape[1] :].tolist()

    # Remove trailing SPEECH_GENERATION_END if present
    if generated and generated[-1] == speech_end:
        generated = generated[:-1]

    # Convert speech token IDs back to xcodec2 codes
    codes = []
    for tok_id in generated:
        code = tok_id - speech_offset
        if 0 <= code < XCODEC2_VOCAB_SIZE:
            codes.append(code)

    return codes if codes else None
