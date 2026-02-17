"""LLASA-style TTS: a single LLM with expanded vocabulary for text-to-speech.

Architecture:
  Text tokens + speech tokens → SmolLM3-3B (all params trainable)
  → next-token prediction, loss on speech tokens only

The LLM's vocabulary is expanded with 65536 speech tokens (<|s_0|>..<|s_65535|>)
plus 2 special tokens for audio delimiters (<|audio_start|>, <|audio_end|>).
Chat structure uses SmolLM3's native ChatML via apply_chat_template. Training uses
trl's get_training_chat_template for prefix-preserving templates. Standard causal LM
loss with labels masked to -100 for the prompt portion.

Dataset: neuphonic/emilia-yodas-english-neucodec (30.6M samples with pre-computed
NeuCodec codes, 65536 vocab FSQ).
"""

import importlib
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_ATTN_IMPL = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"

# NeuCodec constants (65536 vocab FSQ, same format as xcodec2)
CODEC_VOCAB_SIZE = 65536
CODEC_SAMPLE_RATE = 16000

# Only 2 new tokens added to the vocabulary for audio delimiters.
# Chat structure reuses SmolLM3's native ChatML tokens (already in base vocab):
#   <|im_start|>, <|im_end|>
SPECIAL_TOKENS = [
    "<|audio_start|>",
    "<|audio_end|>",
]

# Instruction prefix for the speak direction (user message)
TTS_PREFIX = "Convert the text to speech: "

# Speech tokens: <|s_0|> through <|s_65535|>
SPEECH_TOKENS = [f"<|s_{i}|>" for i in range(CODEC_VOCAB_SIZE)]


def setup_tts_model(
    model_id: str = "HuggingFaceTB/SmolLM3-3B",
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: Optional[str] = None,
) -> tuple:
    """Load an LLM and expand its vocabulary with codec speech tokens.

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
        # Load from a previous training checkpoint (e.g. Stage1 → Stage2).
        # The checkpoint may only have embed_tokens.weight but config may say
        # tie_word_embeddings=False (set by resize_token_embeddings). Force tied
        # loading so embed_tokens weight is shared with lm_head.
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=_ATTN_IMPL,
            tie_word_embeddings=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        logger.info(f"Loaded TTS model from checkpoint: {checkpoint_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Add special tokens + speech tokens (only 2 new special tokens)
        num_added = tokenizer.add_tokens(SPECIAL_TOKENS + SPEECH_TOKENS)
        logger.info(
            f"Added {num_added} tokens to tokenizer ({len(SPECIAL_TOKENS)} special + {CODEC_VOCAB_SIZE} speech)"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=_ATTN_IMPL,
        )
        # Resize embeddings — new tokens are initialized from a multivariate normal
        # fitted to existing embeddings (mean_resizing=True, the default). This gives
        # each new token a unique in-distribution embedding rather than collapsing all
        # to the same mean vector. See: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        orig_vocab_size = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        logger.info(
            f"Initialized {len(tokenizer) - orig_vocab_size} new embeddings via multivariate normal"
        )

    # Build token_ids dict for quick lookup — includes both new and native tokens
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS}
    # Speech token offset: ID of <|s_0|>
    token_ids["speech_token_offset"] = tokenizer.convert_tokens_to_ids("<|s_0|>")
    # Native ChatML tokens (looked up from existing vocab, not added)
    token_ids["<|im_start|>"] = tokenizer.convert_tokens_to_ids("<|im_start|>")
    token_ids["<|im_end|>"] = tokenizer.convert_tokens_to_ids("<|im_end|>")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"TTS model: {total_params:,} params (all trainable)")

    return model, tokenizer, token_ids


def codes_to_speech_text(codes: list[int]) -> str:
    """Convert codec codes to their text representation for chat templates.

    Returns a string like: <|audio_start|><|s_0|><|s_42|>...<|audio_end|>
    These tokens are in the expanded vocabulary and tokenize correctly through
    apply_chat_template.
    """
    return "<|audio_start|>" + "".join(f"<|s_{c}|>" for c in codes) + "<|audio_end|>"


def generate_speech(
    model,
    tokenizer,
    text: str,
    token_ids: dict,
    max_new_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> Optional[list[int]]:
    """Generate speech codes from text using apply_chat_template.

    Uses apply_chat_template to build the prompt, then appends <|audio_start|>
    to begin speech generation.

    Args:
        model: The TTS model with expanded vocabulary.
        tokenizer: Tokenizer with speech tokens added.
        text: Input text to synthesize.
        token_ids: Dict from setup_tts_model with special token IDs.
        max_new_tokens: Maximum speech tokens to generate.
        temperature: Sampling temperature (0 for greedy).
        top_p: Nucleus sampling threshold.

    Returns:
        List of codec codes (0-65535), or None if generation failed.
    """
    speech_start = token_ids["<|audio_start|>"]
    speech_end = token_ids["<|audio_end|>"]
    speech_offset = token_ids["speech_token_offset"]

    # Build ChatML input using apply_chat_template
    # Then append <|audio_start|> to begin speech generation.
    messages = [{"role": "user", "content": TTS_PREFIX + text}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    input_ids = prompt_ids + [speech_start]
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

    # Remove trailing audio_end if present
    if generated and generated[-1] == speech_end:
        generated = generated[:-1]

    # Convert speech token IDs back to codec codes
    codes = []
    for tok_id in generated:
        code = tok_id - speech_offset
        if 0 <= code < CODEC_VOCAB_SIZE:
            codes.append(code)

    return codes if codes else None
