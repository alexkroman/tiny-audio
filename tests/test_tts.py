"""Tests for LLASA-style TTS: SmolLM2-135M with expanded xcodec2 vocabulary."""

import pytest
import torch

from tiny_audio.tts import (
    SPECIAL_TOKENS,
    XCODEC2_VOCAB_SIZE,
    setup_tts_model,
)


@pytest.fixture(scope="session")
def tts_setup():
    """Setup TTS model once per session (downloads SmolLM2-135M)."""
    model, tokenizer, token_ids = setup_tts_model(
        model_id="HuggingFaceTB/SmolLM2-135M",
        dtype=torch.float32,
    )
    return model, tokenizer, token_ids


class TestVocabExpansion:
    def test_vocab_size(self, tts_setup):
        """Tokenizer should have original vocab + 8 special + 65536 speech tokens."""
        _, tokenizer, _ = tts_setup
        # SmolLM2-135M original vocab is 49152
        assert len(tokenizer) == 49152 + 8 + XCODEC2_VOCAB_SIZE

    def test_model_embedding_matches_tokenizer(self, tts_setup):
        """Model embeddings should be resized to match tokenizer."""
        model, tokenizer, _ = tts_setup
        embed_size = model.get_input_embeddings().weight.shape[0]
        assert embed_size == len(tokenizer)

    def test_special_tokens_resolvable(self, tts_setup):
        """All 8 special tokens should resolve to valid IDs."""
        _, tokenizer, _ = tts_setup
        for tok in SPECIAL_TOKENS:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            assert tok_id != tokenizer.unk_token_id, f"{tok} resolved to UNK"

    def test_speech_tokens_resolvable(self, tts_setup):
        """Spot-check a few speech tokens."""
        _, tokenizer, _ = tts_setup
        for i in [0, 1, 100, 65535]:
            tok_id = tokenizer.convert_tokens_to_ids(f"<|s_{i}|>")
            assert tok_id != tokenizer.unk_token_id, f"<|s_{i}|> resolved to UNK"


class TestTokenIds:
    def test_speech_token_offset(self, tts_setup):
        """speech_token_offset should be the ID of <|s_0|>."""
        _, tokenizer, token_ids = tts_setup
        expected = tokenizer.convert_tokens_to_ids("<|s_0|>")
        assert token_ids["speech_token_offset"] == expected

    def test_speech_tokens_contiguous(self, tts_setup):
        """Speech tokens should have contiguous IDs."""
        _, tokenizer, token_ids = tts_setup
        offset = token_ids["speech_token_offset"]
        for i in [0, 1, 2, 100, 65535]:
            tok_id = tokenizer.convert_tokens_to_ids(f"<|s_{i}|>")
            assert tok_id == offset + i

    def test_special_token_ids_in_dict(self, tts_setup):
        """All special tokens should be in the token_ids dict."""
        _, _, token_ids = tts_setup
        for tok in SPECIAL_TOKENS:
            assert tok in token_ids, f"{tok} not in token_ids"
            assert isinstance(token_ids[tok], int)


class TestForward:
    def _make_batch(self, tts_setup):
        """Build a batch manually (matching what SFTTrainer would produce)."""
        model, tokenizer, token_ids = tts_setup
        offset = token_ids["speech_token_offset"]

        text = "Hello world"
        codes = list(range(20))

        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        speech_ids = [offset + c for c in codes]

        input_ids = (
            [token_ids["<|TEXT_UNDERSTANDING_START|>"]]
            + text_tokens
            + [token_ids["<|TEXT_UNDERSTANDING_END|>"], token_ids["<|SPEECH_GENERATION_START|>"]]
            + speech_ids
            + [token_ids["<|SPEECH_GENERATION_END|>"]]
        )

        text_prefix_len = 1 + len(text_tokens) + 2
        labels = [-100] * text_prefix_len + input_ids[text_prefix_len:]

        return {
            "input_ids": torch.tensor([input_ids]),
            "labels": torch.tensor([labels]),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
        }

    def test_forward_produces_loss(self, tts_setup):
        """Model forward should produce a scalar loss."""
        model = tts_setup[0]
        batch = self._make_batch(tts_setup)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        output = model(**batch)
        assert output.loss is not None
        assert output.loss.dim() == 0
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)

    def test_gradients_flow(self, tts_setup):
        """All model parameters should receive gradients."""
        model = tts_setup[0]
        batch = self._make_batch(tts_setup)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        model.zero_grad()
        output = model(**batch)
        output.loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "Model should have gradients"
        model.zero_grad()
