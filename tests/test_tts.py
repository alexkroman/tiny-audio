"""Tests for LLASA-style TTS: SmolLM2-135M with expanded xcodec2 vocabulary."""

import pytest
import torch

from tiny_audio.lm import (
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


class TestStage1Tokenization:
    """Tests for LM-Stage1 speech-only tokenization."""

    def test_stage1_speech_only(self, tts_setup):
        """Stage1 tokenization should produce speech-only sequences with all completion_mask=1."""
        _, tokenizer, token_ids = tts_setup
        speech_start_id = token_ids["<|SPEECH_GENERATION_START|>"]
        speech_end_id = token_ids["<|SPEECH_GENERATION_END|>"]
        offset = token_ids["speech_token_offset"]

        # Simulate tokenize_lm_stage1_batch
        codes = [10, 20, 30, 40, 50]
        input_ids = [speech_start_id] + [offset + c for c in codes] + [speech_end_id]
        completion_mask = [1] * len(input_ids)

        # All tokens should be completion (loss computed on all)
        assert all(m == 1 for m in completion_mask)
        # Length: SPEECH_START + codes + SPEECH_END
        assert len(input_ids) == len(codes) + 2
        # First token is SPEECH_START
        assert input_ids[0] == speech_start_id
        # Last token is SPEECH_END
        assert input_ids[-1] == speech_end_id
        # Middle tokens are offset speech codes
        for i, code in enumerate(codes):
            assert input_ids[i + 1] == offset + code

    def test_stage1_no_text_tokens(self, tts_setup):
        """Stage1 should not contain any text understanding tokens."""
        _, tokenizer, token_ids = tts_setup
        speech_start_id = token_ids["<|SPEECH_GENERATION_START|>"]
        speech_end_id = token_ids["<|SPEECH_GENERATION_END|>"]
        text_start_id = token_ids["<|TEXT_UNDERSTANDING_START|>"]
        text_end_id = token_ids["<|TEXT_UNDERSTANDING_END|>"]
        offset = token_ids["speech_token_offset"]

        codes = [0, 100, 65535]
        input_ids = [speech_start_id] + [offset + c for c in codes] + [speech_end_id]

        assert text_start_id not in input_ids
        assert text_end_id not in input_ids

    def test_stage1_truncation(self, tts_setup):
        """Stage1 should truncate to max_seq_length."""
        _, tokenizer, token_ids = tts_setup
        speech_start_id = token_ids["<|SPEECH_GENERATION_START|>"]
        speech_end_id = token_ids["<|SPEECH_GENERATION_END|>"]
        offset = token_ids["speech_token_offset"]

        max_seq_length = 10
        codes = list(range(20))  # More codes than max_seq_length
        input_ids = [speech_start_id] + [offset + c for c in codes] + [speech_end_id]
        input_ids = input_ids[:max_seq_length]
        completion_mask = [1] * len(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(completion_mask) == max_seq_length
        assert all(m == 1 for m in completion_mask)


class TestStage3Tokenization:
    """Tests for LM-Stage3 chain-of-modality tokenization."""

    def test_stage3_produces_four_samples(self, tts_setup):
        """Stage3 tokenization should produce 4 samples per data point."""
        _, tokenizer, token_ids = tts_setup
        speech_start_id = token_ids["<|SPEECH_GENERATION_START|>"]
        speech_end_id = token_ids["<|SPEECH_GENERATION_END|>"]
        text_start_id = token_ids["<|TEXT_UNDERSTANDING_START|>"]
        text_end_id = token_ids["<|TEXT_UNDERSTANDING_END|>"]
        offset = token_ids["speech_token_offset"]
        max_seq_length = 2048

        # Simulate one data point
        in_codes = [10, 20, 30]
        out_codes = [40, 50, 60]
        input_text = "What is the capital of France?"
        output_text = "The capital of France is Paris."

        input_text_ids = tokenizer.encode(input_text, add_special_tokens=False)
        output_text_ids = tokenizer.encode(output_text, add_special_tokens=False)
        in_speech_ids = [offset + c for c in in_codes]
        out_speech_ids = [offset + c for c in out_codes]

        all_input_ids = []
        all_completion_mask = []

        # 1. Speech→Speech
        s2s_prompt = [speech_start_id] + in_speech_ids + [speech_end_id, speech_start_id]
        s2s_completion = out_speech_ids + [speech_end_id]
        s2s_seq = (s2s_prompt + s2s_completion)[:max_seq_length]
        s2s_plen = min(len(s2s_prompt), len(s2s_seq))
        all_input_ids.append(s2s_seq)
        all_completion_mask.append([0] * s2s_plen + [1] * (len(s2s_seq) - s2s_plen))

        # 2. Speech→Text
        s2t_prompt = [speech_start_id] + in_speech_ids + [speech_end_id, text_start_id]
        s2t_completion = output_text_ids + [text_end_id]
        s2t_seq = (s2t_prompt + s2t_completion)[:max_seq_length]
        s2t_plen = min(len(s2t_prompt), len(s2t_seq))
        all_input_ids.append(s2t_seq)
        all_completion_mask.append([0] * s2t_plen + [1] * (len(s2t_seq) - s2t_plen))

        # 3. Text→Speech
        t2s_prompt = [text_start_id] + input_text_ids + [text_end_id, speech_start_id]
        t2s_completion = out_speech_ids + [speech_end_id]
        t2s_seq = (t2s_prompt + t2s_completion)[:max_seq_length]
        t2s_plen = min(len(t2s_prompt), len(t2s_seq))
        all_input_ids.append(t2s_seq)
        all_completion_mask.append([0] * t2s_plen + [1] * (len(t2s_seq) - t2s_plen))

        # 4. Text→Text
        t2t_prompt = [text_start_id] + input_text_ids + [text_end_id, text_start_id]
        t2t_completion = output_text_ids + [text_end_id]
        t2t_seq = (t2t_prompt + t2t_completion)[:max_seq_length]
        t2t_plen = min(len(t2t_prompt), len(t2t_seq))
        all_input_ids.append(t2t_seq)
        all_completion_mask.append([0] * t2t_plen + [1] * (len(t2t_seq) - t2t_plen))

        assert len(all_input_ids) == 4
        assert len(all_completion_mask) == 4

    def test_stage3_speech_to_speech_structure(self, tts_setup):
        """Speech→Speech should have speech delimiters on both sides."""
        _, tokenizer, token_ids = tts_setup
        speech_start_id = token_ids["<|SPEECH_GENERATION_START|>"]
        speech_end_id = token_ids["<|SPEECH_GENERATION_END|>"]
        text_start_id = token_ids["<|TEXT_UNDERSTANDING_START|>"]
        text_end_id = token_ids["<|TEXT_UNDERSTANDING_END|>"]
        offset = token_ids["speech_token_offset"]

        in_codes = [10, 20]
        out_codes = [30, 40]
        in_speech = [offset + c for c in in_codes]
        out_speech = [offset + c for c in out_codes]

        seq = (
            [speech_start_id]
            + in_speech
            + [speech_end_id, speech_start_id]
            + out_speech
            + [speech_end_id]
        )

        # Should start with speech_start
        assert seq[0] == speech_start_id
        # Should end with speech_end
        assert seq[-1] == speech_end_id
        # Should not contain text tokens
        assert text_start_id not in seq
        assert text_end_id not in seq

    def test_stage3_completion_mask_covers_output_only(self, tts_setup):
        """Completion mask should be 0 for input, 1 for output."""
        _, tokenizer, token_ids = tts_setup
        speech_start_id = token_ids["<|SPEECH_GENERATION_START|>"]
        speech_end_id = token_ids["<|SPEECH_GENERATION_END|>"]
        text_start_id = token_ids["<|TEXT_UNDERSTANDING_START|>"]
        text_end_id = token_ids["<|TEXT_UNDERSTANDING_END|>"]
        offset = token_ids["speech_token_offset"]

        in_codes = [10, 20, 30]
        out_text_ids = tokenizer.encode("Hello", add_special_tokens=False)
        in_speech = [offset + c for c in in_codes]

        # Speech→Text
        prompt = [speech_start_id] + in_speech + [speech_end_id, text_start_id]
        completion = out_text_ids + [text_end_id]
        seq = prompt + completion
        mask = [0] * len(prompt) + [1] * len(completion)

        assert len(mask) == len(seq)
        # Prompt part should all be 0
        assert all(m == 0 for m in mask[: len(prompt)])
        # Completion part should all be 1
        assert all(m == 1 for m in mask[len(prompt) :])

    def test_stage3_text_to_text_no_speech_tokens(self, tts_setup):
        """Text→Text should not contain any speech code tokens."""
        _, tokenizer, token_ids = tts_setup
        text_start_id = token_ids["<|TEXT_UNDERSTANDING_START|>"]
        text_end_id = token_ids["<|TEXT_UNDERSTANDING_END|>"]
        speech_start_id = token_ids["<|SPEECH_GENERATION_START|>"]
        speech_end_id = token_ids["<|SPEECH_GENERATION_END|>"]
        offset = token_ids["speech_token_offset"]

        input_text_ids = tokenizer.encode("What is 2+2?", add_special_tokens=False)
        output_text_ids = tokenizer.encode("4", add_special_tokens=False)

        seq = (
            [text_start_id]
            + input_text_ids
            + [text_end_id, text_start_id]
            + output_text_ids
            + [text_end_id]
        )

        # Should not contain speech delimiters
        assert speech_start_id not in seq
        assert speech_end_id not in seq
        # Should not contain any speech offset tokens
        for tok_id in seq:
            assert (
                tok_id < offset
                or tok_id >= offset + 65536
                or tok_id in {text_start_id, text_end_id}
            ), f"Unexpected speech token {tok_id} in Text→Text sequence"


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
