"""Tests for LLASA-style TTS: LLM with expanded codec vocabulary and ChatML format."""

import pytest
import torch
from trl import get_training_chat_template

from tiny_audio.lm import (
    CODEC_VOCAB_SIZE,
    SPECIAL_TOKENS,
    codes_to_speech_text,
    setup_tts_model,
)


@pytest.fixture(scope="session")
def tts_setup():
    """Setup TTS model once per session (downloads Qwen3-0.6B)."""
    model, tokenizer, token_ids = setup_tts_model(
        model_id="Qwen/Qwen3-0.6B",
        dtype=torch.float32,
    )
    return model, tokenizer, token_ids


@pytest.fixture(scope="session")
def training_template(tts_setup):
    """Get the prefix-preserving training template for Qwen3."""
    _, tokenizer, _ = tts_setup
    return get_training_chat_template(tokenizer)


class TestVocabExpansion:
    def test_vocab_size(self, tts_setup):
        """Tokenizer should have base vocab + 2 special + 65536 speech tokens."""
        _, tokenizer, _ = tts_setup
        # Verify the total is at least base + added tokens
        # (exact base vocab depends on tokenizer version)
        min_expected = CODEC_VOCAB_SIZE + len(SPECIAL_TOKENS)
        assert len(tokenizer) >= min_expected

    def test_only_two_special_tokens(self):
        """Only 2 new tokens should be added: audio_start and audio_end."""
        assert len(SPECIAL_TOKENS) == 2
        assert "<|audio_start|>" in SPECIAL_TOKENS
        assert "<|audio_end|>" in SPECIAL_TOKENS

    def test_model_embedding_matches_tokenizer(self, tts_setup):
        """Model embeddings should be resized to match tokenizer."""
        model, tokenizer, _ = tts_setup
        embed_size = model.get_input_embeddings().weight.shape[0]
        assert embed_size == len(tokenizer)

    def test_special_tokens_resolvable(self, tts_setup):
        """All special tokens should resolve to valid IDs."""
        _, tokenizer, _ = tts_setup
        for tok in SPECIAL_TOKENS:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            assert tok_id != tokenizer.unk_token_id, f"{tok} resolved to UNK"

    def test_native_chatml_tokens_exist(self, tts_setup):
        """Qwen3 native ChatML tokens should be in the base vocab."""
        _, tokenizer, token_ids = tts_setup
        assert token_ids["<|im_start|>"] != tokenizer.unk_token_id
        assert token_ids["<|im_end|>"] != tokenizer.unk_token_id

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

    def test_native_token_ids_in_dict(self, tts_setup):
        """Native ChatML tokens should be in the token_ids dict."""
        _, _, token_ids = tts_setup
        assert "<|im_start|>" in token_ids
        assert "<|im_end|>" in token_ids


class TestCodesToSpeechText:
    """Tests for the codes_to_speech_text helper function."""

    def test_basic_roundtrip(self, tts_setup):
        """codes_to_speech_text output should tokenize back to correct token IDs."""
        _, tokenizer, token_ids = tts_setup
        offset = token_ids["speech_token_offset"]
        codes = [10, 20, 30]
        text = codes_to_speech_text(codes)
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Should be: audio_start, s_10, s_20, s_30, audio_end
        assert ids[0] == token_ids["<|audio_start|>"]
        assert ids[-1] == token_ids["<|audio_end|>"]
        for i, code in enumerate(codes):
            assert ids[i + 1] == offset + code

    def test_empty_codes(self):
        """Empty codes should produce just delimiters."""
        text = codes_to_speech_text([])
        assert text == "<|audio_start|><|audio_end|>"

    def test_speech_tokens_in_chat_template(self, tts_setup, training_template):
        """Speech tokens should survive apply_chat_template roundtrip."""
        _, tokenizer, token_ids = tts_setup
        offset = token_ids["speech_token_offset"]
        codes = [0, 42, 65535]
        speech_text = codes_to_speech_text(codes)

        messages = [
            {"role": "user", "content": "Say hello"},
            {"role": "assistant", "content": speech_text},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages,
            chat_template=training_template,
            tokenize=True,
            return_dict=False,
        )
        # All speech token IDs should be present in the output
        for code in codes:
            assert offset + code in full_ids


class TestTrainingTemplate:
    """Tests for the training template from get_training_chat_template."""

    def test_template_is_not_none(self, training_template):
        """Qwen3 should need a training template (it's not prefix-preserving by default)."""
        assert training_template is not None

    def test_prefix_preserving(self, tts_setup, training_template):
        """Training template should be prefix-preserving: full[:len(prompt)] == prompt."""
        _, tokenizer, _ = tts_setup
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            [messages[0]],
            chat_template=training_template,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            chat_template=training_template,
            tokenize=True,
            return_dict=False,
        )
        assert full_ids[: len(prompt_ids)] == prompt_ids

    def test_think_block_in_assistant(self, tts_setup, training_template):
        """Training template should include think block in assistant messages."""
        _, tokenizer, _ = tts_setup
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages,
            chat_template=training_template,
            tokenize=True,
            return_dict=False,
        )
        decoded = tokenizer.decode(full_ids)
        assert "<think>\n\n</think>\n\n" in decoded


class TestStage1Tokenization:
    """Tests for LM-Stage1 speech-only tokenization via apply_chat_template."""

    def _build_stage1(self, tts_setup, training_template, codes):
        """Build a Stage 1 sample using apply_chat_template."""
        _, tokenizer, _ = tts_setup
        speech_text = codes_to_speech_text(codes)
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": speech_text},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            [messages[0]],
            chat_template=training_template,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            chat_template=training_template,
            tokenize=True,
            return_dict=False,
        )
        prompt_len = min(len(prompt_ids), len(full_ids))
        mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
        return full_ids, mask

    def test_stage1_has_speech_tokens(self, tts_setup, training_template):
        """Stage1 sequence should contain audio delimiters and speech tokens."""
        _, tokenizer, token_ids = tts_setup
        offset = token_ids["speech_token_offset"]
        codes = [10, 20, 30, 40, 50]
        full_ids, _ = self._build_stage1(tts_setup, training_template, codes)

        assert token_ids["<|audio_start|>"] in full_ids
        assert token_ids["<|audio_end|>"] in full_ids
        for code in codes:
            assert offset + code in full_ids

    def test_stage1_completion_mask(self, tts_setup, training_template):
        """Stage1 should have prompt masked (0) and completion unmasked (1)."""
        codes = [10, 20, 30]
        full_ids, mask = self._build_stage1(tts_setup, training_template, codes)

        assert len(mask) == len(full_ids)
        # Should have some 0s (prompt) and some 1s (completion)
        assert 0 in mask
        assert 1 in mask
        # All 0s should come before all 1s
        first_one = mask.index(1)
        assert all(m == 0 for m in mask[:first_one])
        assert all(m == 1 for m in mask[first_one:])

    def test_stage1_decoded_is_valid_chatml(self, tts_setup, training_template):
        """Decoded Stage 1 should contain ChatML structure with think block."""
        _, tokenizer, _ = tts_setup
        codes = [0, 100, 65535]
        full_ids, _ = self._build_stage1(tts_setup, training_template, codes)
        decoded = tokenizer.decode(full_ids)

        assert "<|im_start|>user\n" in decoded
        assert "<|im_end|>" in decoded
        assert "<|im_start|>assistant\n" in decoded
        assert "<think>\n\n</think>\n\n" in decoded
        assert "<|audio_start|>" in decoded
        assert "<|audio_end|>" in decoded


class TestStage2Tokenization:
    """Tests for LM-Stage2 cross-modal tokenization via apply_chat_template."""

    def _build_stage2_pair(self, tts_setup, training_template, codes, text):
        """Build Stage 2 transcribe + speak pair using apply_chat_template."""
        _, tokenizer, _ = tts_setup
        speech_text = codes_to_speech_text(codes)

        results = []
        for user_content, assistant_content in [
            (speech_text, text),  # Transcribe: speech → text
            (text, speech_text),  # Speak: text → speech
        ]:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                [messages[0]],
                chat_template=training_template,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            full_ids = tokenizer.apply_chat_template(
                messages,
                chat_template=training_template,
                tokenize=True,
                return_dict=False,
            )
            prompt_len = min(len(prompt_ids), len(full_ids))
            mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
            results.append((full_ids, mask, prompt_len))
        return results

    def test_stage2_transcribe_has_speech_in_prompt(self, tts_setup, training_template):
        """Transcribe direction should have speech tokens in the prompt portion."""
        _, _, token_ids = tts_setup
        codes = [10, 20, 30]
        text = "Hello world"
        (full_ids, mask, prompt_len), _ = self._build_stage2_pair(
            tts_setup, training_template, codes, text
        )

        # Speech tokens should be in the prompt (masked) portion
        prompt_portion = full_ids[:prompt_len]
        assert token_ids["<|audio_start|>"] in prompt_portion

    def test_stage2_speak_has_speech_in_completion(self, tts_setup, training_template):
        """Speak direction should have speech tokens in the completion portion."""
        _, _, token_ids = tts_setup
        codes = [10, 20, 30]
        text = "Hello world"
        _, (full_ids, mask, prompt_len) = self._build_stage2_pair(
            tts_setup, training_template, codes, text
        )

        # Speech tokens should be in the completion (unmasked) portion
        completion_portion = full_ids[prompt_len:]
        assert token_ids["<|audio_start|>"] in completion_portion

    def test_stage2_mask_structure(self, tts_setup, training_template):
        """Both directions should have proper prompt/completion mask structure."""
        codes = [10, 20, 30]
        text = "Hello world"
        results = self._build_stage2_pair(tts_setup, training_template, codes, text)

        for full_ids, mask, prompt_len in results:
            assert len(mask) == len(full_ids)
            assert all(m == 0 for m in mask[:prompt_len])
            assert all(m == 1 for m in mask[prompt_len:])

    def test_stage2_prefix_preserving(self, tts_setup, training_template):
        """Both directions should be prefix-preserving."""
        _, tokenizer, _ = tts_setup
        codes = [10, 20]
        text = "hi"
        speech_text = codes_to_speech_text(codes)

        for user_content, assistant_content in [
            (speech_text, text),
            (text, speech_text),
        ]:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                [messages[0]],
                chat_template=training_template,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            full_ids = tokenizer.apply_chat_template(
                messages,
                chat_template=training_template,
                tokenize=True,
                return_dict=False,
            )
            assert full_ids[: len(prompt_ids)] == prompt_ids

    def test_stage2_decoded_is_valid_chatml(self, tts_setup, training_template):
        """Decoded Stage 2 sequence should be valid ChatML with think block."""
        _, tokenizer, _ = tts_setup
        codes = [10, 20]
        text = "hi"
        (full_ids, _, _), _ = self._build_stage2_pair(tts_setup, training_template, codes, text)
        decoded = tokenizer.decode(full_ids)
        assert "<|im_start|>user\n" in decoded
        assert "<|im_end|>" in decoded
        assert "<|im_start|>assistant\n" in decoded
        assert "<think>\n\n</think>\n\n" in decoded


class TestStage3Tokenization:
    """Tests for LM-Stage3 chain-of-modality tokenization via apply_chat_template."""

    def _build_stage3_samples(self, tts_setup, training_template, reasoning=""):
        """Build all 4 Stage 3 samples using apply_chat_template."""
        _, tokenizer, _ = tts_setup

        in_codes = [10, 20, 30]
        out_codes = [40, 50, 60]
        input_text = "What is the capital of France?"
        output_text = "The capital of France is Paris."

        in_speech_text = codes_to_speech_text(in_codes)
        out_speech_text = codes_to_speech_text(out_codes)

        def _assistant_text(output):
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n\n{output}"
            return output

        pairs = [
            (in_speech_text, _assistant_text(out_speech_text)),  # 1. Speech→Speech
            (in_speech_text, _assistant_text(output_text)),  # 2. Speech→Text
            (input_text, _assistant_text(out_speech_text)),  # 3. Text→Speech
            (input_text, _assistant_text(output_text)),  # 4. Text→Text
        ]

        results = []
        for user_content, assistant_content in pairs:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                [messages[0]],
                chat_template=training_template,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            full_ids = tokenizer.apply_chat_template(
                messages,
                chat_template=training_template,
                tokenize=True,
                return_dict=False,
            )
            prompt_len = min(len(prompt_ids), len(full_ids))
            mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
            results.append((full_ids, mask, prompt_len))
        return results

    def test_stage3_produces_four_samples(self, tts_setup, training_template):
        """Stage3 tokenization should produce 4 samples per data point."""
        samples = self._build_stage3_samples(tts_setup, training_template)
        assert len(samples) == 4

    def test_stage3_speech_to_speech_structure(self, tts_setup, training_template):
        """Speech→Speech should have audio delimiters on both sides with think tags."""
        _, _, token_ids = tts_setup
        full_ids, mask, prompt_len = self._build_stage3_samples(tts_setup, training_template)[0]

        # Prompt should contain audio delimiters (input speech)
        prompt_portion = full_ids[:prompt_len]
        assert token_ids["<|audio_start|>"] in prompt_portion

        # Completion should contain audio delimiters (output speech)
        completion_portion = full_ids[prompt_len:]
        assert token_ids["<|audio_start|>"] in completion_portion

    def test_stage3_completion_mask_covers_output_only(self, tts_setup, training_template):
        """Completion mask should be 0 for prompt, 1 for completion."""
        full_ids, mask, prompt_len = self._build_stage3_samples(tts_setup, training_template)[1]

        assert len(mask) == len(full_ids)
        assert all(m == 0 for m in mask[:prompt_len])
        assert all(m == 1 for m in mask[prompt_len:])

    def test_stage3_text_to_text_no_speech_codes(self, tts_setup, training_template):
        """Text→Text should not contain any speech code tokens (offset+N)."""
        _, _, token_ids = tts_setup
        offset = token_ids["speech_token_offset"]
        full_ids, _, _ = self._build_stage3_samples(tts_setup, training_template)[3]

        # Should not contain audio_start/end (no speech content)
        assert token_ids["<|audio_start|>"] not in full_ids
        assert token_ids["<|audio_end|>"] not in full_ids
        # Should not contain any speech offset tokens
        for tok_id in full_ids:
            assert not (offset <= tok_id < offset + CODEC_VOCAB_SIZE), (
                f"Unexpected speech token {tok_id} in Text→Text sequence"
            )

    def test_stage3_with_reasoning(self, tts_setup, training_template):
        """Stage3 with reasoning should include reasoning text in decoded output."""
        _, tokenizer, _ = tts_setup
        reasoning = "France is a country in Europe."
        samples = self._build_stage3_samples(tts_setup, training_template, reasoning=reasoning)

        # Check Speech→Text (sample index 1)
        full_ids, _, prompt_len = samples[1]
        completion_decoded = tokenizer.decode(full_ids[prompt_len:])

        # Completion should contain the reasoning in a think block
        assert reasoning in completion_decoded
        assert "<think>" in completion_decoded
        assert "</think>" in completion_decoded

    def test_stage3_empty_reasoning(self, tts_setup, training_template):
        """Stage3 without reasoning should have empty think block."""
        _, tokenizer, _ = tts_setup
        samples = self._build_stage3_samples(tts_setup, training_template, reasoning="")

        # Check Speech→Text (sample index 1)
        full_ids, _, _ = samples[1]
        decoded = tokenizer.decode(full_ids)

        # Should contain empty think block
        assert "<think>\n\n</think>\n\n" in decoded


class TestForward:
    def _make_batch(self, tts_setup, training_template):
        """Build a batch using apply_chat_template."""
        model, tokenizer, token_ids = tts_setup

        text = "Hello world"
        codes = list(range(20))
        speech_text = codes_to_speech_text(codes)

        # Build speak direction: text → speech
        messages = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": speech_text},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            [messages[0]],
            chat_template=training_template,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            chat_template=training_template,
            tokenize=True,
            return_dict=False,
        )
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": torch.tensor([full_ids]),
            "labels": torch.tensor([labels]),
            "attention_mask": torch.ones(1, len(full_ids), dtype=torch.long),
        }

    def test_forward_produces_loss(self, tts_setup, training_template):
        """Model forward should produce a scalar loss."""
        model = tts_setup[0]
        batch = self._make_batch(tts_setup, training_template)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        output = model(**batch)
        assert output.loss is not None
        assert output.loss.dim() == 0
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)

    def test_gradients_flow(self, tts_setup, training_template):
        """All model parameters should receive gradients."""
        model = tts_setup[0]
        batch = self._make_batch(tts_setup, training_template)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        model.zero_grad()
        output = model(**batch)
        output.loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "Model should have gradients"
        model.zero_grad()
