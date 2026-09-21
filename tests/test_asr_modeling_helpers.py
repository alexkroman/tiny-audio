"""Tests for ASRModel helpers that need no weights.

Module-level functions are called directly; instance methods that only touch
the tokenizer or projector are invoked unbound against a stand-in `self`.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tiny_audio.asr_modeling import (
    ASRModel,
    _assert_audio_token_counts,
    _gather_audio_embeds,
    _resolve_attn_implementation,
)


class TestResolveAttnImplementation:
    """FA2 degrades to sdpa off-CUDA, and everything degrades to eager on MPS."""

    @pytest.fixture(autouse=True)
    def no_accelerators(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    @pytest.mark.parametrize("requested", [None, "eager", "sdpa"])
    def test_non_fa2_requests_pass_through(self, requested):
        assert _resolve_attn_implementation(requested) == requested

    def test_fa2_without_cuda_is_sdpa(self):
        assert _resolve_attn_implementation("flash_attention_2") == "sdpa"

    def test_fa2_with_cuda_but_no_flash_attn_is_sdpa(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr("transformers.utils.is_flash_attn_2_available", lambda: False)
        assert _resolve_attn_implementation("flash_attention_2") == "sdpa"

    def test_fa2_with_cuda_and_flash_attn_is_kept(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr("transformers.utils.is_flash_attn_2_available", lambda: True)
        assert _resolve_attn_implementation("flash_attention_2") == "flash_attention_2"

    @pytest.mark.parametrize("requested", [None, "sdpa", "flash_attention_2"])
    def test_mps_forces_eager(self, monkeypatch, requested):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert _resolve_attn_implementation(requested) == "eager"


class TestGatherAudioEmbeds:
    """`max_tokens` skips the device sync but must give identical results."""

    def test_max_tokens_matches_sync_path(self):
        embeds = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        counts = torch.tensor([2, 4])
        assert torch.equal(
            _gather_audio_embeds(embeds, counts), _gather_audio_embeds(embeds, counts, max_tokens=4)
        )

    def test_explicit_max_tokens_beyond_length_zero_pads(self):
        embeds = torch.ones(1, 2, 3)
        out = _gather_audio_embeds(embeds, torch.tensor([5]), max_tokens=5)
        assert out.shape == (5, 3)
        assert torch.equal(out[:2], torch.ones(2, 3))
        assert torch.equal(out[2:], torch.zeros(3, 3))


class TestAssertAudioTokenCounts:
    """The per-sample count check that guards masked_scatter."""

    @staticmethod
    def projector():
        proj = MagicMock()
        proj.get_output_length.side_effect = lambda n: (n - 4) // 4 + 1
        return proj

    def test_consistent_counts_pass(self):
        embeds = torch.zeros(2, 5, 8)
        counts = torch.tensor([5, 3])
        _assert_audio_token_counts(embeds, counts, self.projector())
        _assert_audio_token_counts(embeds, counts, self.projector(), max_tokens=5)

    def test_prompt_longer_than_projector_output_raises(self):
        with pytest.raises(ValueError, match="Projector produced 5 audio frames"):
            _assert_audio_token_counts(torch.zeros(1, 5, 8), torch.tensor([6]), self.projector())

    def test_max_tokens_is_checked_instead_of_counts(self):
        with pytest.raises(ValueError, match="expects up to 9"):
            _assert_audio_token_counts(
                torch.zeros(1, 5, 8), torch.tensor([1]), self.projector(), max_tokens=9
            )

    def test_empty_batch_passes(self):
        _assert_audio_token_counts(torch.zeros(0, 5, 8), torch.tensor([]), self.projector())

    def test_encoder_lengths_agreeing_with_prompt_pass(self):
        # encoder lengths 20 and 8 -> projector 5 and 2 tokens.
        _assert_audio_token_counts(
            torch.zeros(2, 5, 8),
            torch.tensor([5, 2]),
            self.projector(),
            encoder_valid_lengths=torch.tensor([20, 8]),
        )

    def test_encoder_lengths_disagreeing_with_prompt_raise_with_rows(self):
        with pytest.raises(ValueError, match=r"Rows \[1\]: prompt expects \[3\]"):
            _assert_audio_token_counts(
                torch.zeros(2, 5, 8),
                torch.tensor([5, 3]),
                self.projector(),
                encoder_valid_lengths=torch.tensor([20, 8]),
            )

    def test_encoder_lengths_of_wrong_shape_are_ignored(self):
        # A shape mismatch means the encoder mask isn't per-sample; skip the
        # cross-check rather than compare apples to oranges.
        _assert_audio_token_counts(
            torch.zeros(2, 5, 8),
            torch.tensor([5, 3]),
            self.projector(),
            encoder_valid_lengths=torch.tensor([20]),
        )


class TestLeftPadPromptRows:
    """Generation prompts are left-padded so nothing sits before the first token."""

    def test_pads_on_the_left_with_pad_token(self):
        fake = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=9, eos_token_id=2))
        ids, mask = ASRModel._left_pad_prompt_rows(
            fake, [torch.tensor([1, 2]), torch.tensor([3])], torch.device("cpu")
        )
        assert ids.tolist() == [[1, 2], [9, 3]]
        assert mask.tolist() == [[1, 1], [0, 1]]

    def test_falls_back_to_eos_then_zero(self):
        fake = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None, eos_token_id=2))
        ids, _ = ASRModel._left_pad_prompt_rows(
            fake, [torch.tensor([1, 2]), torch.tensor([3])], torch.device("cpu")
        )
        assert ids[1, 0].item() == 2

        fake = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None, eos_token_id=None))
        ids, _ = ASRModel._left_pad_prompt_rows(
            fake, [torch.tensor([1, 2]), torch.tensor([3])], torch.device("cpu")
        )
        assert ids[1, 0].item() == 0


class TestRenderAudioPrompt:
    """Placeholder count and instruction land in one user turn."""

    def test_content_and_template_kwargs(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int32)
        )
        fake = SimpleNamespace(
            tokenizer=tokenizer, audio_token="<audio>", TRANSCRIBE_PROMPT="Transcribe"
        )
        row = ASRModel._render_audio_prompt(fake, 3)
        call = tokenizer.apply_chat_template.call_args
        assert call.args[0] == [{"role": "user", "content": "<audio><audio><audio> Transcribe"}]
        assert call.kwargs["add_generation_prompt"] is True
        assert call.kwargs["enable_thinking"] is False
        assert row.tolist() == [1, 2, 3]
        assert row.dtype == torch.long

    def test_empty_instruction_leaves_placeholders_alone(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = SimpleNamespace(input_ids=torch.tensor([1]))
        fake = SimpleNamespace(tokenizer=tokenizer, audio_token="<a>", TRANSCRIBE_PROMPT="")
        ASRModel._render_audio_prompt(fake, 2)
        assert tokenizer.apply_chat_template.call_args.args[0][0]["content"] == "<a><a>"


class TestGetNumAudioTokens:
    """The batch-max token count chains encoder lengths into the projector."""

    def test_uses_longest_sample(self):
        fake = SimpleNamespace(
            _compute_encoder_output_lengths=lambda mask: torch.tensor([10, 20]),
            projector=SimpleNamespace(get_output_length=lambda n: (n - 4) // 4 + 1),
        )
        assert ASRModel._get_num_audio_tokens(fake, torch.ones(2, 40)) == 5


class TestCreateOrUpdateModelCard:
    """`Trainer.create_model_card` calls this PEFT method on the unwrapped model.

    It fires whenever the README in `output_dir` says `library_name: peft`,
    which `save_pretrained` guarantees on a LoRA run. The outer model is a
    plain `PreTrainedModel`, so the method has to exist here and forward to
    the adapter-bearing language model.
    """

    def test_delegates_to_peft_language_model(self, tmp_path):
        card_fn = MagicMock()
        fake = SimpleNamespace(language_model=SimpleNamespace(create_or_update_model_card=card_fn))

        ASRModel.create_or_update_model_card(fake, tmp_path)

        card_fn.assert_called_once_with(str(tmp_path))

    def test_no_adapter_is_a_noop(self, tmp_path):
        """A stale peft README from an earlier run must not take the save down."""
        fake = SimpleNamespace(language_model=SimpleNamespace())

        ASRModel.create_or_update_model_card(fake, tmp_path)

    def test_method_is_reachable_on_the_class(self):
        """nn.Module.__getattr__ is what raised; guard the attribute itself."""
        assert callable(getattr(ASRModel, "create_or_update_model_card", None))
