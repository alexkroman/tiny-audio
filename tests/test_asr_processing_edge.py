"""Edge cases for ASRProcessor prompt rendering and batching."""

from unittest.mock import MagicMock

import pytest
import torch

from tiny_audio.asr_processing import ASRProcessor


def make_processor(pad_token_id=0, eos_token_id=2, template_return=None) -> ASRProcessor:
    """Processor over mocks; `template_return` controls apply_chat_template."""
    tok = MagicMock()
    tok.pad_token_id = pad_token_id
    tok.eos_token_id = eos_token_id
    tok.convert_tokens_to_ids.return_value = 777
    tok.apply_chat_template.return_value = (
        torch.tensor([[1, 2, 3]]) if template_return is None else template_return
    )
    fe = MagicMock()
    fe.sampling_rate = 16000
    proj = MagicMock()
    proj.get_output_length.side_effect = lambda n: (n - 4) // 4 + 1
    return ASRProcessor(fe, tok, projector=proj)


class TestStackPromptRows:
    """Ragged prompt rows are left-padded with the right id."""

    def test_uniform_rows_are_stacked(self):
        proc = make_processor()
        ids, mask = proc._stack_prompt_rows([torch.tensor([1, 2]), torch.tensor([3, 4])])
        assert ids.tolist() == [[1, 2], [3, 4]]
        assert mask.tolist() == [[1, 1], [1, 1]]

    def test_ragged_rows_use_pad_token(self):
        proc = make_processor(pad_token_id=9)
        ids, mask = proc._stack_prompt_rows([torch.tensor([1]), torch.tensor([3, 4, 5])])
        assert ids.tolist() == [[9, 9, 1], [3, 4, 5]]
        assert mask.tolist() == [[0, 0, 1], [1, 1, 1]]
        assert ids.dtype == torch.long

    def test_missing_pad_token_falls_back_to_eos(self):
        proc = make_processor(pad_token_id=None, eos_token_id=2)
        ids, _ = proc._stack_prompt_rows([torch.tensor([1]), torch.tensor([3, 4])])
        assert ids[0, 0].item() == 2

    def test_missing_pad_and_eos_falls_back_to_zero(self):
        proc = make_processor(pad_token_id=None, eos_token_id=None)
        ids, _ = proc._stack_prompt_rows([torch.tensor([1]), torch.tensor([3, 4])])
        assert ids[0, 0].item() == 0


class TestRenderPrompt:
    """Prompt content and the shapes apply_chat_template may return."""

    def test_audio_placeholders_precede_instruction(self):
        proc = make_processor()
        proc._render_prompt(3, None)
        messages = proc.tokenizer.apply_chat_template.call_args.args[0]
        assert messages == [
            {"role": "user", "content": "<audio><audio><audio> " + proc.TRANSCRIBE_PROMPT}
        ]
        assert proc.tokenizer.apply_chat_template.call_args.kwargs["add_generation_prompt"] is True

    def test_no_audio_means_instruction_only(self):
        proc = make_processor()
        proc._render_prompt(0, None)
        messages = proc.tokenizer.apply_chat_template.call_args.args[0]
        assert messages[0]["content"] == proc.TRANSCRIBE_PROMPT

    def test_target_text_appends_assistant_turn(self):
        proc = make_processor()
        proc._render_prompt(1, "hello")
        call = proc.tokenizer.apply_chat_template.call_args
        assert call.args[0][-1] == {"role": "assistant", "content": "hello"}
        assert call.kwargs["add_generation_prompt"] is False

    def test_batch_encoding_return_is_unwrapped(self):
        proc = make_processor(template_return={"input_ids": torch.tensor([[4, 5, 6]])})
        row = proc._render_prompt(1, None)
        assert row.tolist() == [4, 5, 6]
        assert row.dtype == torch.long

    def test_one_dimensional_return_is_kept(self):
        proc = make_processor(template_return=torch.tensor([7, 8], dtype=torch.int32))
        row = proc._render_prompt(1, None)
        assert row.tolist() == [7, 8]
        assert row.dtype == torch.long


class TestCallWithoutAudio:
    """Text-only calls still produce a single prompt row."""

    def test_returns_prompt_only(self):
        proc = make_processor()
        out = proc(text="target")
        assert set(out) == {"input_ids", "attention_mask"}
        assert out["input_ids"].shape == (1, 3)
        proc.feature_extractor.assert_not_called()


class TestEncoderLength:
    """Custom conv layers flow through the processor's length helper."""

    def test_custom_layers(self):
        proc = make_processor()
        proc.encoder_conv_layers = [(0, 2, 2), (0, 2, 2)]
        assert proc._compute_encoder_output_length(100) == 25

    def test_audio_token_id_comes_from_tokenizer(self):
        proc = make_processor()
        proc.tokenizer.convert_tokens_to_ids.assert_called_once_with("<audio>")
        assert proc.audio_token_id == 777


@pytest.mark.parametrize("mel_len", [4, 8, 100])
def test_token_count_matches_projector_for_each_row(mel_len):
    """Per-row token counts come from each row's own mel length."""
    proc = make_processor()
    proc.feature_extractor.return_value = {
        "input_features": torch.zeros(2, 80, mel_len),
        "attention_mask": torch.stack(
            [torch.ones(mel_len), torch.cat([torch.ones(4), torch.zeros(mel_len - 4)])]
        ).long(),
    }
    proc(audio=[torch.zeros(16000), torch.zeros(8000)])
    expected_full = (((mel_len + 2 - 2 - 1) // 1 + 1 + 2 - 2 - 1) // 2 + 1 - 4) // 4 + 1
    contents = [
        call.args[0][0]["content"] for call in proc.tokenizer.apply_chat_template.call_args_list
    ]
    assert contents[0].count("<audio>") == expected_full
    # The 4-frame row: encoder 4 -> 2, projector (2-4)//4+1 = 0 placeholders.
    assert contents[1].count("<audio>") == 0
