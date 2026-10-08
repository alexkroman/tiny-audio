"""Tests for ASRProcessor.

Uses pytest-mock and shared fixtures from conftest.py.
"""

import functools
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from tiny_audio.asr_config import DEFAULT_ENCODER_CONV_LAYERS
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.asr_processing import ASRProcessor


def _quarter(x: int) -> int:
    return x // 4


class TestProcessorConstants:
    """Tests for ASRProcessor constants."""

    def test_audio_token_defined(self) -> None:
        """AUDIO_TOKEN constant should be defined."""
        assert ASRProcessor.AUDIO_TOKEN == "<audio>"

    def test_transcribe_prompt_defined(self) -> None:
        """TRANSCRIBE_PROMPT must match the prompt used in training (scripts/train.py)."""
        assert ASRProcessor.TRANSCRIBE_PROMPT == "Transcribe the speech to text"

    def test_default_conv_layers(self) -> None:
        """DEFAULT_ENCODER_CONV_LAYERS should match Whisper."""
        expected = [(1, 3, 1), (1, 3, 2)]
        assert expected == DEFAULT_ENCODER_CONV_LAYERS


class TestProcessorInit:
    """Tests for ASRProcessor initialization."""

    def test_init_sets_attributes(
        self,
        mock_feature_extractor: MagicMock,
        mock_tokenizer: MagicMock,
        mock_projector: MagicMock,
    ) -> None:
        """Processor should store all attributes."""
        processor = ASRProcessor(mock_feature_extractor, mock_tokenizer, mock_projector)

        assert processor.feature_extractor is mock_feature_extractor
        assert processor.tokenizer is mock_tokenizer
        assert processor.projector is mock_projector
        assert processor.audio_token_id == 12345

    def test_init_uses_default_conv_layers(
        self,
        mock_feature_extractor: MagicMock,
        mock_tokenizer: MagicMock,
        mock_projector: MagicMock,
    ) -> None:
        """Should use default conv layers when not specified."""
        processor = ASRProcessor(mock_feature_extractor, mock_tokenizer, mock_projector)

        assert processor.encoder_conv_layers == DEFAULT_ENCODER_CONV_LAYERS

    def test_init_accepts_custom_conv_layers(
        self,
        mock_feature_extractor: MagicMock,
        mock_tokenizer: MagicMock,
        mock_projector: MagicMock,
    ) -> None:
        """Should accept custom conv layer configuration."""
        custom_layers = [(0, 3, 2), (0, 3, 2)]
        processor = ASRProcessor(
            mock_feature_extractor,
            mock_tokenizer,
            mock_projector,
            encoder_conv_layers=custom_layers,
        )

        assert processor.encoder_conv_layers == custom_layers


class TestProcessorCall:
    """Tests for ASRProcessor.__call__ method."""

    @pytest.fixture
    def mock_processor(self, mocker: MockerFixture) -> ASRProcessor:
        """Create processor with mocked components."""
        fe = mocker.MagicMock()
        fe.sampling_rate = 16000
        fe.return_value = {
            "input_features": torch.randn(1, 80, 100),
            "attention_mask": torch.ones(1, 100),
        }

        tok = mocker.MagicMock()
        tok.convert_tokens_to_ids.return_value = 12345
        tok.apply_chat_template.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        proj = mocker.MagicMock()
        proj.get_output_length.side_effect = functools.partial(torch.full_like, fill_value=50)

        return ASRProcessor(fe, tok, proj)

    def test_call_with_audio_only(self, mock_processor: ASRProcessor) -> None:
        """Should process audio and build prompt."""
        audio = torch.randn(16000)
        result = mock_processor(audio=audio)

        assert "input_features" in result
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_call_builds_user_message(self, mock_processor: ASRProcessor) -> None:
        """Should build message with audio tokens."""
        audio = torch.randn(16000)
        mock_processor(audio=audio)

        cast(MagicMock, mock_processor.tokenizer).apply_chat_template.assert_called()
        call_args = cast(MagicMock, mock_processor.tokenizer).apply_chat_template.call_args
        messages = call_args[0][0]

        user_msg = next(m for m in messages if m["role"] == "user")
        assert "<audio>" in user_msg["content"]

    def test_call_without_audio(self, mock_processor: ASRProcessor) -> None:
        """Should handle call without audio (text-only)."""
        result = mock_processor(text="hello")

        assert "input_ids" in result
        assert "input_features" not in result

    def test_call_with_text_adds_assistant_message(self, mock_processor: ASRProcessor) -> None:
        """Text should be added as assistant response."""
        audio = torch.randn(16000)
        mock_processor(audio=audio, text="hello world")

        call_args = cast(MagicMock, mock_processor.tokenizer).apply_chat_template.call_args
        messages = call_args[0][0]

        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "hello world"


class TestProcessorAudioTokenCount:
    """Tests for correct audio token counting."""

    @pytest.fixture
    def processor_with_variable_attention(self, mocker: MockerFixture) -> ASRProcessor:
        """Create processor that returns variable attention masks."""
        fe = mocker.MagicMock()
        fe.sampling_rate = 16000

        tok = mocker.MagicMock()
        tok.convert_tokens_to_ids.return_value = 12345
        tok.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])

        proj = mocker.MagicMock()
        proj.get_output_length.side_effect = _quarter

        return ASRProcessor(fe, tok, proj)

    def test_audio_token_count_from_attention_mask(
        self, processor_with_variable_attention: ASRProcessor
    ) -> None:
        """Audio token count should be based on actual audio length."""
        processor = processor_with_variable_attention

        # Create attention mask with 80 valid frames
        cast(MagicMock, processor.feature_extractor).return_value = {
            "input_features": torch.randn(1, 80, 100),
            "attention_mask": torch.cat([torch.ones(1, 80), torch.zeros(1, 20)], dim=1),
        }

        audio = torch.randn(16000)
        processor(audio=audio)

        call_args = cast(MagicMock, processor.tokenizer).apply_chat_template.call_args
        messages = call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")

        audio_tokens = user_msg["content"].count("<audio>")

        # Encoder output: 80 -> conv1 -> 80 -> conv2 -> 40
        # Projector: 40 -> 40 // 4 = 10
        assert audio_tokens == 10


class TestProcessorRaggedBatch:
    """A batch of unequal-length audio must get one prompt per sample."""

    @pytest.fixture
    def batch_processor(self, mocker: MockerFixture) -> ASRProcessor:
        """Processor over a 2-sample batch: 80 valid mel frames vs 40."""
        fe = mocker.MagicMock()
        fe.sampling_rate = 16000
        fe.return_value = {
            "input_features": torch.randn(2, 80, 100),
            "attention_mask": torch.stack(
                [
                    torch.cat([torch.ones(80), torch.zeros(20)]),
                    torch.cat([torch.ones(40), torch.zeros(60)]),
                ]
            ),
        }

        tok = mocker.MagicMock()
        tok.convert_tokens_to_ids.return_value = 12345
        tok.pad_token_id = 0

        # One id per audio placeholder, so row length tracks the count.
        def render(messages: list[dict[str, str]], **kw: Any) -> torch.Tensor:
            return torch.tensor([[12345] * messages[0]["content"].count("<audio>")])

        tok.apply_chat_template.side_effect = render

        proj = mocker.MagicMock()
        proj.get_output_length.side_effect = _quarter

        return ASRProcessor(fe, tok, proj)

    def test_prompt_rows_match_batch_size(self, batch_processor: ASRProcessor) -> None:
        """`input_ids` must have one row per audio sample, not one row total."""
        result = batch_processor(audio=[torch.randn(16000), torch.randn(8000)])

        assert result["input_ids"].shape[0] == result["input_features"].shape[0] == 2
        assert result["attention_mask"].shape == result["input_ids"].shape

    def test_each_row_gets_its_own_token_count(self, batch_processor: ASRProcessor) -> None:
        """Sizing every row from the batch max over-counts the shorter rows.

        Encoder: 80 -> 40 and 40 -> 20; projector: // 4 -> 10 and 5. The
        shorter row must carry 5 placeholders, not the batch max of 10 --
        `masked_scatter` mis-scatters when a row claims more audio positions
        than the projector produced for it.
        """
        result = batch_processor(audio=[torch.randn(16000), torch.randn(8000)])

        counts = [int((row == batch_processor.audio_token_id).sum()) for row in result["input_ids"]]
        assert counts == [10, 5]

    def test_shorter_row_is_left_padded(self, batch_processor: ASRProcessor) -> None:
        """Padding goes on the left so it never sits before the first new token."""
        result = batch_processor(audio=[torch.randn(16000), torch.randn(8000)])

        short = result["attention_mask"][1]
        assert short.tolist() == [0] * 5 + [1] * 5

    def test_projector_is_required_for_audio(
        self, mock_feature_extractor: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Without a projector the token count is unknowable -- say so."""
        processor = ASRProcessor(mock_feature_extractor, mock_tokenizer)

        with pytest.raises(ValueError, match="needs a projector"):
            processor(audio=torch.randn(16000))


class TestProcessorAudioToken:
    """The placeholder token must come from the config, not a hardcoded default."""

    def test_defaults_to_class_constant(
        self,
        mock_feature_extractor: MagicMock,
        mock_tokenizer: MagicMock,
        mock_projector: MagicMock,
    ) -> None:
        processor = ASRProcessor(mock_feature_extractor, mock_tokenizer, mock_projector)

        assert processor.audio_token == ASRProcessor.AUDIO_TOKEN

    def test_accepts_native_decoder_token(
        self,
        mock_feature_extractor: MagicMock,
        mock_tokenizer: MagicMock,
        mock_projector: MagicMock,
    ) -> None:
        """Gemma 4's placeholder is its own pretrained "<|audio|>", not "<audio>"."""
        processor = ASRProcessor(
            mock_feature_extractor,
            mock_tokenizer,
            mock_projector,
            audio_token="<|audio|>",
        )

        assert processor.audio_token == "<|audio|>"
        mock_tokenizer.convert_tokens_to_ids.assert_called_with("<|audio|>")

    def test_prompt_uses_configured_token(
        self,
        mock_feature_extractor: MagicMock,
        mock_tokenizer: MagicMock,
        mock_projector: MagicMock,
    ) -> None:
        """The prompt must repeat the configured token, or masked_scatter mismatches."""
        mock_projector.get_output_length.side_effect = functools.partial(
            torch.full_like, fill_value=3
        )
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])

        processor = ASRProcessor(
            mock_feature_extractor,
            mock_tokenizer,
            mock_projector,
            audio_token="<|audio|>",
        )
        processor(audio=[0.0] * 16000)

        messages = mock_tokenizer.apply_chat_template.call_args[0][0]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert user_content.startswith("<|audio|>" * 3)
        assert "<audio>" not in user_content

    def test_model_processor_inherits_config_token(self, base_asr_model: ASRModel) -> None:
        """ASRModel.get_processor must forward its resolved audio_token."""
        processor = base_asr_model.get_processor()

        assert processor.audio_token == base_asr_model.audio_token
        assert processor.audio_token_id == base_asr_model.audio_token_id
