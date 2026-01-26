"""Tests for ASRProcessor.

Uses pytest-mock and shared fixtures from conftest.py.
"""

import pytest
import torch


class TestComputeEncoderOutputLength:
    """Tests for ASRProcessor._compute_encoder_output_length method."""

    @pytest.fixture
    def processor_method(self):
        """Get the method without creating full processor."""
        from tiny_audio.asr_processing import ASRProcessor

        class MockProcessor:
            encoder_conv_layers = ASRProcessor.DEFAULT_ENCODER_CONV_LAYERS

            def _compute_encoder_output_length(self, mel_length: int) -> int:
                length = mel_length
                for padding, kernel_size, stride in self.encoder_conv_layers:
                    length = (length + 2 * padding - (kernel_size - 1) - 1) // stride + 1
                return length

        return MockProcessor()

    def test_default_conv_layers(self, processor_method):
        """Should compute correct length with default Whisper conv layers."""
        # Whisper default: [(1, 3, 1), (1, 3, 2)]
        # Layer 1: (100 + 2*1 - 2 - 1) // 1 + 1 = 100
        # Layer 2: (100 + 2*1 - 2 - 1) // 2 + 1 = 50
        result = processor_method._compute_encoder_output_length(100)
        assert result == 50

    def test_single_frame(self, processor_method):
        """Should handle single frame input."""
        result = processor_method._compute_encoder_output_length(1)
        assert result == 1

    def test_large_input(self, processor_method):
        """Should handle large input lengths."""
        result = processor_method._compute_encoder_output_length(3000)
        assert result == 1500  # Halved by stride=2


class TestProcessorConstants:
    """Tests for ASRProcessor constants."""

    def test_audio_token_defined(self):
        """AUDIO_TOKEN constant should be defined."""
        from tiny_audio.asr_processing import ASRProcessor

        assert hasattr(ASRProcessor, "AUDIO_TOKEN")
        assert ASRProcessor.AUDIO_TOKEN == "<audio>"

    def test_transcribe_prompt_defined(self):
        """TRANSCRIBE_PROMPT constant should be defined (empty for instruction-free)."""
        from tiny_audio.asr_processing import ASRProcessor

        assert hasattr(ASRProcessor, "TRANSCRIBE_PROMPT")
        # Instruction-free (AZEROS approach) - prompt is empty
        assert ASRProcessor.TRANSCRIBE_PROMPT == ""

    def test_default_conv_layers(self):
        """DEFAULT_ENCODER_CONV_LAYERS should match Whisper."""
        from tiny_audio.asr_processing import ASRProcessor

        expected = [(1, 3, 1), (1, 3, 2)]
        assert expected == ASRProcessor.DEFAULT_ENCODER_CONV_LAYERS


class TestProcessorInit:
    """Tests for ASRProcessor initialization."""

    def test_init_sets_attributes(self, mock_feature_extractor, mock_tokenizer, mock_projector):
        """Processor should store all attributes."""
        from tiny_audio.asr_processing import ASRProcessor

        processor = ASRProcessor(mock_feature_extractor, mock_tokenizer, mock_projector)

        assert processor.feature_extractor is mock_feature_extractor
        assert processor.tokenizer is mock_tokenizer
        assert processor.projector is mock_projector
        assert processor.audio_token_id == 12345

    def test_init_uses_default_conv_layers(
        self, mock_feature_extractor, mock_tokenizer, mock_projector
    ):
        """Should use default conv layers when not specified."""
        from tiny_audio.asr_processing import ASRProcessor

        processor = ASRProcessor(mock_feature_extractor, mock_tokenizer, mock_projector)

        assert processor.encoder_conv_layers == ASRProcessor.DEFAULT_ENCODER_CONV_LAYERS

    def test_init_accepts_custom_conv_layers(
        self, mock_feature_extractor, mock_tokenizer, mock_projector
    ):
        """Should accept custom conv layer configuration."""
        from tiny_audio.asr_processing import ASRProcessor

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
    def mock_processor(self, mocker):
        """Create processor with mocked components."""
        from tiny_audio.asr_processing import ASRProcessor

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
        proj.get_output_length.return_value = 50

        return ASRProcessor(fe, tok, proj)

    def test_call_with_audio_only(self, mock_processor):
        """Should process audio and build prompt."""
        audio = torch.randn(16000)
        result = mock_processor(audio=audio)

        assert "input_features" in result
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_call_builds_user_message(self, mock_processor):
        """Should build message with audio tokens."""
        audio = torch.randn(16000)
        mock_processor(audio=audio)

        mock_processor.tokenizer.apply_chat_template.assert_called()
        call_args = mock_processor.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        user_msg = [m for m in messages if m["role"] == "user"][0]
        assert "<audio>" in user_msg["content"]

    def test_call_with_system_prompt(self, mock_processor):
        """Should include system prompt when provided."""
        audio = torch.randn(16000)
        mock_processor(audio=audio, system_prompt="You are helpful.")

        call_args = mock_processor.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are helpful."

    def test_call_without_audio(self, mock_processor):
        """Should handle call without audio (text-only)."""
        result = mock_processor(text="hello")

        assert "input_ids" in result
        assert "input_features" not in result

    def test_call_with_text_adds_assistant_message(self, mock_processor):
        """Text should be added as assistant response."""
        audio = torch.randn(16000)
        mock_processor(audio=audio, text="hello world")

        call_args = mock_processor.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "hello world"


class TestProcessorAudioTokenCount:
    """Tests for correct audio token counting."""

    @pytest.fixture
    def processor_with_variable_attention(self, mocker):
        """Create processor that returns variable attention masks."""
        from tiny_audio.asr_processing import ASRProcessor

        fe = mocker.MagicMock()
        fe.sampling_rate = 16000

        tok = mocker.MagicMock()
        tok.convert_tokens_to_ids.return_value = 12345
        tok.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])

        proj = mocker.MagicMock()
        proj.get_output_length.side_effect = lambda x: x // 4

        return ASRProcessor(fe, tok, proj)

    def test_audio_token_count_from_attention_mask(self, processor_with_variable_attention):
        """Audio token count should be based on actual audio length."""
        processor = processor_with_variable_attention

        # Create attention mask with 80 valid frames
        processor.feature_extractor.return_value = {
            "input_features": torch.randn(1, 80, 100),
            "attention_mask": torch.cat([torch.ones(1, 80), torch.zeros(1, 20)], dim=1),
        }

        audio = torch.randn(16000)
        processor(audio=audio)

        call_args = processor.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        user_msg = [m for m in messages if m["role"] == "user"][0]

        audio_tokens = user_msg["content"].count("<audio>")

        # Encoder output: 80 -> conv1 -> 80 -> conv2 -> 40
        # Projector: 40 -> 40 // 4 = 10
        assert audio_tokens == 10
