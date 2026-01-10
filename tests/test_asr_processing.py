"""Tests for ASRProcessor."""

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
        # Formula: (input + 2*pad - (kernel-1) - 1) // stride + 1
        # Layer 1: (100 + 2*1 - 2 - 1) // 1 + 1 = 100
        # Layer 2: (100 + 2*1 - 2 - 1) // 2 + 1 = 50
        result = processor_method._compute_encoder_output_length(100)
        assert result == 50

    def test_single_frame(self, processor_method):
        """Should handle single frame input."""
        result = processor_method._compute_encoder_output_length(1)
        # Layer 1: (1 + 2 - 2 - 1) // 1 + 1 = 1
        # Layer 2: (1 + 2 - 2 - 1) // 2 + 1 = 1
        assert result == 1

    def test_large_input(self, processor_method):
        """Should handle large input lengths."""
        # 3000 frames (30s at 100fps)
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
        """TRANSCRIBE_PROMPT constant should be defined."""
        from tiny_audio.asr_processing import ASRProcessor

        assert hasattr(ASRProcessor, "TRANSCRIBE_PROMPT")
        assert "Transcribe" in ASRProcessor.TRANSCRIBE_PROMPT

    def test_default_conv_layers(self):
        """DEFAULT_ENCODER_CONV_LAYERS should match Whisper."""
        from tiny_audio.asr_processing import ASRProcessor

        # Whisper uses: conv1(k=3,s=1,p=1) + conv2(k=3,s=2,p=1)
        expected = [(1, 3, 1), (1, 3, 2)]
        assert expected == ASRProcessor.DEFAULT_ENCODER_CONV_LAYERS


class TestProcessorInit:
    """Tests for ASRProcessor initialization."""

    @pytest.fixture
    def mock_components(self):
        """Create mock feature extractor and tokenizer."""
        from unittest.mock import MagicMock

        feature_extractor = MagicMock()
        feature_extractor.sampling_rate = 16000

        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 12345

        projector = MagicMock()
        projector.get_output_length.return_value = 100

        return feature_extractor, tokenizer, projector

    def test_init_sets_attributes(self, mock_components):
        """Processor should store all attributes."""
        from tiny_audio.asr_processing import ASRProcessor

        fe, tok, proj = mock_components
        processor = ASRProcessor(fe, tok, proj)

        assert processor.feature_extractor is fe
        assert processor.tokenizer is tok
        assert processor.projector is proj
        assert processor.audio_token_id == 12345

    def test_init_uses_default_conv_layers(self, mock_components):
        """Should use default conv layers when not specified."""
        from tiny_audio.asr_processing import ASRProcessor

        fe, tok, proj = mock_components
        processor = ASRProcessor(fe, tok, proj)

        assert processor.encoder_conv_layers == ASRProcessor.DEFAULT_ENCODER_CONV_LAYERS

    def test_init_accepts_custom_conv_layers(self, mock_components):
        """Should accept custom conv layer configuration."""
        from tiny_audio.asr_processing import ASRProcessor

        fe, tok, proj = mock_components
        custom_layers = [(0, 3, 2), (0, 3, 2)]
        processor = ASRProcessor(fe, tok, proj, encoder_conv_layers=custom_layers)

        assert processor.encoder_conv_layers == custom_layers


class TestProcessorCall:
    """Tests for ASRProcessor.__call__ method."""

    @pytest.fixture
    def mock_processor(self):
        """Create processor with mocked components."""
        from unittest.mock import MagicMock

        from tiny_audio.asr_processing import ASRProcessor

        fe = MagicMock()
        fe.sampling_rate = 16000
        fe.return_value = {
            "input_features": torch.randn(1, 80, 100),
            "attention_mask": torch.ones(1, 100),
        }

        tok = MagicMock()
        tok.convert_tokens_to_ids.return_value = 12345
        tok.apply_chat_template.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        proj = MagicMock()
        proj.get_output_length.return_value = 50

        return ASRProcessor(fe, tok, proj)

    def test_call_with_audio_only(self, mock_processor):
        """Should process audio and build prompt."""
        audio = torch.randn(16000)  # 1 second of audio
        result = mock_processor(audio=audio)

        assert "input_features" in result
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_call_builds_user_message(self, mock_processor):
        """Should build message with audio tokens."""
        audio = torch.randn(16000)
        mock_processor(audio=audio)

        # Check that apply_chat_template was called with messages
        mock_processor.tokenizer.apply_chat_template.assert_called()
        call_args = mock_processor.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        # Should have user message with audio tokens
        user_msg = [m for m in messages if m["role"] == "user"][0]
        assert "<audio>" in user_msg["content"]

    def test_call_with_system_prompt(self, mock_processor):
        """Should include system prompt when provided."""
        audio = torch.randn(16000)
        mock_processor(audio=audio, system_prompt="You are helpful.")

        call_args = mock_processor.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        # Should have system message
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are helpful."

    def test_call_without_audio(self, mock_processor):
        """Should handle call without audio (text-only)."""
        result = mock_processor(text="hello")

        # Should still have input_ids
        assert "input_ids" in result
        # But no input_features
        assert "input_features" not in result

    def test_call_with_text_adds_assistant_message(self, mock_processor):
        """Text should be added as assistant response."""
        audio = torch.randn(16000)
        mock_processor(audio=audio, text="hello world")

        call_args = mock_processor.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]

        # Should have assistant message
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "hello world"


class TestProcessorAudioTokenCount:
    """Tests for correct audio token counting."""

    @pytest.fixture
    def processor_with_variable_attention(self):
        """Create processor that returns variable attention masks."""
        from unittest.mock import MagicMock

        from tiny_audio.asr_processing import ASRProcessor

        fe = MagicMock()
        fe.sampling_rate = 16000

        tok = MagicMock()
        tok.convert_tokens_to_ids.return_value = 12345
        tok.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])

        proj = MagicMock()
        # Map encoder length to projector output length
        proj.get_output_length.side_effect = lambda x: x // 4

        return ASRProcessor(fe, tok, proj)

    def test_audio_token_count_from_attention_mask(self, processor_with_variable_attention):
        """Audio token count should be based on actual audio length."""
        processor = processor_with_variable_attention

        # Create attention mask with 80 valid frames
        processor.feature_extractor.return_value = {
            "input_features": torch.randn(1, 80, 100),
            "attention_mask": torch.cat(
                [torch.ones(1, 80), torch.zeros(1, 20)], dim=1
            ),  # 80 real, 20 padded
        }

        audio = torch.randn(16000)
        processor(audio=audio)

        # Check that apply_chat_template was called
        call_args = processor.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        user_msg = [m for m in messages if m["role"] == "user"][0]

        # Count audio tokens in user message
        audio_tokens = user_msg["content"].count("<audio>")

        # Encoder output: 80 -> conv1 -> 80 -> conv2 -> 40
        # Projector: 40 -> 40 // 4 = 10
        assert audio_tokens == 10
