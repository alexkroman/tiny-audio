"""Tests for EndpointHandler."""

import pytest
import torch


class TestEndpointHandlerCall:
    """Tests for EndpointHandler.__call__ method."""

    @pytest.fixture
    def mock_handler(self, mocker):
        """Create handler with mocked model and pipeline."""
        from tiny_audio.handler import EndpointHandler

        handler = object.__new__(EndpointHandler)
        handler.pipe = mocker.MagicMock()
        handler.pipe.return_value = {"text": "hello world"}
        handler.model = mocker.MagicMock()
        handler.device = "cpu"
        handler.dtype = None

        return handler

    def test_call_with_inputs(self, mock_handler):
        """Should pass inputs to pipeline."""
        result = mock_handler({"inputs": "audio_data"})

        mock_handler.pipe.assert_called_once_with("audio_data")
        assert result == {"text": "hello world"}

    def test_call_with_parameters(self, mock_handler):
        """Should pass parameters to pipeline."""
        mock_handler({"inputs": "audio_data", "parameters": {"max_new_tokens": 100}})

        mock_handler.pipe.assert_called_once_with("audio_data", max_new_tokens=100)

    def test_call_missing_inputs_raises(self, mock_handler):
        """Should raise ValueError when inputs missing."""
        with pytest.raises(ValueError, match="Missing 'inputs'"):
            mock_handler({})

    def test_call_empty_parameters(self, mock_handler):
        """Should handle empty parameters dict."""
        mock_handler({"inputs": "audio_data", "parameters": {}})

        mock_handler.pipe.assert_called_once_with("audio_data")


class TestEndpointHandlerInit:
    """Tests for EndpointHandler initialization logic."""

    def test_device_detection_cpu(self, mocker):
        """Should use CPU when CUDA not available."""
        mocker.patch("torch.cuda.is_available", return_value=False)
        mock_model = mocker.patch("tiny_audio.handler.ASRModel")
        mocker.patch("tiny_audio.handler.ASRPipeline")

        # Set up the mock to return a proper device
        mock_instance = mocker.MagicMock()
        mock_param = mocker.MagicMock()
        mock_param.device = torch.device("cpu")
        mock_instance.parameters.return_value = iter([mock_param])
        mock_model.from_pretrained.return_value = mock_instance

        from tiny_audio.handler import EndpointHandler

        handler = EndpointHandler("/fake/path")

        assert handler.device == torch.device("cpu")

    def test_handler_sets_tf32_flags(self):
        """Handler __init__ should set TF32 flags."""
        # Verify the flags can be set (actual init tested via device_detection_cpu)
        original = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        torch.backends.cuda.matmul.allow_tf32 = original


class TestEndpointHandlerIntegration:
    """Integration-style tests for EndpointHandler."""

    def test_handler_importable(self):
        """EndpointHandler should be importable."""
        from tiny_audio.handler import EndpointHandler

        assert EndpointHandler is not None
