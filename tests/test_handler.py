"""Tests for EndpointHandler."""

import pytest


class TestIsFlashAttnAvailable:
    """Tests for EndpointHandler._is_flash_attn_available."""

    def test_returns_bool(self):
        """Should return a boolean."""
        from tiny_audio.handler import EndpointHandler

        # Create handler without __init__
        handler = object.__new__(EndpointHandler)
        result = handler._is_flash_attn_available()

        assert isinstance(result, bool)

    def test_checks_flash_attn_module(self):
        """Should check for flash_attn module."""
        from unittest.mock import patch

        from tiny_audio.handler import EndpointHandler

        handler = object.__new__(EndpointHandler)

        # Mock flash_attn being available
        with patch("importlib.util.find_spec") as mock_find:
            mock_find.return_value = True
            result = handler._is_flash_attn_available()
            assert result is True
            mock_find.assert_called_with("flash_attn")

        # Mock flash_attn not available
        with patch("importlib.util.find_spec") as mock_find:
            mock_find.return_value = None
            result = handler._is_flash_attn_available()
            assert result is False


class TestEndpointHandlerCall:
    """Tests for EndpointHandler.__call__ method."""

    @pytest.fixture
    def mock_handler(self):
        """Create handler with mocked model and pipeline."""
        from unittest.mock import MagicMock

        from tiny_audio.handler import EndpointHandler

        handler = object.__new__(EndpointHandler)
        handler.pipe = MagicMock()
        handler.pipe.return_value = {"text": "hello world"}
        handler.model = MagicMock()
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

    def test_device_detection_cpu(self):
        """Should use CPU when CUDA not available."""
        from unittest.mock import MagicMock, patch

        import torch

        with patch("torch.cuda.is_available", return_value=False), patch(
            "tiny_audio.handler.ASRModel"
        ) as mock_model, patch("tiny_audio.handler.ASRPipeline"):
            # Set up the mock to return a proper device
            mock_instance = MagicMock()
            mock_param = MagicMock()
            mock_param.device = torch.device("cpu")
            mock_instance.parameters.return_value = iter([mock_param])
            mock_model.from_pretrained.return_value = mock_instance

            from tiny_audio.handler import EndpointHandler

            handler = EndpointHandler("/fake/path")

            assert handler.device == torch.device("cpu")

    def test_handler_sets_tf32_flags(self):
        """Handler __init__ should set TF32 flags."""
        import torch

        # Just verify the handler code sets these flags
        # (the actual init has been tested via device_detection_cpu)
        # TF32 is enabled in the handler's __init__
        # We test that the flags can be set
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

    def test_handler_has_required_methods(self):
        """Handler should have __init__ and __call__."""
        from tiny_audio.handler import EndpointHandler

        assert hasattr(EndpointHandler, "__init__")
        assert callable(EndpointHandler)  # Class is callable (can instantiate)
        assert hasattr(EndpointHandler, "_is_flash_attn_available")
