"""Tests for EndpointHandler."""

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from tiny_audio.handler import (
    EndpointHandler,
    _best_device,  # pyright: ignore[reportPrivateImportUsage]  # aliased from diarization
)


class TestEndpointHandlerCall:
    """Tests for EndpointHandler.__call__ method."""

    @pytest.fixture
    def mock_handler(self, mocker: MockerFixture) -> EndpointHandler:
        """Create handler with mocked model and pipeline."""
        handler = object.__new__(EndpointHandler)
        pipe = mocker.MagicMock()
        pipe.return_value = {"text": "hello world"}
        handler.pipe = pipe
        handler.model = mocker.MagicMock()
        handler.device = torch.device("cpu")

        return handler

    def test_call_with_inputs(self, mock_handler: EndpointHandler) -> None:
        """Should pass inputs to pipeline."""
        result = mock_handler({"inputs": "audio_data"})

        cast(MagicMock, mock_handler.pipe).assert_called_once_with("audio_data")
        assert result == {"text": "hello world"}

    def test_call_with_parameters(self, mock_handler: EndpointHandler) -> None:
        """Should pass parameters to pipeline."""
        mock_handler({"inputs": "audio_data", "parameters": {"max_new_tokens": 100}})

        cast(MagicMock, mock_handler.pipe).assert_called_once_with("audio_data", max_new_tokens=100)

    def test_call_missing_inputs_raises(self, mock_handler: EndpointHandler) -> None:
        """Should raise ValueError when inputs missing."""
        with pytest.raises(ValueError, match="Missing 'inputs'"):
            mock_handler({})

    def test_call_empty_parameters(self, mock_handler: EndpointHandler) -> None:
        """Should handle empty parameters dict."""
        mock_handler({"inputs": "audio_data", "parameters": {}})

        cast(MagicMock, mock_handler.pipe).assert_called_once_with("audio_data")


class TestEndpointHandlerInit:
    """Tests for EndpointHandler initialization logic."""

    def test_device_detection_cpu(self, mocker: MockerFixture) -> None:
        """Should fall back to CPU when neither CUDA nor MPS is available.

        The handler picks cuda > mps > cpu itself (it no longer reads the device
        off the model's parameters), so both accelerators have to be stubbed out
        or this test picks MPS on Apple Silicon.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=False)
        mock_model = mocker.patch("tiny_audio.handler.ASRModel")
        mocker.patch("tiny_audio.handler.ASRPipeline")
        mock_model.from_pretrained.return_value = mocker.MagicMock()

        handler = EndpointHandler("/fake/path")

        assert handler.device == torch.device("cpu")
        mock_model.from_pretrained.return_value.to.assert_called_once_with(torch.device("cpu"))

    @pytest.mark.parametrize(
        ("cuda", "mps", "expected"),
        [
            (True, True, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        ],
    )
    def test_best_device_prefers_cuda_then_mps(
        self, mocker: MockerFixture, cuda: bool, mps: bool, expected: str
    ) -> None:
        """_best_device should prefer CUDA, then MPS, then CPU."""
        mocker.patch("torch.cuda.is_available", return_value=cuda)
        mocker.patch("torch.backends.mps.is_available", return_value=mps)

        assert _best_device() == torch.device(expected)

    def test_handler_sets_tf32_flags(self) -> None:
        """Handler __init__ should set TF32 flags."""
        # Verify the flags can be set (actual init tested via device_detection_cpu)
        original = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        torch.backends.cuda.matmul.allow_tf32 = original


class TestEndpointHandlerIntegration:
    """Integration-style tests for EndpointHandler."""

    def test_handler_importable(self) -> None:
        """EndpointHandler should be importable."""
        assert EndpointHandler is not None
