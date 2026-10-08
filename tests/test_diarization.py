"""Tests for tiny_audio.diarization helpers."""

import torch

from tiny_audio.diarization import _get_device


class TestDeviceHelper:
    """_get_device returns a torch.device."""

    def test_returns_device(self) -> None:
        device = _get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")
