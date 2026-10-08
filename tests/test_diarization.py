"""Tests for tiny_audio.diarization helpers."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tiny_audio import diarization
from tiny_audio.diarization import NemotronDiarizer, get_device


class TestDeviceHelper:
    """get_device returns a torch.device."""

    def test_returns_device(self) -> None:
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")


class TestNemotronGetInstance:
    """Model loading is guarded by the transformers feature check and cached."""

    @pytest.fixture(autouse=True)
    def reset_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(NemotronDiarizer, "_model", None)
        monkeypatch.setattr(NemotronDiarizer, "_processor", None)

    @staticmethod
    def _find_spec(spec: object) -> Callable[[str], object]:
        def find_spec(_name: str) -> object:
            return spec

        return find_spec

    def test_missing_model_type_raises_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(diarization.importlib.util, "find_spec", self._find_spec(None))
        with pytest.raises(ImportError, match="nemotron3_diarization"):
            NemotronDiarizer.get_instance()

    def test_loads_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model_cls, processor_cls = MagicMock(), MagicMock()
        monkeypatch.setattr(diarization.importlib.util, "find_spec", self._find_spec(object()))
        monkeypatch.setattr(diarization, "AutoModelForAudioFrameClassification", model_cls)
        monkeypatch.setattr(diarization, "AutoProcessor", processor_cls)
        monkeypatch.setattr(diarization, "get_device", lambda: torch.device("cpu"))

        assert NemotronDiarizer.get_instance() == NemotronDiarizer.get_instance()
        model_cls.from_pretrained.assert_called_once_with(NemotronDiarizer.MODEL_ID)
        model_cls.from_pretrained.return_value.eval.assert_called_once()
        processor_cls.from_pretrained.assert_called_once_with(NemotronDiarizer.MODEL_ID)


class TestNemotronActivity:
    """activity() resamples to 16 kHz and returns sigmoid probabilities."""

    @pytest.fixture
    def processor(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        model = MagicMock()
        model.device = torch.device("cpu")
        model.dtype = torch.float32
        model.return_value.logits = torch.zeros(1, 5, 8)
        processor = MagicMock()

        def get_instance(_: object) -> tuple[MagicMock, MagicMock]:
            return model, processor

        monkeypatch.setattr(NemotronDiarizer, "get_instance", classmethod(get_instance))
        return processor

    def test_returns_sigmoid_probs(self, processor: MagicMock) -> None:
        probs = NemotronDiarizer.activity(np.zeros(1600))
        assert probs.shape == (5, 8)
        np.testing.assert_allclose(probs, 0.5)
        audio = processor.call_args.args[0]
        assert audio.dtype == np.float32
        assert audio.shape == (1600,)
        assert processor.call_args.kwargs == {"sampling_rate": 16000}

    def test_resamples_other_rates(self, processor: MagicMock) -> None:
        NemotronDiarizer.activity(np.zeros(8000), sample_rate=8000)
        assert processor.call_args.args[0].shape == (16000,)
