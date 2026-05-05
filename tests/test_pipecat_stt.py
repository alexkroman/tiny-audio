"""Tests for TinyAudioSTTService pipecat integration.

Note: These tests require pipecat to be installed. Tests for _ensure_model
and run_stt would require extensive mocking of the lazy ASRModel import
and are better tested via integration tests.
"""

import pytest

# Skip all tests if pipecat is not installed
pipecat = pytest.importorskip("pipecat")


class TestTinyAudioSTTServiceInit:
    """Tests for TinyAudioSTTService initialization."""

    def test_init_stores_model_id(self):
        """Should store model_id for lazy loading."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService(model_id="test/model")

        assert service._model_id == "test/model"
        assert service._model is None  # Lazy loaded

    def test_init_default_model_id(self):
        """Should use default model_id."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService()

        assert service._model_id == "mazesmazes/tiny-audio"

    def test_init_stores_streaming_flag(self):
        """Should store streaming flag."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService(streaming=False)

        assert service._streaming is False

    def test_init_default_streaming(self):
        """Should default to streaming=True."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService()

        assert service._streaming is True

    def test_init_stores_device(self):
        """Should store device."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService(device="cpu")

        assert service._device == "cpu"

    def test_init_default_device_none(self):
        """Should default device to None (auto-detect)."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService()

        assert service._device is None


class TestTinyAudioSTTServiceClass:
    """Tests for TinyAudioSTTService class structure."""

    def test_inherits_segmented_stt_service(self):
        """Should inherit from SegmentedSTTService."""
        from pipecat.services.stt_service import SegmentedSTTService

        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        assert issubclass(TinyAudioSTTService, SegmentedSTTService)

    def test_has_ensure_model_method(self):
        """Should have _ensure_model method."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        assert hasattr(TinyAudioSTTService, "_ensure_model")

    def test_has_run_stt_method(self):
        """Should have run_stt method."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        assert hasattr(TinyAudioSTTService, "run_stt")

    def test_run_stt_is_async_generator(self):
        """run_stt should be an async generator method."""
        import inspect

        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        assert inspect.isasyncgenfunction(TinyAudioSTTService.run_stt)


class TestTinyAudioSTTServiceDocstring:
    """Tests for TinyAudioSTTService documentation."""

    def test_has_docstring(self):
        """Class should have a docstring."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        assert TinyAudioSTTService.__doc__ is not None
        assert len(TinyAudioSTTService.__doc__) > 50  # Meaningful docstring

    def test_docstring_mentions_pipecat(self):
        """Docstring should mention Pipecat."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        assert "Pipecat" in TinyAudioSTTService.__doc__

    def test_docstring_has_example(self):
        """Docstring should have usage example."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        assert "Example:" in TinyAudioSTTService.__doc__


class TestEnsureModel:
    """_ensure_model lazy-loads the ASRModel and selects a device."""

    def test_ensure_model_calls_from_pretrained_once(self):
        from unittest.mock import MagicMock, patch

        import torch

        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService(model_id="test/model", device="cpu")

        fake_model = MagicMock()
        fake_param = MagicMock()
        fake_param.dtype = torch.float32
        fake_model.parameters.return_value = iter([fake_param])

        with patch("tiny_audio.ASRModel.from_pretrained", return_value=fake_model) as mock_fp:
            service._ensure_model()
            service._ensure_model()  # second call should be a no-op

        assert mock_fp.call_count == 1
        assert service._model is fake_model
        assert service._model_dtype == torch.float32

    def test_ensure_model_auto_detects_device(self):
        from unittest.mock import MagicMock, patch

        import torch

        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService(model_id="test/model")  # device=None

        fake_model = MagicMock()
        fake_param = MagicMock()
        fake_param.dtype = torch.float32
        fake_model.parameters.return_value = iter([fake_param])

        with patch("tiny_audio.ASRModel.from_pretrained", return_value=fake_model), patch(
            "torch.backends.mps.is_available", return_value=False
        ), patch("torch.cuda.is_available", return_value=False):
            service._ensure_model()

        assert service._device == torch.device("cpu")


class TestRunStt:
    """run_stt yields TranscriptionFrame for empty audio and InterimFrames during streaming."""

    @pytest.mark.asyncio
    async def test_run_stt_empty_audio_yields_empty_frame(self):
        from unittest.mock import MagicMock

        import torch
        from pipecat.frames.frames import TranscriptionFrame

        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService(model_id="test/model", device="cpu")
        # Pre-set _model so _ensure_model is a no-op
        service._model = MagicMock()
        service._model_dtype = torch.float32
        service._user_id = "test_user"

        frames = []
        async for frame in service.run_stt(b""):
            frames.append(frame)

        assert len(frames) == 1
        assert isinstance(frames[0], TranscriptionFrame)
        assert frames[0].text == ""

    @pytest.mark.asyncio
    async def test_run_stt_streaming_yields_interim_frames(self):
        from unittest.mock import MagicMock

        import numpy as np
        import torch
        from pipecat.frames.frames import (
            InterimTranscriptionFrame,
            TranscriptionFrame,
        )

        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService(model_id="test/model", device="cpu", streaming=True)

        # Mock the model and its components
        fake_model = MagicMock()
        fake_inputs = MagicMock()
        fake_inputs.input_features = torch.zeros(1, 80, 100)
        fake_inputs.attention_mask = torch.ones(1, 100, dtype=torch.long)
        fake_model.feature_extractor.return_value = fake_inputs
        fake_model.generate_streaming.return_value = iter(["hello", " world"])

        service._model = fake_model
        service._model_dtype = torch.float32
        service._user_id = "test_user"

        # 16-bit PCM, 1 second of silence
        audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()

        frames = []
        async for frame in service.run_stt(audio_bytes):
            frames.append(frame)

        # Should produce 2 InterimTranscriptionFrames + 1 final TranscriptionFrame
        interim = [f for f in frames if isinstance(f, InterimTranscriptionFrame)]
        final = [f for f in frames if isinstance(f, TranscriptionFrame)]
        assert len(interim) == 2
        assert len(final) == 1
        assert final[0].text == "hello world"

    @pytest.mark.asyncio
    async def test_run_stt_non_streaming_yields_single_frame(self):
        from unittest.mock import MagicMock

        import numpy as np
        import torch
        from pipecat.frames.frames import TranscriptionFrame

        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        service = TinyAudioSTTService(model_id="test/model", device="cpu", streaming=False)

        fake_model = MagicMock()
        fake_inputs = MagicMock()
        fake_inputs.input_features = torch.zeros(1, 80, 100)
        fake_inputs.attention_mask = torch.ones(1, 100, dtype=torch.long)
        fake_model.feature_extractor.return_value = fake_inputs
        fake_model.generate.return_value = torch.tensor([[1, 2, 3]])
        fake_model.tokenizer.decode.return_value = "  hello world  "

        service._model = fake_model
        service._model_dtype = torch.float32
        service._user_id = "test_user"

        audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()

        frames = []
        async for frame in service.run_stt(audio_bytes):
            frames.append(frame)

        assert len(frames) == 1
        assert isinstance(frames[0], TranscriptionFrame)
        assert frames[0].text == "hello world"  # stripped
