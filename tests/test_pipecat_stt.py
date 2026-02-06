"""Tests for TinyAudioSTTService pipecat integration.

Note: These tests require pipecat to be installed. Tests for _ensure_pipeline
and run_stt would require extensive mocking of the lazy ASRModel/ASRPipeline import
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
        assert service._pipeline is None  # Lazy loaded

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

    def test_has_ensure_pipeline_method(self):
        """Should have _ensure_pipeline method."""
        from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

        assert hasattr(TinyAudioSTTService, "_ensure_pipeline")

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
