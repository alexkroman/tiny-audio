"""Integration adapters for voice agent frameworks."""

try:
    from .pipecat_stt import TinyAudioSTTService
except ImportError:
    # pipecat not installed - that's okay, make import conditional
    TinyAudioSTTService = None

__all__ = ["TinyAudioSTTService"]
