"""Pipecat STT service adapter for Tiny Audio model."""

from typing import Optional

try:
    from pipecat.frames.frames import (
        InterimTranscriptionFrame,
        TranscriptionFrame,
    )
    from pipecat.services.stt_service import SegmentedSTTService
except ImportError as err:
    raise ImportError(
        "pipecat-ai is required for this integration. Install with: pip install pipecat-ai[silero]"
    ) from err


class TinyAudioSTTService(SegmentedSTTService):
    """Pipecat STT service for Tiny Audio model with streaming output.

    This service integrates the Tiny Audio ASR model with Pipecat's
    voice agent pipeline. It supports streaming token output for
    reduced latency in voice applications.

    Args:
        model_id: HuggingFace model ID or local path (default: "mazesmazes/tiny-audio")
        streaming: Enable streaming token output for lower latency (default: True)
        device: Device to run model on (default: auto-detect)
        **kwargs: Additional arguments passed to SegmentedSTTService

    Example:
        >>> from tiny_audio.integrations import TinyAudioSTTService
        >>> stt = TinyAudioSTTService(model_id="mazesmazes/tiny-audio")
        >>> # Use in Pipecat pipeline
        >>> pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])
    """

    def __init__(
        self,
        *,
        model_id: str = "mazesmazes/tiny-audio",
        streaming: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pipeline = None
        self._model_id = model_id
        self._streaming = streaming
        self._device = device

    def _ensure_pipeline(self):
        """Lazy-load the ASR pipeline on first use."""
        if self._pipeline is not None:
            return

        try:
            from tiny_audio import ASRModel, ASRPipeline
        except ImportError:
            # Fallback for local development
            import sys
            from pathlib import Path

            src_path = Path(__file__).parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from asr_modeling import ASRModel
            from asr_pipeline import ASRPipeline

        # Load model and create pipeline (HF Pipeline handles device detection)
        model = ASRModel.from_pretrained(self._model_id)
        self._pipeline = ASRPipeline(model, device=self._device)
        model.eval()

    async def run_stt(self, audio: bytes):
        """Transcribe audio segment, yielding interim results if streaming enabled.

        Args:
            audio: Raw audio bytes (16-bit PCM at 16kHz - pipecat default)

        Yields:
            InterimTranscriptionFrame for partial results (if streaming)
            TranscriptionFrame for final result
        """
        self._ensure_pipeline()

        # Handle empty audio
        if len(audio) == 0:
            yield TranscriptionFrame(text="", user_id=self._user_id, timestamp="")
            return

        # Pass raw PCM bytes directly - pipeline handles conversion
        audio_input = {"raw_bytes": audio, "dtype": "int16", "sampling_rate": 16000}

        if self._streaming:
            # Stream tokens as generated for lower latency
            partial_text = ""
            for token_text in self._pipeline.transcribe_streaming(audio_input):
                partial_text += token_text
                yield InterimTranscriptionFrame(
                    text=partial_text,
                    user_id=self._user_id,
                    timestamp="",
                )
            # Yield final result
            yield TranscriptionFrame(
                text=partial_text.strip(),
                user_id=self._user_id,
                timestamp="",
            )
        else:
            # Non-streaming: use pipeline's __call__ method
            result = self._pipeline(audio_input)
            yield TranscriptionFrame(
                text=result.get("text", "").strip(),
                user_id=self._user_id,
                timestamp="",
            )
