"""Pipecat STT service adapter for Tiny Audio model."""

from typing import Optional

import numpy as np

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
        self._model = None
        self._model_id = model_id
        self._streaming = streaming
        self._device = device

    def _ensure_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            import torch

            try:
                from tiny_audio import ASRModel  # pyright: ignore[reportMissingImports]
            except ImportError:
                # Fallback for local development
                import sys
                from pathlib import Path

                src_path = Path(__file__).parent.parent
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                from asr_modeling import ASRModel

            # Determine device: MPS > CUDA > CPU
            if self._device is None:
                if torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                elif torch.cuda.is_available():
                    self._device = torch.device("cuda")
                else:
                    self._device = torch.device("cpu")

            # Load model and move to device
            self._model = ASRModel.from_pretrained(self._model_id)
            self._model.to(self._device)
            self._model.eval()

    async def run_stt(self, audio: bytes):
        """Transcribe audio segment, yielding interim results if streaming enabled.

        Args:
            audio: Raw audio bytes (16-bit PCM at 16kHz - pipecat default)

        Yields:
            InterimTranscriptionFrame for partial results (if streaming)
            TranscriptionFrame for final result
        """
        self._ensure_model()

        # Convert bytes to float32 array (16-bit PCM at 16kHz)
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Handle empty audio
        if len(audio_array) == 0:
            yield TranscriptionFrame(text="", user_id=self._user_id, timestamp="")
            return

        # Preprocess audio through feature extractor
        import torch

        inputs = self._model.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Get model dtype and cast inputs to match
        model_dtype = next(self._model.parameters()).dtype
        input_features = inputs.input_features.to(self._device, dtype=model_dtype)
        attention_mask = inputs.attention_mask.to(self._device)

        if self._streaming:
            # Stream tokens as generated for lower latency
            partial_text = ""
            for token_text in self._model.generate_streaming(
                input_features=input_features,
                audio_attention_mask=attention_mask,
            ):
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
            # Non-streaming: wait for full result
            with torch.no_grad():
                output = self._model.generate(
                    input_features=input_features,
                    audio_attention_mask=attention_mask,
                )

            text = self._model.tokenizer.decode(output[0], skip_special_tokens=True)
            yield TranscriptionFrame(
                text=text.strip(),
                user_id=self._user_id,
                timestamp="",
            )
