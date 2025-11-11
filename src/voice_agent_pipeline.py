"""
Voice Agent Pipeline - Combines ASR and TTS in a single pipeline.

This pipeline replaces the process_audio_for_agent method by providing
a clean pipeline interface for voice agents that need both speech-to-text
and text-to-speech functionality.
"""

import time
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

try:
    from .asr_pipeline import ASRPipeline
except ImportError:
    from asr_pipeline import ASRPipeline  # type: ignore[no-redef]


class VoiceAgentPipeline(ASRPipeline):
    """
    Pipeline for voice agents that combines ASR and TTS.

    This pipeline extends ASRPipeline to add TTS generation capabilities,
    making it perfect for conversational voice agents.
    """

    def __call__(
        self,
        inputs: Optional[Union[Dict, np.ndarray, torch.Tensor, str]] = None,
        text: Optional[str] = None,
        return_audio: bool = True,
        tts_voice: Optional[str] = None,
        tts_speed: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process audio to text and optionally generate TTS response.

        Args:
            inputs: Audio input (dict with 'raw' and 'sampling_rate', or array/tensor)
                   OR text string for TTS-only mode
            text: Direct text input for TTS-only mode (alternative to inputs)
            return_audio: Whether to generate TTS audio from the result
            tts_voice: Optional TTS voice override
            tts_speed: Optional TTS speed override
            **kwargs: Additional arguments passed to ASR pipeline

        Returns:
            Dictionary with:
                - text: Generated or input text
                - audio: TTS audio array (if return_audio=True and TTS enabled)
                - audio_sample_rate: Sample rate of TTS audio (usually 24000)
                - processing_time: Total processing time
        """
        start_time = time.time()
        result = {}

        # Handle text-only input for direct TTS
        if isinstance(inputs, str):
            text = inputs
            inputs = None

        # Process audio if provided
        if inputs is not None and text is None:
            # Use parent ASR pipeline
            asr_result = super().__call__(inputs, **kwargs)
            text = asr_result.get("text", "")
            result["text"] = text
        elif text is not None:
            # Direct text provided
            result["text"] = text
        else:
            raise ValueError("Either audio inputs or text must be provided")

        # Generate TTS if requested and text is available
        if return_audio and text and self.model.tts_enabled:
            try:
                # Configure TTS if parameters provided
                if tts_voice or tts_speed:
                    original_voice = self.model.tts_voice
                    original_speed = self.model.tts_speed

                    if tts_voice:
                        self.model.tts_voice = tts_voice
                    if tts_speed:
                        self.model.tts_speed = tts_speed

                    try:
                        tts_result = self.model._generate_tts(text)
                        if tts_result is not None:
                            tts_audio, _ = tts_result  # _generate_tts returns (audio, sample_rate)
                        else:
                            tts_audio = None
                    finally:
                        # Restore original settings
                        self.model.tts_voice = original_voice
                        self.model.tts_speed = original_speed
                else:
                    tts_result = self.model._generate_tts(text)
                    if tts_result is not None:
                        tts_audio, _ = tts_result  # _generate_tts returns (audio, sample_rate)
                    else:
                        tts_audio = None

                if tts_audio is not None:
                    result["audio"] = tts_audio
                    result["audio_sample_rate"] = getattr(self.model, "tts_sample_rate", 24000)

            except Exception as e:
                # Log but don't fail if TTS fails
                import logging

                logging.warning(f"TTS generation failed: {e}")

        result["processing_time"] = time.time() - start_time
        return result

    def process_streaming(self, audio_stream, task: str = "continue", **kwargs):
        """
        Process streaming audio for real-time agents.

        This is useful for LiveKit or other real-time applications.

        Args:
            audio_stream: Iterator or generator of audio chunks
            task: Task to perform (continue, transcribe, etc.)
            **kwargs: Additional pipeline arguments

        Yields:
            Dict with text and optionally audio for each processed chunk
        """
        for audio_chunk in audio_stream:
            result = self(audio_chunk, task=task, **kwargs)
            yield result

    @staticmethod
    def list_available_voices() -> Dict[str, list]:
        """
        List available TTS voices organized by language.

        Returns:
            Dictionary mapping language codes to voice lists with descriptions
        """
        return {
            "american": [  # American English (code: 'a')
                {"id": "af_heart", "name": "Heart", "gender": "female"},
                {"id": "af_bella", "name": "Bella", "gender": "female"},
                {"id": "af_sarah", "name": "Sarah", "gender": "female"},
                {"id": "af_nicole", "name": "Nicole", "gender": "female"},
                {"id": "af_sky", "name": "Sky", "gender": "female"},
                {"id": "am_adam", "name": "Adam", "gender": "male"},
                {"id": "am_michael", "name": "Michael", "gender": "male"},
            ],
            "british": [  # British English (code: 'b')
                {"id": "bf_emma", "name": "Emma", "gender": "female"},
                {"id": "bf_isabella", "name": "Isabella", "gender": "female"},
                {"id": "bm_george", "name": "George", "gender": "male"},
                {"id": "bm_lewis", "name": "Lewis", "gender": "male"},
            ],
            # Additional languages can be added as Kokoro supports them
            "spanish": [],  # code: 'e'
            "french": [],  # code: 'f'
            "hindi": [],  # code: 'h'
            "italian": [],  # code: 'i'
            "japanese": [],  # code: 'j'
            "portuguese": [],  # code: 'p'
            "chinese": [],  # code: 'z'
        }


# Register the pipeline
import transformers

try:
    from .asr_config import ASRConfig
except ImportError:
    pass  # type: ignore[no-redef]

# Register for custom pipeline task
try:
    PIPELINE_REGISTRY = transformers.pipelines.PIPELINE_REGISTRY
    if "voice-agent" not in PIPELINE_REGISTRY.supported_tasks:
        PIPELINE_REGISTRY.register_pipeline(
            "voice-agent",
            pipeline_class=VoiceAgentPipeline,
            default={"model": {"pt": ("mazesmazes/tiny-audio", "main")}},
        )
except Exception:
    # Registration might fail in some environments, that's ok
    pass
