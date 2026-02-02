"""Pipecat Speech-to-Speech service that wraps Tiny Audio Omni + Kokoro TTS.

This service presents tiny-audio-omni as a unified speech-to-speech model,
taking audio input and producing audio output directly.

Uses streaming generation + streaming TTS for minimal latency.
"""

import re
from typing import Optional

import numpy as np

try:
    from pipecat.frames.frames import (
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
    )
    from pipecat.services.stt_service import SegmentedSTTService
except ImportError as err:
    raise ImportError(
        "pipecat-ai is required for this integration. Install with: pip install pipecat-ai[silero]"
    ) from err


# Sentence-ending punctuation for chunking text to TTS
SENTENCE_END_PATTERN = re.compile(r"[.!?]+\s*")


class TinyAudioS2SService(SegmentedSTTService):
    """Speech-to-Speech service using Tiny Audio Omni + Kokoro TTS.

    Uses streaming for both LLM generation and TTS to minimize latency.
    Text is streamed token-by-token and sent to TTS at sentence boundaries.
    """

    def __init__(
        self,
        *,
        model_id: str = "mazesmazes/tiny-audio-omni",
        tts_voice: str = "af_heart",
        system_prompt: str = "",
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = None
        self._feature_extractor = None
        self._tts_pipeline = None
        self._model_id = model_id
        self._tts_voice = tts_voice
        self._system_prompt = system_prompt
        self._device = device
        self._tts_sample_rate = 24000
        self._conversation_history: list[dict[str, str]] = []  # [{role, content}, ...]
        self._max_history_turns = 10  # Keep last N turns to avoid context overflow

    def _ensure_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        import torch

        from tiny_audio.asr_modeling import ASRModel

        if self._device is None:
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

        self._model = ASRModel.from_pretrained(self._model_id)
        self._model.to(self._device)
        self._model.eval()

        if self._system_prompt:
            self._model.system_prompt = self._system_prompt

        # Get feature extractor for audio preprocessing
        self._feature_extractor = self._model.get_processor().feature_extractor

    def _ensure_tts(self):
        """Lazy-load Kokoro TTS on first use."""
        if self._tts_pipeline is not None:
            return

        from kokoro import KPipeline

        self._tts_pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def _preprocess_audio(self, audio_array: np.ndarray, sample_rate: int = 16000):
        """Convert audio array to model input features."""
        # Use feature extractor to get mel spectrogram
        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Match model dtype (usually bfloat16)
        model_dtype = next(self._model.audio_tower.parameters()).dtype
        input_features = inputs.input_features.to(self._device, dtype=model_dtype)
        attention_mask = inputs.attention_mask.to(self._device)

        return input_features, attention_mask

    def _build_system_prompt(self) -> str:
        """Build system prompt with conversation history."""
        prompt = self._system_prompt

        if self._conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for turn in self._conversation_history[-self._max_history_turns :]:
                history_text += f"{turn['role'].title()}: {turn['content']}\n"
            prompt = prompt + history_text

        return prompt

    async def run_stt(self, audio: bytes):
        """Process audio and stream speech response.

        Uses streaming generation to get text incrementally, then streams
        each sentence to TTS as soon as it's complete.
        """
        self._ensure_model()
        self._ensure_tts()

        # Convert bytes to float array
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_array) == 0:
            return

        # Preprocess audio to mel spectrogram
        input_features, attention_mask = self._preprocess_audio(audio_array)

        # Build system prompt with conversation history
        system_prompt = self._build_system_prompt()

        # Stream text generation
        text_buffer = ""
        full_response = ""  # Collect full response for history
        started = False

        for text_chunk in self._model.generate_streaming(
            input_features=input_features,
            audio_attention_mask=attention_mask,
            system_prompt=system_prompt,
        ):
            text_buffer += text_chunk
            full_response += text_chunk

            # Check for complete sentences
            while True:
                match = SENTENCE_END_PATTERN.search(text_buffer)
                if not match:
                    break

                # Extract complete sentence
                sentence = text_buffer[: match.end()].strip()
                text_buffer = text_buffer[match.end() :]

                if not sentence:
                    continue

                # Stream TTS for this sentence
                async for frame in self._stream_tts(sentence, started):
                    if not started:
                        started = True
                    yield frame

        # Handle any remaining text (no sentence-ending punctuation)
        if text_buffer.strip():
            async for frame in self._stream_tts(text_buffer.strip(), started):
                if not started:
                    started = True
                yield frame

        # Send stop frame if we started
        if started:
            yield TTSStoppedFrame()

        # Add to conversation history
        if full_response.strip():
            self._conversation_history.append({"role": "user", "content": "[audio input]"})
            self._conversation_history.append(
                {"role": "assistant", "content": full_response.strip()}
            )

    async def _stream_tts(self, text: str, already_started: bool = False):
        """Stream TTS audio for a piece of text.

        Yields audio frames as Kokoro generates them.
        """
        if not text:
            return

        if not already_started:
            yield TTSStartedFrame()

        # Kokoro yields (graphemes, phonemes, audio) tuples
        for _, _, audio_chunk in self._tts_pipeline(text, voice=self._tts_voice):
            if audio_chunk is None or len(audio_chunk) == 0:
                continue

            # Convert tensor to numpy if needed
            if hasattr(audio_chunk, "cpu"):
                audio_chunk = audio_chunk.cpu().numpy()

            # Convert to int16 and yield in small chunks for smooth playback
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            chunk_samples = int(self._tts_sample_rate * 0.02)  # 20ms chunks

            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i : i + chunk_samples]
                yield TTSAudioRawFrame(
                    audio=chunk.tobytes(),
                    sample_rate=self._tts_sample_rate,
                    num_channels=1,
                )

    async def say(self, text: str):
        """Synthesize text to speech directly with streaming."""
        self._ensure_model()
        self._ensure_tts()

        if not text:
            return

        yield TTSStartedFrame()

        for _, _, audio_chunk in self._tts_pipeline(text, voice=self._tts_voice):
            if audio_chunk is None or len(audio_chunk) == 0:
                continue

            # Convert tensor to numpy if needed
            if hasattr(audio_chunk, "cpu"):
                audio_chunk = audio_chunk.cpu().numpy()

            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            chunk_samples = int(self._tts_sample_rate * 0.02)

            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i : i + chunk_samples]
                yield TTSAudioRawFrame(
                    audio=chunk.tobytes(),
                    sample_rate=self._tts_sample_rate,
                    num_channels=1,
                )

        yield TTSStoppedFrame()
