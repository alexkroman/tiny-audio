"""Full-duplex audio session for speech-to-speech.

Implements Freeze-Omni style full-duplex conversation where the model
can listen and speak simultaneously, with support for user interruption.

Architecture:
- Dual queue system: PCMQueue (input) + AudioQueue (output)
- Multi-threaded: Listen thread + Generate thread run concurrently
- State machine: listen -> speak -> (interrupt) -> listen
- VAD-based turn detection using model's built-in Silero VAD

Usage (sync):
    session = FullDuplexSession(model)
    session.start()

    while has_audio:
        session.push_audio(audio_chunk)
        output = session.pop_audio()
        if output is not None:
            speaker.play(output)

    session.stop()

Usage (async/web):
    session = FullDuplexSession(
        model,
        on_state_change=lambda s: send_status(s),
        on_text=lambda t: send_text(t),
        on_audio=lambda a: send_audio(a),
    )
    session.start()

    # In your receive loop:
    session.push_audio(audio_chunk)
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from .asr_modeling import ASRModel

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """State machine for full-duplex conversation."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass
class FullDuplexConfig:
    """Configuration for full-duplex session."""

    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 512  # Samples per chunk (32ms at 16kHz)
    output_sample_rate: int = 24000  # Mimi output rate

    # VAD settings
    vad_threshold: float = 0.5
    silence_duration_ms: float = 700  # Silence to end turn
    min_speech_duration_ms: float = 100  # Minimum speech to trigger

    # Generation settings
    audio_chunk_size: int = 4  # Tokens per audio chunk

    # Timing
    poll_interval: float = 0.01


class PCMQueue:
    """Thread-safe queue for streaming PCM audio input."""

    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()

    def put(self, audio: np.ndarray) -> None:
        with self.lock:
            self.buffer = np.concatenate([self.buffer, audio.astype(np.float32)])

    def get(self, length: int) -> Optional[np.ndarray]:
        with self.lock:
            if len(self.buffer) < length:
                return None
            result = self.buffer[:length]
            self.buffer = self.buffer[length:]
            return result

    def clear(self) -> None:
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)


class AudioQueue:
    """Thread-safe queue for output audio chunks."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()

    def put(self, audio: torch.Tensor) -> None:
        self._queue.put(audio)

    def get(self) -> Optional[torch.Tensor]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def is_empty(self) -> bool:
        return self._queue.empty()


@dataclass
class _SessionState:
    """Internal state for full-duplex session."""

    state: ConversationState = ConversationState.IDLE
    speech_buffer: list = field(default_factory=list)
    speech_start_time: float = 0.0
    last_speech_time: float = 0.0
    silence_frames: int = 0
    stop_generate: bool = False
    is_generating: bool = False
    generated_text: str = ""


class FullDuplexSession:
    """Full-duplex speech-to-speech session (Freeze-Omni style).

    Manages simultaneous listening and speaking with VAD-based turn detection.
    Designed to be easy to integrate with both sync and async (web) applications.

    Args:
        model: ASRModel with audio_head configured
        config: FullDuplexConfig for session parameters
        on_state_change: Callback when state changes (state: ConversationState)
        on_text: Callback when text is generated (text: str, interim: bool)
        on_audio: Callback when audio chunk is ready (audio: torch.Tensor)
            If provided, audio is sent here instead of output_queue
        on_interrupted: Callback when generation is interrupted
    """

    def __init__(
        self,
        model: "ASRModel",
        config: Optional[FullDuplexConfig] = None,
        on_state_change: Optional[Callable[[ConversationState], None]] = None,
        on_text: Optional[Callable[[str, bool], None]] = None,
        on_audio: Optional[Callable[[torch.Tensor], None]] = None,
        on_interrupted: Optional[Callable[[], None]] = None,
    ):
        self.model = model
        self.config = config or FullDuplexConfig()

        # Callbacks
        self.on_state_change = on_state_change
        self.on_text = on_text
        self.on_audio = on_audio
        self.on_interrupted = on_interrupted

        # Queues
        self.input_queue = PCMQueue()
        self.output_queue = AudioQueue()

        # State
        self._state = _SessionState()
        self._running = False
        self._state_lock = threading.Lock()

        # Threads
        self._listen_thread: Optional[threading.Thread] = None
        self._generate_thread: Optional[threading.Thread] = None

        # Precompute timing thresholds
        ms_per_chunk = self.config.chunk_size / self.config.sample_rate * 1000
        self._silence_threshold = int(self.config.silence_duration_ms / ms_per_chunk)
        self._min_speech_chunks = int(self.config.min_speech_duration_ms / ms_per_chunk)

        # Ensure VAD is loaded
        self.model.load_vad()

    @property
    def state(self) -> ConversationState:
        with self._state_lock:
            return self._state.state

    def _set_state(self, value: ConversationState) -> None:
        with self._state_lock:
            old_state = self._state.state
            self._state.state = value
        if old_state != value:
            logger.debug(f"State: {old_state.value} -> {value.value}")
            if self.on_state_change:
                try:
                    self.on_state_change(value)
                except Exception as e:
                    logger.error(f"on_state_change callback error: {e}")

    @property
    def is_generating(self) -> bool:
        with self._state_lock:
            return self._state.is_generating

    @property
    def generated_text(self) -> str:
        with self._state_lock:
            return self._state.generated_text

    def start(self) -> None:
        """Start the full-duplex session."""
        if self._running:
            return

        self._running = True
        self._set_state(ConversationState.LISTENING)

        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

        logger.info("Full-duplex session started")

    def stop(self) -> None:
        """Stop the full-duplex session."""
        self._running = False

        with self._state_lock:
            self._state.stop_generate = True

        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        if self._generate_thread:
            self._generate_thread.join(timeout=2.0)

        self.input_queue.clear()
        self.output_queue.clear()
        self._set_state(ConversationState.IDLE)

        logger.info("Full-duplex session stopped")

    def push_audio(self, audio: np.ndarray) -> None:
        """Push audio samples to the input queue.

        Args:
            audio: Audio samples as numpy array (float32 normalized or int16)
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        self.input_queue.put(audio)

    def pop_audio(self) -> Optional[torch.Tensor]:
        """Pop generated audio from the output queue.

        Only used if on_audio callback is not set.

        Returns:
            Audio tensor [samples] or None
        """
        return self.output_queue.get()

    def interrupt(self) -> None:
        """Interrupt current generation and return to listening."""
        with self._state_lock:
            self._state.stop_generate = True

        # Wait for generation to stop
        timeout = 2.0
        start = time.time()
        while self._state.is_generating and (time.time() - start) < timeout:
            time.sleep(self.config.poll_interval)

        # Clear output queue
        self.output_queue.clear()

        # Reset state
        with self._state_lock:
            self._state.stop_generate = False
            self._state.generated_text = ""
            self._state.speech_buffer.clear()
            self._state.silence_frames = 0

        self._set_state(ConversationState.LISTENING)
        self.model.reset_vad_state()

        if self.on_interrupted:
            try:
                self.on_interrupted()
            except Exception as e:
                logger.error(f"on_interrupted callback error: {e}")

        logger.debug("Generation interrupted")

    def _emit_audio(self, audio: torch.Tensor) -> None:
        """Send audio to callback or queue."""
        if self.on_audio:
            try:
                self.on_audio(audio)
            except Exception as e:
                logger.error(f"on_audio callback error: {e}")
        else:
            self.output_queue.put(audio)

    def _emit_text(self, text: str, interim: bool = False) -> None:
        """Send text to callback."""
        if self.on_text:
            try:
                self.on_text(text, interim)
            except Exception as e:
                logger.error(f"on_text callback error: {e}")

    def _listen_loop(self) -> None:
        """Main listening loop - processes audio and detects speech."""
        is_speaking = False

        while self._running:
            audio = self.input_queue.get(self.config.chunk_size)
            if audio is None:
                time.sleep(self.config.poll_interval)
                continue

            # Run VAD
            audio_tensor = torch.from_numpy(audio)
            is_speech, prob = self.model.detect_speech(
                audio_tensor,
                self.config.sample_rate,
                self.config.vad_threshold,
            )

            current_time = time.time()

            # Check for interruption during generation
            if self._state.is_generating and is_speech:
                logger.debug(f"Interruption detected (prob={prob:.2f})")
                self.interrupt()
                # Start new utterance with this chunk
                is_speaking = True
                with self._state_lock:
                    self._state.speech_buffer = [audio]
                    self._state.speech_start_time = current_time
                    self._state.last_speech_time = current_time
                    self._state.silence_frames = 0
                continue

            # Normal VAD state machine
            if is_speech:
                if not is_speaking:
                    is_speaking = True
                    with self._state_lock:
                        self._state.speech_buffer = []
                        self._state.speech_start_time = current_time
                with self._state_lock:
                    self._state.speech_buffer.append(audio)
                    self._state.last_speech_time = current_time
                    self._state.silence_frames = 0

            elif is_speaking:
                with self._state_lock:
                    self._state.speech_buffer.append(audio)
                    self._state.silence_frames += 1

                    if self._state.silence_frames >= self._silence_threshold:
                        is_speaking = False

                        # Check minimum speech duration
                        if len(self._state.speech_buffer) >= self._min_speech_chunks:
                            speech_audio = np.concatenate(self._state.speech_buffer)
                            self._state.speech_buffer = []
                            self._state.silence_frames = 0

                            # Start generation
                            self._generate_thread = threading.Thread(
                                target=self._generate_loop,
                                args=(speech_audio,),
                                daemon=True,
                            )
                            self._generate_thread.start()
                        else:
                            self._state.speech_buffer = []
                            self._state.silence_frames = 0

    def _generate_loop(self, speech_audio: np.ndarray) -> None:
        """Generation loop - produces text and audio response."""
        with self._state_lock:
            self._state.is_generating = True
            self._state.generated_text = ""
            self._state.stop_generate = False

        try:
            self._set_state(ConversationState.PROCESSING)

            # Process input audio
            device = next(self.model.language_model.parameters()).device
            inputs = self.model._process_audio(speech_audio, self.config.sample_rate)
            input_features = inputs["input_features"]
            audio_attention_mask = inputs["attention_mask"]

            # Encode
            audio_embeds = self.model._encode_audio(input_features, audio_attention_mask)
            input_ids, attention_mask = self.model._build_audio_prompt(
                audio_attention_mask, 1, device
            )
            inputs_embeds = self.model._inject_audio_embeddings(input_ids, audio_embeds)

            # Check for interruption
            if self._state.stop_generate:
                return

            # Generate text
            with torch.no_grad():
                output = self.model.language_model.generate(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    generation_config=self.model.generation_config,
                )

            if self._state.stop_generate:
                return

            # Extract text
            text_ids = output[:, input_ids.shape[1] :]
            text = self.model.tokenizer.decode(text_ids[0], skip_special_tokens=True)

            with self._state_lock:
                self._state.generated_text = text

            self._emit_text(text, interim=False)

            if self._state.stop_generate:
                return

            # Generate audio
            if self.model.audio_head is not None:
                self._set_state(ConversationState.SPEAKING)

                with torch.no_grad():
                    lm_output = self.model.language_model(
                        input_ids=text_ids,
                        output_hidden_states=True,
                    )
                    embeddings = lm_output.hidden_states[-1]

                for audio_chunk in self.model.audio_head.generate_streaming(
                    embeddings=embeddings,
                    chunk_size=self.config.audio_chunk_size,
                ):
                    if self._state.stop_generate:
                        return
                    self._emit_audio(audio_chunk)

            self._set_state(ConversationState.LISTENING)

        except Exception as e:
            logger.error(f"Generation error: {e}")
            self._set_state(ConversationState.LISTENING)

        finally:
            with self._state_lock:
                self._state.is_generating = False
