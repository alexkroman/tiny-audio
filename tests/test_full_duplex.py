"""Tests for full-duplex audio session.

Uses lightweight mocks to avoid loading real models.
"""

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tiny_audio.full_duplex import (
    AudioQueue,
    ConversationState,
    FullDuplexConfig,
    FullDuplexSession,
    PCMQueue,
)


class TestConversationState:
    """Tests for the ConversationState enum."""

    def test_states_exist(self):
        assert ConversationState.IDLE.value == "idle"
        assert ConversationState.LISTENING.value == "listening"
        assert ConversationState.PROCESSING.value == "processing"
        assert ConversationState.SPEAKING.value == "speaking"

    def test_all_states(self):
        assert len(ConversationState) == 4


class TestFullDuplexConfig:
    """Tests for FullDuplexConfig defaults."""

    def test_defaults(self):
        config = FullDuplexConfig()
        assert config.sample_rate == 16000
        assert config.chunk_size == 512
        assert config.output_sample_rate == 44100
        assert config.vad_threshold == 0.5
        assert config.silence_duration_ms == 700
        assert config.min_speech_duration_ms == 100

    def test_custom_values(self):
        config = FullDuplexConfig(sample_rate=8000, vad_threshold=0.3)
        assert config.sample_rate == 8000
        assert config.vad_threshold == 0.3


class TestPCMQueue:
    """Tests for the thread-safe PCM audio input queue."""

    def test_put_and_get(self):
        q = PCMQueue()
        audio = np.ones(100, dtype=np.float32)
        q.put(audio)
        result = q.get(100)
        assert result is not None
        np.testing.assert_array_equal(result, audio)

    def test_get_returns_none_when_insufficient(self):
        q = PCMQueue()
        q.put(np.ones(50, dtype=np.float32))
        assert q.get(100) is None

    def test_get_consumes_data(self):
        q = PCMQueue()
        q.put(np.ones(200, dtype=np.float32))
        q.get(100)
        assert len(q) == 100

    def test_put_accumulates(self):
        q = PCMQueue()
        q.put(np.ones(50, dtype=np.float32))
        q.put(np.ones(50, dtype=np.float32))
        assert len(q) == 100

    def test_clear(self):
        q = PCMQueue()
        q.put(np.ones(100, dtype=np.float32))
        q.clear()
        assert len(q) == 0

    def test_int16_conversion(self):
        """Test that int16 audio is converted to float32."""
        q = PCMQueue()
        audio_int16 = np.array([16384, -16384], dtype=np.int16)
        q.put(audio_int16.astype(np.float32))
        result = q.get(2)
        assert result.dtype == np.float32

    def test_thread_safety(self):
        """Test concurrent put/get from multiple threads."""
        q = PCMQueue()
        results: list[int] = []

        def producer():
            for _ in range(100):
                q.put(np.ones(10, dtype=np.float32))

        def consumer():
            collected = 0
            for _ in range(200):
                chunk = q.get(10)
                if chunk is not None:
                    collected += len(chunk)
                time.sleep(0.001)
            results.append(collected)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Consumer should have gotten some data
        assert results[0] > 0


class TestAudioQueue:
    """Tests for the thread-safe output audio queue."""

    def test_put_and_get(self):
        q = AudioQueue()
        audio = torch.randn(1000)
        q.put(audio)
        result = q.get()
        assert result is not None
        torch.testing.assert_close(result, audio)

    def test_get_returns_none_when_empty(self):
        q = AudioQueue()
        assert q.get() is None

    def test_is_empty(self):
        q = AudioQueue()
        assert q.is_empty()
        q.put(torch.randn(10))
        assert not q.is_empty()

    def test_clear(self):
        q = AudioQueue()
        q.put(torch.randn(10))
        q.put(torch.randn(10))
        q.clear()
        assert q.is_empty()

    def test_fifo_order(self):
        q = AudioQueue()
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        q.put(a)
        q.put(b)
        torch.testing.assert_close(q.get(), a)
        torch.testing.assert_close(q.get(), b)


def _make_mock_model() -> MagicMock:
    """Create a mock ASRModel for FullDuplexSession tests."""
    model = MagicMock()

    # Mock language_model with parameters on CPU
    param = torch.nn.Parameter(torch.zeros(1))
    model.language_model.parameters.return_value = iter([param])

    # Mock VAD
    model.load_vad.return_value = None
    model.detect_speech.return_value = (False, 0.0)
    model.reset_vad_state.return_value = None

    # Mock audio processing
    model._process_audio.return_value = {
        "input_features": torch.randn(1, 80, 100),
        "attention_mask": torch.ones(1, 100),
    }
    model._encode_audio.return_value = torch.randn(1, 25, 512)
    model._build_audio_prompt.return_value = (
        torch.ones(1, 30, dtype=torch.long),
        torch.ones(1, 30, dtype=torch.long),
    )
    model._inject_audio_embeddings.return_value = torch.randn(1, 30, 512)

    # Mock generation
    model.language_model.generate.return_value = torch.ones(1, 40, dtype=torch.long)
    model.tokenizer.decode.return_value = "Hello world"

    # No audio head by default
    model.audio_head = None
    model.generation_config = MagicMock()

    return model


class TestFullDuplexSession:
    """Tests for FullDuplexSession."""

    def test_init_defaults(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        assert session.state == ConversationState.IDLE
        assert session.config.sample_rate == 16000
        model.load_vad.assert_called_once()

    def test_init_custom_config(self):
        model = _make_mock_model()
        config = FullDuplexConfig(sample_rate=8000)
        session = FullDuplexSession(model, config=config)
        assert session.config.sample_rate == 8000

    def test_start_sets_listening(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        session.start()
        try:
            assert session.state == ConversationState.LISTENING
            assert session._running
        finally:
            session.stop()

    def test_stop_sets_idle(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        session.start()
        session.stop()
        assert session.state == ConversationState.IDLE
        assert not session._running

    def test_start_idempotent(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        session.start()
        session.start()  # Should not start a second listen thread
        try:
            assert session._running
        finally:
            session.stop()

    def test_push_audio_float32(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        audio = np.ones(512, dtype=np.float32)
        session.push_audio(audio)
        assert len(session.input_queue) == 512

    def test_push_audio_int16_conversion(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        audio = np.array([16384, -16384], dtype=np.int16)
        session.push_audio(audio)
        assert len(session.input_queue) == 2

    def test_pop_audio_empty(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        assert session.pop_audio() is None

    def test_state_change_callback(self):
        states: list[ConversationState] = []
        model = _make_mock_model()
        session = FullDuplexSession(model, on_state_change=states.append)
        session.start()
        try:
            assert ConversationState.LISTENING in states
        finally:
            session.stop()
        assert ConversationState.IDLE in states

    def test_interrupt_clears_state(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        session.start()
        try:
            # Simulate some speech buffer
            with session._state_lock:
                session._state.speech_buffer = [np.ones(100)]
                session._state.silence_frames = 5
            session.interrupt()
            assert session.state == ConversationState.LISTENING
            assert len(session._state.speech_buffer) == 0
            assert session._state.silence_frames == 0
            assert session._state.generated_text == ""
            model.reset_vad_state.assert_called()
        finally:
            session.stop()

    def test_interrupt_callback(self):
        interrupted: list[bool] = []
        model = _make_mock_model()
        session = FullDuplexSession(model, on_interrupted=lambda: interrupted.append(True))
        session.start()
        try:
            session.interrupt()
            assert len(interrupted) == 1
        finally:
            session.stop()

    def test_emit_audio_to_queue(self):
        """Without on_audio callback, audio goes to output_queue."""
        model = _make_mock_model()
        session = FullDuplexSession(model)
        audio = torch.randn(1000)
        session._emit_audio(audio)
        result = session.pop_audio()
        torch.testing.assert_close(result, audio)

    def test_emit_audio_to_callback(self):
        """With on_audio callback, audio goes to callback instead of queue."""
        chunks: list[torch.Tensor] = []
        model = _make_mock_model()
        session = FullDuplexSession(model, on_audio=chunks.append)
        audio = torch.randn(1000)
        session._emit_audio(audio)
        assert len(chunks) == 1
        assert session.pop_audio() is None  # Queue should be empty

    def test_emit_text_callback(self):
        texts: list[tuple[str, bool]] = []
        model = _make_mock_model()
        session = FullDuplexSession(model, on_text=lambda t, interim: texts.append((t, interim)))
        session._emit_text("hello", interim=False)
        assert texts == [("hello", False)]

    def test_emit_text_no_callback(self):
        """No crash when on_text is not set."""
        model = _make_mock_model()
        session = FullDuplexSession(model)
        session._emit_text("hello")  # Should not raise

    def test_is_generating_property(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        assert not session.is_generating
        with session._state_lock:
            session._state.is_generating = True
        assert session.is_generating

    def test_generated_text_property(self):
        model = _make_mock_model()
        session = FullDuplexSession(model)
        assert session.generated_text == ""
        with session._state_lock:
            session._state.generated_text = "Hello"
        assert session.generated_text == "Hello"


class TestGenerateLoop:
    """Tests for the _generate_loop method."""

    def test_generates_text_and_emits(self):
        """Test that _generate_loop produces text and calls on_text."""
        texts: list[str] = []
        model = _make_mock_model()
        session = FullDuplexSession(model, on_text=lambda t, _: texts.append(t))

        speech = np.random.randn(16000).astype(np.float32)
        session._generate_loop(speech)

        assert len(texts) == 1
        assert texts[0] == "Hello world"
        assert session.generated_text == "Hello world"
        assert not session.is_generating  # Should be False after loop ends

    def test_state_transitions_without_audio_head(self):
        """Without audio_head, goes PROCESSING -> LISTENING (no SPEAKING)."""
        states: list[ConversationState] = []
        model = _make_mock_model()
        session = FullDuplexSession(model, on_state_change=states.append)

        speech = np.random.randn(16000).astype(np.float32)
        session._generate_loop(speech)

        assert ConversationState.PROCESSING in states
        assert ConversationState.LISTENING in states
        assert ConversationState.SPEAKING not in states

    def test_state_transitions_with_audio_head(self):
        """With audio_head, goes PROCESSING -> SPEAKING -> LISTENING."""
        states: list[ConversationState] = []
        model = _make_mock_model()
        audio_head = MagicMock()
        audio_head.generate_streaming.return_value = iter([torch.randn(1000), torch.randn(1000)])
        model.audio_head = audio_head

        session = FullDuplexSession(model, on_state_change=states.append)

        speech = np.random.randn(16000).astype(np.float32)
        session._generate_loop(speech)

        assert ConversationState.PROCESSING in states
        assert ConversationState.SPEAKING in states
        assert ConversationState.LISTENING in states

    def test_audio_chunks_emitted(self):
        """Test that audio chunks from audio_head are emitted."""
        chunks: list[torch.Tensor] = []
        model = _make_mock_model()
        audio_head = MagicMock()
        chunk1 = torch.randn(1000)
        chunk2 = torch.randn(500)
        audio_head.generate_streaming.return_value = iter([chunk1, chunk2])
        model.audio_head = audio_head

        session = FullDuplexSession(model, on_audio=chunks.append)

        speech = np.random.randn(16000).astype(np.float32)
        session._generate_loop(speech)

        assert len(chunks) == 2
        torch.testing.assert_close(chunks[0], chunk1)
        torch.testing.assert_close(chunks[1], chunk2)

    def test_stop_generate_aborts_before_audio(self):
        """Setting stop_generate during text gen should skip audio generation."""
        model = _make_mock_model()
        audio_head = MagicMock()
        model.audio_head = audio_head

        session = FullDuplexSession(model)

        # Set stop flag when language_model.generate is called (after text, before audio)
        def set_stop_on_generate(**_: Any) -> torch.Tensor:
            with session._state_lock:
                session._state.stop_generate = True
            return torch.ones(1, 40, dtype=torch.long)

        model.language_model.generate.side_effect = set_stop_on_generate

        speech = np.random.randn(16000).astype(np.float32)
        session._generate_loop(speech)

        # Should not have called generate_streaming since stop was set
        audio_head.generate_streaming.assert_not_called()

    def test_exception_returns_to_listening(self):
        """Errors in generate loop should return to LISTENING state."""
        model = _make_mock_model()
        model._process_audio.side_effect = RuntimeError("test error")

        states: list[ConversationState] = []
        session = FullDuplexSession(model, on_state_change=states.append)

        speech = np.random.randn(16000).astype(np.float32)
        session._generate_loop(speech)

        assert states[-1] == ConversationState.LISTENING
        assert not session.is_generating


def _make_vad_side_effect(speech_chunks: int):
    """Create a VAD side_effect that returns speech for N chunks then silence."""
    call_count = [0]

    def side_effect(*_: Any) -> tuple[bool, float]:
        call_count[0] += 1
        if call_count[0] <= speech_chunks:
            return (True, 0.9)
        return (False, 0.1)

    return side_effect


class TestListenLoop:
    """Tests for VAD-based speech detection in _listen_loop."""

    def test_speech_triggers_generation(self):
        """Speech followed by silence should trigger _generate_loop."""
        model = _make_mock_model()
        config = FullDuplexConfig(
            chunk_size=512,
            silence_duration_ms=100,
            min_speech_duration_ms=50,
        )
        session = FullDuplexSession(model, config=config)

        model.detect_speech.side_effect = _make_vad_side_effect(speech_chunks=5)

        # Patch _generate_loop to just record the call
        generated: list[int] = []
        session._generate_loop = lambda audio: generated.append(len(audio))

        session._running = True

        # Feed audio chunks
        for _ in range(5):
            session.input_queue.put(np.ones(512, dtype=np.float32))
        # Silence chunks (enough to exceed threshold)
        for _ in range(10):
            session.input_queue.put(np.zeros(512, dtype=np.float32))

        # Run listen loop in a thread, stop after processing
        def run_and_stop():
            time.sleep(0.5)
            session._running = False

        stopper = threading.Thread(target=run_and_stop)
        stopper.start()
        session._listen_loop()
        stopper.join()

        # Should have triggered generation with concatenated speech
        assert len(generated) > 0

    def test_short_speech_ignored(self):
        """Speech shorter than min_speech_duration_ms should not trigger generation."""
        model = _make_mock_model()
        config = FullDuplexConfig(
            chunk_size=512,
            silence_duration_ms=100,
            min_speech_duration_ms=500,  # Very high minimum
        )
        session = FullDuplexSession(model, config=config)

        model.detect_speech.side_effect = _make_vad_side_effect(speech_chunks=1)

        generated: list[int] = []
        session._generate_loop = lambda audio: generated.append(len(audio))

        session._running = True
        session.input_queue.put(np.ones(512, dtype=np.float32))
        for _ in range(10):
            session.input_queue.put(np.zeros(512, dtype=np.float32))

        def run_and_stop():
            time.sleep(0.5)
            session._running = False

        stopper = threading.Thread(target=run_and_stop)
        stopper.start()
        session._listen_loop()
        stopper.join()

        # Should NOT have triggered generation
        assert len(generated) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
