#!/usr/bin/env python3
"""
Simple streaming transcription demo using Tiny Audio and Pipecat.

This demo captures audio from your microphone, uses VAD to detect speech,
and streams the transcription in real-time after each turn.

Requirements:
    pip install pipecat-ai[silero,local]
    # On macOS: brew install portaudio

Usage:
    python demo/streaming_demo.py

Press Ctrl+C to stop.
"""

import asyncio
import os
import sys

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


async def main():
    # Import pipecat components
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.frames.frames import (
            Frame,
            InterimTranscriptionFrame,
            TranscriptionFrame,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.transports.local.audio import (
            LocalAudioTransport,
            LocalAudioTransportParams,
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install pipecat with required extras:")
        print("  pip install pipecat-ai[silero,local]")
        print("\nOn macOS, also install portaudio:")
        print("  brew install portaudio")
        sys.exit(1)

    # Import our STT service
    from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

    print("Initializing streaming transcription...")
    print("  - Loading Tiny Audio model (this may take a moment)...")

    # STT: Tiny Audio model with streaming
    stt = TinyAudioSTTService(
        model_id="mazesmazes/tiny-audio",
        streaming=True,
    )
    # Load model now instead of lazily on first transcription
    stt._ensure_model()
    print("  - Model loaded!")

    print("  - Setting up audio...")

    # Custom frame processor to print transcriptions
    class TranscriptionPrinter(FrameProcessor):
        def __init__(self):
            super().__init__()
            self._current_text = ""

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, InterimTranscriptionFrame):
                # Only print the new characters (delta)
                if frame.text.startswith(self._current_text):
                    delta = frame.text[len(self._current_text):]
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                self._current_text = frame.text
            elif isinstance(frame, TranscriptionFrame):
                # Print any remaining text and newline
                if frame.text.startswith(self._current_text):
                    delta = frame.text[len(self._current_text):]
                    if delta:
                        sys.stdout.write(delta)
                if frame.text.strip():
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                self._current_text = ""

            await self.push_frame(frame, direction)

    # Transport: local microphone with VAD (reduced silence threshold)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=0.3)  # Default is 0.8s
            ),
        )
    )

    # Transcription printer
    printer = TranscriptionPrinter()

    # Build pipeline: mic -> VAD -> STT -> print
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            printer,
        ]
    )

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    print("\n" + "=" * 50)
    print("Streaming transcription ready!")
    print("Speak into your microphone...")
    print("Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        print("\n\nStopping transcription...")


if __name__ == "__main__":
    asyncio.run(main())
