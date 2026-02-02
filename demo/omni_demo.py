#!/usr/bin/env python3
"""
Omni Voice Agent Demo - Speech-to-Speech using Tiny Audio Omni + Kokoro TTS.

This demo creates a local voice agent that appears as a unified speech-to-speech
model, combining Tiny Audio Omni (for understanding and response) with Kokoro TTS
(for speech synthesis) into a single seamless experience.

No external APIs required - runs entirely locally.

Requirements:
    pip install pipecat-ai[silero,local]
    # On macOS: brew install portaudio
    # On Linux: apt-get install espeak-ng

Usage:
    python demo/omni_demo.py
    python demo/omni_demo.py --model mazesmazes/tiny-audio-omni
    python demo/omni_demo.py --voice af_bella

Press Ctrl+C to stop.
"""

import asyncio
import os
import sys

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable MPS (Apple Silicon GPU) for Kokoro TTS
# See: https://github.com/hexgrad/kokoro
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Omni Voice Agent - Speech-to-Speech Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo/omni_demo.py
    python demo/omni_demo.py --model mazesmazes/tiny-audio-omni
    python demo/omni_demo.py --voice af_bella

Available voices (see https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md):
    af_heart, af_bella, af_nicole, af_sarah, af_sky, am_adam, am_michael, ...
        """,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mazesmazes/tiny-audio-omni",
        help="HuggingFace model ID (default: mazesmazes/tiny-audio-omni)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        default="af_heart",
        help="Kokoro TTS voice ID (default: af_heart)",
    )

    args = parser.parse_args()

    # Import pipecat components
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
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

    # Import our S2S service
    from tiny_audio.integrations.pipecat_s2s import TinyAudioS2SService

    print("Initializing Omni voice agent...")
    print(f"  - Model: {args.model}")
    print("  - Loading model (this may take a moment)...")

    # Voice agent system prompt
    system_prompt = (
        "You are a helpful voice assistant. Keep your responses brief and "
        "conversational - aim for 1-2 sentences. Be friendly and natural. "
        "Do not use emojis or special characters."
    )

    # Speech-to-Speech service: Tiny Audio Omni + Kokoro TTS
    # This appears as a single unified S2S model to pipecat
    s2s = TinyAudioS2SService(
        model_id=args.model,
        tts_voice=args.voice,
        system_prompt=system_prompt,
    )
    # Load model now instead of lazily on first transcription
    s2s._ensure_model()
    print("  - Model loaded!")

    print("  - Setting up audio...")
    # Transport: local microphone/speaker with VAD for end-of-turn detection
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # Build pipeline: mic -> S2S -> speaker
    pipeline = Pipeline(
        [
            transport.input(),
            s2s,
            transport.output(),
        ]
    )

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    # Play greeting after a short delay
    async def play_greeting():
        import asyncio
        await asyncio.sleep(1)  # Wait for pipeline to be ready
        print("  - Playing greeting...")
        async for frame in s2s.say("Hello! How can I help you?"):
            await task.queue_frame(frame)
        print("  - Greeting done")

    print("\n" + "=" * 50)
    print("Omni voice agent ready!")
    print("This is a unified Speech-to-Speech model.")
    print("Speak into your microphone...")
    print("Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    try:
        # Start greeting task
        asyncio.create_task(play_greeting())
        await runner.run(task)
    except KeyboardInterrupt:
        print("\nStopping voice agent...")


if __name__ == "__main__":
    asyncio.run(main())
