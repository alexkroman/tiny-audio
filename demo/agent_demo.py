#!/usr/bin/env python3
"""
Simple voice agent demo using Tiny Audio STT + OpenAI LLM + TTS.

This demo creates a local voice agent that:
1. Captures audio from your microphone
2. Transcribes speech using the Tiny Audio model (with streaming)
3. Generates responses using OpenAI GPT
4. Speaks responses using OpenAI TTS

Requirements:
    pip install pipecat-ai[silero,openai,local]
    # On macOS: brew install portaudio

Usage:
    export OPENAI_API_KEY=your-key-here
    python demo/pipecat_demo.py

Press Ctrl+C to stop.
"""

import asyncio
import os
import sys

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# tiny_audio package should be installed via: poetry install


async def main():
    # Import pipecat components
    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        from pipecat.transports.local.audio import (
            LocalAudioTransport,
            LocalAudioTransportParams,
        )
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.tts import OpenAITTSService
        from pipecat.audio.vad.silero import SileroVADAnalyzer
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install pipecat with required extras:")
        print("  pip install pipecat-ai[silero,openai,local]")
        print("\nOn macOS, also install portaudio:")
        print("  brew install portaudio")
        sys.exit(1)

    # Import our STT service
    from tiny_audio.integrations.pipecat_stt import TinyAudioSTTService

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("  export OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    print("Initializing voice agent...")
    print("  - Loading Tiny Audio STT model (this may take a moment)...")

    # STT: Tiny Audio model with streaming - load model FIRST before audio
    stt = TinyAudioSTTService(
        model_id="mazesmazes/tiny-audio",
        streaming=True,
    )
    # Load model now instead of lazily on first transcription
    stt._ensure_model()
    print("  - Model loaded!")

    print("  - Setting up audio...")
    # Transport: local microphone/speaker with VAD
    # Let pipecat handle sample rates automatically
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # LLM: OpenAI GPT-4o-mini
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    # Set system prompt for the LLM
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

    context = OpenAILLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. Keep your responses brief "
                    "and conversational - aim for 1-2 sentences. Be friendly and natural. "
                    "Do not use emojis or special characters."
                ),
            }
        ]
    )
    context_aggregator = llm.create_context_aggregator(context)

    # TTS: OpenAI TTS (let pipecat handle sample rate)
    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="alloy",
    )

    # Build pipeline: mic -> VAD -> STT -> LLM -> TTS -> speaker
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    print("\n" + "=" * 50)
    print("Voice agent ready!")
    print("Speak into your microphone...")
    print("Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        print("\nStopping voice agent...")


if __name__ == "__main__":
    asyncio.run(main())
