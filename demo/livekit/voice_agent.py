#!/usr/bin/env python3
"""
LiveKit Voice Agent using Tiny Audio ASR Model with integrated TTS.

This agent:
1. Receives audio from LiveKit participants
2. Processes it with the ASR model using the "continue" task
3. Generates TTS audio from the model's response
4. Sends the TTS audio back to LiveKit
"""

import asyncio
import logging
import sys
import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from livekit import agents, rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import silero
from transformers import AutoModel, pipeline

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env file from the same directory as this script
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logging.info(f"Loaded environment from {env_path}")
else:
    logging.warning(f"No .env file found at {env_path}")
    # Try loading from current directory as fallback
    load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the voice agent."""
    model_path: str = os.getenv("MODEL_PATH", "mazesmazes/tiny-audio")
    use_local_model: bool = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
    tts_voice: str = os.getenv("TTS_VOICE", "af_heart")
    tts_speed: float = float(os.getenv("TTS_SPEED", "1.0"))
    sample_rate: int = 16000  # ASR model expects 16kHz
    tts_sample_rate: int = 24000  # Kokoro outputs 24kHz
    vad_enabled: bool = True
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


class TinyAudioVoiceAgent:
    """Voice agent using Tiny Audio ASR model with integrated TTS."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = None
        self.pipeline = None
        self.audio_buffer = []
        self.is_processing = False

    async def initialize(self):
        """Initialize the ASR model with TTS."""
        logger.info(f"Loading model from {self.config.model_path}")

        try:
            # Load model using AutoModel
            # Choose dtype based on device - MPS doesn't fully support bfloat16
            if torch.backends.mps.is_available():
                # Use float32 on Apple Silicon (MPS)
                dtype = torch.float32
            elif torch.cuda.is_available():
                # Use bfloat16 on CUDA for better performance
                dtype = torch.bfloat16
            else:
                # Use float32 on CPU
                dtype = torch.float32

            self.model = AutoModel.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                torch_dtype=dtype
            )

            # Ensure all model components use the same dtype
            self.model = self.model.to(dtype)

            # Put model in eval mode for inference
            self.model.eval()

            # Initialize TTS (needs to be done after eval mode since it's skipped in training mode)
            if hasattr(self.model, '_initialize_tts'):
                self.model._initialize_tts()

            # Configure TTS if available
            if hasattr(self.model, 'configure_tts') and self.model.tts_enabled:
                self.model.configure_tts(
                    enabled=True,
                    voice=self.config.tts_voice,
                    speed=self.config.tts_speed
                )
                logger.info(f"TTS configured: voice={self.config.tts_voice}, speed={self.config.tts_speed}")
            else:
                logger.warning("TTS not available in model - will return text only")

            # Create pipeline from the model
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                trust_remote_code=True
            )

            logger.info("Model and pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def process_audio(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio with ASR model and return TTS audio.

        Args:
            audio_data: Input audio as numpy array (16kHz)

        Returns:
            TTS audio as numpy array (24kHz) or None
        """
        if self.model is None or self.pipeline is None:
            logger.error("Model not initialized")
            return None

        if self.is_processing:
            logger.debug("Already processing, skipping")
            return None

        self.is_processing = True

        try:
            logger.debug(f"Processing audio shape: {audio_data.shape}")

            # Use pipeline for ASR - it handles all preprocessing
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    {"raw": audio_data, "sampling_rate": self.config.sample_rate},
                    task="continue",
                    max_new_tokens=50,  # Short responses for low latency
                    num_beams=1,  # Greedy decoding for fastest response
                    chunk_length_s=30
                    # Don't set return_timestamps to avoid CTC errors
                )
            )

            text = result.get('text', '')

            if text:
                logger.info(f"Generated response: {text}")

            # Generate TTS using the pipeline
            if text:
                try:
                    # Use the pipeline for TTS generation
                    tts_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.pipeline(
                            text=text,
                            return_audio=True,
                            tts_voice=self.config.tts_voice,
                            tts_speed=self.config.tts_speed
                        )
                    )
                    tts_audio = tts_result.get('audio')
                    if tts_audio is not None:
                        logger.debug(f"Generated TTS audio shape: {tts_audio.shape}")
                        return tts_audio
                except Exception as tts_error:
                    logger.warning(f"TTS generation failed: {tts_error}")

            logger.warning("No TTS audio generated")
            return None

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None

        finally:
            self.is_processing = False


class LiveKitVoiceAgent:
    """LiveKit integration for the voice agent."""

    def __init__(self, ctx: JobContext, agent: TinyAudioVoiceAgent):
        self.ctx = ctx
        self.agent = agent
        self.room = ctx.room
        self.participant: Optional[rtc.RemoteParticipant] = None
        self.audio_source = rtc.AudioSource(
            self.agent.config.tts_sample_rate,
            1  # Mono audio
        )
        self.audio_track: Optional[rtc.LocalAudioTrack] = None
        self.vad = silero.VAD.load() if agent.config.vad_enabled else None
        self.audio_buffer = []
        self.buffer_sample_rate = 48000  # LiveKit default
        self.min_audio_duration = 0.5  # Minimum seconds of audio before processing

    async def start(self):
        """Start the agent and connect to the room."""
        # Subscribe to room events
        self.room.on("participant_connected", self.on_participant_connected)
        self.room.on("track_published", self.on_track_published)
        self.room.on("track_subscribed", self.on_track_subscribed)

        # Create and publish audio track for TTS output
        self.audio_track = rtc.LocalAudioTrack.create_audio_track(
            "agent-voice",
            self.audio_source
        )

        # Publish the track
        await self.room.local_participant.publish_track(
            self.audio_track
        )

        logger.info("Agent started and audio track published")

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle participant connection."""
        logger.info(f"Participant connected: {participant.identity}")
        self.participant = participant

    def on_track_published(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ):
        """Handle track publication."""
        if publication.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track published by {participant.identity}")

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ):
        """Handle track subscription."""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to audio from {participant.identity}")
            asyncio.create_task(self.process_audio_track(track))

    async def process_audio_track(self, track: rtc.AudioTrack):
        """Process incoming audio track."""
        audio_stream = rtc.AudioStream(track)

        async for event in audio_stream:
            if isinstance(event, rtc.AudioFrameEvent):
                frame = event.frame

                # Convert frame to numpy array
                audio_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

                # Store the original sample rate - pipeline will handle resampling
                self.buffer_sample_rate = frame.sample_rate

                # Add to buffer
                self.audio_buffer.extend(audio_data)

                # Process if we have enough audio
                buffer_duration = len(self.audio_buffer) / self.buffer_sample_rate
                if buffer_duration >= self.min_audio_duration:
                    # Check VAD if enabled
                    if self.vad:
                        audio_array = np.array(self.audio_buffer)
                        # VAD expects 16kHz, so we need to resample for VAD check
                        if self.buffer_sample_rate != 16000:
                            # Simple resampling for VAD
                            target_length = int(len(audio_array) * 16000 / self.buffer_sample_rate)
                            indices = np.linspace(0, len(audio_array) - 1, target_length)
                            audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
                        if not self.vad.is_speech(audio_array, 16000):
                            # No speech detected, clear buffer and continue
                            self.audio_buffer = []
                            continue

                    # Process the buffered audio
                    await self.process_buffered_audio()

    async def process_buffered_audio(self):
        """Process the buffered audio and send TTS response."""
        if not self.audio_buffer:
            return

        # Convert buffer to numpy array
        audio_array = np.array(self.audio_buffer, dtype=np.float32)
        self.audio_buffer = []  # Clear buffer

        logger.debug(f"Processing {len(audio_array) / self.buffer_sample_rate:.2f}s of audio")

        # Update the agent's pipeline to use the correct sample rate
        # The pipeline will handle resampling internally
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.agent.pipeline(
                {"raw": audio_array, "sampling_rate": self.buffer_sample_rate},
                task="continue",
                max_new_tokens=50,
                num_beams=1,
                chunk_length_s=30
                # Don't set return_timestamps to avoid CTC errors
            )
        )

        text = result.get('text', '')

        if text:
            logger.info(f"Generated response: {text}")

            # Generate TTS using the pipeline
            try:
                tts_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.agent.pipeline(
                        text=text,
                        return_audio=True,
                        tts_voice=self.agent.config.tts_voice,
                        tts_speed=self.agent.config.tts_speed
                    )
                )
                tts_audio = tts_result.get('audio')
                if tts_audio is not None:
                    # Send TTS audio back to LiveKit
                    await self.send_audio(tts_audio)
            except Exception as tts_error:
                logger.warning(f"TTS generation failed: {tts_error}")

    async def send_audio(self, audio_data: np.ndarray):
        """Send audio to LiveKit."""
        try:
            # Ensure audio is float32 in range [-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Clip to valid range
            audio_data = np.clip(audio_data, -1.0, 1.0)

            # Convert to int16 for LiveKit
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Create audio frame
            frame = rtc.AudioFrame.create(
                sample_rate=self.agent.config.tts_sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_int16)
            )

            # Copy data to frame
            frame_data = np.frombuffer(frame.data, dtype=np.int16)
            frame_data[:] = audio_int16

            # Send frame
            await self.audio_source.capture_frame(frame)

            logger.info(f"Sent {len(audio_data) / self.agent.config.tts_sample_rate:.2f}s of TTS audio")

        except Exception as e:
            logger.error(f"Error sending audio: {e}")



async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent."""
    logger.info("Starting Tiny Audio Voice Agent")

    # Create configuration
    config = AgentConfig()

    if config.debug:
        logger.setLevel(logging.DEBUG)

    # Create and initialize the agent
    agent = TinyAudioVoiceAgent(config)
    await agent.initialize()

    # Connect to the room FIRST
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info("Connected to room")

    # Create LiveKit integration AFTER connecting
    lk_agent = LiveKitVoiceAgent(ctx, agent)

    # Start the agent (now it can publish tracks)
    await lk_agent.start()

    logger.info("Agent connected and ready")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )