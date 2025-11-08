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

import numpy as np
from livekit import agents, rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero

# Add parent directory to path to import model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.asr_modeling import ASRModel
from src.asr_config import ASRConfig

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
        self.model: Optional[ASRModel] = None
        self.audio_buffer = []
        self.is_processing = False

    async def initialize(self):
        """Initialize the ASR model with TTS."""
        logger.info(f"Loading model from {self.config.model_path}")

        try:
            # Load the model
            if self.config.use_local_model:
                # Load from local path
                self.model = ASRModel.from_pretrained(self.config.model_path)
            else:
                # Load from HuggingFace Hub
                self.model = ASRModel.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True
                )

            # Put model in eval mode for inference
            self.model.eval()

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

            logger.info("Model loaded successfully")

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
        if self.model is None:
            logger.error("Model not initialized")
            return None

        if self.is_processing:
            logger.debug("Already processing, skipping")
            return None

        self.is_processing = True

        try:
            logger.debug(f"Processing audio shape: {audio_data.shape}")

            # Use the new simplified method
            result = self.model.process_audio_for_agent(
                audio=audio_data,
                sample_rate=self.config.sample_rate,
                task="continue",  # Use continue task as requested
                return_audio=True,
                max_new_tokens=150,  # Reasonable response length
                num_beams=3,  # Balance quality and speed
            )

            # Get the generated text and audio
            text = result.get('text', '')
            audio = result.get('audio')
            processing_time = result.get('processing_time', 0)

            if text:
                logger.info(f"Generated response: {text}")
                logger.debug(f"Processing took {processing_time:.2f}s")

            if audio is not None:
                logger.debug(f"Generated TTS audio shape: {audio.shape}")
                return audio
            else:
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
            self.audio_track,
            rtc.TrackPublishOptions(name="agent-voice")
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

                # Resample if necessary (LiveKit usually provides 48kHz)
                if frame.sample_rate != self.agent.config.sample_rate:
                    audio_data = self.resample_audio(
                        audio_data,
                        frame.sample_rate,
                        self.agent.config.sample_rate
                    )

                # Add to buffer
                self.audio_buffer.extend(audio_data)

                # Process if we have enough audio
                buffer_duration = len(self.audio_buffer) / self.agent.config.sample_rate
                if buffer_duration >= self.min_audio_duration:
                    # Check VAD if enabled
                    if self.vad:
                        audio_array = np.array(self.audio_buffer)
                        if not self.vad.is_speech(audio_array, self.agent.config.sample_rate):
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

        logger.debug(f"Processing {len(audio_array) / self.agent.config.sample_rate:.2f}s of audio")

        # Process with ASR model
        tts_audio = await self.agent.process_audio(audio_array)

        if tts_audio is not None:
            # Send TTS audio back to LiveKit
            await self.send_audio(tts_audio)

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

    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple audio resampling."""
        if orig_sr == target_sr:
            return audio

        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)

        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)

        return resampled


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

    # Create LiveKit integration
    lk_agent = LiveKitVoiceAgent(ctx, agent)

    # Start the agent
    await lk_agent.start()

    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

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
            prewarm=False,  # Don't prewarm to save resources
        )
    )