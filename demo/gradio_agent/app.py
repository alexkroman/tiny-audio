#!/usr/bin/env python3
"""
Gradio Voice Agent Demo using Tiny Audio Model.

This demo provides a web interface for voice-based interaction with the Tiny Audio model.
Features:
- Real-time audio recording and processing
- Continuous voice conversation using the "continue" task
- Text-to-speech synthesis for agent responses
- Visual feedback and conversation history
"""

import gradio as gr
import numpy as np
import torch
import logging
import time
import sys
import os
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import threading
import queue
from transformers import AutoModel, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the Gradio voice agent."""
    model_path: str = "mazesmazes/tiny-audio"
    use_local_model: bool = False
    sample_rate: int = 16000  # ASR model expects 16kHz
    tts_sample_rate: int = 24000  # TTS outputs 24kHz
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    max_new_tokens: int = 100
    temperature: float = 0.7
    num_beams: int = 1
    enable_vad: bool = True
    vad_threshold: float = 0.5


class VoiceAgent:
    """Voice agent using Tiny Audio model with continuous conversation."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = None
        self.pipeline = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_processing = False
        self.audio_queue = queue.Queue()
        self.stop_listening = threading.Event()

    def load_model(self):
        """Load the ASR model with TTS capabilities."""
        try:
            logger.info(f"Loading model from {self.config.model_path}")

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

            # Put model in eval mode
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
                logger.warning("TTS not available - will return text only")

            # Create pipeline - try to load VoiceAgentPipeline for TTS support
            try:
                # Import from src if running locally
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
                from voice_agent_pipeline import VoiceAgentPipeline

                self.pipeline = VoiceAgentPipeline(
                    model=self.model,
                    feature_extractor=self.model.feature_extractor,
                    tokenizer=self.model.tokenizer
                )
                logger.info("Using VoiceAgentPipeline with TTS support")
            except ImportError:
                # Fall back to standard ASR pipeline (no TTS)
                self.pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    trust_remote_code=True
                )
                logger.warning("Using standard ASR pipeline (TTS may not work)")

            logger.info("Model and pipeline loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def process_audio(
        self,
        audio_input: Tuple[int, np.ndarray],
        task: str = "continue",
        conversation_context: str = ""
    ) -> Tuple[str, Optional[Tuple[int, np.ndarray]], List[Dict]]:
        """
        Process audio input and generate response with TTS.

        Args:
            audio_input: Tuple of (sample_rate, audio_data)
            task: Task type for the model
            conversation_context: Previous conversation for context

        Returns:
            Tuple of (transcribed_text, tts_audio, conversation_history)
        """
        if self.model is None or self.pipeline is None:
            return "Model not loaded. Please wait...", None, []

        if audio_input is None:
            return "No audio input received", None, self.conversation_history

        sample_rate, audio_data = audio_input

        # Handle stereo to mono conversion if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        try:
            logger.info(f"Processing audio with task: {task}")
            start_time = time.time()

            # Use VoiceAgentPipeline - supports return_audio, tts_voice, etc.
            result = self.pipeline(
                {"raw": audio_data, "sampling_rate": sample_rate},
                task=task,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                num_beams=self.config.num_beams,
                chunk_length_s=30,
                return_audio=True,  # Get TTS audio back
                tts_voice=self.config.tts_voice,
                tts_speed=self.config.tts_speed
            )
            generated_text = result.get('text', '')
            tts_audio = result.get('audio')

            processing_time = time.time() - start_time

            # Add to conversation history with transcribed text
            user_message = generated_text if task == "transcribe" else f"[Audio input - {len(audio_data)/sample_rate:.1f}s]"
            self.conversation_history.append({
                'role': 'user',
                'content': user_message,
                'task': task
            })

            if generated_text:
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': generated_text,
                    'processing_time': processing_time
                })
                logger.info(f"Generated: {generated_text} (in {processing_time:.2f}s)")

            # Prepare audio output if available
            audio_output = None
            if tts_audio is not None:
                # Convert to int16 format that Gradio expects
                if tts_audio.dtype == np.float32 or tts_audio.dtype == np.float64:
                    # Convert from float [-1, 1] to int16 [-32768, 32767]
                    tts_audio = np.clip(tts_audio, -1.0, 1.0)
                    tts_audio = (tts_audio * 32767).astype(np.int16)
                elif tts_audio.dtype != np.int16:
                    tts_audio = tts_audio.astype(np.int16)
                audio_output = (self.config.tts_sample_rate, tts_audio)

            return generated_text, audio_output, self.conversation_history

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return f"Error: {str(e)}", None, self.conversation_history

    def continuous_conversation(
        self,
        audio_stream: Any,
        conversation_state: List[Dict]
    ) -> Tuple[str, Optional[Tuple[int, np.ndarray]], List[Dict]]:
        """
        Handle continuous voice conversation.

        Args:
            audio_stream: Streaming audio input
            conversation_state: Current conversation state

        Returns:
            Tuple of (response_text, tts_audio, updated_conversation)
        """
        if audio_stream is None:
            return "", None, conversation_state

        # Process with continue task for ongoing conversation
        return self.process_audio(
            audio_stream,
            task="continue",
            conversation_context=self._format_conversation(conversation_state)
        )

    def _format_conversation(self, history: List[Dict]) -> str:
        """Format conversation history as context."""
        if not history:
            return ""

        context = []
        for entry in history[-5:]:  # Last 5 exchanges for context
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            context.append(f"{role}: {content}")

        return "\n".join(context)

    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


def create_gradio_interface():
    """Create the Gradio interface for the voice agent."""

    # Initialize agent
    config = AgentConfig()
    agent = VoiceAgent(config)

    # Load model immediately
    print("Loading model...")
    model_loaded = agent.load_model()

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .conversation-box {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: #f5f5f5;
        border-radius: 8px;
        margin: 10px 0;
    }
    .message {
        margin: 8px 0;
        padding: 8px 12px;
        border-radius: 6px;
    }
    .user-message {
        background: #e3f2fd;
        text-align: right;
    }
    .assistant-message {
        background: #f3e5f5;
        text-align: left;
    }
    """

    with gr.Blocks(title="Tiny Audio Voice Agent", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🎤 Tiny Audio Voice Agent
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Task selection
                task_selector = gr.Radio(
                    choices=["continue", "transcribe"],
                    value="continue",
                    label="Task Mode",
                    info="Select the task for audio processing"
                )

                # TTS settings
                with gr.Accordion("TTS Settings", open=False):
                    tts_voice = gr.Dropdown(
                        choices=["af", "af_bella", "af_nicole", "af_heart", "af_sarah",
                                "am_adam", "am_michael", "bf_emma", "bf_isabella",
                                "bm_george", "bm_lewis"],
                        value="af_heart",
                        label="Voice",
                        info="Select TTS voice"
                    )

                    tts_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speech Speed",
                        info="Adjust TTS speed"
                    )

                # Processing settings
                with gr.Accordion("Advanced Settings", open=False):
                    max_tokens = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Max Response Length",
                        info="Maximum tokens to generate"
                    )

                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in generation"
                    )

            with gr.Column(scale=2):
                # Audio input
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎙️ Record Audio (click stop to process)",
                    elem_id="audio-input"
                )

                # Conversation display with audio playback
                conversation_display = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    elem_id="conversation",
                    type="messages"
                )

                # Hidden audio component for autoplay
                audio_output = gr.Audio(
                    visible=False,
                    autoplay=True,
                    elem_id="audio-output"
                )

        # State management
        conversation_state = gr.State([])

        # Event handlers
        def process_audio_handler(audio, task, state):
            if audio is None:
                return None, [], state

            text, tts, history = agent.process_audio(audio, task)

            # Format for chatbot display using OpenAI-style messages format
            chat_history = []
            for entry in history:
                role = entry.get('role')
                content = entry.get('content')
                if role in ['user', 'assistant']:
                    chat_history.append({
                        'role': role,
                        'content': content
                    })

            return tts, chat_history, history

        def update_tts_settings(voice, speed):
            if agent.model is not None:
                agent.config.tts_voice = voice
                agent.config.tts_speed = speed
                if hasattr(agent.model, 'configure_tts'):
                    agent.model.configure_tts(
                        enabled=True,
                        voice=voice,
                        speed=speed
                    )

        # Wire up events
        # Auto-process when audio is recorded
        audio_input.stop_recording(
            fn=process_audio_handler,
            inputs=[audio_input, task_selector, conversation_state],
            outputs=[audio_output, conversation_display, conversation_state]
        )

        # Update TTS settings
        tts_voice.change(
            fn=update_tts_settings,
            inputs=[tts_voice, tts_speed]
        )

        tts_speed.change(
            fn=update_tts_settings,
            inputs=[tts_voice, tts_speed]
        )

    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()

    # Launch with options
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
        debug=True,
        show_error=True
    )