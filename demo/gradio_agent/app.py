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

# Add parent directory to path to import model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.asr_modeling import ASRModel
from src.asr_config import ASRConfig

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
        self.model: Optional[ASRModel] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_processing = False
        self.audio_queue = queue.Queue()
        self.stop_listening = threading.Event()

    def load_model(self):
        """Load the ASR model with TTS capabilities."""
        try:
            logger.info(f"Loading model from {self.config.model_path}")

            if self.config.use_local_model:
                self.model = ASRModel.from_pretrained(self.config.model_path)
            else:
                self.model = ASRModel.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True
                )

            # Put model in eval mode
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
                logger.warning("TTS not available - will return text only")

            logger.info("Model loaded successfully")
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
        if self.model is None:
            return "Model not loaded. Please wait...", None, []

        if audio_input is None:
            return "No audio input received", None, self.conversation_history

        sample_rate, audio_data = audio_input

        # Handle stereo to mono conversion if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Normalize audio to float32 in range [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        # Resample if necessary
        if sample_rate != self.config.sample_rate:
            audio_data = self.resample_audio(audio_data, sample_rate, self.config.sample_rate)

        try:
            logger.info(f"Processing audio with task: {task}")
            start_time = time.time()

            # Process with the model
            result = self.model.process_audio_for_agent(
                audio=audio_data,
                sample_rate=self.config.sample_rate,
                task=task,
                return_audio=True,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                num_beams=self.config.num_beams,
            )

            processing_time = time.time() - start_time

            # Extract results
            generated_text = result.get('text', '')
            tts_audio = result.get('audio')

            # Add to conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': f"[Audio input - {len(audio_data)/self.config.sample_rate:.1f}s]",
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
                # Ensure proper format for Gradio
                if tts_audio.dtype != np.float32:
                    tts_audio = tts_audio.astype(np.float32)
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

    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple audio resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)

        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)

        return resampled

    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


def create_gradio_interface():
    """Create the Gradio interface for the voice agent."""

    # Initialize agent
    config = AgentConfig()
    agent = VoiceAgent(config)

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

            Interactive voice agent powered by the Tiny Audio model with continuous conversation capabilities.

            ## Features:
            - 🎙️ Real-time voice recording and processing
            - 💬 Continuous conversation with context
            - 🔊 Text-to-speech synthesis for responses
            - 📝 Conversation history tracking
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Model loading status
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Click 'Load Model' to start",
                    interactive=False
                )

                load_btn = gr.Button("Load Model", variant="primary")

                # Task selection
                task_selector = gr.Radio(
                    choices=["continue", "transcribe", "translate"],
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
                    label="🎙️ Record Audio",
                    elem_id="audio-input"
                )

                # Process button
                process_btn = gr.Button("Process Audio", variant="primary", size="lg")

                # Response display
                response_text = gr.Textbox(
                    label="Response",
                    placeholder="Agent's response will appear here...",
                    lines=3,
                    interactive=False
                )

                # TTS output
                audio_output = gr.Audio(
                    label="🔊 Agent Voice",
                    type="numpy",
                    autoplay=True,
                    elem_id="audio-output"
                )

                # Conversation history
                with gr.Accordion("Conversation History", open=True):
                    conversation_display = gr.Chatbot(
                        label="Conversation",
                        height=300,
                        elem_id="conversation"
                    )

                # Clear button
                clear_btn = gr.Button("Clear Conversation", variant="secondary")

        # Continuous conversation mode
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    ## 🔄 Continuous Conversation Mode

                    Enable continuous listening for natural back-and-forth conversation.
                    The agent will use the "continue" task to maintain context.
                    """
                )

                continuous_audio = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="🎤 Continuous Recording",
                    elem_id="continuous-audio"
                )

                continuous_output = gr.Audio(
                    label="🔊 Live Response",
                    type="numpy",
                    autoplay=True,
                    streaming=True,
                    elem_id="continuous-output"
                )

        # State management
        conversation_state = gr.State([])

        # Event handlers
        def load_model_handler():
            success = agent.load_model()
            if success:
                agent.config.tts_voice = tts_voice.value
                agent.config.tts_speed = tts_speed.value
                return "✅ Model loaded successfully!"
            else:
                return "❌ Failed to load model. Check logs."

        def process_audio_handler(audio, task, state):
            if audio is None:
                return "", None, [], state

            text, tts, history = agent.process_audio(audio, task)

            # Format for chatbot display
            chat_history = []
            for entry in history:
                role = entry.get('role')
                content = entry.get('content')
                if role == 'user':
                    chat_history.append((content, None))
                elif role == 'assistant':
                    if len(chat_history) > 0:
                        chat_history[-1] = (chat_history[-1][0], content)
                    else:
                        chat_history.append((None, content))

            return text, tts, chat_history, history

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
            return f"TTS updated: {voice} at {speed}x speed"

        def clear_conversation_handler():
            agent.clear_conversation()
            return [], [], None, None

        # Continuous conversation handler
        def continuous_handler(audio_stream, state):
            if audio_stream is None:
                return None, state

            # Process continuous audio with continue task
            _, tts_audio, updated_state = agent.continuous_conversation(
                audio_stream,
                state
            )

            return tts_audio, updated_state

        # Wire up events
        load_btn.click(
            fn=load_model_handler,
            outputs=model_status
        )

        process_btn.click(
            fn=process_audio_handler,
            inputs=[audio_input, task_selector, conversation_state],
            outputs=[response_text, audio_output, conversation_display, conversation_state]
        )

        # Auto-process when audio is recorded
        audio_input.stop_recording(
            fn=process_audio_handler,
            inputs=[audio_input, task_selector, conversation_state],
            outputs=[response_text, audio_output, conversation_display, conversation_state]
        )

        # Update TTS settings
        tts_voice.change(
            fn=update_tts_settings,
            inputs=[tts_voice, tts_speed],
            outputs=model_status
        )

        tts_speed.change(
            fn=update_tts_settings,
            inputs=[tts_voice, tts_speed],
            outputs=model_status
        )

        clear_btn.click(
            fn=clear_conversation_handler,
            outputs=[conversation_display, conversation_state, response_text, audio_output]
        )

        # Continuous conversation streaming
        continuous_audio.stream(
            fn=continuous_handler,
            inputs=[continuous_audio, conversation_state],
            outputs=[continuous_output, conversation_state]
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