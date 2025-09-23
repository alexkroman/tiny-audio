#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Loading from HuggingFace Hub
- Loading from local directory
- Microphone recording
- File upload
- Processing audio files from outputs directory
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import soundfile as sf
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.modeling import ASRModel  # noqa: E402


class ASRDemo:
    def __init__(self, model_path: str = "mazesmazes/tiny-audio", outputs_dir: str = "wav_outputs"):
        """Initialize the ASR demo with a model from HuggingFace Hub or local path.

        Args:
            model_path: HuggingFace model repository name or local directory path
            outputs_dir: Directory containing audio files to process
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Handle outputs directory
        self.outputs_dir = Path(outputs_dir)
        if not self.outputs_dir.is_absolute():
            self.outputs_dir = Path.cwd() / outputs_dir
        logger.info(f"Using outputs directory: {self.outputs_dir.absolute()}")

        # Load model with feature extractor
        logger.info(f"Loading model from: {model_path}")
        self.model = ASRModel.from_pretrained(model_path)

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully with dtype: {self.model.dtype}")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text
        """
        try:
            with torch.no_grad():
                result = self.model.transcribe(audio_path)
            return result if result else "Could not generate transcription"
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return f"Error: {str(e)}"

    def process_audio(self, audio_input: Union[str, Tuple[int, np.ndarray], None]) -> str:
        """Process audio from microphone or file upload.

        Args:
            audio_input: Audio input from Gradio (file path or tuple of sample_rate, audio_data)

        Returns:
            Transcribed text
        """
        if audio_input is None:
            return "Please provide audio input"

        # Handle microphone input (tuple of sample_rate, audio_data)
        if isinstance(audio_input, tuple):
            sample_rate, audio_data = audio_input

            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name
                sf.write(temp_path, audio_data, sample_rate)

            # Transcribe the temporary file
            result = self.transcribe(temp_path)

            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
            return result

        # Handle file upload (string path)
        return self.transcribe(audio_input)

    def get_output_files(self) -> List[str]:
        """Get list of audio files from outputs directory."""
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

        if not self.outputs_dir.exists():
            return []

        audio_files: List[Path] = []
        for ext in audio_extensions:
            audio_files.extend(self.outputs_dir.glob(f"**/*{ext}"))

        # Return relative paths as strings
        return [str(f.relative_to(self.outputs_dir)) for f in sorted(audio_files)]

    def process_from_outputs(self, selected_file: Optional[str]) -> str:
        """Process a file from the outputs directory."""
        if not selected_file:
            return "Please select a file"

        file_path = self.outputs_dir / selected_file

        if not file_path.exists():
            return f"File not found: {file_path}"

        return self.transcribe(str(file_path))


def create_demo(model_path: str = "mazesmazes/tiny-audio", outputs_dir: str = "wav_outputs"):
    """Create and return the Gradio interface.

    Args:
        model_path: HuggingFace model repository name or local path
        outputs_dir: Directory containing audio files to process

    Returns:
        Gradio Blocks interface
    """
    # Initialize the demo
    asr_demo = ASRDemo(model_path, outputs_dir)

    # Create the interface
    with gr.Blocks(title="ASR Demo - Whisper + SmolLM2") as demo:
        gr.Markdown("# 🎤 ASR Demo: Whisper Encoder + SmolLM2 Decoder")
        gr.Markdown(
            "Upload an audio file, record from microphone, or select from outputs directory"
        )

        with gr.Tabs():
            with gr.Tab("Microphone/Upload"), gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Audio Input", type="filepath", sources=["microphone", "upload"]
                    )
                    submit_btn = gr.Button("Transcribe", variant="primary")

                with gr.Column():
                    output_text = gr.Textbox(
                        label="Transcription",
                        lines=5,
                        placeholder="Transcription will appear here...",
                    )

            with gr.Tab("From Outputs Directory"):
                with gr.Row():
                    with gr.Column():
                        # Dropdown for selecting files from outputs
                        output_files = asr_demo.get_output_files()
                        file_dropdown = gr.Dropdown(
                            choices=output_files,
                            label=f"Select audio from {outputs_dir}/",
                            value=output_files[0] if output_files else None,
                        )
                        refresh_btn = gr.Button("🔄 Refresh List")
                        process_btn = gr.Button("Transcribe Selected", variant="primary")

                    with gr.Column():
                        output_text_2 = gr.Textbox(
                            label="Transcription",
                            lines=5,
                            placeholder="Transcription will appear here...",
                        )

                # Refresh button updates the dropdown
                def refresh_files():
                    files = asr_demo.get_output_files()
                    return gr.Dropdown(choices=files, value=files[0] if files else None)

                refresh_btn.click(refresh_files, outputs=[file_dropdown])

                process_btn.click(
                    asr_demo.process_from_outputs, inputs=[file_dropdown], outputs=[output_text_2]
                )

        # Connect the main transcribe button
        submit_btn.click(asr_demo.process_audio, inputs=[audio_input], outputs=[output_text])

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Gradio Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="mazesmazes/tiny-audio",
        help="Path to model (HuggingFace Hub or local directory)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="wav_outputs",
        help="Directory containing audio files to process",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run the demo on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")

    args = parser.parse_args()

    # Create and launch demo
    demo = create_demo(args.model, args.outputs_dir)
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")
