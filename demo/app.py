#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Loading from HuggingFace Hub
- Loading from local directory
- Microphone recording
- File upload
- Processing audio files from outputs directory
"""

import os

# Set matplotlib config dir to avoid warning in Hugging Face Spaces
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq

# Try to import spaces for GPU allocation on Hugging Face Spaces
try:
    import spaces
except ImportError:
    # If not running on Hugging Face Spaces, create a dummy decorator
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(func):  # noqa: N802
            return func


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global variable to store model
_model = None


def get_model(model_path: str = "mazesmazes/tiny-audio"):
    """Initialize and cache the model."""
    global _model
    if _model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load model
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, trust_remote_code=True)
        _model = _model.to(device)
        _model.eval()

    return _model


@spaces.GPU
def transcribe_audio(audio_path: str, model_path: str = "mazesmazes/tiny-audio") -> str:
    """Transcribe audio using the model's built-in transcribe method."""
    model = get_model(model_path)

    # Simply use the model's transcribe method
    return model.transcribe(audio_path)


class ASRDemo:
    def __init__(self, model_path: str = "mazesmazes/tiny-audio", outputs_dir: str = "wav_outputs"):
        self.model_path = model_path
        self.outputs_dir = Path(outputs_dir)
        if not self.outputs_dir.is_absolute():
            script_dir = Path(__file__).parent
            potential_path = script_dir / outputs_dir
            if potential_path.exists():
                self.outputs_dir = potential_path
            else:
                self.outputs_dir = Path.cwd() / outputs_dir

        # Initialize model on startup
        get_model(model_path)

    def transcribe(self, audio_path: str) -> str:
        return transcribe_audio(audio_path, self.model_path)

    def process_audio(self, audio_input: Union[str, Tuple[int, np.ndarray], None]) -> str:
        if audio_input is None:
            return "Please provide audio input"

        if isinstance(audio_input, tuple):
            sample_rate, audio_data = audio_input

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name
                sf.write(temp_path, audio_data, sample_rate)

            result = self.transcribe(temp_path)

            Path(temp_path).unlink(missing_ok=True)
            return result

        return self.transcribe(audio_input)

    def get_output_files(self) -> List[str]:
        """Get list of audio files from outputs directory."""
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

        if not self.outputs_dir.exists():
            return []

        audio_files: List[Path] = []
        for ext in audio_extensions:
            audio_files.extend(self.outputs_dir.glob(f"**/*{ext}"))

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
    asr_demo = ASRDemo(model_path, outputs_dir)
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

                def refresh_files():
                    files = asr_demo.get_output_files()
                    return gr.Dropdown(choices=files, value=files[0] if files else None)

                refresh_btn.click(refresh_files, outputs=[file_dropdown])

                process_btn.click(
                    asr_demo.process_from_outputs, inputs=[file_dropdown], outputs=[output_text_2]
                )

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

    demo = create_demo(args.model, args.outputs_dir)
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")
