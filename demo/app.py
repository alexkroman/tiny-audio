#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Loading from HuggingFace Hub
- Loading from local directory
- Processing audio files from outputs directory
"""

import os

# Set matplotlib config dir to avoid warning in Hugging Face Spaces
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import logging
from pathlib import Path
from typing import List, Optional

import gradio as gr
import torch
from transformers import AutoModel

try:
    import spaces
except ImportError:
    # If not running on Hugging Face Spaces, create a dummy decorator
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(func):  # noqa: N802
            return func


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_model = None


def get_model(model_path: str = "mazesmazes/tiny-audio"):
    """Initialize and cache the model."""
    global _model
    if _model is None:
        # Load model from HuggingFace Hub with trust_remote_code
        _model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        # Optionally move to GPU/MPS if available for better performance
        if torch.cuda.is_available():
            _model = _model.cuda()
            # Try to compile for faster inference on GPU
            try:
                _model = torch.compile(_model, mode="reduce-overhead")
            except Exception:
                # Compilation might fail in some environments
                pass
        elif torch.backends.mps.is_available():
            _model = _model.to("mps")
            # Try to compile for faster inference on MPS
            try:
                _model = torch.compile(_model, mode="reduce-overhead")
            except Exception:
                # Compilation might fail in some environments
                pass
        # Otherwise stays on CPU (default)

    return _model


@spaces.GPU
def transcribe_audio(audio_path: str, model_path: str = "mazesmazes/tiny-audio") -> str:
    """Transcribe audio using the model's built-in transcribe method."""
    model = get_model(model_path)
    return model.transcribe(audio_path, max_new_tokens=64)


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

        get_model(model_path)

    def transcribe(self, audio_path: str) -> str:
        return transcribe_audio(audio_path, self.model_path)

    def get_output_files(self) -> List[str]:
        """Get list of audio files from outputs directory."""
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

        if not self.outputs_dir.exists():
            return []

        audio_files: List[Path] = []
        for ext in audio_extensions:
            audio_files.extend(self.outputs_dir.glob(f"**/*{ext}"))

        return [str(f.relative_to(self.outputs_dir)) for f in sorted(audio_files)]

    def get_audio_path(self, selected_file: Optional[str]) -> str:
        """Get the full path for the selected audio file."""
        if not selected_file:
            return ""
        file_path = self.outputs_dir / selected_file
        return str(file_path) if file_path.exists() else ""

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
    with gr.Blocks(title="Tiny Audio") as demo:
        gr.Markdown("# 🎤 Tiny Audio")

        with gr.Row():
            with gr.Column():
                output_files = asr_demo.get_output_files()
                file_dropdown = gr.Dropdown(
                    choices=output_files,
                    label="Select audio file",
                    value=output_files[0] if output_files else None,
                )
                audio_player = gr.Audio(
                    label="Audio Preview",
                    type="filepath",
                    interactive=False,
                    value=asr_demo.get_audio_path(output_files[0]) if output_files else None,
                )
                process_btn = gr.Button("Transcribe Selected", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(
                    label="Transcription",
                    lines=5,
                    placeholder="Transcription will appear here...",
                )

        file_dropdown.change(
            asr_demo.get_audio_path, inputs=[file_dropdown], outputs=[audio_player]
        )

        process_btn.click(
            asr_demo.process_from_outputs, inputs=[file_dropdown], outputs=[output_text]
        )

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
