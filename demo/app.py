#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Microphone input
- File upload
"""

import os

# Set matplotlib config dir to avoid warning in Hugging Face Spaces
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import gradio as gr
import torch
from transformers import pipeline


def create_demo(model_path: str = "mazesmazes/tiny-audio"):
    """Create Gradio demo interface using transformers pipeline."""

    # Determine device
    device = 0 if torch.cuda.is_available() else -1

    # Load ASR pipeline with trust_remote_code
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        trust_remote_code=True,
        device=device,
    )

    def transcribe(audio):
        """Transcribe audio file or recording."""
        if audio is None:
            return ""
        result = pipe(audio)
        return result["text"]

    # Create Gradio interface
    demo = gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(type="filepath"),
        outputs=gr.Textbox(label="Transcription"),
        title="Tiny Audio ASR",
        description="Upload an audio file or record from microphone to transcribe.",
        examples=[],
    )

    return demo



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Gradio Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="mazesmazes/tiny-audio",
        help="HuggingFace Hub model ID (e.g., mazesmazes/tiny-audio)",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run the demo on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")

    args = parser.parse_args()

    demo = create_demo(args.model)
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")
