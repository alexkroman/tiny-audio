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

    # Load ASR pipeline directly from the model - it will load everything needed
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        trust_remote_code=True,
        device=device,
    )

    def process_audio(audio, task):
        """Process audio file with selected task."""
        if audio is None:
            return ""

        # Pass the task parameter to the pipeline
        result = pipe(audio, task=task)
        return result["text"]

    # Create Gradio interface with task selection
    demo = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(type="filepath", label="Audio Input"),
            gr.Dropdown(
                choices=["transcribe", "describe", "continue", "emotion"],
                value="transcribe",
                label="Task",
                info="Select the task to perform on the audio"
            )
        ],
        outputs=gr.Textbox(label="Output"),
        title="Tiny Audio - Multi-Task Audio Processing",
        description="Upload an audio file or record from microphone. Select a task to perform:\n"
                   "• **transcribe**: Convert speech to text\n"
                   "• **describe**: Describe the audio content\n"
                   "• **continue**: Continue/complete the audio\n"
                   "• **emotion**: Analyze emotional content",
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
