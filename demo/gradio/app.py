#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Microphone input
- File upload
"""

import os

# Set matplotlib config dir to avoid warning in Hugging Face Spaces
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import torch
from transformers import pipeline


def create_demo(model_path: str = "mazesmazes/tiny-audio"):
    """Create Gradio demo interface using transformers pipeline."""

    # Determine device - prioritize CUDA > MPS > CPU
    # Note: For pipelines, we use integer device IDs or -1 for CPU
    # MPS device requires special handling
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        # For MPS, use device="mps" string for pipeline
        device = "mps"
    else:
        device = -1

    # Load pipeline directly - this ensures the custom ASRPipeline is used
    # The model's from_pretrained will handle the feature extractor configuration
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        trust_remote_code=True,
        device=device,
    )

    def process_audio(audio):
        """Process audio file for transcription."""
        if audio is None:
            return "Please provide audio input"
        # Transcribe the audio
        result = pipe(audio)
        return result["text"]

    # Create Gradio interface
    with gr.Blocks(title="Tiny Audio") as demo:
        gr.Markdown("# Tiny Audio")

        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Audio Input (Record from microphone or upload file)",
        )

        output_text = gr.Textbox(label="Output")
        process_btn = gr.Button("Process")

        # Wire up events
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=output_text,
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
