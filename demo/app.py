#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Microphone input
- File upload
- Sample audio files
"""

import os

# Set matplotlib config dir to avoid warning in Hugging Face Spaces
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import logging

import gradio as gr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

_client = None


def get_client(model_path: str = "mazesmazes/tiny-audio"):
    """Initialize and cache the InferenceClient."""
    global _client
    if _client is None:
        endpoint_url = os.environ.get("INFERENCE_ENDPOINT_URL", model_path)
        from huggingface_hub import InferenceClient

        _client = InferenceClient(model=endpoint_url)
    return _client


def transcribe_audio(audio, model_path: str = "mazesmazes/tiny-audio") -> str:
    """Transcribe audio using InferenceClient."""
    if audio is None:
        return "No audio provided"

    client = get_client(model_path)

    try:
        # InferenceClient accepts file paths directly
        # This may take longer on first request while GPU spins up
        result = client.automatic_speech_recognition(audio)
        return result.get("text", str(result))
    except Exception as e:
        return f"Error: {str(e)}"


def create_demo(model_path: str = "mazesmazes/tiny-audio"):
    """Create Gradio demo interface."""
    # Pre-load client
    get_client(model_path)

    return gr.Interface(
        fn=lambda audio: transcribe_audio(audio, model_path),
        inputs=gr.Audio(sources=["upload"], type="filepath", label="Audio"),
        outputs=gr.Textbox(label="Transcription"),
        title="Tiny Audio",
        description="Upload an audio file to transcribe. **Note:** First request may take 30-60 seconds while GPU spins up.",
        article="Powered by [Tiny Audio](https://huggingface.co/mazesmazes/tiny-audio) model.",
    )



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
