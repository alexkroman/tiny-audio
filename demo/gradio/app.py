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

    def process_audio(audio, task, text_input, custom_prompt):
        """Process audio file or text input with selected task."""
        if task == "text":
            if not text_input or not text_input.strip():
                return "Please provide text input for text task"
            # Use text-only mode
            result = pipe(None, task="text", text_input=text_input)
            return result["text"]
        elif task == "custom":
            if audio is None:
                return "Please provide audio input"
            if not custom_prompt or not custom_prompt.strip():
                return "Please provide a custom prompt"
            # Use custom prompt with user_prompt parameter
            result = pipe(audio, user_prompt=f"{custom_prompt.strip()}: <audio>")
            return result["text"]
        else:
            if audio is None:
                return "Please provide audio input"
            # Pass the task parameter to the pipeline
            result = pipe(audio, task=task)
            return result["text"]

    def update_inputs(task):
        """Update input visibility based on selected task."""
        if task == "text":
            # Show text input, hide audio and custom prompt
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        elif task == "custom":
            # Show audio and custom prompt, hide text input
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
        else:
            # Show audio, hide text input and custom prompt
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    # Create Gradio interface with task selection
    with gr.Blocks(title="Tiny Audio") as demo:
        gr.Markdown("# Tiny Audio")

        with gr.Row():
            with gr.Column():
                task_selector = gr.Dropdown(
                    choices=["transcribe", "describe", "emotion", "custom", "text"],
                    value="transcribe",
                    label="Task",
                    info="Select the task to perform"
                )
                custom_prompt = gr.Textbox(
                    label="Custom Prompt",
                    placeholder="Enter your custom prompt (e.g., 'Classify', 'Summarize', 'Translate to Spanish')...",
                    lines=2,
                    visible=False
                )

        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Audio Input (Record from microphone or upload file)",
                visible=True
            )
            text_input = gr.Textbox(
                label="Text Input (for text task)",
                placeholder="Enter text to process...",
                lines=3,
                visible=False
            )

        output_text = gr.Textbox(label="Output")
        process_btn = gr.Button("Process")

        # Wire up events
        task_selector.change(
            fn=update_inputs,
            inputs=[task_selector],
            outputs=[audio_input, text_input, custom_prompt]
        )

        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, task_selector, text_input, custom_prompt],
            outputs=output_text
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
