#!/usr/bin/env python3
"""Gradio TTS demo for tiny-audio-tts (LLASA-style text-to-speech)."""

import os
import warnings
from typing import Annotated

# Suppress noisy deprecation warnings from dependencies
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated.*")

import gradio as gr
import numpy as np
import torch
import typer

from tiny_audio.lm import generate_speech, setup_tts_model

app = typer.Typer(help="TTS Gradio Demo")

NEUCODEC_SAMPLE_RATE = 24000


def create_demo(model_path: str = "mazesmazes/tiny-audio-tts"):
    """Create Gradio TTS demo interface."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load TTS model
    model, tokenizer, token_ids = setup_tts_model(checkpoint_path=model_path)
    model = model.to(device)
    model.eval()

    # Load NeuCodec for decoding codes to audio
    from neucodec import NeuCodec

    neucodec = NeuCodec.from_pretrained("neuphonic/neucodec")
    neucodec = neucodec.to(device)
    neucodec.eval()

    def synthesize(text: str, temperature: float, top_p: float):
        """Generate speech from text."""
        if not text.strip():
            return None

        codes = generate_speech(
            model, tokenizer, text, token_ids,
            temperature=temperature, top_p=top_p,
        )
        if codes is None:
            return None

        # Decode codes to audio waveform (same pattern as AudioHead.decode_to_audio)
        codes_tensor = torch.tensor([codes], device=device)
        codes_3d = codes_tensor.unsqueeze(1)
        with torch.no_grad():
            audio_values = neucodec.decode_code(codes_3d)
        waveform = audio_values[0, 0].cpu().numpy()
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform = (waveform * 32767).astype(np.int16)

        return (NEUCODEC_SAMPLE_RATE, waveform)

    with gr.Blocks(title="Tiny Audio TTS") as demo:
        gr.Markdown("# Tiny Audio TTS")
        gr.Markdown("Text-to-speech powered by LLASA-style LLM with NeuCodec.")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to synthesize...",
                    lines=3,
                )
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.5, value=0.9, step=0.05,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.95, step=0.05,
                        label="Top-p",
                    )
                speak_btn = gr.Button("Speak", variant="primary")

            with gr.Column(scale=3):
                audio_output = gr.Audio(label="Output", type="numpy")

        speak_btn.click(
            fn=synthesize,
            inputs=[text_input, temperature, top_p],
            outputs=[audio_output],
        )

    return demo


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="HuggingFace Hub model ID or checkpoint path"),
    ] = os.environ.get("MODEL_ID", "mazesmazes/tiny-audio-tts"),
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 7861,
    share: Annotated[bool, typer.Option("--share", "-s", help="Create public share link")] = False,
):
    """Launch TTS Gradio demo."""
    if ctx.invoked_subcommand is not None:
        return
    demo = create_demo(model)
    demo.launch(server_port=port, share=share, server_name="0.0.0.0")


if __name__ == "__main__":
    app()
