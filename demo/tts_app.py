#!/usr/bin/env python3
"""Gradio demo for TTS model.

Loads a trained TTS model (SmolLM2 + expanded xcodec2 vocab) from HuggingFace Hub,
generates speech codes from text, decodes them with xcodec2, and plays audio.

Usage:
    poetry run python demo/tts_app.py
    poetry run python demo/tts_app.py --model mazesmazes/tiny-audio-tts
"""

import os
import tempfile
from typing import Annotated

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

app = typer.Typer(help="TTS Gradio Demo")

# xcodec2 constants (must match tiny_audio/tts.py)
XCODEC2_VOCAB_SIZE = 65536
XCODEC2_SAMPLE_RATE = 16000


def load_tts_model(model_path: str, device: str = "auto"):
    """Load trained TTS model and tokenizer from Hub or local path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # Resolve token IDs
    token_ids = {}
    for tok in [
        "<|TEXT_UNDERSTANDING_START|>",
        "<|TEXT_UNDERSTANDING_END|>",
        "<|SPEECH_GENERATION_START|>",
        "<|SPEECH_GENERATION_END|>",
    ]:
        token_ids[tok] = tokenizer.convert_tokens_to_ids(tok)
    token_ids["speech_token_offset"] = tokenizer.convert_tokens_to_ids("<|s_0|>")

    return model, tokenizer, token_ids


def load_xcodec2_decoder(device: str = "auto"):
    """Load xcodec2 model for decoding codes to audio."""
    from xcodec2.modeling_xcodec2 import XCodec2Model

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    codec = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2")
    codec.eval().to(device)
    return codec


def generate_speech(model, tokenizer, token_ids, text, max_new_tokens, temperature, top_p):
    """Generate xcodec2 codes from text."""
    text_start = token_ids["<|TEXT_UNDERSTANDING_START|>"]
    text_end = token_ids["<|TEXT_UNDERSTANDING_END|>"]
    speech_start = token_ids["<|SPEECH_GENERATION_START|>"]
    speech_end = token_ids["<|SPEECH_GENERATION_END|>"]
    speech_offset = token_ids["speech_token_offset"]

    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = [text_start] + text_tokens + [text_end, speech_start]
    input_ids = torch.tensor([input_ids], device=model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            eos_token_id=speech_end,
        )

    generated = output[0, input_ids.shape[1] :].tolist()
    if generated and generated[-1] == speech_end:
        generated = generated[:-1]

    codes = []
    for tok_id in generated:
        code = tok_id - speech_offset
        if 0 <= code < XCODEC2_VOCAB_SIZE:
            codes.append(code)

    return codes


def decode_codes_to_audio(codec_model, codes: list[int]) -> np.ndarray:
    """Decode xcodec2 codes to 16kHz audio waveform."""
    device = next(codec_model.parameters()).device
    vq_code = torch.LongTensor(codes).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        audio = codec_model.decode_code(vq_code).cpu()

    return audio[0, 0, :].float().numpy()


def create_demo(model_path: str = "mazesmazes/tiny-audio-tts"):
    """Create Gradio TTS demo interface."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, token_ids = load_tts_model(model_path, device=device)
    codec_model = load_xcodec2_decoder(device=device)

    def synthesize(text, max_new_tokens, temperature, top_p):
        if not text or not text.strip():
            return None, "Please enter some text."

        codes = generate_speech(
            model, tokenizer, token_ids, text.strip(),
            int(max_new_tokens), temperature, top_p,
        )

        if not codes:
            return None, "Generation failed - no speech codes produced."

        audio = decode_codes_to_audio(codec_model, codes)

        # Save to temp file for Gradio
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, XCODEC2_SAMPLE_RATE)

        info = f"Generated {len(codes)} speech codes ({len(codes) / 50:.1f}s at 50 tokens/sec)"
        return tmp.name, info

    with gr.Blocks(title="Tiny Audio TTS") as demo:
        gr.Markdown("# Tiny Audio TTS")
        gr.Markdown("Text-to-speech using a language model with xcodec2 speech tokens.")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to synthesize...",
                    lines=3,
                )

                with gr.Accordion("Settings", open=False):
                    max_tokens = gr.Slider(
                        minimum=100, maximum=4000, value=2000, step=100,
                        label="Max speech tokens",
                    )
                    temp = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.8, step=0.05,
                        label="Temperature (0 = greedy)",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.95, step=0.05,
                        label="Top-p",
                    )

                synth_btn = gr.Button("Synthesize", variant="primary")

            with gr.Column(scale=3):
                audio_output = gr.Audio(label="Output", type="filepath")
                info_output = gr.Textbox(label="Info", lines=1)

        synth_btn.click(
            fn=synthesize,
            inputs=[text_input, max_tokens, temp, top_p_slider],
            outputs=[audio_output, info_output],
        )

        gr.Examples(
            examples=[
                ["Hello, welcome to tiny audio text to speech."],
                ["The quick brown fox jumps over the lazy dog."],
                ["This is a test of the speech synthesis system."],
            ],
            inputs=[text_input],
        )

    return demo


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="HuggingFace Hub model ID or local path"),
    ] = os.environ.get("TTS_MODEL_ID", "mazesmazes/tiny-audio-tts"),
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
