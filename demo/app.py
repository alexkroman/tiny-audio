#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Microphone input
- File upload
- Word-level timestamps
- Speaker diarization
- Streaming transcription
- Custom system prompts
"""

import os
from typing import Annotated

# Fix OpenMP environment variable if invalid
if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

# Set matplotlib config dir to avoid warning in Hugging Face Spaces
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import typer
from transformers import AutoModel, AutoProcessor

from tiny_audio.asr_pipeline import ASRPipeline

app = typer.Typer(help="ASR Gradio Demo")

# Default transcribe prompt (matches training)
DEFAULT_TRANSCRIBE_PROMPT = "Transcribe: "


def format_timestamp(seconds):
    """Format seconds as MM:SS.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def format_words_with_timestamps(words):
    """Format word timestamps as readable text."""
    if not words:
        return ""

    lines = []
    for w in words:
        start = format_timestamp(w["start"])
        end = format_timestamp(w["end"])
        speaker = w.get("speaker", "")
        if speaker:
            lines.append(f"[{start} - {end}] ({speaker}) {w['word']}")
        else:
            lines.append(f"[{start} - {end}] {w['word']}")

    return "\n".join(lines)


def format_words_with_speakers(words):
    """Format words grouped by speaker."""
    if not words:
        return ""

    lines = []
    current_speaker = None
    current_words = []

    for w in words:
        speaker = w.get("speaker", "Unknown")
        if speaker != current_speaker:
            # Output previous speaker's words
            if current_words:
                lines.append(f"{current_speaker}: {' '.join(current_words)}")
            current_speaker = speaker
            current_words = [w["word"]]
        else:
            current_words.append(w["word"])

    # Output last speaker's words
    if current_words:
        lines.append(f"{current_speaker}: {' '.join(current_words)}")

    return "\n".join(lines)


def format_speaker_segments(segments):
    """Format speaker segments as readable text."""
    if not segments:
        return ""

    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"[{start} - {end}] {seg['speaker']}")

    return "\n".join(lines)


def create_demo(model_path="mazesmazes/tiny-audio"):
    """Create Gradio demo interface using transformers pipeline."""

    # Load model with automatic device placement
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto", torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    feature_extractor = processor.feature_extractor
    pipe = ASRPipeline(model=model, feature_extractor=feature_extractor)

    def process_audio(audio, show_timestamps, show_diarization, transcribe_prompt):
        """Process audio file for transcription."""
        if audio is None:
            return "Please provide audio input"

        # Build kwargs
        kwargs = {}
        # Only pass user_prompt if different from default
        prompt = transcribe_prompt.strip() if transcribe_prompt else None
        if prompt and prompt != DEFAULT_TRANSCRIBE_PROMPT:
            kwargs["user_prompt"] = prompt
        # Always get timestamps if diarization is requested (needed to assign speakers to words)
        if show_timestamps or show_diarization:
            kwargs["return_timestamps"] = True
        if show_diarization:
            kwargs["return_speakers"] = True

        # Transcribe the audio
        result = pipe(audio, **kwargs)

        # Return timestamps (with speaker labels if diarization also enabled)
        if show_timestamps and "words" in result:
            return format_words_with_timestamps(result["words"])
        if show_timestamps and "timestamp_error" in result:
            return f"Error: {result['timestamp_error']}"

        # Return diarization only (words with speakers, no timestamps shown)
        if show_diarization and "words" in result:
            return format_words_with_speakers(result["words"])
        if show_diarization and "diarization_error" in result:
            return f"Error: {result['diarization_error']}"

        # Default: return transcript
        return result.get("text", "")

    # Create Gradio interface
    with gr.Blocks(title="Tiny Audio") as demo:
        gr.Markdown("# Tiny Audio")
        gr.Markdown("Speech recognition with word timestamps and speaker diarization.")

        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Audio Input",
                )

                with gr.Row():
                    show_timestamps = gr.Checkbox(
                        label="Word Timestamps",
                        value=False,
                    )
                    show_diarization = gr.Checkbox(
                        label="Speaker Diarization",
                        value=False,
                    )

                transcribe_prompt_input = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_TRANSCRIBE_PROMPT,
                    lines=1,
                )
                process_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=3):
                output_text = gr.Textbox(
                    label="Transcript",
                    lines=12,
                )

        # Wire up events
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, show_timestamps, show_diarization, transcribe_prompt_input],
            outputs=[output_text],
        )

    return demo


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="HuggingFace Hub model ID"),
    ] = os.environ.get("MODEL_ID", "mazesmazes/tiny-audio"),
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 7860,
    share: Annotated[bool, typer.Option("--share", "-s", help="Create public share link")] = False,
):
    """Launch ASR Gradio demo."""
    if ctx.invoked_subcommand is not None:
        return
    demo = create_demo(model)
    demo.launch(server_port=port, share=share, server_name="0.0.0.0")


if __name__ == "__main__":
    app()
