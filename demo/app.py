#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Microphone input
- File upload
- Word-level timestamps
- Speaker diarization
"""

import os

# Fix OpenMP environment variable if invalid
if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

# Set matplotlib config dir to avoid warning in Hugging Face Spaces
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Annotated

import gradio as gr
import torch
import typer
from transformers import pipeline

app = typer.Typer(add_completion=False)


def format_timestamp(seconds):
    """Format seconds as MM:SS.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def word_rows(words):
    """Word timestamps as table rows: start, end, speaker, word."""
    return [
        [format_timestamp(w["start"]), format_timestamp(w["end"]), w.get("speaker", ""), w["word"]]
        for w in words or []
    ]


def segment_rows(segments):
    """Speaker segments as table rows: start, end, speaker."""
    return [
        [format_timestamp(seg["start"]), format_timestamp(seg["end"]), seg["speaker"]]
        for seg in segments or []
    ]


THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
#header h1 { margin-bottom: 0.25rem; }
#header p { color: var(--body-text-color-subdued); margin-top: 0; }
"""

HEADER = """
<div id="header">
<h1>Tiny Audio</h1>
<p>Speech recognition with optional word timestamps and speaker diarization.
Model: <a href="https://huggingface.co/{model}" target="_blank">{model}</a></p>
</div>
"""


def create_demo(model_path="mazesmazes/tiny-audio"):
    """Create Gradio demo interface using transformers pipeline."""

    # Determine device
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1

    # Load pipeline - uses custom ASRPipeline from the model repo
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        trust_remote_code=True,
        device=device,
    )

    def process_audio(audio, show_timestamps, show_diarization, num_speakers=0):
        """Process audio file for transcription."""
        if audio is None:
            raise gr.Error("Record or upload some audio first.")

        # Build kwargs
        kwargs = {}
        if show_timestamps:
            kwargs["return_timestamps"] = True
        if show_diarization:
            kwargs["return_speakers"] = True
            # Auto-detection tends to over-split short clips; a known count
            # pins the clustering. 0 means auto.
            if num_speakers and int(num_speakers) > 0:
                kwargs["num_speakers"] = int(num_speakers)

        result = pipe(audio, **kwargs)

        if "timestamp_error" in result:
            gr.Warning(f"Word timestamps failed: {result['timestamp_error']}")
        if "diarization_error" in result:
            gr.Warning(f"Diarization failed: {result['diarization_error']}")

        words = word_rows(result.get("words")) if show_timestamps else []
        segments = segment_rows(result.get("speaker_segments")) if show_diarization else []
        return result.get("text", ""), words, segments

    with gr.Blocks(title="Tiny Audio") as demo:
        gr.HTML(HEADER.format(model=model_path))

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio",
                )

                with gr.Group():
                    show_timestamps = gr.Checkbox(
                        label="Word timestamps",
                        info="Align each word to the audio",
                        value=False,
                    )
                    show_diarization = gr.Checkbox(
                        label="Speaker diarization",
                        info="Label who spoke when",
                        value=False,
                    )
                    num_speakers = gr.Slider(
                        label="Number of speakers",
                        info="0 detects the count automatically",
                        value=0,
                        minimum=0,
                        maximum=10,
                        step=1,
                        visible=False,
                    )

                process_btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column(scale=3, min_width=400), gr.Tabs():
                with gr.Tab("Transcript"):
                    output_text = gr.Textbox(
                        show_label=False,
                        placeholder="Your transcript will appear here.",
                        lines=12,
                        buttons=["copy"],
                    )
                with gr.Tab("Words"):
                    timestamps_output = gr.Dataframe(
                        headers=["Start", "End", "Speaker", "Word"],
                        show_label=False,
                        interactive=False,
                        max_height=420,
                    )
                with gr.Tab("Speakers"):
                    diarization_output = gr.Dataframe(
                        headers=["Start", "End", "Speaker"],
                        show_label=False,
                        interactive=False,
                        max_height=420,
                    )

        # The speaker count only matters when diarization is on
        show_diarization.change(
            fn=lambda on: gr.update(visible=on),
            inputs=show_diarization,
            outputs=num_speakers,
        )
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, show_timestamps, show_diarization, num_speakers],
            outputs=[output_text, timestamps_output, diarization_output],
            api_name="transcribe",
        )

    return demo


@app.command()
def main(
    model: Annotated[
        str,
        typer.Option("--model", "-m", envvar="MODEL_ID", help="HuggingFace Hub model ID"),
    ] = "mazesmazes/tiny-audio",
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 7860,
    share: Annotated[bool, typer.Option("--share", help="Create public share link")] = False,
):
    """Launch ASR Gradio demo."""
    demo = create_demo(model)
    demo.launch(server_port=port, share=share, server_name="0.0.0.0", theme=THEME, css=CSS)


if __name__ == "__main__":
    app()
