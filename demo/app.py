#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Microphone input
- File upload
- Word-level timestamps
- Speaker diarization
"""

import os

# ZeroGPU: `spaces` must be imported before anything that touches CUDA (torch,
# transformers). It is preinstalled on Spaces; locally the decorator is a no-op.
try:
    import spaces
except ImportError:
    zero_gpu = None
else:
    zero_gpu = spaces.GPU


# Fix OpenMP environment variable if invalid
if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

# Set matplotlib config dir to avoid warning in Hugging Face Spaces
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import html
import sys
from pathlib import Path
from typing import Annotated, Any, cast

import gradio as gr
import soundfile
import torch
import typer
from gradio import themes
from transformers import pipeline


def gpu_seconds(audio, kwargs):
    """GPU time to reserve for one request, from the audio's length.

    ZeroGPU charges each visitor's daily quota the RESERVED duration up front,
    not the time used, so a flat 120 s locked visitors out after a request or
    two. Measured on the A10G: about 0.2 s of GPU per second of audio with
    timestamps and diarization (46 s of speech in 8.7 s). Reserve twice that
    plus load overhead, within ZeroGPU's 20-120 s.
    """
    try:
        seconds = soundfile.info(audio).duration
    except Exception:  # unreadable here: let the pipeline report it, reserve the cap
        return 120
    per_second = 0.4 if kwargs.get("return_timestamps") or kwargs.get("return_speakers") else 0.2
    return int(min(120, max(20, 10 + per_second * seconds)))


def gpu(fn):
    return zero_gpu(duration=gpu_seconds)(fn) if zero_gpu else fn


app = typer.Typer(add_completion=False)


def format_timestamp(seconds):
    """Format seconds as MM:SS.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def speaker_label(speaker):
    """The pipeline's "SPEAKER_0" as "Speaker 1" for people; anything else unchanged."""
    prefix, _, index = (speaker or "").rpartition("_")
    return f"Speaker {int(index) + 1}" if prefix == "SPEAKER" and index.isdigit() else speaker


def word_rows(words):
    """Word timestamps as table rows: start, end, speaker, word."""
    return [
        [
            format_timestamp(w["start"]),
            format_timestamp(w["end"]),
            speaker_label(w.get("speaker", "")),
            w["word"],
        ]
        for w in words or []
    ]


# One color per speaker, in the order they first speak (Nemotron tracks at most 8).
SPEAKER_COLORS = [
    "#4f46e5",
    "#0d9488",
    "#d97706",
    "#db2777",
    "#2563eb",
    "#65a30d",
    "#9333ea",
    "#dc2626",
]


def speaker_turns(words):
    """Consecutive words of one speaker as turns: (speaker, start, end, text)."""
    turns: list[tuple[str, float, float, str]] = []
    for w in words or []:
        if turns and turns[-1][0] == w["speaker"]:
            speaker, start, _, text = turns[-1]
            turns[-1] = (speaker, start, w["end"], f"{text} {w['word']}")
        else:
            turns.append((w["speaker"], w["start"], w["end"], w["word"]))
    return turns


def conversation_html(words):
    """Speaker-attributed transcript: one color-coded block per speaker turn.

    Styled inline rather than through the app's CSS: on Spaces this block
    rendered without the stylesheet's rules (only inline colors survived).
    """
    turns = speaker_turns(words)
    if not turns:
        return (
            '<p style="color:var(--body-text-color-subdued)">'
            "Turn on speaker diarization to see who said what.</p>"
        )
    colors: dict[str, str] = {}
    for speaker, *_ in turns:
        colors.setdefault(speaker, SPEAKER_COLORS[len(colors) % len(SPEAKER_COLORS)])
    blocks = [
        f'<div style="border-left:4px solid {colors[s]};padding:0.35rem 0.75rem;'
        'background:var(--background-fill-secondary);border-radius:0 6px 6px 0">'
        f'<div style="color:{colors[s]};font-weight:600;font-size:0.85rem;margin-bottom:0.15rem">'
        f"{html.escape(speaker_label(s))}"
        '<span style="font-weight:400;color:var(--body-text-color-subdued);margin-left:0.5rem">'
        f"{format_timestamp(start)} \u2013 {format_timestamp(end)}</span></div>"
        f"<div>{html.escape(text)}</div></div>"
        for s, start, end, text in turns
    ]
    return (
        '<div style="display:flex;flex-direction:column;gap:0.6rem;'
        'max-height:420px;overflow-y:auto">' + "".join(blocks) + "</div>"
    )


def segment_rows(segments):
    """Speaker segments as table rows: start, end, speaker."""
    return [
        [
            format_timestamp(seg["start"]),
            format_timestamp(seg["end"]),
            speaker_label(seg["speaker"]),
        ]
        for seg in segments or []
    ]


THEME = themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
#header h1 { margin-bottom: 0.25rem; }
#header p { color: var(--body-text-color-subdued); margin-top: 0; }
"""

EXAMPLE = Path(__file__).parent / "examples" / "ami_meeting.wav"

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
    device: int | str
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
    # Load the aligner and diarizer now, not on the first request: on ZeroGPU
    # each request runs in a forked worker, so a model first loaded there is
    # thrown away afterwards and reloaded on every call.
    pipeline_module = sys.modules[type(pipe).__module__]
    pipeline_module.QwenForcedAligner.get_instance()
    pipeline_module.NemotronDiarizer.get_instance()

    @gpu
    def run_pipeline(audio, kwargs) -> dict[str, Any]:
        # One audio input gives one result dict; transformers annotates the
        # pipeline's __call__ with the batched (list) return type.
        return cast(dict[str, Any], pipe(audio, **kwargs))

    def process_audio(audio, show_timestamps, show_diarization, num_speakers=0, max_speakers=0):
        """Process audio file for transcription."""
        if audio is None:
            msg = "Record or upload some audio first."
            raise gr.Error(msg)

        # Build kwargs
        kwargs: dict[str, Any] = {}
        if show_timestamps:
            kwargs["return_timestamps"] = True
        if show_diarization:
            kwargs["return_speakers"] = True
            # An exact count, or an upper bound, caps the speakers Nemotron
            # keeps; the exact count wins if both are set. 0 means auto.
            if num_speakers and int(num_speakers) > 0:
                kwargs["num_speakers"] = int(num_speakers)
            elif max_speakers and int(max_speakers) > 0:
                kwargs["max_speakers"] = int(max_speakers)

        # Transcribe the audio (on the GPU, on ZeroGPU)
        result = run_pipeline(audio, kwargs)

        if "timestamp_error" in result:
            gr.Warning(f"Word timestamps failed: {result['timestamp_error']}")
        if "diarization_error" in result:
            gr.Warning(f"Diarization failed: {result['diarization_error']}")

        words = word_rows(result.get("words")) if show_timestamps or show_diarization else []
        segments = segment_rows(result.get("speaker_segments")) if show_diarization else []
        conversation = conversation_html(result.get("words") if show_diarization else None)
        # Open the conversation view when there are speakers to show.
        tab = gr.Tabs(selected="conversation" if show_diarization else "transcript")
        return result.get("text", ""), conversation, words, segments, tab

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
                        info="Exact count if you know it; 0 detects it automatically",
                        value=0,
                        minimum=0,
                        maximum=8,
                        step=1,
                        visible=False,
                    )
                    max_speakers = gr.Slider(
                        label="Maximum speakers",
                        info="Upper bound when the exact count is unknown; 0 means no limit",
                        value=0,
                        minimum=0,
                        maximum=8,
                        step=1,
                        visible=False,
                    )

                process_btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column(scale=3, min_width=400), gr.Tabs(selected="transcript") as tabs:
                with gr.Tab("Transcript", id="transcript"):
                    output_text = gr.Textbox(
                        show_label=False,
                        placeholder="Your transcript will appear here.",
                        lines=12,
                        buttons=["copy"],
                    )
                with gr.Tab("Conversation", id="conversation"):
                    conversation_output = gr.HTML(conversation_html(None))
                with gr.Tab("Words", id="words"):
                    timestamps_output = gr.Dataframe(
                        headers=["Start", "End", "Speaker", "Word"],
                        show_label=False,
                        interactive=False,
                        max_height=420,
                    )
                with gr.Tab("Speakers", id="speakers"):
                    diarization_output = gr.Dataframe(
                        headers=["Start", "End", "Speaker"],
                        show_label=False,
                        interactive=False,
                        max_height=420,
                    )

        # The speaker controls only matter when diarization is on
        show_diarization.change(
            fn=lambda on: (gr.update(visible=on), gr.update(visible=on)),
            inputs=show_diarization,
            outputs=[num_speakers, max_speakers],
            api_visibility="private",
        )
        inputs = [audio_input, show_timestamps, show_diarization, num_speakers, max_speakers]
        outputs = [output_text, conversation_output, timestamps_output, diarization_output, tabs]
        process_btn.click(fn=process_audio, inputs=inputs, outputs=outputs, api_name="transcribe")

        if EXAMPLE.exists():
            gr.Examples(
                examples=[[str(EXAMPLE), True, True, 0, 0]],
                inputs=inputs,
                outputs=outputs,
                fn=process_audio,
                label="Try a two-person meeting (AMI Meeting Corpus, CC BY 4.0)",
                cache_examples=False,
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
