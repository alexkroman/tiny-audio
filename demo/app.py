#!/usr/bin/env python3
"""
Gradio app for ASR model with support for:
- Microphone input
- File upload
- Word-level timestamps
- Speaker diarization

With ENDPOINT_URL set, requests go to a tiny-audio server (`ta serve`) and this
app needs no GPU, torch or transformers; without it the model runs in-process.
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

import base64
import html
import sys
from collections.abc import Callable, Mapping, Sequence
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict, cast

import gradio as gr
import httpx
import soundfile
import typer
from gradio import themes


class Word(TypedDict):
    """One aligned word from the pipeline's `words` output."""

    word: str
    start: float
    end: float
    speaker: NotRequired[str]


class SpeakerWord(TypedDict):
    """A word from a diarized result, where every word carries its speaker."""

    word: str
    start: float
    end: float
    speaker: str


class SpeakerSegment(TypedDict):
    start: float
    end: float
    speaker: str


def gpu_seconds(audio: str, kwargs: Mapping[str, Any]) -> int:
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


def gpu[F: Callable[..., Any]](fn: F) -> F:
    return zero_gpu(duration=gpu_seconds)(fn) if zero_gpu else fn


app = typer.Typer(add_completion=False)


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def speaker_label(speaker: str) -> str:
    """The pipeline's "SPEAKER_0" as "Speaker 1" for people; anything else unchanged."""
    prefix, _, index = (speaker or "").rpartition("_")
    return f"Speaker {int(index) + 1}" if prefix == "SPEAKER" and index.isdigit() else speaker


def word_rows(words: Sequence[Word] | None) -> list[list[str]]:
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


def speaker_turns(words: Sequence[SpeakerWord] | None) -> list[tuple[str, float, float, str]]:
    """Consecutive words of one speaker as turns: (speaker, start, end, text)."""
    turns: list[tuple[str, float, float, str]] = []
    for speaker, group in groupby(words or [], key=itemgetter("speaker")):
        run = list(group)
        turns.append((speaker, run[0]["start"], run[-1]["end"], " ".join(w["word"] for w in run)))
    return turns


def conversation_html(words: Sequence[SpeakerWord] | None) -> str:
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


def segment_rows(segments: Sequence[SpeakerSegment] | None) -> list[list[str]]:
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


def pick_device() -> int | str:
    """The pipeline device: CUDA index 0, Apple "mps", or -1 for CPU."""
    import torch  # noqa: PLC0415 -- server mode runs without torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


def pipeline_kwargs(
    show_timestamps: bool, show_diarization: bool, num_speakers: float, max_speakers: float
) -> dict[str, Any]:
    """Pipeline call options for the requested outputs and speaker-count hints."""
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
    return kwargs


def warn_partial_failures(result: Mapping[str, Any]) -> None:
    """Surface a failed timestamp or diarization stage as a UI warning."""
    if "timestamp_error" in result:
        gr.Warning(f"Word timestamps failed: {result['timestamp_error']}")
    if "diarization_error" in result:
        gr.Warning(f"Diarization failed: {result['diarization_error']}")


def toggle_speaker_controls(on: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Show the speaker-count sliders only while diarization is on: they only matter then."""
    return gr.update(visible=on), gr.update(visible=on)


def build_options() -> tuple[gr.Checkbox, gr.Checkbox, gr.Slider, gr.Slider]:
    """The output toggles and speaker-count sliders, in one group."""
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
    return show_timestamps, show_diarization, num_speakers, max_speakers


def build_output_tabs() -> tuple[gr.Tabs, gr.Textbox, gr.HTML, gr.Dataframe, gr.Dataframe]:
    """The result tabs: transcript, conversation, word and speaker tables."""
    with gr.Tabs(selected="transcript") as tabs:
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
    return tabs, output_text, conversation_output, timestamps_output, diarization_output


def local_runner(model_path: str) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Run the pipeline in this process (on the GPU, on ZeroGPU)."""
    from transformers import pipeline  # noqa: PLC0415 -- server mode runs without it

    # Load pipeline - uses custom ASRPipeline from the model repo
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        trust_remote_code=True,
        device=pick_device(),
    )
    # Load the aligner and diarizer now, not on the first request: on ZeroGPU
    # each request runs in a forked worker, so a model first loaded there is
    # thrown away afterwards and reloaded on every call.
    pipeline_module = sys.modules[type(pipe).__module__]
    pipeline_module.QwenForcedAligner.get_instance()
    pipeline_module.NemotronDiarizer.get_instance()

    @gpu
    def run_pipeline(audio: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        # One audio input gives one result dict; transformers annotates the
        # pipeline's __call__ with the batched (list) return type.
        return cast(dict[str, Any], pipe(audio, **kwargs))

    return run_pipeline


# The RunPod proxy answers 502-504 while the pod's server is down or still
# loading its models; anything else is a real failure.
WAKING_UP = "The model server is starting up or offline. Please try again in a few minutes."


def remote_runner(
    endpoint_url: str, api_key: str | None
) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """POST each request to a `ta serve` server (scripts/serve.py)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def run_pipeline(audio: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        # A JSON body is the only way to send `parameters` with the audio,
        # so the audio travels base64-encoded.
        payload = {
            "inputs": base64.b64encode(Path(audio).read_bytes()).decode(),
            "parameters": kwargs,
        }
        try:
            response = httpx.post(endpoint_url, json=payload, headers=headers, timeout=600)
        except httpx.TimeoutException as e:
            msg = "The model server took too long to respond."
            raise gr.Error(msg) from e
        except httpx.HTTPError as e:
            msg = f"Could not reach the model server: {e}"
            raise gr.Error(msg) from e
        if response.status_code in (502, 503, 504):
            raise gr.Error(WAKING_UP)
        if response.is_error:
            msg = f"Model server error {response.status_code}: {response.text[:300]}"
            raise gr.Error(msg)
        return cast(dict[str, Any], response.json())

    return run_pipeline


def create_demo(
    model_path: str = "mazesmazes/tiny-audio",
    endpoint_url: str | None = None,
    api_key: str | None = None,
) -> gr.Blocks:
    """Create the Gradio demo, backed by a `ta serve` server or a local pipeline."""
    run_pipeline = (
        remote_runner(endpoint_url, api_key) if endpoint_url else local_runner(model_path)
    )

    def process_audio(
        audio: str | None,
        show_timestamps: bool,
        show_diarization: bool,
        num_speakers: float = 0,
        max_speakers: float = 0,
    ) -> tuple[str, str, list[list[str]], list[list[str]], gr.Tabs]:
        """Process audio file for transcription."""
        if audio is None:
            msg = "Record or upload some audio first."
            raise gr.Error(msg)

        kwargs = pipeline_kwargs(show_timestamps, show_diarization, num_speakers, max_speakers)

        result = run_pipeline(audio, kwargs)
        warn_partial_failures(result)

        words = word_rows(result.get("words")) if show_timestamps or show_diarization else []
        segments = segment_rows(result.get("speaker_segments")) if show_diarization else []
        conversation = conversation_html(result.get("words") if show_diarization else None)
        # Open the conversation view when there are speakers to show.
        tab = gr.Tabs(selected="conversation" if show_diarization else "transcript")
        text: str = result.get("text", "")
        return text, conversation, words, segments, tab

    demo: gr.Blocks
    with gr.Blocks(title="Tiny Audio") as demo:
        gr.HTML(HEADER.format(model=model_path))

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio",
                )

                show_timestamps, show_diarization, num_speakers, max_speakers = build_options()

                process_btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column(scale=3, min_width=400):
                tabs, output_text, conversation_output, timestamps_output, diarization_output = (
                    build_output_tabs()
                )

        # gradio attaches event listeners at runtime and only writes the .pyi
        # stubs declaring them on its first import, so a fresh install has none.
        show_diarization.change(  # pyright: ignore[reportAttributeAccessIssue]
            fn=toggle_speaker_controls,
            inputs=show_diarization,
            outputs=[num_speakers, max_speakers],
            api_visibility="private",
        )
        inputs = [audio_input, show_timestamps, show_diarization, num_speakers, max_speakers]
        outputs = [output_text, conversation_output, timestamps_output, diarization_output, tabs]
        process_btn.click(  # pyright: ignore[reportAttributeAccessIssue]
            fn=process_audio, inputs=inputs, outputs=outputs, api_name="transcribe"
        )

        if EXAMPLE.exists():
            gr.Examples(
                examples=[[str(EXAMPLE), True, True, 0, 0]],
                inputs=inputs,
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
    endpoint_url: Annotated[
        str | None,
        typer.Option(
            "--endpoint-url",
            envvar="ENDPOINT_URL",
            help="`ta serve` URL to send requests to instead of loading the model",
        ),
    ] = None,
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 7860,
    share: Annotated[bool, typer.Option("--share", help="Create public share link")] = False,
) -> None:
    """Launch ASR Gradio demo."""
    demo = create_demo(model, endpoint_url, os.environ.get("TINY_AUDIO_API_KEY"))
    demo.launch(server_port=port, share=share, server_name="0.0.0.0", theme=THEME, css=CSS)


if __name__ == "__main__":
    app()
