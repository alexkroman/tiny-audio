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

# Fix OpenMP environment variable if invalid
if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

# Set matplotlib config dir to avoid warning in Hugging Face Spaces
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import torch
from transformers import pipeline

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

    # Determine device
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1

    # Load pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        trust_remote_code=True,
        device=device,
    )

    # Get the underlying model for streaming
    model = pipe.model
    feature_extractor = pipe.feature_extractor

    def process_audio(audio, show_timestamps, show_diarization, transcribe_prompt):
        """Process audio file for transcription."""
        if audio is None:
            return "Please provide audio input", "", ""

        # Build kwargs
        kwargs = {"user_prompt": transcribe_prompt.strip() if transcribe_prompt else None}
        if show_timestamps:
            kwargs["return_timestamps"] = True
        if show_diarization:
            kwargs["return_speakers"] = True

        # Transcribe the audio
        result = pipe(audio, **kwargs)

        # Format outputs
        transcript = result.get("text", "")

        # Format timestamps
        if show_timestamps and "words" in result:
            timestamps_text = format_words_with_timestamps(result["words"])
        elif "timestamp_error" in result:
            timestamps_text = f"Error: {result['timestamp_error']}"
        else:
            timestamps_text = ""

        # Format diarization
        if show_diarization and "speaker_segments" in result:
            diarization_text = format_speaker_segments(result["speaker_segments"])
        elif "diarization_error" in result:
            diarization_text = f"Error: {result['diarization_error']}"
        else:
            diarization_text = ""

        return transcript, timestamps_text, diarization_text

    def transcribe_streaming(audio, transcribe_prompt):
        """Stream transcription word by word."""
        if audio is None:
            yield "Please provide audio input"
            return

        # Load and preprocess audio
        import librosa

        audio_array, _ = librosa.load(audio, sr=16000)

        # Get features
        inputs = feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to model's device
        model_device = next(model.parameters()).device
        input_features = inputs.input_features.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device)

        # Set transcribe prompt
        if transcribe_prompt and transcribe_prompt.strip():
            model.TRANSCRIBE_PROMPT = transcribe_prompt.strip()
        else:
            model.TRANSCRIBE_PROMPT = DEFAULT_TRANSCRIBE_PROMPT

        # Stream generation
        accumulated = ""
        try:
            for chunk in model.generate_streaming(
                input_features=input_features,
                audio_attention_mask=attention_mask,
            ):
                accumulated += chunk
                yield accumulated
        except Exception as e:
            yield f"Error during streaming: {e}"

        # Final yield with complete text
        if accumulated:
            yield accumulated

    # Create Gradio interface with tabs
    with gr.Blocks(title="Tiny Audio") as demo:
        gr.Markdown("# Tiny Audio")
        gr.Markdown("Speech recognition with streaming, word timestamps, and speaker diarization.")

        with gr.Tabs():
            # Standard transcription tab
            with gr.TabItem("Standard"):
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
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

                        process_btn = gr.Button("Transcribe", variant="primary")

                    with gr.Column(scale=3):
                        output_text = gr.Textbox(
                            label="Transcript",
                            lines=5,
                        )
                        timestamps_output = gr.Textbox(
                            label="Word Timestamps",
                            lines=8,
                        )
                        diarization_output = gr.Textbox(
                            label="Speaker Segments",
                            lines=5,
                        )

            # Streaming transcription tab
            with gr.TabItem("Streaming"):
                gr.Markdown("Watch your words appear in real-time as the model transcribes.")

                with gr.Row():
                    with gr.Column(scale=2):
                        streaming_audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Audio Input",
                        )
                        stream_btn = gr.Button("Start Streaming", variant="primary")

                    with gr.Column(scale=3):
                        streaming_output = gr.Textbox(
                            label="Live Transcript",
                            lines=10,
                            placeholder="Words will appear here as they are transcribed...",
                        )

        # Advanced options (shared across tabs)
        with gr.Accordion("Advanced Options", open=False):
            transcribe_prompt_input = gr.Textbox(
                label="Transcribe Prompt",
                value=DEFAULT_TRANSCRIBE_PROMPT,
                lines=1,
                placeholder="Enter a custom transcribe prompt...",
            )
            gr.Markdown(
                "*The transcribe prompt is sent to the model before the audio. "
                "The default matches the prompt used during training.*"
            )

        # Wire up events
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, show_timestamps, show_diarization, transcribe_prompt_input],
            outputs=[output_text, timestamps_output, diarization_output],
        )

        stream_btn.click(
            fn=transcribe_streaming,
            inputs=[streaming_audio_input, transcribe_prompt_input],
            outputs=[streaming_output],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Gradio Demo")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_ID", "mazesmazes/tiny-audio"),
        help="HuggingFace Hub model ID",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()

    demo = create_demo(args.model)
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")
