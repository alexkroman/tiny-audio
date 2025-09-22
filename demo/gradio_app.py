#!/usr/bin/env python3
"""
Gradio demo for ASR model with support for:
- Microphone recording
- File upload
- Processing audio files from outputs directory
"""

import os
import sys
from pathlib import Path
import gradio as gr
import torch
import torchaudio
from transformers import AutoTokenizer, WhisperFeatureExtractor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.modeling import ASRModel

logger = logging.getLogger(__name__)


class ASRDemo:
    def __init__(self, model_path: str = None, outputs_dir: str = "wav_outputs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Handle relative paths - make sure we find the directory
        outputs_path = Path(outputs_dir)
        if not outputs_path.is_absolute():
            # Check if we're in the demo directory or parent
            if Path("wav_outputs").exists():
                outputs_path = Path("wav_outputs")
            elif Path("demo/wav_outputs").exists():
                outputs_path = Path("demo/wav_outputs")
            else:
                outputs_path = Path(outputs_dir)

        self.outputs_dir = outputs_path
        logger.info(f"Using outputs directory: {self.outputs_dir.absolute()}")

        # Load model and associated components
        if not model_path or not Path(model_path).exists():
            raise ValueError(f"Model path is required and must exist. Provided path: {model_path}")

        logger.info(f"Loading model from {model_path}")
        # Load tokenizer and feature extractor from saved model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

        # Load the model using from_pretrained
        logger.info("Loading ASR model...")
        self.model = ASRModel.from_pretrained(model_path)

        # Move to device (keep the model's original dtype)
        self.model = self.model.to(self.device)

        # Ensure dtype consistency after loading
        self.model.post_init()

        self.model.eval()
        logger.info(f"Model loaded with dtype: {self.model.dtype}")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Prepare audio features using WhisperFeatureExtractor
            inputs = self.feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move to device
            audio_features = inputs.input_features.to(self.device)

            # Generate transcription
            with torch.no_grad():
                # Generate text using the model's generate method
                generated_ids = self.model.generate(
                    input_features=audio_features,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # Decode text
                transcription = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            return transcription

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return f"Error: {str(e)}"

    def process_audio(self, audio_input):
        """Process audio from microphone or file upload."""
        if audio_input is None:
            return "Please provide audio input"

        # If audio_input is a tuple (sample_rate, audio_data) from microphone
        if isinstance(audio_input, tuple):
            sample_rate, audio_data = audio_input
            # Save temporary file
            temp_path = "temp_audio.wav"
            torchaudio.save(temp_path, torch.tensor(audio_data).unsqueeze(0), sample_rate)
            result = self.transcribe(temp_path)
            os.remove(temp_path)
            return result
        else:
            # File path from upload
            return self.transcribe(audio_input)

    def get_output_files(self):
        """Get list of audio files from outputs directory."""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']

        if not self.outputs_dir.exists():
            return []

        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(self.outputs_dir.glob(f"**/*{ext}"))

        # Return relative paths as strings
        return [str(f.relative_to(self.outputs_dir)) for f in sorted(audio_files)]

    def process_from_outputs(self, selected_file):
        """Process a file from the outputs directory."""
        if not selected_file:
            return "Please select a file"

        file_path = self.outputs_dir / selected_file

        if not file_path.exists():
            return f"File not found: {file_path}"

        return self.transcribe(str(file_path))


def create_demo(model_path: str = None, outputs_dir: str = "wav_outputs"):
    """Create and return the Gradio interface."""

    # Initialize the demo
    asr_demo = ASRDemo(model_path, outputs_dir)

    # Create the interface
    with gr.Blocks(title="ASR Demo - Whisper + SmolLM2") as demo:
        gr.Markdown("# 🎤 ASR Demo: Whisper Encoder + SmolLM2 Decoder")
        gr.Markdown("Upload an audio file, record from microphone, or select from outputs directory")

        with gr.Tabs():
            with gr.Tab("Microphone/Upload"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Audio Input",
                            type="filepath",
                            sources=["microphone", "upload"]
                        )
                        submit_btn = gr.Button("Transcribe", variant="primary")

                    with gr.Column():
                        output_text = gr.Textbox(
                            label="Transcription",
                            lines=5,
                            placeholder="Transcription will appear here..."
                        )

                # Add examples
                gr.Examples(
                    examples=[
                        # You can add example audio files here
                    ],
                    inputs=audio_input
                )

            with gr.Tab("From Outputs Directory"):
                with gr.Row():
                    with gr.Column():
                        # Dropdown for selecting files from outputs
                        output_files = asr_demo.get_output_files()
                        file_dropdown = gr.Dropdown(
                            choices=output_files,
                            label=f"Select audio from {outputs_dir}/",
                            value=output_files[0] if output_files else None
                        )
                        refresh_btn = gr.Button("🔄 Refresh List")
                        process_btn = gr.Button("Transcribe Selected", variant="primary")

                    with gr.Column():
                        output_text_2 = gr.Textbox(
                            label="Transcription",
                            lines=5,
                            placeholder="Transcription will appear here..."
                        )

                # Refresh button updates the dropdown
                def refresh_files():
                    files = asr_demo.get_output_files()
                    return gr.Dropdown(choices=files, value=files[0] if files else None)

                refresh_btn.click(
                    refresh_files,
                    outputs=[file_dropdown]
                )

                process_btn.click(
                    asr_demo.process_from_outputs,
                    inputs=[file_dropdown],
                    outputs=[output_text_2]
                )

        # Connect the main transcribe button
        submit_btn.click(
            asr_demo.process_audio,
            inputs=[audio_input],
            outputs=[output_text]
        )

        # Add model info
        with gr.Accordion("Model Information", open=False):
            gr.Markdown("""
            **Model Architecture:**
            - Encoder: Whisper-small (frozen)
            - Decoder: SmolLM2-135M with LoRA adapters
            - Audio Projector: RMSNorm + GELU activation

            **Configuration:**
            - LoRA rank: 32
            - Target modules: q_proj, v_proj, k_proj, o_proj
            - Max audio length: 30 seconds
            """)

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Gradio Demo")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory (required)"
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="wav_outputs",
        help="Directory containing audio files to process"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )

    args = parser.parse_args()

    # Create and launch demo
    demo = create_demo(args.model, args.outputs_dir)
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )