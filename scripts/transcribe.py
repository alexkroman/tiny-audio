#!/usr/bin/env python3
"""Transcribe an audio file using the tiny-audio model."""

import argparse

import torch

from src.asr_modeling import ASRModel
from src.asr_pipeline import ASRPipeline


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file")
    parser.add_argument("audio_file", type=str, help="Path to audio file to transcribe")
    parser.add_argument(
        "--model",
        type=str,
        default="mazesmazes/tiny-audio",
        help="Model ID or path (default: mazesmazes/tiny-audio)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu). Auto-detected if not specified.",
    )
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")

    # Load model directly to device
    model = ASRModel.from_pretrained(
        args.model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=None,
        device=device,
    )

    # Create pipeline
    pipe = ASRPipeline(
        model=model,
        tokenizer=model.tokenizer,
        feature_extractor=model.feature_extractor,
        device=device,
    )

    print(f"\nTranscribing: {args.audio_file}")

    try:
        result = pipe(args.audio_file)
        print("\n" + "=" * 80)
        print("TRANSCRIPTION:")
        print("=" * 80)
        print(result["text"])
        print("=" * 80)
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
