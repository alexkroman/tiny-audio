"""
Demo: What does Mimi codec sound like with only 1 codebook?

Mimi uses 16 codebooks total. The first codebook captures semantic information
(trained via distillation from WavLM), while subsequent codebooks add acoustic detail.

This script encodes audio, keeps only the first N codebooks, and decodes to hear
the quality degradation.
"""

from pathlib import Path

import torch
import torchaudio


def demo_mimi_codebooks(
    audio_path: str | None = None,
    num_codebooks: int = 1,
    output_dir: str = "mimi_demo_outputs",
):
    """
    Encode audio with Mimi and decode using only the first N codebooks.

    Args:
        audio_path: Path to input audio file. If None, uses a sample from LibriSpeech.
        num_codebooks: Number of codebooks to use (1-16)
        output_dir: Directory to save output files
    """
    from transformers import AutoFeatureExtractor, MimiModel

    print("Loading Mimi model...")
    model = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load audio
    if audio_path is None:
        print("Loading sample from LibriSpeech...")
        from datasets import Audio, load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        audio_sample = ds[0]["audio"]["array"]
        sample_rate = feature_extractor.sampling_rate
    else:
        print(f"Loading audio from {audio_path}...")
        waveform, sample_rate = torchaudio.load(audio_path)
        # Resample if needed
        if sample_rate != feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, feature_extractor.sampling_rate)
            waveform = resampler(waveform)
            sample_rate = feature_extractor.sampling_rate
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_sample = waveform.squeeze().numpy()

    # Pre-process
    inputs = feature_extractor(
        raw_audio=audio_sample, sampling_rate=sample_rate, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Encode to get codebook indices
    print("Encoding audio...")
    with torch.no_grad():
        encoder_outputs = model.encode(inputs["input_values"])
        audio_codes = encoder_outputs.audio_codes  # Shape: [batch, num_codebooks, seq_len]

    print(f"Audio codes shape: {audio_codes.shape}")
    print(f"Total codebooks: {audio_codes.shape[1]}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save original for comparison
    print("Decoding with all codebooks (original quality)...")
    with torch.no_grad():
        full_audio = model.decode(audio_codes)[0]

    # Squeeze batch dimension: [1, samples] -> [1, samples] (channel, samples)
    full_audio = full_audio.squeeze(0)  # Remove batch dim, keep channel dim
    if full_audio.dim() == 1:
        full_audio = full_audio.unsqueeze(0)  # Add channel dim if missing

    torchaudio.save(output_path / "mimi_all_codebooks.wav", full_audio.cpu(), sample_rate)
    print(f"Saved: {output_path / 'mimi_all_codebooks.wav'}")

    # Decode with limited codebooks
    for n in range(1, min(num_codebooks + 1, audio_codes.shape[1] + 1)):
        print(f"Decoding with {n} codebook(s)...")

        # Zero out codebooks beyond n
        limited_codes = audio_codes.clone()
        if n < audio_codes.shape[1]:
            # Set unused codebooks to 0 (the first entry in each codebook)
            limited_codes[:, n:, :] = 0

        with torch.no_grad():
            limited_audio = model.decode(limited_codes)[0]

        # Fix tensor shape for torchaudio
        limited_audio = limited_audio.squeeze(0)
        if limited_audio.dim() == 1:
            limited_audio = limited_audio.unsqueeze(0)

        output_file = output_path / f"mimi_{n}_codebook{'s' if n > 1 else ''}.wav"
        torchaudio.save(output_file, limited_audio.cpu(), sample_rate)
        print(f"Saved: {output_file}")

    print(f"\nDone! Check the '{output_dir}' directory for audio files.")
    print("\nExpected quality progression:")
    print("  1 codebook  = Semantic only (intelligible but robotic/distorted)")
    print("  2-4 codebooks = Adding acoustic detail")
    print("  8+ codebooks = Near-transparent quality")
    print("  16 codebooks = Full quality (~1.1 kbps)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo Mimi codec with limited codebooks")
    parser.add_argument("--audio", "-a", type=str, help="Path to input audio file")
    parser.add_argument(
        "--codebooks",
        "-c",
        type=int,
        default=4,
        help="Max number of codebooks to demo (default: 4)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="mimi_demo_outputs", help="Output directory"
    )

    args = parser.parse_args()
    demo_mimi_codebooks(args.audio, args.codebooks, args.output)
