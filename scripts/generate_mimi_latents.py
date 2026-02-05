#!/usr/bin/env python3
"""Generate continuous Mimi latents for datasets.

This script encodes audio to continuous Mimi latents (before quantization)
for use in flow matching S2S training.

Usage:
    # Generate latents for a HuggingFace dataset
    python scripts/generate_mimi_latents.py \
        --input-dataset mazesmazes/libritts-r \
        --output-dataset mazesmazes/libritts-mimi-latents \
        --audio-column audio \
        --splits train.clean.100,dev.clean

    # Generate latents from a local directory
    python scripts/generate_mimi_latents.py \
        --input-dir /path/to/audio \
        --output-dir /path/to/latents
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from datasets import Audio, load_dataset
from tqdm.auto import tqdm
from transformers import MimiModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MimiLatentEncoder:
    """Encodes audio to continuous Mimi latents (before quantization)."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.mimi = MimiModel.from_pretrained("kyutai/mimi")
        self.mimi.to(device)
        self.mimi.eval()
        self.sample_rate = self.mimi.config.sampling_rate  # 24000 Hz

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Encode audio waveform to continuous latents.

        Args:
            audio: Waveform tensor, shape [samples] or [channels, samples]
            sr: Sample rate of input audio

        Returns:
            Latents tensor, shape [seq_len, latent_dim]
        """
        # Ensure correct shape: [batch, channels, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            if audio.shape[0] > audio.shape[1]:
                # Likely [samples, channels], transpose
                audio = audio.T
            audio = audio.unsqueeze(0)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Move to device with correct dtype (Mimi expects float32)
        audio = audio.to(self.device, dtype=torch.float32)

        # Pad to frame boundary (matching pocket-tts)
        frame_size = int(self.sample_rate / self.mimi.config.frame_rate)
        if audio.shape[-1] % frame_size != 0:
            pad_len = frame_size - (audio.shape[-1] % frame_size)
            audio = torch.nn.functional.pad(audio, (0, pad_len))

        # Encode through Mimi encoder + transformer + downsample
        # This gives us the continuous embeddings before quantization
        embeddings = self.mimi.encoder(audio)

        # Pass through encoder transformer
        embeddings = embeddings.transpose(1, 2)  # [batch, seq, dim]
        encoder_out = self.mimi.encoder_transformer(embeddings)
        if hasattr(encoder_out, "last_hidden_state"):
            embeddings = encoder_out.last_hidden_state
        else:
            embeddings = encoder_out[0]
        embeddings = embeddings.transpose(1, 2)  # [batch, dim, seq]

        # Downsample to final frame rate
        embeddings = self.mimi.downsample(embeddings)

        # Return as [seq_len, latent_dim]
        return embeddings.squeeze(0).transpose(0, 1).cpu()


def process_dataset(
    input_dataset: str,
    output_dataset: str,
    dataset_config: Optional[str] = None,
    audio_column: str = "audio",
    text_column: str = "text",
    splits: Optional[list[str]] = None,
    batch_size: int = 1,
    push_to_hub: bool = True,
    max_samples: Optional[int] = None,
):
    """Process a HuggingFace dataset to add Mimi latents."""
    encoder = MimiLatentEncoder()

    def encode_latents(batch):
        """Encode a batch of audio to latents."""
        latents_list = []
        for audio_item in batch[audio_column]:
            audio = torch.tensor(audio_item["array"])
            sr = audio_item["sampling_rate"]
            latents = encoder.encode(audio, sr)
            latents_list.append(latents.numpy())
        batch["latents"] = latents_list
        return batch

    # Process each split
    for split in splits or ["train"]:
        logger.info(f"Processing split: {split}")

        ds = load_dataset(
            input_dataset,
            name=dataset_config,
            split=split,
            trust_remote_code=True,
        )

        # Limit samples for testing
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
            logger.info(f"Limited to {len(ds)} samples")

        # Cast audio column to Audio type with target sample rate
        ds = ds.cast_column(audio_column, Audio(sampling_rate=encoder.sample_rate))

        # Encode latents
        ds = ds.map(
            encode_latents,
            batched=True,
            batch_size=batch_size,
            desc=f"Encoding {split}",
        )

        # Keep relevant columns
        keep_cols = ["latents"]
        if text_column in ds.column_names:
            keep_cols.append(text_column)
        if audio_column in ds.column_names:
            keep_cols.append(audio_column)

        ds = ds.select_columns(keep_cols)

        if push_to_hub:
            ds.push_to_hub(output_dataset, split=split)
            logger.info(f"Pushed {split} to {output_dataset}")
        else:
            output_path = Path(output_dataset) / split
            output_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(output_path))
            logger.info(f"Saved {split} to {output_path}")


def process_directory(
    input_dir: str,
    output_dir: str,
    extensions: list[str] = None,
):
    """Process audio files in a directory to generate latents."""
    encoder = MimiLatentEncoder()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = extensions or [".wav", ".mp3", ".flac", ".ogg"]
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_path.glob(f"**/*{ext}"))

    logger.info(f"Found {len(audio_files)} audio files")

    for audio_file in tqdm(audio_files, desc="Encoding"):
        try:
            audio, sr = torchaudio.load(str(audio_file))
            audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio.squeeze(0)

            latents = encoder.encode(audio, sr)

            # Save as .pt file with same relative path
            rel_path = audio_file.relative_to(input_path)
            out_file = output_path / rel_path.with_suffix(".pt")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(latents, out_file)

        except Exception as e:
            logger.warning(f"Failed to process {audio_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate Mimi latents for audio")
    parser.add_argument(
        "--input-dataset",
        type=str,
        help="Input HuggingFace dataset ID",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config/name (e.g., 'clean' for libritts_r_filtered)",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        help="Output HuggingFace dataset ID or local path",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory with audio files (alternative to dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for latent files",
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default="audio",
        help="Name of audio column in dataset",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in dataset",
    )
    parser.add_argument(
        "--splits",
        type=str,
        help="Comma-separated list of splits to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push to HuggingFace Hub, save locally",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)",
    )

    args = parser.parse_args()

    if args.input_dataset:
        splits = args.splits.split(",") if args.splits else None
        process_dataset(
            input_dataset=args.input_dataset,
            output_dataset=args.output_dataset,
            dataset_config=args.dataset_config,
            audio_column=args.audio_column,
            text_column=args.text_column,
            splits=splits,
            batch_size=args.batch_size,
            push_to_hub=not args.no_push,
            max_samples=args.max_samples,
        )
    elif args.input_dir:
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        raise ValueError("Must specify either --input-dataset or --input-dir")


if __name__ == "__main__":
    main()
