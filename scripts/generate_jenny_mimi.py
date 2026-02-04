#!/usr/bin/env python3
"""Generate Jenny TTS dataset with Mimi codec codes.

Takes the reach-vb/jenny_tts_dataset and adds a 'codes' column containing
Mimi codec tokens encoded from the audio.

Usage:
    python -m scripts.generate_jenny_mimi --output-repo user/jenny-mimi

    # Test with limited samples
    python -m scripts.generate_jenny_mimi --max-samples 100

    # Use specific batch size for encoding
    python -m scripts.generate_jenny_mimi --batch-size 32
"""

import os
import platform
from typing import Annotated

import torch
import typer

# Enable fast HuggingFace transfers
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import Audio, Dataset, load_dataset
from huggingface_hub import DatasetCard, DatasetCardData
from tqdm import tqdm

app = typer.Typer(help="Generate Jenny TTS dataset with Mimi codec codes")

# Mimi codec settings
MIMI_SAMPLE_RATE = 24000
JENNY_SAMPLE_RATE = 48000


def create_dataset_card(repo_id: str, num_samples: int) -> None:
    """Create and push a dataset card with proper metadata."""
    card_data = DatasetCardData(
        language=["en"],
        license="mit",
        task_categories=["text-to-speech", "audio-to-audio"],
        tags=["audio", "speech", "mimi", "codec", "tts", "jenny"],
        pretty_name="Jenny TTS with Mimi Codes",
    )

    card_content = f"""---
{card_data.to_yaml()}
---

# Jenny TTS with Mimi Codes

This dataset adds Mimi codec codes to [reach-vb/jenny_tts_dataset](https://huggingface.co/datasets/reach-vb/jenny_tts_dataset).

## Dataset Description

Each sample contains:
- **audio**: Original Jenny TTS audio resampled to 24kHz (Mimi's native rate)
- **codes**: 8-layer Mimi codec codes (list of 8 lists of integers)
- **transcription**: Original text transcription
- **transcription_normalised**: Normalized transcription

## Stats

- **Samples**: {num_samples:,}
- **Audio Sample Rate**: 24kHz (resampled from original 48kHz)
- **Codec**: Mimi (kyutai/mimi) with 8 codebooks

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")

# Access audio and codes together
sample = ds[0]
audio = sample["audio"]  # {{'array': [...], 'sampling_rate': 24000}}
codes = sample["codes"]  # 8 lists of codec indices
text = sample["transcription_normalised"]

# Decode codes back to audio (requires moshi_mlx or transformers)
import torch
from transformers import MimiModel

mimi = MimiModel.from_pretrained("kyutai/mimi")
codes_tensor = torch.tensor(codes).unsqueeze(0)  # (1, 8, seq_len)
with torch.no_grad():
    decoded = mimi.decode(codes_tensor)
    waveform = decoded.audio_values  # (1, 1, samples) at 24kHz
```

## Source Dataset

- [reach-vb/jenny_tts_dataset](https://huggingface.co/datasets/reach-vb/jenny_tts_dataset) - Original Jenny TTS recordings

## License

MIT (same as source dataset)
"""

    card = DatasetCard(card_content)
    card.push_to_hub(repo_id)


class MimiEncoder:
    """Encodes audio to Mimi codec tokens."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.feature_extractor = None
        self._use_mlx = False

    def _load_model(self):
        """Lazy load Mimi model."""
        if self.model is not None:
            return

        # Try MLX on Mac first (faster)
        if platform.system() == "Darwin":
            try:
                from moshi_mlx import models

                typer.echo("Loading Mimi (MLX)...")
                self.model = models.mimi_202412()
                self.model.load_weights()
                self._use_mlx = True
                typer.echo("Mimi loaded (MLX backend)")
                return
            except ImportError:
                pass

        # Fall back to transformers
        typer.echo("Loading Mimi (transformers)...")
        from transformers import AutoFeatureExtractor, MimiModel

        self.model = MimiModel.from_pretrained("kyutai/mimi").to(self.device)
        self.model.eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        self._use_mlx = False
        typer.echo(f"Mimi loaded on {self.device}")

        # Verify model is on correct device
        param_device = next(self.model.parameters()).device
        typer.echo(f"Model parameters on: {param_device}")

    def encode(self, audio: torch.Tensor) -> list[list[int]]:
        """Encode audio tensor to Mimi codes.

        Args:
            audio: Audio tensor (samples,) at 24kHz

        Returns:
            List of 8 lists of codec indices (one per codebook)
        """
        self._load_model()

        # Convert to numpy for feature extractor
        if audio.dim() > 1:
            audio = audio.squeeze()
        audio_np = audio.numpy()

        if self._use_mlx:
            import mlx.core as mx

            # MLX expects (batch, channels, samples)
            audio_mlx = mx.array(audio_np[None, None, :])
            codes_mlx = self.model.encode(audio_mlx)
            codes = torch.from_numpy(codes_mlx.__array__())
        else:
            # Use feature extractor for proper preprocessing
            inputs = self.feature_extractor(
                raw_audio=audio_np,
                sampling_rate=MIMI_SAMPLE_RATE,
                return_tensors="pt",
            )
            # Move ALL inputs to the device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Encode with 8 codebooks (matching Mimi's default)
                encoded = self.model.encode(**inputs, num_quantizers=8)
                codes = encoded.audio_codes  # (batch, codebooks, seq_len)

        # Convert to list of lists: 8 codebooks, each with seq_len indices
        # Shape: (1, 8, seq_len) -> list of 8 lists
        return codes.squeeze(0).cpu().tolist()  # (8, seq_len)

    def encode_batch(
        self, audios: list[torch.Tensor], show_progress: bool = True
    ) -> list[list[list[int]]]:
        """Encode multiple audio tensors.

        Args:
            audios: List of audio tensors at 24kHz
            show_progress: Whether to show progress bar

        Returns:
            List of codes for each audio
        """
        results = []
        iterator = tqdm(audios, desc="Encoding") if show_progress else audios

        for audio in iterator:
            try:
                codes = self.encode(audio)
                results.append(codes)
            except Exception as e:
                typer.echo(f"Warning: Encoding failed: {e}")
                # Return empty codes on failure
                results.append([[] for _ in range(8)])

        return results


def process_dataset(
    max_samples: int | None = None,
    batch_size: int = 1,
) -> Dataset | None:
    """Load Jenny dataset and add Mimi codes.

    Args:
        max_samples: Optional limit on number of samples
        batch_size: Batch size for progress reporting

    Returns:
        Dataset with codes column added
    """
    import gc

    typer.echo("Loading reach-vb/jenny_tts_dataset...")
    ds = load_dataset("reach-vb/jenny_tts_dataset", split="train")
    typer.echo(f"Loaded {len(ds)} samples")

    # Limit samples if requested
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        typer.echo(f"Limited to {len(ds)} samples")

    # Resample audio to 24kHz (Mimi's native rate)
    typer.echo(f"Resampling audio from {JENNY_SAMPLE_RATE}Hz to {MIMI_SAMPLE_RATE}Hz...")
    ds = ds.cast_column("audio", Audio(sampling_rate=MIMI_SAMPLE_RATE))

    # Initialize encoder on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"Using device: {device}")
    encoder = MimiEncoder(device=device)

    # Encode all audio samples
    typer.echo("Encoding audio to Mimi codes...")
    all_codes = []

    for i in tqdm(range(0, len(ds), batch_size), desc="Processing"):
        batch_end = min(i + batch_size, len(ds))
        batch = ds.select(range(i, batch_end))

        for sample in batch:
            audio_array = sample["audio"]["array"]
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

            try:
                codes = encoder.encode(audio_tensor)
                all_codes.append(codes)
            except Exception as e:
                typer.echo(f"Warning: Failed to encode sample {i}: {e}")
                all_codes.append([[] for _ in range(8)])

        # Periodic garbage collection
        if (i // batch_size) % 100 == 0:
            gc.collect()

    # Add codes column
    typer.echo("Adding codes column...")
    ds = ds.add_column("codes", all_codes)

    # Remove file_name column if present (not needed)
    if "file_name" in ds.column_names:
        ds = ds.remove_columns(["file_name"])

    typer.echo(f"Final dataset: {len(ds)} samples with codes")
    return ds


@app.command()
def main(
    output_repo: Annotated[
        str, typer.Option(help="HuggingFace repo ID for output")
    ] = "mazesmazes/jenny-mimi",
    max_samples: Annotated[int | None, typer.Option(help="Max samples (for testing)")] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size for processing")
    ] = 1,
    push: Annotated[bool, typer.Option(help="Push to HuggingFace Hub")] = True,
):
    """Generate Jenny TTS dataset with Mimi codec codes.

    Loads reach-vb/jenny_tts_dataset, resamples audio to 24kHz,
    encodes with Mimi codec, and pushes to HuggingFace Hub.
    """
    typer.echo(f"Output: {output_repo}")
    if max_samples:
        typer.echo(f"Max samples: {max_samples}")

    ds = process_dataset(
        max_samples=max_samples,
        batch_size=batch_size,
    )

    if ds is None:
        typer.echo("Failed to process dataset")
        raise typer.Exit(1)

    if push:
        typer.echo(f"\nPushing to {output_repo}...")
        ds.push_to_hub(output_repo, private=False)

        typer.echo("Updating dataset card...")
        create_dataset_card(output_repo, len(ds))

    typer.echo("Done!")


if __name__ == "__main__":
    app()
