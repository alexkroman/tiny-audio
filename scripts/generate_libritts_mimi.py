#!/usr/bin/env python3
"""Generate LibriTTS dataset with Mimi codec codes.

Takes the parler-tts/libritts_r_filtered clean/train.clean.100 split and adds a 'codes' column
containing Mimi codec tokens encoded from the audio.

Usage:
    python -m scripts.generate_libritts_mimi --output-repo user/libritts-mimi

    # Test with limited samples
    python -m scripts.generate_libritts_mimi --max-samples 100
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

app = typer.Typer(help="Generate LibriTTS dataset with Mimi codec codes")

# Mimi codec settings
MIMI_SAMPLE_RATE = 24000
LIBRITTS_SAMPLE_RATE = 24000  # LibriTTS-R is already at 24kHz


def create_dataset_card(repo_id: str, num_samples: int) -> None:
    """Create and push a dataset card with proper metadata."""
    card_data = DatasetCardData(
        language=["en"],
        license="cc-by-4.0",
        task_categories=["text-to-speech", "audio-to-audio"],
        tags=["audio", "speech", "mimi", "codec", "tts", "libritts"],
        pretty_name="LibriTTS with Mimi Codes",
    )

    card_content = f"""---
{card_data.to_yaml()}
---

# LibriTTS with Mimi Codes

This dataset adds Mimi codec codes to [parler-tts/libritts_r_filtered](https://huggingface.co/datasets/parler-tts/libritts_r_filtered) clean/train.clean.100 split.

## Dataset Description

Each sample contains:
- **audio**: Original LibriTTS-R audio at 24kHz (Mimi's native rate)
- **codes**: 8-layer Mimi codec codes (list of 8 lists of integers)
- **text_normalized**: Normalized text transcription
- **text_original**: Original text transcription
- **speaker_id**: Speaker identifier
- **chapter_id**: Chapter identifier

## Stats

- **Samples**: {num_samples:,}
- **Audio Sample Rate**: 24kHz
- **Codec**: Mimi (kyutai/mimi) with 8 codebooks

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")

# Access audio and codes together
sample = ds[0]
audio = sample["audio"]  # {{'array': [...], 'sampling_rate': 24000}}
codes = sample["codes"]  # 8 lists of codec indices
text = sample["text_normalized"]

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

- [parler-tts/libritts_r_filtered](https://huggingface.co/datasets/parler-tts/libritts_r_filtered) - LibriTTS-R filtered recordings

## License

CC-BY-4.0 (same as source dataset)
"""

    card = DatasetCard(card_content)
    card.push_to_hub(repo_id)


class MimiEncoder:
    """Encodes audio to Mimi codec tokens."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
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
        from transformers import MimiModel

        # Load in half precision for faster inference
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = MimiModel.from_pretrained("kyutai/mimi", torch_dtype=dtype).to(self.device)
        self.model.eval()
        self._use_mlx = False
        self._dtype = dtype
        typer.echo(f"Mimi loaded on {self.device} ({dtype})")

    def encode_single(self, audio_np) -> list[list[int]]:
        """Encode single audio array to Mimi codes.

        Args:
            audio_np: Numpy array at 24kHz

        Returns:
            List of 8 lists of codec indices (one per codebook)
        """
        self._load_model()

        if self._use_mlx:
            import mlx.core as mx

            audio_mlx = mx.array(audio_np[None, None, :])
            codes_mlx = self.model.encode(audio_mlx)
            codes = torch.from_numpy(codes_mlx.__array__())
            return codes.squeeze(0).tolist()

        # Pass raw tensor directly - skip feature_extractor overhead
        # Shape: (batch=1, channels=1, samples)
        audio_tensor = torch.as_tensor(audio_np, dtype=self._dtype, device=self.device)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            encoded = self.model.encode(audio_tensor, num_quantizers=8)
            codes = encoded.audio_codes  # (1, 8, seq_len)

        return codes.squeeze(0).cpu().tolist()


def process_dataset(
    max_samples: int | None = None,
) -> Dataset | None:
    """Load LibriTTS dataset and add Mimi codes.

    Args:
        max_samples: Optional limit on number of samples

    Returns:
        Dataset with codes column added
    """
    from tqdm import tqdm

    typer.echo("Loading parler-tts/libritts_r_filtered clean/train.clean.100...")
    ds = load_dataset("parler-tts/libritts_r_filtered", "clean", split="train.clean.100")
    typer.echo(f"Loaded {len(ds)} samples")

    # Limit samples if requested
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        typer.echo(f"Limited to {len(ds)} samples")

    # Ensure audio is at 24kHz (should already be, but cast to be sure)
    typer.echo(f"Ensuring audio is at {MIMI_SAMPLE_RATE}Hz...")
    ds = ds.cast_column("audio", Audio(sampling_rate=MIMI_SAMPLE_RATE))

    # Initialize encoder on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"Using device: {device}")
    encoder = MimiEncoder(device=device)
    encoder._load_model()

    # Encode all samples
    typer.echo("Encoding audio to Mimi codes...")
    all_codes = []
    for sample in tqdm(ds, desc="Encoding"):
        audio_array = sample["audio"]["array"]
        try:
            codes = encoder.encode_single(audio_array)
            all_codes.append(codes)
        except Exception as e:
            typer.echo(f"Warning: Encoding failed: {e}")
            all_codes.append([[] for _ in range(8)])

    # Add codes column
    ds = ds.add_column("codes", all_codes)

    # Keep only essential columns
    keep_cols = ["audio", "codes", "text_normalized", "text_original", "speaker_id", "chapter_id"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)

    typer.echo(f"Final dataset: {len(ds)} samples with codes")
    return ds


@app.command()
def main(
    output_repo: Annotated[
        str, typer.Option(help="HuggingFace repo ID for output")
    ] = "mazesmazes/libritts-mimi",
    max_samples: Annotated[int | None, typer.Option(help="Max samples (for testing)")] = None,
    push: Annotated[bool, typer.Option(help="Push to HuggingFace Hub")] = True,
):
    """Generate LibriTTS dataset with Mimi codec codes.

    Loads parler-tts/libritts_r_filtered clean/train.clean.100, ensures audio is at 24kHz,
    encodes with Mimi codec, and pushes to HuggingFace Hub.
    """
    typer.echo(f"Output: {output_repo}")
    if max_samples:
        typer.echo(f"Max samples: {max_samples}")

    ds = process_dataset(max_samples=max_samples)

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
