#!/usr/bin/env python3
"""Generate dataset with Mimi codec codes.

Encodes audio from any HuggingFace dataset to Mimi codec tokens (8 codebooks).
Each split is pushed separately, so running again with a new split will add/update
only that split without overwriting existing splits.

Usage:
    # LibriTTS (default)
    python -m scripts.generate_mimi --output-repo user/libritts-mimi

    # Process specific splits (each pushed as separate HF split)
    python -m scripts.generate_mimi \
        --input-dataset parler-tts/libritts_r_filtered \
        --dataset-config clean \
        --splits train.clean.100,train.clean.360 \
        --output-repo user/libritts-mimi

    # Add another split later (won't overwrite existing splits)
    python -m scripts.generate_mimi \
        --splits train.clean.500 \
        --output-repo user/libritts-mimi

    # Process a different dataset
    python -m scripts.generate_mimi \
        --input-dataset librispeech_asr \
        --dataset-config clean \
        --splits train.100 \
        --audio-column audio \
        --text-column text \
        --output-repo user/librispeech-mimi

    # Test with limited samples
    python -m scripts.generate_mimi --max-samples 100
"""

import gc
import os
import platform
from typing import Annotated

import torch
import typer

# Enable fast HuggingFace transfers
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import Audio, Dataset, Features, Value, load_dataset
from huggingface_hub import DatasetCard, DatasetCardData
from tqdm import tqdm

app = typer.Typer(help="Generate dataset with Mimi codec codes")

# Mimi codec settings
MIMI_SAMPLE_RATE = 24000


def create_dataset_card(
    repo_id: str,
    input_dataset: str,
    splits: list[str],
    num_samples: int,
    text_column: str,
) -> None:
    """Create and push a dataset card with proper metadata."""
    card_data = DatasetCardData(
        language=["en"],
        license="cc-by-4.0",
        task_categories=["text-to-speech", "audio-to-audio"],
        tags=["audio", "speech", "mimi", "codec", "s2s"],
        pretty_name="Dataset with Mimi Codes",
    )

    splits_str = ", ".join(splits)
    card_content = f"""---
{card_data.to_yaml()}
---

# Dataset with Mimi Codes

This dataset adds Mimi codec codes to [{input_dataset}](https://huggingface.co/datasets/{input_dataset}).

## Dataset Description

Each sample contains:
- **audio**: Audio resampled to 24kHz (Mimi's native rate)
- **codes**: 8-layer Mimi codec codes (list of 8 lists of integers)
- **text**: Text transcription (from `{text_column}` column)
- Additional columns preserved from source dataset

## Stats

- **Source**: {input_dataset}
- **Splits**: {splits_str}
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
text = sample["text"]

# Decode codes back to audio
import torch
from transformers import MimiModel

mimi = MimiModel.from_pretrained("kyutai/mimi")
codes_tensor = torch.tensor(codes).unsqueeze(0)  # (1, 8, seq_len)
with torch.no_grad():
    decoded = mimi.decode(codes_tensor)
    waveform = decoded.audio_values  # (1, 1, samples) at 24kHz
```

## Source Dataset

- [{input_dataset}](https://huggingface.co/datasets/{input_dataset})

## License

Same as source dataset.
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


def process_split(
    ds: Dataset,
    encoder: MimiEncoder,
    audio_column: str,
    text_column: str,
    max_samples: int | None = None,
    batch_size: int = 1,
) -> Dataset:
    """Process a single dataset split and add Mimi codes.

    Args:
        ds: Input dataset split
        encoder: MimiEncoder instance
        audio_column: Name of audio column
        text_column: Name of text column
        max_samples: Optional limit on number of samples
        batch_size: Batch size for progress reporting

    Returns:
        Dataset with codes column added
    """
    # Limit samples if requested
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        typer.echo(f"Limited to {len(ds)} samples")

    # Ensure audio is at 24kHz
    typer.echo(f"Ensuring audio is at {MIMI_SAMPLE_RATE}Hz...")
    ds = ds.cast_column(audio_column, Audio(sampling_rate=MIMI_SAMPLE_RATE))

    # Encode all audio samples
    typer.echo("Encoding audio to Mimi codes...")
    all_codes = []

    for i in tqdm(range(0, len(ds), batch_size), desc="Processing"):
        batch_end = min(i + batch_size, len(ds))
        batch = ds.select(range(i, batch_end))

        for sample in batch:
            audio_array = sample[audio_column]["array"]
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

    # Rename text column to standardized 'text' if different
    if text_column != "text" and text_column in ds.column_names:
        ds = ds.rename_column(text_column, "text")

    # Rename audio column to standardized 'audio' if different
    if audio_column != "audio" and audio_column in ds.column_names:
        ds = ds.rename_column(audio_column, "audio")

    # Keep only essential columns to avoid schema mismatch issues
    keep_cols = {"audio", "text", "text_original", "speaker_id", "chapter_id", "codes"}
    extra_cols = [c for c in ds.column_names if c not in keep_cols]
    if extra_cols:
        typer.echo(f"Removing extra columns: {extra_cols}")
        ds = ds.remove_columns(extra_cols)

    # Cast to explicit features to ensure correct schema in HuggingFace
    features = Features(
        {
            "audio": Audio(sampling_rate=MIMI_SAMPLE_RATE),
            "text": Value("string"),
            "text_original": Value("string"),
            "speaker_id": Value("string"),
            "chapter_id": Value("string"),
            "codes": [[Value("int64")]],
        }
    )
    ds = ds.cast(features)

    typer.echo(f"Processed {len(ds)} samples")
    return ds


@app.command()
def main(
    input_dataset: Annotated[
        str, typer.Option("--input-dataset", "-i", help="Input HuggingFace dataset")
    ] = "parler-tts/libritts_r_filtered",
    dataset_config: Annotated[
        str | None, typer.Option("--dataset-config", "-c", help="Dataset config name")
    ] = "clean",
    splits: Annotated[
        str, typer.Option("--splits", "-s", help="Comma-separated list of splits")
    ] = "train.clean.360",
    audio_column: Annotated[
        str, typer.Option("--audio-column", help="Audio column name")
    ] = "audio",
    text_column: Annotated[
        str, typer.Option("--text-column", help="Text column name")
    ] = "text_normalized",
    output_repo: Annotated[
        str, typer.Option("--output-repo", "-o", help="HuggingFace repo ID for output")
    ] = "mazesmazes/libritts-mimi",
    max_samples: Annotated[
        int | None, typer.Option("--max-samples", "-n", help="Max samples per split")
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size for processing")
    ] = 1,
    push: Annotated[bool, typer.Option(help="Push to HuggingFace Hub")] = True,
):
    """Generate dataset with Mimi codec codes.

    Loads audio from any HuggingFace dataset, encodes with Mimi codec (8 codebooks),
    and optionally pushes to HuggingFace Hub.
    """
    split_list = [s.strip() for s in splits.split(",")]

    typer.echo(f"Input: {input_dataset}" + (f" ({dataset_config})" if dataset_config else ""))
    typer.echo(f"Splits: {', '.join(split_list)}")
    typer.echo(f"Output: {output_repo}")
    if max_samples:
        typer.echo(f"Max samples per split: {max_samples}")

    # Initialize encoder on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"Using device: {device}")
    encoder = MimiEncoder(device=device)

    # Process each split
    processed_datasets = []
    total_samples = 0

    for split in split_list:
        typer.echo(f"\n{'=' * 50}")
        typer.echo(f"Processing split: {split}")
        typer.echo(f"{'=' * 50}")

        # Load dataset split
        typer.echo(f"Loading {input_dataset} {dataset_config or ''} {split}...")
        ds = load_dataset(input_dataset, dataset_config, split=split)
        typer.echo(f"Loaded {len(ds)} samples")

        # Process split
        processed = process_split(
            ds,
            encoder,
            audio_column,
            text_column,
            max_samples,
            batch_size,
        )
        processed_datasets.append(processed)
        total_samples += len(processed)

        # Clear memory between splits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Push each split separately (allows incremental updates)
    if push:
        for split_name, processed in zip(split_list, processed_datasets):
            typer.echo(f"\nPushing split '{split_name}' to {output_repo}...")
            processed.push_to_hub(output_repo, split=split_name, private=False)

        typer.echo("Updating dataset card...")
        create_dataset_card(
            output_repo,
            input_dataset,
            split_list,
            total_samples,
            text_column,
        )

    typer.echo("\nDone!")


if __name__ == "__main__":
    app()
