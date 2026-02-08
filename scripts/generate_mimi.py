#!/usr/bin/env python3
"""Generate dataset with Mimi codec codes (for trainable AR decoder training).

Encodes audio from any HuggingFace dataset to Mimi codec tokens (8 codebooks).
Each split is pushed separately, so running again with a new split will add/update
only that split without overwriting existing splits.

Usage:
    # LibriTTS (default)
    python -m scripts.generate_mimi --output-repo user/libritts-mimi

    # Process specific splits
    python -m scripts.generate_mimi \
        --input-dataset parler-tts/libritts_r_filtered \
        --dataset-config clean \
        --splits train.clean.100,train.clean.360 \
        --output-repo user/libritts-mimi

    # Test with limited samples
    python -m scripts.generate_mimi --max-samples 100
"""

import os
from typing import Annotated

import torch
import typer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import Audio, Dataset, Features, Value, load_dataset
from huggingface_hub import DatasetCard, DatasetCardData
from tqdm import tqdm

app = typer.Typer(help="Generate dataset with Mimi codec codes")

MIMI_SAMPLE_RATE = 24000
NUM_CODEBOOKS = 8


def create_dataset_card(
    repo_id: str,
    input_dataset: str,
    splits: list[str],
    num_samples: int,
    text_column: str,
) -> None:
    """Create and push a dataset card."""
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
- **codes**: 8-layer Mimi codec codes (list of 8 lists of integers, vocab 0-2047)
- **text**: Text transcription (from `{text_column}` column)

## Stats

- **Source**: {input_dataset}
- **Splits**: {splits_str}
- **Samples**: {num_samples:,}
- **Audio Sample Rate**: 24kHz
- **Codec**: Mimi (kyutai/mimi) with 8 codebooks, vocab size 2048, 12.5 tokens/sec

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
sample = ds[0]
codes = sample["codes"]  # 8 lists of codec indices
text = sample["text"]
```

## License

Same as source dataset.
"""
    card = DatasetCard(card_content)
    card.push_to_hub(repo_id)


class MimiEncoder:
    """Encodes audio to Mimi codec tokens using transformers MimiModel."""

    MIMI_MODEL_ID = "kyutai/mimi"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    def _load_model(self):
        """Lazy load Mimi model from transformers."""
        if self.model is not None:
            return

        from transformers import MimiModel

        typer.echo("Loading Mimi model...")
        self.model = MimiModel.from_pretrained(self.MIMI_MODEL_ID, torch_dtype=torch.float32).to(
            self.device
        )
        self.model.eval()
        typer.echo(f"Mimi loaded on {self.device}")

    def encode(self, audio: torch.Tensor) -> list[list[int]]:
        """Encode a single audio tensor to Mimi codes.

        Args:
            audio: 1-D audio tensor at MIMI_SAMPLE_RATE (24kHz)

        Returns:
            List of 8 lists of codec indices (first 8 codebooks)
        """
        self._load_model()

        input_values = audio.unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            encoder_outputs = self.model.encode(input_values)
            codes = encoder_outputs.audio_codes  # [1, num_codebooks, seq_len]

        # Take first 8 codebooks only
        return codes.squeeze(0)[:NUM_CODEBOOKS].cpu().tolist()  # [8, seq_len]


def process_split(
    ds: Dataset,
    encoder: MimiEncoder,
    audio_column: str,
    text_column: str,
    max_samples: int | None = None,
) -> Dataset:
    """Process a single dataset split and add Mimi codes."""
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        typer.echo(f"Limited to {len(ds)} samples")

    # Ensure audio is at Mimi sample rate
    typer.echo(f"Ensuring audio is at {MIMI_SAMPLE_RATE}Hz...")
    ds = ds.cast_column(audio_column, Audio(sampling_rate=MIMI_SAMPLE_RATE))

    typer.echo("Encoding audio to Mimi codes...")
    all_codes = []

    for i in tqdm(range(len(ds)), desc="Processing"):
        sample = ds[i]
        audio_array = sample[audio_column]["array"]
        audio_tensor = torch.as_tensor(audio_array, dtype=torch.float32)

        try:
            codes = encoder.encode(audio_tensor)
            all_codes.append(codes)
        except Exception as e:
            typer.echo(f"Warning: Failed to encode sample {i}: {e}")
            all_codes.append([[] for _ in range(NUM_CODEBOOKS)])

    ds = ds.add_column("codes", all_codes)

    if text_column != "text" and text_column in ds.column_names:
        ds = ds.rename_column(text_column, "text")
    if audio_column != "audio" and audio_column in ds.column_names:
        ds = ds.rename_column(audio_column, "audio")

    # Keep only essential columns
    keep_cols = {"audio", "text", "text_original", "speaker_id", "chapter_id", "codes"}
    extra_cols = [c for c in ds.column_names if c not in keep_cols]
    if extra_cols:
        typer.echo(f"Removing extra columns: {extra_cols}")
        ds = ds.remove_columns(extra_cols)

    # Cast to explicit features
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
    push: Annotated[bool, typer.Option(help="Push to HuggingFace Hub")] = True,
):
    """Generate dataset with Mimi codec codes for AR decoder training."""
    split_list = [s.strip() for s in splits.split(",")]

    typer.echo(f"Input: {input_dataset}" + (f" ({dataset_config})" if dataset_config else ""))
    typer.echo(f"Splits: {', '.join(split_list)}")
    typer.echo(f"Output: {output_repo}")
    if max_samples:
        typer.echo(f"Max samples per split: {max_samples}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"Using device: {device}")
    encoder = MimiEncoder(device=device)

    processed_datasets = []
    total_samples = 0

    for split in split_list:
        typer.echo(f"\n{'=' * 50}")
        typer.echo(f"Processing split: {split}")
        typer.echo(f"{'=' * 50}")

        ds = load_dataset(input_dataset, dataset_config, split=split)
        typer.echo(f"Loaded {len(ds)} samples")

        processed = process_split(
            ds,
            encoder,
            audio_column,
            text_column,
            max_samples=max_samples,
        )
        processed_datasets.append(processed)
        total_samples += len(processed)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if push:
        for split_name, processed in zip(split_list, processed_datasets):
            typer.echo(f"\nPushing split '{split_name}' to {output_repo}...")
            processed.push_to_hub(output_repo, split=split_name, private=False)

        typer.echo("Updating dataset card...")
        create_dataset_card(output_repo, input_dataset, split_list, total_samples, text_column)

    typer.echo("\nDone!")


if __name__ == "__main__":
    app()
