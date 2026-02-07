#!/usr/bin/env python3
"""Generate dataset with DAC codec codes (for Dia TTS decoder training).

Encodes audio from any HuggingFace dataset to DAC codec tokens (9 codebooks).
Each split is pushed separately, so running again with a new split will add/update
only that split without overwriting existing splits.

Usage:
    # LibriTTS (default)
    python -m scripts.generate_dac --output-repo user/libritts-dac

    # Process specific splits
    python -m scripts.generate_dac \
        --input-dataset parler-tts/libritts_r_filtered \
        --dataset-config clean \
        --splits train.clean.100,train.clean.360 \
        --output-repo user/libritts-dac

    # Test with limited samples
    python -m scripts.generate_dac --max-samples 100
"""

import gc
import os
from typing import Annotated

import torch
import typer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import Audio, Dataset, Features, Value, load_dataset
from huggingface_hub import DatasetCard, DatasetCardData
from tqdm import tqdm

app = typer.Typer(help="Generate dataset with DAC codec codes")

DAC_SAMPLE_RATE = 44100
NUM_CODEBOOKS = 9


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
        tags=["audio", "speech", "dac", "codec", "s2s", "dia"],
        pretty_name="Dataset with DAC Codes",
    )

    splits_str = ", ".join(splits)
    card_content = f"""---
{card_data.to_yaml()}
---

# Dataset with DAC Codes

This dataset adds DAC codec codes to [{input_dataset}](https://huggingface.co/datasets/{input_dataset}).

## Dataset Description

Each sample contains:
- **audio**: Audio resampled to 44.1kHz (DAC's native rate)
- **codes**: 9-layer DAC codec codes (list of 9 lists of integers, vocab 0-1027)
- **text**: Text transcription (from `{text_column}` column)

## Stats

- **Source**: {input_dataset}
- **Splits**: {splits_str}
- **Samples**: {num_samples:,}
- **Audio Sample Rate**: 44.1kHz
- **Codec**: DAC (descript-audio-codec) with 9 codebooks, vocab size 1028

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
sample = ds[0]
codes = sample["codes"]  # 9 lists of codec indices
text = sample["text"]
```

## License

Same as source dataset.
"""
    card = DatasetCard(card_content)
    card.push_to_hub(repo_id)


class DACEncoder:
    """Encodes audio to DAC codec tokens."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    def _load_model(self):
        """Lazy load DAC model."""
        if self.model is not None:
            return

        import dac

        typer.echo("Loading DAC model...")
        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path).to(self.device)
        self.model.eval()
        typer.echo(f"DAC loaded on {self.device}")

    def encode(self, audio: torch.Tensor) -> list[list[int]]:
        """Encode a single audio tensor to DAC codes.

        Args:
            audio: 1-D audio tensor at DAC_SAMPLE_RATE

        Returns:
            List of 9 lists of codec indices (one per codebook)
        """
        self._load_model()

        audio = audio.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, samples]

        with torch.no_grad():
            x = self.model.preprocess(audio, DAC_SAMPLE_RATE)
            _, codes, _, _, _ = self.model.encode(x)

        return codes.squeeze(0).cpu().tolist()  # [9, seq_len]


def process_split(
    ds: Dataset,
    encoder: DACEncoder,
    audio_column: str,
    text_column: str,
    max_samples: int | None = None,
) -> Dataset:
    """Process a single dataset split and add DAC codes."""
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        typer.echo(f"Limited to {len(ds)} samples")

    # Ensure audio is at DAC sample rate
    typer.echo(f"Ensuring audio is at {DAC_SAMPLE_RATE}Hz...")
    ds = ds.cast_column(audio_column, Audio(sampling_rate=DAC_SAMPLE_RATE))

    typer.echo("Encoding audio to DAC codes...")
    all_codes = []

    for i in tqdm(range(len(ds)), desc="Processing"):
        sample = ds[i]
        audio_array = sample[audio_column]["array"]
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

        try:
            codes = encoder.encode(audio_tensor)
            all_codes.append(codes)
        except Exception as e:
            typer.echo(f"Warning: Failed to encode sample {i}: {e}")
            all_codes.append([[] for _ in range(NUM_CODEBOOKS)])

        if i % 100 == 0:
            gc.collect()

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
            "audio": Audio(sampling_rate=DAC_SAMPLE_RATE),
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
    ] = "mazesmazes/libritts-dac",
    max_samples: Annotated[
        int | None, typer.Option("--max-samples", "-n", help="Max samples per split")
    ] = None,
    push: Annotated[bool, typer.Option(help="Push to HuggingFace Hub")] = True,
):
    """Generate dataset with DAC codec codes for Dia TTS training."""
    split_list = [s.strip() for s in splits.split(",")]

    typer.echo(f"Input: {input_dataset}" + (f" ({dataset_config})" if dataset_config else ""))
    typer.echo(f"Splits: {', '.join(split_list)}")
    typer.echo(f"Output: {output_repo}")
    if max_samples:
        typer.echo(f"Max samples per split: {max_samples}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"Using device: {device}")
    encoder = DACEncoder(device=device)

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

        gc.collect()
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
