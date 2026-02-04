#!/usr/bin/env python3
"""Generate LibriTTS-R-MIMI dataset with audio column.

Takes the jkeisling/libritts-r-mimi dataset (which has mimi codec codes but no audio)
and joins it with the original blabble-io/libritts_r dataset to add the audio column.

Usage:
    python -m scripts.generate_libritts_mimi_with_audio --output-repo user/libritts-r-mimi-audio

    # Process specific splits only
    python -m scripts.generate_libritts_mimi_with_audio --splits dev.clean,test.clean

    # Test with limited samples
    python -m scripts.generate_libritts_mimi_with_audio --max-samples 100
"""

import os
from typing import Annotated

import typer

# Enable fast HuggingFace transfers
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import Audio, DatasetDict, load_dataset
from huggingface_hub import DatasetCard, DatasetCardData
from tqdm import tqdm

app = typer.Typer(help="Generate LibriTTS-R-MIMI dataset with audio column")

# Split name mapping: mimi dataset uses dots, libritts_r uses dashes
SPLIT_MAPPING = {
    "dev.clean": "dev.clean",
    "test.clean": "test.clean",
    "train.clean.100": "train.clean.100",
    "train.clean.360": "train.clean.360",
}

# LibriTTS-R uses different split names
LIBRITTS_SPLIT_MAPPING = {
    "dev.clean": "dev.clean",
    "test.clean": "test.clean",
    "train.clean.100": "train.clean.100",
    "train.clean.360": "train.clean.360",
}


def create_dataset_card(repo_id: str, splits: list[str]) -> None:
    """Create and push a dataset card with proper metadata."""
    card_data = DatasetCardData(
        language=["en"],
        license="cc-by-4.0",
        task_categories=["automatic-speech-recognition", "audio-to-audio"],
        tags=["audio", "speech", "mimi", "codec", "libritts"],
        pretty_name="LibriTTS-R MIMI with Audio",
    )

    splits_list = "\n".join(f"- `{split}`" for split in sorted(splits))

    card_content = f"""---
{card_data.to_yaml()}
---

# LibriTTS-R MIMI with Audio

This dataset combines [jkeisling/libritts-r-mimi](https://huggingface.co/datasets/jkeisling/libritts-r-mimi)
(which contains MIMI codec codes) with audio from [mythicinfinity/libritts_r](https://huggingface.co/datasets/mythicinfinity/libritts_r).

## Dataset Description

Each sample contains:
- **audio**: Original LibriTTS-R audio at 24kHz
- **codes**: 8-layer MIMI codec codes
- **text_normalized**: Normalized transcription
- **text_original**: Original transcription
- **speaker_id**, **chapter_id**, **id**: Identifiers

## Splits

{splits_list}

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="dev.clean")

# Access audio and codes together
sample = ds[0]
audio = sample["audio"]  # {{'array': [...], 'sampling_rate': 24000}}
codes = sample["codes"]  # 8 lists of codec indices
text = sample["text_normalized"]
```

## Source Datasets

- [jkeisling/libritts-r-mimi](https://huggingface.co/datasets/jkeisling/libritts-r-mimi) - MIMI codec codes
- [mythicinfinity/libritts_r](https://huggingface.co/datasets/mythicinfinity/libritts_r) - Audio files

## License

CC-BY-4.0 (same as LibriTTS-R)
"""

    card = DatasetCard(card_content)
    card.push_to_hub(repo_id)


def process_split(
    mimi_split_name: str,
    max_samples: int | None = None,
) -> dict | None:
    """Process a single split by joining mimi codes with libritts_r audio.

    Args:
        mimi_split_name: Name of the split (e.g., "dev.clean")
        max_samples: Optional limit on number of samples

    Returns:
        Dataset with audio column added, or None on failure
    """
    typer.echo(f"\nProcessing split: {mimi_split_name}")

    # Load mimi dataset (has codes but no audio)
    typer.echo("  Loading mimi dataset...")
    try:
        mimi_ds = load_dataset(
            "jkeisling/libritts-r-mimi",
            split=mimi_split_name,
            trust_remote_code=True,
        )
    except Exception as e:
        typer.echo(f"  Error loading mimi split {mimi_split_name}: {e}")
        return None

    # Load libritts_r dataset (has audio)
    # LibriTTS-R requires a config name: "clean", "other", or "all"
    libritts_split = LIBRITTS_SPLIT_MAPPING.get(mimi_split_name, mimi_split_name)
    # Determine config based on split name
    if "clean" in libritts_split:
        libritts_config = "clean"
    elif "other" in libritts_split:
        libritts_config = "other"
    else:
        libritts_config = "all"
    typer.echo(
        f"  Loading libritts_r dataset (config: {libritts_config}, split: {libritts_split})..."
    )
    try:
        libritts_ds = load_dataset(
            "mythicinfinity/libritts_r",
            libritts_config,
            split=libritts_split,
            trust_remote_code=True,
        )
    except Exception as e:
        typer.echo(f"  Error loading libritts_r split {libritts_split}: {e}")
        return None

    typer.echo(f"  MIMI samples: {len(mimi_ds)}, LibriTTS-R samples: {len(libritts_ds)}")

    # Limit samples if requested (for testing)
    if max_samples and len(mimi_ds) > max_samples:
        mimi_ds = mimi_ds.select(range(max_samples))
        typer.echo(f"  Limited to {len(mimi_ds)} samples")

    # Build lookup from id -> audio
    typer.echo("  Building audio lookup...")
    id_to_audio = {}
    for sample in tqdm(libritts_ds, desc="  Indexing audio"):
        id_to_audio[sample["id"]] = sample["audio"]

    # Add audio column to mimi dataset
    typer.echo("  Adding audio column...")
    audio_list = []
    missing_count = 0

    for sample in tqdm(mimi_ds, desc="  Matching audio"):
        sample_id = sample["id"]
        if sample_id in id_to_audio:
            audio_list.append(id_to_audio[sample_id])
        else:
            # Audio not found - this shouldn't happen if datasets are aligned
            missing_count += 1
            audio_list.append(None)

    if missing_count > 0:
        typer.echo(f"  Warning: {missing_count} samples missing audio")

    # Filter out samples with missing audio
    if missing_count > 0:
        valid_indices = [i for i, a in enumerate(audio_list) if a is not None]
        mimi_ds = mimi_ds.select(valid_indices)
        audio_list = [a for a in audio_list if a is not None]
        typer.echo(f"  Filtered to {len(mimi_ds)} samples with audio")

    # Add audio column
    result_ds = mimi_ds.add_column("audio", audio_list)

    # Remove the path column (not needed anymore, was just cache path)
    if "path" in result_ds.column_names:
        result_ds = result_ds.remove_columns(["path"])

    # Ensure audio column has proper type
    result_ds = result_ds.cast_column("audio", Audio(sampling_rate=24000))

    typer.echo(f"  Final dataset: {len(result_ds)} samples")
    return result_ds


@app.command()
def main(
    output_repo: Annotated[
        str, typer.Option(help="HuggingFace repo ID for output")
    ] = "mazesmazes/libritts-r-mimi-audio",
    splits: Annotated[
        list[str] | None,
        typer.Option(help="Specific splits to process (comma-separated). Default: all"),
    ] = None,
    max_samples: Annotated[
        int | None, typer.Option(help="Max samples per split (for testing)")
    ] = None,
    push_every: Annotated[int, typer.Option(help="Push to hub every N splits")] = 1,
):
    """Generate LibriTTS-R-MIMI dataset with audio column.

    Joins the mimi dataset (codec codes only) with LibriTTS-R (audio) using
    the shared 'id' field to create a complete dataset with both audio and codes.
    """
    # Parse splits
    if splits:
        # Expand comma-separated values
        expanded = []
        for s in splits:
            expanded.extend(s.split(","))
        split_names = [s.strip() for s in expanded if s.strip()]
    else:
        split_names = list(SPLIT_MAPPING.keys())

    typer.echo(f"Processing {len(split_names)} splits: {split_names}")
    typer.echo(f"Output: {output_repo}")

    all_datasets = {}
    splits_processed = 0

    for split_name in split_names:
        try:
            ds = process_split(
                mimi_split_name=split_name,
                max_samples=max_samples,
            )

            if ds is not None:
                all_datasets[split_name] = ds
                splits_processed += 1

                # Push periodically
                if splits_processed % push_every == 0:
                    typer.echo(f"\nPushing {len(all_datasets)} splits to {output_repo}...")
                    dataset_dict = DatasetDict(all_datasets)
                    dataset_dict.push_to_hub(output_repo, private=False)

        except Exception as e:
            import traceback

            typer.echo(f"Error processing {split_name}: {e}")
            traceback.print_exc()
            continue

    # Final push
    if all_datasets:
        typer.echo(f"\nFinal push: {len(all_datasets)} splits to {output_repo}...")
        dataset_dict = DatasetDict(all_datasets)
        dataset_dict.push_to_hub(output_repo, private=False)

        # Update dataset card
        typer.echo("Updating dataset card...")
        create_dataset_card(output_repo, list(all_datasets.keys()))

        typer.echo("Done!")
    else:
        typer.echo("No splits were successfully processed.")


if __name__ == "__main__":
    app()
