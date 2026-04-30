#!/usr/bin/env python3
"""Generate SIFT datasets for paralinguistic training."""

import os
import re
from dataclasses import dataclass
from typing import Annotated

import typer

# Enable fast HuggingFace transfers (must be set before importing HF libraries)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import Audio, DatasetDict, Value, load_dataset
from huggingface_hub import DatasetCard, DatasetCardData
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, pipeline
from transformers.pipelines.pt_utils import KeyDataset

SIFT_SYSTEM_PROMPT = (
    'Describe the audio in one sentence starting with "Sounds like".\n'
    "Include: emotion, speaker gender, what they said (quoted), and voice quality.\n"
    "Example: \"Sounds like an angry man saying 'leave me alone' in a harsh, loud voice.\""
)
SIFT_INSTRUCTION = "/no_think"

MISSING_VALUE_SENTINELS = frozenset({"", "na", "null", "unk", "unknown", "nan", "none"})


def _clean_string(value: object) -> str | None:
    """Lowercase and strip a value; return None if it represents a missing value."""
    if value is None:
        return None
    cleaned = str(value).lower().strip()
    return None if cleaned in MISSING_VALUE_SENTINELS else cleaned


@dataclass
class DatasetConfig:
    """Configuration for a source dataset."""

    name: str
    hf_path: str
    hf_config: str | None = None
    split: str = "train"
    audio_field: str = "audio"
    text_field: str | None = None
    emotion_field: str | None = None
    gender_field: str | None = None
    age_field: str | None = None
    pace_field: str | None = None
    accent_field: str | None = None
    volume_field: str | None = None  # relative_db from AbstractTTS datasets
    max_samples: int | None = None  # Per-dataset sample limit (overrides --max-samples)
    # Flag for datasets with integer emotion labels (like MELD)
    emotion_is_int: bool = False


# HuggingFace datasets with paralinguistic labels (English only)
DATASET_CONFIGS = [
    # CREMA-D: emotion + gender + speaking_rate + volume (AbstractTTS version)
    DatasetConfig(
        name="crema_d",
        hf_path="AbstractTTS/CREMA-D",
        split="train",
        audio_field="audio",
        text_field="transcription",
        emotion_field="major_emotion",
        gender_field="gender",
        age_field=None,
        pace_field="speaking_rate",
        volume_field="relative_db",
    ),
    # RAVDESS: emotion, gender, speaking_rate, volume (AbstractTTS version)
    DatasetConfig(
        name="ravdess",
        hf_path="AbstractTTS/RAVDESS",
        split="train",
        audio_field="audio",
        text_field="transcription",
        emotion_field="emotion",
        gender_field="gender",
        age_field=None,
        pace_field="speaking_rate",
        volume_field="relative_db",
    ),
    # TESS: emotion + gender + speaking_rate + volume (all female speakers)
    # 2,800 samples with 7 emotion classes
    DatasetConfig(
        name="tess",
        hf_path="AbstractTTS/TESS",
        split="train",
        audio_field="audio",
        text_field="transcription",
        emotion_field="emotion",
        gender_field="gender",
        age_field=None,
        pace_field="speaking_rate",
        volume_field="relative_db",
    ),
    # SAVEE: emotion + gender + speaking_rate + volume (all male speakers)
    # 480 samples with 7 emotion classes
    DatasetConfig(
        name="savee",
        hf_path="AbstractTTS/SAVEE",
        split="train",
        audio_field="audio",
        text_field="transcription",
        emotion_field="emotion",
        gender_field="gender",
        age_field=None,
        pace_field="speaking_rate",
        volume_field="relative_db",
    ),
    # ESD English: emotion + gender + speaking_rate + volume (male and female)
    # 17,500 samples with 5 emotion classes
    DatasetConfig(
        name="esd",
        hf_path="AbstractTTS/ESD_english",
        split="train",
        audio_field="audio",
        text_field="transcription",
        emotion_field="emotion",
        gender_field="gender",
        age_field=None,
        pace_field="speaking_rate",
        volume_field="relative_db",
    ),
    # PODCAST: emotion, gender, speaking_rate, volume from podcast recordings
    # 149k samples with 16 emotion classes
    DatasetConfig(
        name="podcast",
        hf_path="AbstractTTS/PODCAST",
        split="train",
        audio_field="audio",
        text_field="transcription",
        emotion_field="major_emotion",
        gender_field="gender",
        age_field=None,
        pace_field="speaking_rate",
        volume_field="relative_db",
    ),
    # CommonVoice: Large-scale multilingual dataset (English subset)
    # Has age, gender, accent metadata
    DatasetConfig(
        name="commonvoice",
        hf_path="fixie-ai/common_voice_17_0",
        hf_config="en",
        split="train",
        audio_field="audio",
        text_field="sentence",
        emotion_field=None,
        gender_field="gender",
        age_field="age",
        accent_field="accent",
        max_samples=100000,
    ),
    # MELD: Multimodal EmotionLines Dataset (Friends TV show)
    # Has emotion (7 classes)
    DatasetConfig(
        name="meld",
        hf_path="garam-icecream/MELD",
        split="train",
        audio_field="audio",
        text_field="text",
        emotion_field="emotion",
        emotion_is_int=True,  # MELD uses integer labels
    ),
]


def age_to_group(age: str | int | None) -> str | None:
    """Convert numeric age to age group. Returns None for missing/invalid values."""
    if age is None:
        return None
    try:
        age_int = int(age)
    except (ValueError, TypeError):
        return _clean_string(age)
    if 0 < age_int < 18:
        return "teenager"
    if age_int < 40:
        return "young adult"
    if age_int <= 60:
        return "middle-age adult"
    if 60 < age_int < 200:
        return "senior"
    return None


def volume_to_label(relative_db: float | None) -> str | None:
    """Convert relative_db to volume label.

    Thresholds (AbstractTTS dataset distributions):
      - quiet: < -16.4 dB (below 25th percentile)
      - loud:  > -10.0 dB (above 75th percentile)
      - normal: between (returns None — don't mention unremarkable features)
    """
    if relative_db is None:
        return None
    if relative_db < -16.4:
        return "quiet"
    if relative_db > -10.0:
        return "loud"
    return None


def pace_to_label(rate: str | float | None) -> str | None:
    """Convert numeric speaking rate to text label.

    Thresholds (AbstractTTS dataset distributions, range ~3-17):
      - slow:   < 6.0
      - normal: 6.0 - 9.0
      - fast:   > 9.0
    """
    if rate is None:
        return None
    try:
        rate_float = float(rate)
    except (ValueError, TypeError):
        return _clean_string(rate)
    if rate_float <= 0:
        return None
    if rate_float < 6.0:
        return "slow"
    if rate_float <= 9.0:
        return "normal"
    return "fast"


def normalize_label(value: str | None) -> str | None:
    """Normalize a label value. Returns None for missing/invalid values."""
    return _clean_string(value)


EMOTION_NORMALIZATION = {
    "anger": "angry",
    "happiness": "happy",
    "sadness": "sad",
    "surprised": "surprise",
    "pleasant surprise": "surprise",
}


def normalize_emotion(value: str | None) -> str | None:
    """Normalize dataset-specific emotion labels (e.g. anger→angry, happiness→happy)."""
    cleaned = _clean_string(value)
    if cleaned is None:
        return None
    return EMOTION_NORMALIZATION.get(cleaned, cleaned)


# MELD uses integer emotion labels; joy maps to "happy" for consistency with other datasets.
MELD_EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


def normalize_meld_emotion(value: int | str | None) -> str | None:
    """Convert MELD emotion integer (or string) to canonical label."""
    if isinstance(value, int):
        return normalize_emotion(MELD_EMOTION_MAP.get(value))
    return normalize_emotion(value)


METADATA_KEYS = ("text", "emotion", "gender", "age", "pace", "accent", "volume")


def extract_metadata(sample: dict, config: DatasetConfig) -> dict:
    """Extract paralinguistic metadata from a sample.

    Returns empty strings instead of None to ensure consistent schema
    across multiprocessing batches in datasets.map().
    """
    metadata: dict = dict.fromkeys(METADATA_KEYS, "")

    if config.text_field and (text := sample.get(config.text_field)):
        metadata["text"] = str(text).strip().lower()

    if config.emotion_field and config.emotion_field in sample:
        raw = sample[config.emotion_field]
        metadata["emotion"] = (
            normalize_meld_emotion(raw) if config.emotion_is_int else normalize_emotion(raw)
        )

    if config.gender_field and config.gender_field in sample:
        gender = normalize_label(sample[config.gender_field])
        if gender in ("m", "male"):
            gender = "male"
        elif gender in ("f", "female"):
            gender = "female"
        metadata["gender"] = gender

    if config.age_field and config.age_field in sample:
        metadata["age"] = age_to_group(sample[config.age_field])

    if config.pace_field and config.pace_field in sample:
        metadata["pace"] = pace_to_label(sample[config.pace_field])

    if config.accent_field and config.accent_field in sample:
        metadata["accent"] = normalize_label(sample[config.accent_field])

    if config.volume_field and config.volume_field in sample:
        metadata["volume"] = volume_to_label(sample[config.volume_field])

    return metadata


# Order: demographics first, then voice characteristics, then content-related.
PARA_ORDER = ("age", "gender", "volume", "pace", "emotion", "accent")


def build_audio_context(metadata: dict) -> str:
    """Build audio context with metadata for the LLM."""
    para_parts = [
        f"{key.replace('_', ' ')}: {value}" for key in PARA_ORDER if (value := metadata.get(key))
    ]
    inner = ""
    if para_parts:
        inner += f"<meta>{', '.join(para_parts)}</meta>"
    if metadata["text"]:
        inner += f"<text>{metadata['text']}</text>"
    return f"<audio>{inner}</audio>"


def create_dataset_card(repo_id: str, splits: list[str]) -> None:
    """Create and push a dataset card with proper metadata."""
    card_data = DatasetCardData(
        language=["en"],
        license="cc-by-nc-sa-4.0",
        task_categories=["automatic-speech-recognition", "audio-classification"],
        tags=["audio", "speech", "sift", "instruction-tuning", "emotion-recognition"],
        pretty_name="SIFT Audio Dataset",
    )

    splits_list = "\n".join(f"- `{split}`" for split in sorted(splits))

    card_content = f"""---
{card_data.to_yaml()}
---

# SIFT Audio Dataset

Self-Instruction Fine-Tuning (SIFT) dataset for training audio understanding models.

## Dataset Description

This dataset contains audio samples paired with varied instruction-response pairs generated
using LLM-based data augmentation. Each audio sample includes:

- **Transcription**: What was spoken in the audio
- **Speaker metadata**: Gender, emotion, speaking rate (where available)
- **Instruction**: A natural language question or command about the audio
- **Response**: A natural language response answering the instruction

## Splits

{splits_list}

## Usage

```python
from datasets import load_dataset

# Load a specific split
ds = load_dataset("{repo_id}", split="loquacious")

# Access a sample
sample = ds[0]
print(sample["sift_response"])
```

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `audio` | Audio | Audio waveform |
| `text` | string | Transcription of the audio |
| `emotion` | string | Detected emotion (if available) |
| `gender` | string | Speaker gender (if available) |
| `age` | string | Speaker age group (if available) |
| `pace` | string | Speaking pace: slow, normal, fast (if available) |
| `volume` | string | Volume level: quiet, loud (if notable) |
| `accent` | string | Speaker accent (if available) |
| `sift_response` | string | Generated description of the audio |
| `source_dataset` | string | Original dataset source |

## License

Apache 2.0
"""

    card = DatasetCard(card_content)
    card.push_to_hub(repo_id)


def process_dataset(
    config: DatasetConfig,
    pipe,
    tokenizer,
    batch_size: int,
    max_samples: int | None,
    max_new_tokens: int,
):
    """Process a single dataset and generate SIFT responses using datasets.map()."""
    print(f"\nProcessing {config.name}...")

    # Load dataset
    ds = load_dataset(
        config.hf_path,
        name=config.hf_config,
        split=config.split,
        trust_remote_code=True,
    )

    caps = [m for m in (config.max_samples, max_samples) if m is not None]
    effective_max = min(caps) if caps else None
    if effective_max and len(ds) > effective_max:
        ds = ds.select(range(effective_max))

    print(f"  Loaded {len(ds)} samples")
    print("  Preparing prompts...")

    prompts = []
    metadata_lists: dict[str, list] = {key: [] for key in METADATA_KEYS}

    for sample in tqdm(ds, desc="Preparing", total=len(ds)):
        metadata = extract_metadata(sample, config)
        for key in metadata_lists:
            metadata_lists[key].append(metadata[key])

        user_content = f"{build_audio_context(metadata)}\n\n{SIFT_INSTRUCTION}"
        messages = [
            {"role": "system", "content": SIFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )

    ds = ds.add_column("prompt", prompts)
    for key, values in metadata_lists.items():
        ds = ds.add_column(f"meta_{key}", values)

    print("  Generating instructions and responses...")
    thinking_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    responses = []
    for out in tqdm(
        pipe(
            KeyDataset(ds, "prompt"),
            generation_config=generation_config,
            batch_size=batch_size,
            return_full_text=False,
        ),
        total=len(ds),
        desc="Generating",
    ):
        responses.append(thinking_pattern.sub("", out[0]["generated_text"]).strip())

    ds = ds.add_column("sift_response", responses)

    conflict_cols = [c for c in METADATA_KEYS if c in ds.column_names]
    if conflict_cols:
        ds = ds.remove_columns(conflict_cols)

    ds = ds.rename_columns({f"meta_{key}": key for key in METADATA_KEYS})
    ds = ds.add_column("source_dataset", [config.name] * len(ds))

    keep_cols = ("audio", *METADATA_KEYS, "sift_response", "source_dataset")
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)

    for col in ("emotion", "gender", "age", "pace", "accent", "volume"):
        ds = ds.cast_column(col, Value("string"))
    return ds.cast_column("audio", Audio(sampling_rate=16000))


app = typer.Typer(help="Generate SIFT datasets for paralinguistic training")


@app.command()
def main(
    output_repo: Annotated[
        str, typer.Option(help="HuggingFace repo ID for output")
    ] = "mazesmazes/sift-audio-2",
    model_name: Annotated[
        str, typer.Option(help="LLM model for response generation")
    ] = "Qwen/Qwen3-4B",
    batch_size: Annotated[int, typer.Option(help="Batch size for generation")] = 2048,
    max_samples: Annotated[
        int | None, typer.Option(help="Max samples per dataset (for testing)")
    ] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Max new tokens for generation")] = 80,
    datasets: Annotated[list[str] | None, typer.Option(help="Specific datasets to process")] = None,
    push_every: Annotated[int, typer.Option(help="Push to hub every N datasets")] = 1,
):
    """Generate SIFT datasets for paralinguistic training."""
    configs = DATASET_CONFIGS
    if datasets:
        dataset_names = {name.strip() for d in datasets for name in d.split(",") if name.strip()}
        configs = [c for c in configs if c.name in dataset_names]
        if not configs:
            typer.echo(
                f"No matching datasets found. Available: {[c.name for c in DATASET_CONFIGS]}"
            )
            raise typer.Exit(1)

    typer.echo(f"Processing {len(configs)} datasets")
    typer.echo(f"Model: {model_name}")
    typer.echo(f"Output: {output_repo}")

    typer.echo(f"\nLoading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        dtype="bfloat16",
        device_map="auto",
        model_kwargs={"attn_implementation": "flash_attention_2"},
    )

    all_datasets = {}
    for i, config in enumerate(configs, start=1):
        try:
            ds = process_dataset(
                config=config,
                pipe=pipe,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_samples=max_samples,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            import traceback

            typer.echo(f"Error processing {config.name}: {e}")
            traceback.print_exc()
            continue

        if ds is None:
            continue
        all_datasets[config.name] = ds

        if i % push_every == 0:
            typer.echo(f"\nPushing {len(all_datasets)} splits to {output_repo}...")
            DatasetDict(all_datasets).push_to_hub(output_repo, private=False)

    if not all_datasets:
        typer.echo("No datasets were successfully processed.")
        return

    typer.echo(f"\nFinal push: {len(all_datasets)} splits to {output_repo}...")
    DatasetDict(all_datasets).push_to_hub(output_repo, private=False)

    typer.echo("Updating dataset card...")
    create_dataset_card(output_repo, list(all_datasets.keys()))

    typer.echo("Done!")


if __name__ == "__main__":
    app()
