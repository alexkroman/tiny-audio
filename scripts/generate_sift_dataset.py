#!/usr/bin/env python3
"""Generate SIFT (Self-Instruction Fine-Tuning) datasets for paralinguistic training.

This script processes audio datasets with paralinguistic metadata (emotion, gender, age)
and generates natural language responses using an LLM, following the Azeros/Auden approach.

Uses SIT_SSP mode: System message + instruction + semantic + paralinguistic

Usage:
    python -m scripts.generate_sift_dataset --output-repo user/sift-dataset

    # On RunPod via CLI:
    ta runpod sift <host> <port> --output-repo user/sift-dataset
"""

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

# System prompt for audio understanding
SIFT_SYSTEM_PROMPT = (
    "You are an audio assistant that can hear speech. "
    "Audio is provided in <audio> tags. "
    "Respond as if you directly heard the audio. "
    "Never say 'the text says' or mention metadata - just describe what you hear."
)

# Fixed instruction for all samples
SIFT_INSTRUCTION = "Describe all information you can hear."


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
    speaking_rate_field: str | None = None
    accent_field: str | None = None
    sentiment_field: str | None = None
    human_labels_field: str | None = None  # For audioset-style labels (list of strings)
    max_samples: int | None = None  # Per-dataset sample limit (overrides --max-samples)
    # Flag for datasets with integer emotion labels (like MELD)
    emotion_is_int: bool = False


# HuggingFace datasets with paralinguistic labels (English only)
DATASET_CONFIGS = [
    # CREMA-D: emotion + gender + speaking_rate (AbstractTTS version)
    DatasetConfig(
        name="crema_d",
        hf_path="AbstractTTS/CREMA-D",
        split="train",
        audio_field="audio",
        text_field="transcription",
        emotion_field="major_emotion",
        gender_field="gender",
        age_field=None,
        speaking_rate_field="speaking_rate",
    ),
    # RAVDESS: emotion, gender, speaking_rate (AbstractTTS version)
    DatasetConfig(
        name="ravdess",
        hf_path="AbstractTTS/RAVDESS",
        split="train",
        audio_field="audio",
        text_field="transcription",
        emotion_field="emotion",
        gender_field="gender",
        age_field=None,
        speaking_rate_field="speaking_rate",
    ),
    # TESS: emotion + gender + speaking_rate (all female speakers)
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
        speaking_rate_field="speaking_rate",
    ),
    # SAVEE: emotion + gender + speaking_rate (all male speakers)
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
        speaking_rate_field="speaking_rate",
    ),
    # ESD English: emotion + gender + speaking_rate (male and female)
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
        speaking_rate_field="speaking_rate",
    ),
    # PODCAST: emotion, gender, speaking_rate from podcast recordings
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
        speaking_rate_field="speaking_rate",
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
    # Has emotion (7 classes) and sentiment (3 classes)
    DatasetConfig(
        name="meld",
        hf_path="garam-icecream/MELD",
        split="train",
        audio_field="audio",
        text_field="text",
        emotion_field="emotion",
        sentiment_field="sentiment",
        emotion_is_int=True,  # MELD uses integer labels
    ),
    # AudioSet Humans: Sound event detection with human-readable labels
    # Has human_labels (list of sound categories like 'Music', 'Speech', etc.)
    DatasetConfig(
        name="audioset_humans",
        hf_path="enyoukai/audioset-humans-reprocessed",
        split="train",
        audio_field="audio",
        text_field=None,  # No transcription
        human_labels_field="human_labels",
    ),
]


def age_to_group(age: str | int | None) -> str | None:
    """Convert numeric age to age group. Returns None for missing/invalid values."""
    if age is None:
        return None
    try:
        age_int = int(age)
        if 0 < age_int < 18:
            return "teenager"
        if age_int < 40:
            return "young adult"
        if age_int <= 60:
            return "middle-age adult"
        if 60 < age_int < 200:
            return "senior"
        return None
    except (ValueError, TypeError):
        # Already a string like "young adult"
        if isinstance(age, str) and age.lower() not in ("", "na", "null", "unk", "unknown", "nan"):
            return age.lower()
        return None


def speaking_rate_to_label(rate: str | float | None) -> str | None:
    """Convert numeric speaking rate to text label. Returns None for missing/invalid values.

    Speaking rate thresholds based on AbstractTTS dataset distributions:
    - Data ranges from ~3-17 with median ~7-10
    - Slow: < 6.0 (bottom ~25%)
    - Normal: 6.0 - 9.0 (middle ~50%)
    - Fast: > 9.0 (top ~25%)
    """
    if rate is None:
        return None
    try:
        rate_float = float(rate)
        if rate_float <= 0:
            return None
        if rate_float < 6.0:
            return "slow"
        if rate_float <= 9.0:
            return "normal"
        return "fast"
    except (ValueError, TypeError):
        # Already a string like "fast", "slow", "normal"
        if isinstance(rate, str) and rate.lower() not in (
            "",
            "na",
            "null",
            "unk",
            "unknown",
            "nan",
        ):
            return rate.lower()
        return None


def normalize_label(value: str | None) -> str | None:
    """Normalize a label value. Returns None for missing/invalid values."""
    if value is None:
        return None
    value_str = str(value).lower().strip()
    if value_str in ("", "na", "null", "unk", "unknown", "nan", "none"):
        return None
    return value_str


# Emotion label normalization mapping
# Maps various dataset-specific labels to consistent canonical forms
EMOTION_NORMALIZATION = {
    # Anger variants
    "angry": "angry",
    "anger": "angry",
    # Happiness variants
    "happy": "happy",
    "happiness": "happy",
    # Sadness variants
    "sad": "sad",
    "sadness": "sad",
    # Surprise variants
    "surprise": "surprise",
    "surprised": "surprise",
    "pleasant surprise": "surprise",
    # Standard labels (no change needed)
    "neutral": "neutral",
    "fear": "fear",
    "disgust": "disgust",
}


def normalize_emotion(value: str | None) -> str | None:
    """Normalize emotion labels to consistent canonical forms.

    Different datasets use different labels for the same emotion:
    - TESS: angry, happy, sad, pleasant surprise
    - SAVEE: anger, happiness, sadness, surprise
    - CREMA-D: anger, happy, sad

    This normalizes them to: angry, happy, sad, surprise, neutral, fear, disgust
    """
    if value is None:
        return None
    value_lower = str(value).lower().strip()
    if value_lower in ("", "na", "null", "unk", "unknown", "nan", "none"):
        return None
    return EMOTION_NORMALIZATION.get(value_lower, value_lower)


# MELD emotion mapping (integer to string)
MELD_EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",  # joy -> happy for consistency
    4: "neutral",
    5: "sad",
    6: "surprise",
}

# Sentiment mapping (integer to string)
SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def normalize_meld_emotion(value: int | str | None) -> str | None:
    """Convert MELD emotion integer to string label."""
    if value is None:
        return None
    if isinstance(value, int):
        label = MELD_EMOTION_MAP.get(value)
        return normalize_emotion(label) if label else None
    return normalize_emotion(value)


def normalize_sentiment(value: int | str | None) -> str | None:
    """Convert sentiment integer to string label."""
    if value is None:
        return None
    if isinstance(value, int):
        return SENTIMENT_MAP.get(value)
    value_lower = str(value).lower().strip()
    if value_lower in ("", "na", "null", "unk", "unknown", "nan", "none"):
        return None
    return value_lower


def extract_metadata(sample: dict, config: DatasetConfig) -> dict:
    """Extract paralinguistic metadata from a sample.

    Returns empty strings instead of None to ensure consistent schema
    across multiprocessing batches in datasets.map().
    """
    metadata = {
        "text": "",
        "emotion": "",
        "gender": "",
        "age": "",
        "speaking_rate": "",
        "accent": "",
        "sentiment": "",
        "human_labels": "",
    }

    # Extract text transcription
    if config.text_field and config.text_field in sample:
        text = sample[config.text_field]
        if text:
            metadata["text"] = str(text).strip().lower()

    # Extract emotion (handle integer labels for MELD)
    if config.emotion_field and config.emotion_field in sample:
        raw_emotion = sample[config.emotion_field]
        if config.emotion_is_int:
            metadata["emotion"] = normalize_meld_emotion(raw_emotion)
        else:
            metadata["emotion"] = normalize_emotion(raw_emotion)

    # Extract gender
    if config.gender_field and config.gender_field in sample:
        gender = normalize_label(sample[config.gender_field])
        # Normalize gender values
        if gender in ("m", "male"):
            gender = "male"
        elif gender in ("f", "female"):
            gender = "female"
        metadata["gender"] = gender

    # Extract age
    if config.age_field and config.age_field in sample:
        metadata["age"] = age_to_group(sample[config.age_field])

    # Extract speaking rate (convert numeric to label if needed)
    if config.speaking_rate_field and config.speaking_rate_field in sample:
        raw_rate = sample[config.speaking_rate_field]
        metadata["speaking_rate"] = speaking_rate_to_label(raw_rate)

    # Extract accent
    if config.accent_field and config.accent_field in sample:
        accent = normalize_label(sample[config.accent_field])
        metadata["accent"] = accent

    # Extract sentiment (handle integer labels)
    if config.sentiment_field and config.sentiment_field in sample:
        raw_sentiment = sample[config.sentiment_field]
        metadata["sentiment"] = normalize_sentiment(raw_sentiment)

    # Extract human labels (list of sound/audio categories)
    if config.human_labels_field and config.human_labels_field in sample:
        labels = sample[config.human_labels_field]
        if isinstance(labels, list) and labels:
            # Join list into comma-separated string
            metadata["human_labels"] = ", ".join(str(lbl).lower() for lbl in labels)
        elif labels:
            metadata["human_labels"] = str(labels).lower()

    return metadata


def build_audio_context(metadata: dict) -> str:
    """Build audio context with metadata for the LLM.

    Args:
        metadata: Extracted metadata from the audio sample

    Returns:
        Audio context string with metadata in tags
    """
    # Build paralinguistic metadata
    para_parts = []
    for key in ["age", "gender", "emotion", "speaking_rate", "accent", "sentiment", "human_labels"]:
        value = metadata.get(key)
        if value:  # Skip empty strings and None
            display_key = key.replace("_", " ")
            para_parts.append(f"{display_key}: {value}")

    # Format input with audio tags
    if para_parts and metadata["text"]:
        para_text = ", ".join(para_parts)
        return f"<audio><meta>{para_text}</meta><text>{metadata['text']}</text></audio>"
    if para_parts:
        para_text = ", ".join(para_parts)
        return f"<audio><meta>{para_text}</meta></audio>"
    if metadata["text"]:
        return f"<audio><text>{metadata['text']}</text></audio>"
    return "<audio></audio>"


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
| `speaking_rate` | string | Speaking pace: slow, normal, fast (if available) |
| `accent` | string | Speaker accent (if available) |
| `sentiment` | string | Sentiment: positive, neutral, negative (if available) |
| `human_labels` | string | Sound event labels from AudioSet (if available) |
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

    # Limit samples first (faster than filtering all)
    if config.max_samples is not None and max_samples is not None:
        effective_max = min(config.max_samples, max_samples)
    elif config.max_samples is not None:
        effective_max = config.max_samples
    elif max_samples is not None:
        effective_max = max_samples
    else:
        effective_max = None

    if effective_max and len(ds) > effective_max:
        ds = ds.select(range(effective_max))

    print(f"  Loaded {len(ds)} samples")

    # Extract metadata and build prompts (simple loop is faster than multiprocess map for this)
    print("  Preparing prompts...")
    prompts = []
    metadata_lists = {
        "text": [],
        "emotion": [],
        "gender": [],
        "age": [],
        "speaking_rate": [],
        "accent": [],
        "sentiment": [],
        "human_labels": [],
    }

    for sample in tqdm(ds, desc="Preparing", total=len(ds)):
        metadata = extract_metadata(sample, config)
        for key in metadata_lists:
            metadata_lists[key].append(metadata[key])

        audio_context = build_audio_context(metadata)
        user_content = f"{audio_context}\n\n{SIFT_INSTRUCTION}"
        messages = [
            {"role": "system", "content": SIFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )

    # Add columns to dataset
    ds = ds.add_column("prompt", prompts)
    ds = ds.add_column("meta_text", metadata_lists["text"])
    ds = ds.add_column("meta_emotion", metadata_lists["emotion"])
    ds = ds.add_column("meta_gender", metadata_lists["gender"])
    ds = ds.add_column("meta_age", metadata_lists["age"])
    ds = ds.add_column("meta_speaking_rate", metadata_lists["speaking_rate"])
    ds = ds.add_column("meta_accent", metadata_lists["accent"])
    ds = ds.add_column("meta_sentiment", metadata_lists["sentiment"])
    ds = ds.add_column("meta_human_labels", metadata_lists["human_labels"])

    # Generate responses using pipeline with dataset (more efficient than .map())
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
        text = thinking_pattern.sub("", out[0]["generated_text"]).strip()
        responses.append(text)

    ds = ds.add_column("sift_response", responses)

    # Remove original columns that would conflict with our renamed columns
    conflict_cols = [
        c
        for c in [
            "text",
            "emotion",
            "gender",
            "age",
            "speaking_rate",
            "accent",
            "sentiment",
            "human_labels",
        ]
        if c in ds.column_names
    ]
    if conflict_cols:
        ds = ds.remove_columns(conflict_cols)

    # Rename meta columns to final names
    ds = ds.rename_columns(
        {
            "meta_text": "text",
            "meta_emotion": "emotion",
            "meta_gender": "gender",
            "meta_age": "age",
            "meta_speaking_rate": "speaking_rate",
            "meta_accent": "accent",
            "meta_sentiment": "sentiment",
            "meta_human_labels": "human_labels",
        }
    )
    ds = ds.add_column("source_dataset", [config.name] * len(ds))

    # Keep only needed columns
    keep_cols = [
        "audio",
        "text",
        "emotion",
        "gender",
        "age",
        "speaking_rate",
        "accent",
        "sentiment",
        "human_labels",
        "sift_response",
        "source_dataset",
    ]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)

    # Cast columns to consistent types
    ds = ds.cast_column("emotion", Value("string"))
    ds = ds.cast_column("gender", Value("string"))
    ds = ds.cast_column("age", Value("string"))
    ds = ds.cast_column("speaking_rate", Value("string"))
    ds = ds.cast_column("accent", Value("string"))
    ds = ds.cast_column("sentiment", Value("string"))
    ds = ds.cast_column("human_labels", Value("string"))
    return ds.cast_column("audio", Audio(sampling_rate=16000))


app = typer.Typer(help="Generate SIFT datasets for paralinguistic training")


@app.command()
def main(
    output_repo: Annotated[
        str, typer.Option(help="HuggingFace repo ID for output")
    ] = "mazesmazes/sift-audio",
    model_name: Annotated[
        str, typer.Option(help="LLM model for response generation")
    ] = "Qwen/Qwen3-1.7B",
    batch_size: Annotated[
        int, typer.Option(help="Batch size for generation")
    ] = 1024,  # Tuned for A40 48GB
    max_samples: Annotated[
        int | None, typer.Option(help="Max samples per dataset (for testing)")
    ] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Max new tokens for generation")] = 256,
    datasets: Annotated[list[str] | None, typer.Option(help="Specific datasets to process")] = None,
    push_every: Annotated[int, typer.Option(help="Push to hub every N datasets")] = 1,
):
    """Generate SIFT datasets for paralinguistic training."""
    # Filter datasets if specified (support comma-separated values)
    configs = DATASET_CONFIGS
    if datasets:
        # Expand comma-separated values: ["a,b", "c"] -> ["a", "b", "c"]
        expanded = []
        for d in datasets:
            expanded.extend(d.split(","))
        dataset_names = [d.strip() for d in expanded if d.strip()]

        configs = [c for c in configs if c.name in dataset_names]
        if not configs:
            typer.echo(
                f"No matching datasets found. Available: {[c.name for c in DATASET_CONFIGS]}"
            )
            raise typer.Exit(1)

    typer.echo(f"Processing {len(configs)} datasets")
    typer.echo(f"Model: {model_name}")
    typer.echo(f"Output: {output_repo}")

    # Load tokenizer and create pipeline
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

    # Process each dataset
    all_datasets = {}
    datasets_processed = 0

    for config in configs:
        try:
            ds = process_dataset(
                config=config,
                pipe=pipe,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_samples=max_samples,
                max_new_tokens=max_new_tokens,
            )

            if ds is not None:
                all_datasets[config.name] = ds
                datasets_processed += 1

                # Push periodically
                if datasets_processed % push_every == 0:
                    typer.echo(f"\nPushing {len(all_datasets)} splits to {output_repo}...")
                    dataset_dict = DatasetDict(all_datasets)
                    dataset_dict.push_to_hub(output_repo, private=False)

        except Exception as e:
            import traceback

            typer.echo(f"Error processing {config.name}: {e}")
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
        typer.echo("No datasets were successfully processed.")


if __name__ == "__main__":
    app()
