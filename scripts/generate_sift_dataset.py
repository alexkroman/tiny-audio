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

import torch
from datasets import Audio, Dataset, DatasetDict, Value, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, pipeline
from transformers.pipelines.pt_utils import KeyDataset

# System message for paralinguistic-aware responses
SIFT_SYSTEM_MESSAGE = (
    "You are a speech perception system that describes audio content in third person. "
    "Audio input is provided in <audio> tags with transcription and speaker characteristics. "
    "Describe what is being said and how, using phrases like 'A person is saying...' or 'Someone says...'. "
    "Do NOT use first person ('I hear'). Do NOT echo metadata directly ('the emotion is X'). "
    "Naturally incorporate the speaker's tone and emotion into your description."
)

# Instruction for SIT mode
SIFT_INSTRUCTION = "Describe the audio in 1-2 sentences using third person."


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
    weight: float = 1.0
    max_samples: int | None = None  # Per-dataset sample limit (overrides --max-samples)
    emotion_map: dict | None = None  # Map numeric labels to emotion strings


# Emotion label mappings for datasets with numeric labels
CREMA_D_EMOTIONS = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad"}
# HuggingFace datasets with paralinguistic labels (English only)
DATASET_CONFIGS = [
    # CREMA-D: emotion + text (gender/age not in this dataset version)
    DatasetConfig(
        name="crema_d",
        hf_path="myleslinder/crema-d",
        split="train",
        audio_field="audio",
        text_field="sentence",  # Spoken sentence
        emotion_field="label",
        gender_field=None,
        age_field=None,
        weight=2.0,
        emotion_map=CREMA_D_EMOTIONS,
    ),
    # RAVDESS: emotion, gender, text
    DatasetConfig(
        name="ravdess",
        hf_path="narad/ravdess",
        split="train",
        audio_field="audio",
        text_field="text",  # Spoken utterance
        emotion_field="emotion",
        gender_field="speaker_gender",
        age_field=None,
        weight=2.0,
    ),
    # MELD: emotion (from TV show Friends)
    # Note: Uses trust_remote_code=True
    DatasetConfig(
        name="meld",
        hf_path="ajyy/MELD_audio",
        split="train",
        audio_field="audio",
        text_field="text",  # Dialogue text
        emotion_field="emotion",
        gender_field=None,
        age_field=None,
        weight=2.0,
    ),
    # TESS: emotion + gender (all female speakers)
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
        weight=2.0,
    ),
    # SAVEE: emotion + gender (all male speakers)
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
        weight=2.0,
    ),
    # ESD English: emotion + gender (male and female)
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
        weight=2.0,
        max_samples=5000,  # Limit since it's large
    ),
    # LoquaciousSet: ASR dataset with gender metadata
    # Limited to 10k samples since it's much larger than paralinguistic datasets
    DatasetConfig(
        name="loquacious",
        hf_path="speechbrain/LoquaciousSet",
        hf_config="small",
        split="train",
        audio_field="wav",
        text_field="text",
        emotion_field=None,
        gender_field="sex",  # Has male/female labels
        age_field=None,
        weight=1.0,
        max_samples=10000,
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


def extract_metadata(sample: dict, config: DatasetConfig) -> dict:
    """Extract paralinguistic metadata from a sample."""
    metadata = {
        "text": "",
        "emotion": None,
        "gender": None,
        "age": None,
    }

    # Extract text transcription
    if config.text_field and config.text_field in sample:
        text = sample[config.text_field]
        if text:
            metadata["text"] = str(text).strip().lower()

    # Extract emotion
    if config.emotion_field and config.emotion_field in sample:
        raw_emotion = sample[config.emotion_field]
        # Map numeric labels to emotion strings if mapping provided
        if config.emotion_map and isinstance(raw_emotion, int):
            mapped = config.emotion_map.get(raw_emotion)
            metadata["emotion"] = normalize_emotion(mapped)
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

    return metadata


def build_prompt(metadata: dict) -> tuple[str, str]:
    """Build input text and user content for SIT_SSP mode.

    Returns:
        (input_text, user_content) - input_text is for metadata, user_content is full prompt
    """
    # Build paralinguistic metadata
    para_parts = []
    for key in ["age", "gender", "emotion"]:
        value = metadata.get(key)
        if value is not None:
            para_parts.append(f"{key}: {value}")

    # Format input with audio tags
    if para_parts and metadata["text"]:
        para_text = ", ".join(para_parts)
        input_text = f"<audio><meta>{para_text}</meta><text>{metadata['text']}</text></audio>"
    elif para_parts:
        para_text = ", ".join(para_parts)
        input_text = f"<audio><meta>{para_text}</meta></audio>"
    elif metadata["text"]:
        input_text = f"<audio><text>{metadata['text']}</text></audio>"
    else:
        input_text = "<audio></audio>"

    # Add instruction
    user_content = f"{input_text} {SIFT_INSTRUCTION}"

    return input_text, user_content


def process_dataset(
    config: DatasetConfig,
    pipe,
    tokenizer,
    max_samples: int | None,
    max_new_tokens: int,
    num_proc: int = 32,
) -> Dataset | None:
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

    # Filter by duration using ds.filter()
    audio_field = config.audio_field

    def is_valid_duration(example):
        try:
            audio = example.get(audio_field)
            if audio and isinstance(audio, dict) and "array" in audio:
                duration = len(audio["array"]) / audio.get("sampling_rate", 16000)
                return 1.0 < duration <= 30.0
            return True  # Keep if audio not decoded yet
        except (FileNotFoundError, OSError):
            return False

    ds = ds.filter(is_valid_duration, num_proc=num_proc, desc=f"Filtering {config.name}")

    if len(ds) == 0:
        print(f"  No valid samples found in {config.name}")
        return None

    print(f"  Loaded {len(ds)} samples")

    # Extract metadata and build prompts
    def prepare_examples(examples):
        """Extract metadata and build prompts for batch."""
        num_examples = len(examples[config.audio_field])
        prompts = []
        metadata_lists = {"text": [], "emotion": [], "gender": [], "age": []}

        for i in range(num_examples):
            # Build sample dict for extract_metadata
            sample = {k: examples[k][i] for k in examples}
            metadata = extract_metadata(sample, config)

            # Store metadata
            for key in metadata_lists:
                metadata_lists[key].append(metadata[key])

            # Build prompt
            _, user_content = build_prompt(metadata)
            messages = [
                {"role": "system", "content": SIFT_SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)

        return {
            "prompt": prompts,
            "meta_text": metadata_lists["text"],
            "meta_emotion": metadata_lists["emotion"],
            "meta_gender": metadata_lists["gender"],
            "meta_age": metadata_lists["age"],
        }

    # Apply transformations
    print("  Preparing prompts...")
    ds = ds.map(prepare_examples, batched=True, num_proc=num_proc, desc="Preparing")

    # Generate responses using pipeline with dataset (more efficient than .map())
    print("  Generating responses...")
    thinking_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Use KeyDataset to iterate efficiently over the prompt column
    responses = []
    for out in tqdm(
        pipe(
            KeyDataset(ds, "prompt"),
            generation_config=generation_config,
            return_full_text=False,
        ),
        total=len(ds),
        desc="Generating",
    ):
        text = thinking_pattern.sub("", out[0]["generated_text"]).strip()
        responses.append(text)

    ds = ds.add_column("sift_response", responses)

    # Compute duration from audio
    def compute_duration(examples):
        durations = []
        for audio in examples[config.audio_field]:
            if audio and isinstance(audio, dict) and "array" in audio:
                sr = audio.get("sampling_rate", 16000)
                durations.append(len(audio["array"]) / sr)
            else:
                durations.append(0.0)
        return {"duration": durations}

    ds = ds.map(compute_duration, batched=True, num_proc=num_proc)

    # Rename and select final columns
    ds = ds.rename_columns(
        {
            "meta_text": "text",
            "meta_emotion": "emotion",
            "meta_gender": "gender",
            "meta_age": "age",
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
        "sift_response",
        "source_dataset",
        "duration",
    ]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)

    # Cast columns to consistent types
    ds = ds.cast_column("emotion", Value("string"))
    ds = ds.cast_column("gender", Value("string"))
    ds = ds.cast_column("age", Value("string"))
    return ds.cast_column("audio", Audio(sampling_rate=16000))


app = typer.Typer(help="Generate SIFT datasets for paralinguistic training")


@app.command()
def main(
    output_repo: Annotated[
        str, typer.Option(help="HuggingFace repo ID for output")
    ] = "mazesmazes/sift-audio",
    model_name: Annotated[
        str, typer.Option(help="LLM model for response generation")
    ] = "Qwen/Qwen3-4B",
    max_samples: Annotated[
        int | None, typer.Option(help="Max samples per dataset (for testing)")
    ] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Max new tokens for generation")] = 256,
    datasets: Annotated[list[str] | None, typer.Option(help="Specific datasets to process")] = None,
    num_proc: Annotated[int, typer.Option(help="Number of CPU processes for preprocessing")] = 32,
    push_every: Annotated[int, typer.Option(help="Push to hub every N datasets")] = 1,
):
    """Generate SIFT datasets for paralinguistic training."""
    # Filter datasets if specified
    configs = DATASET_CONFIGS
    if datasets:
        configs = [c for c in configs if c.name in datasets]
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
        dtype=torch.bfloat16,
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
                max_samples=max_samples,
                max_new_tokens=max_new_tokens,
                num_proc=num_proc,
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
        typer.echo("Done!")
    else:
        typer.echo("No datasets were successfully processed.")


if __name__ == "__main__":
    app()
