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
from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import DatasetCard, DatasetCardData
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, pipeline
from transformers.pipelines.pt_utils import KeyDataset

# System message for generating varied instructions and responses
# Aligned with MMAU evaluation categories: Acoustic Source Inference, Temporal Reasoning, Emotion
SIFT_SYSTEM_MESSAGE = (
    "You generate training data for audio understanding models. "
    "Given audio metadata in <audio> tags, generate an INSTRUCTION and RESPONSE pair.\n\n"
    "CRITICAL: Vary your outputs significantly. Each call should feel different.\n\n"
    "INSTRUCTION TYPES (rotate between these, use MCQ format when appropriate):\n\n"
    "SPEAKER IDENTIFICATION (25%):\n"
    "- 'Is the speaker male or female?'\n"
    "- 'Identify the source of the speaking voice: (A) Man (B) Woman (C) Child'\n"
    "- 'What is the gender of the speaker?'\n"
    "- 'Who is speaking in this audio?'\n\n"
    "TRANSCRIPTION (25%):\n"
    "- 'What did they say?'\n"
    "- 'Transcribe the audio.'\n"
    "- 'What words were spoken?'\n"
    "- 'Write out what you hear.'\n\n"
    "EMOTION/TONE (20%):\n"
    "- 'What emotion is conveyed?'\n"
    "- 'How does the speaker sound: (A) Happy (B) Sad (C) Angry (D) Neutral?'\n"
    "- 'Describe the emotional tone.'\n"
    "- 'What is the speaker's mood?'\n\n"
    "SPEAKER ATTRIBUTES (15%):\n"
    "- 'Is the speaker speaking quickly or slowly?'\n"
    "- 'What is the speaking pace: (A) Slow (B) Normal (C) Fast?'\n"
    "- 'Does the speaker have an accent?'\n"
    "- 'Describe the speaker's voice characteristics.'\n\n"
    "COMBINED ANALYSIS (15%):\n"
    "- 'Describe who is speaking and how they sound.'\n"
    "- 'What can you tell about the speaker from this audio?'\n"
    "- 'Summarize everything about this audio clip.'\n\n"
    "RESPONSE GUIDELINES:\n"
    "- For MCQ instructions: respond with the letter and brief explanation, e.g., '(A) Man - The deep voice indicates a male speaker.'\n"
    "- For open questions: vary between short ('A happy woman'), medium (1 sentence), and detailed (2-3 sentences)\n"
    "- Be direct and factual. Avoid verbose explanations.\n\n"
    "Format: INSTRUCTION: <text>\nRESPONSE: <text>"
)

# Fallback instruction if parsing fails
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
    speaking_rate_field: str | None = None
    accent_field: str | None = None
    weight: float = 1.0
    max_samples: int | None = None  # Per-dataset sample limit (overrides --max-samples)
    emotion_map: dict | None = None  # Map numeric labels to emotion strings
    num_responses: int | None = None  # Per-dataset override for responses per sample
    max_down_votes: int | None = None  # Filter out samples with down_votes above this value


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
        weight=2.0,
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
        weight=2.0,
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
        weight=2.0,
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
        weight=2.0,
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
        weight=1.0,
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
        weight=1.0,
        max_samples=300000,
        max_down_votes=0,  # Only use samples with no downvotes
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

    Speaking rate thresholds based on typical speech rates:
    - Slow: < 3.0 words/sec (< 180 wpm)
    - Normal: 3.0 - 4.5 words/sec (180-270 wpm)
    - Fast: > 4.5 words/sec (> 270 wpm)
    """
    if rate is None:
        return None
    try:
        rate_float = float(rate)
        if rate_float <= 0:
            return None
        if rate_float < 3.0:
            return "slow"
        if rate_float <= 4.5:
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

    # Extract speaking rate (convert numeric to label if needed)
    if config.speaking_rate_field and config.speaking_rate_field in sample:
        raw_rate = sample[config.speaking_rate_field]
        metadata["speaking_rate"] = speaking_rate_to_label(raw_rate)

    # Extract accent
    if config.accent_field and config.accent_field in sample:
        accent = normalize_label(sample[config.accent_field])
        metadata["accent"] = accent

    return metadata


def build_prompt(metadata: dict) -> str:
    """Build input text with audio metadata for LLM to generate instruction and response.

    Returns:
        user_content - the prompt for the LLM containing audio metadata
    """
    # Build paralinguistic metadata
    para_parts = []
    for key in ["age", "gender", "emotion", "speaking_rate", "accent"]:
        value = metadata.get(key)
        if value:  # Skip empty strings and None
            # Use more natural key names in the prompt
            display_key = key.replace("_", " ")
            para_parts.append(f"{display_key}: {value}")

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

    # Ask LLM to generate both instruction and response
    return f"{input_text}\n\nGenerate a varied INSTRUCTION and RESPONSE for this audio."


def parse_instruction_response(text: str) -> tuple[str, str]:
    """Parse LLM output to extract instruction and response.

    Returns:
        (instruction, response) - parsed values or fallbacks if parsing fails
    """
    instruction = SIFT_INSTRUCTION  # Fallback
    response = text.strip()  # Fallback to full text

    # Try to parse INSTRUCTION: and RESPONSE: format
    instruction_match = re.search(
        r"INSTRUCTION:\s*(.+?)(?=\nRESPONSE:|$)", text, re.DOTALL | re.IGNORECASE
    )
    response_match = re.search(r"RESPONSE:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)

    if instruction_match:
        instruction = instruction_match.group(1).strip()
    if response_match:
        response = response_match.group(1).strip()

    return instruction, response


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
print(sample["sift_instruction"])
print(sample["sift_response"])
```

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `audio` | Audio | Audio waveform |
| `text` | string | Transcription of the audio |
| `emotion` | string | Detected emotion (if available) |
| `gender` | string | Speaker gender (if available) |
| `speaking_rate` | string | Speaking pace (if available) |
| `sift_instruction` | string | Generated instruction/question |
| `sift_response` | string | Generated response |
| `response_idx` | int | Response index (for multiple responses per audio) |
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
    num_responses: int = 1,
    num_proc: int = 32,
) -> Dataset | None:
    """Process a single dataset and generate SIFT responses using datasets.map().

    Args:
        num_responses: Number of instruction-response pairs to generate per audio sample.
                      Each audio will appear num_responses times in the output with different
                      instructions and responses.
    """
    # Use per-dataset override if specified
    effective_num_responses = (
        config.num_responses if config.num_responses is not None else num_responses
    )
    print(
        f"\nProcessing {config.name} (generating {effective_num_responses} responses per sample)..."
    )

    # Load dataset
    ds = load_dataset(
        config.hf_path,
        name=config.hf_config,
        split=config.split,
        trust_remote_code=True,
    )

    # Filter by down_votes if configured (using select for speed)
    if config.max_down_votes is not None and "down_votes" in ds.column_names:
        import numpy as np

        original_len = len(ds)
        down_votes = np.array(ds["down_votes"])
        valid_indices = np.where(down_votes <= config.max_down_votes)[0].tolist()
        ds = ds.select(valid_indices)
        print(f"  Filtered by down_votes: {original_len} -> {len(ds)}")

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
            if audio is None:
                return False
            # Check duration if audio is decoded
            if isinstance(audio, dict) and "array" in audio:
                arr = audio["array"]
                if arr is None or len(arr) == 0:
                    return False
                duration = len(arr) / audio.get("sampling_rate", 16000)
                return 1.0 < duration <= 30.0
            # If audio exists but not decoded yet, keep it
            return audio is not None
        except Exception:
            return False

    # Filter with single process
    ds = ds.filter(is_valid_duration, num_proc=1, desc=f"Filtering {config.name}")

    if len(ds) == 0:
        print(f"  No valid samples found in {config.name}")
        return None

    print(f"  Loaded {len(ds)} samples")

    # Extract metadata and build prompts
    def prepare_examples(examples):
        """Extract metadata and build prompts for batch."""
        num_examples = len(examples[config.audio_field])
        prompts = []
        metadata_lists = {
            "text": [],
            "emotion": [],
            "gender": [],
            "age": [],
            "speaking_rate": [],
            "accent": [],
        }

        for i in range(num_examples):
            # Build sample dict for extract_metadata
            sample = {k: examples[k][i] for k in examples}
            metadata = extract_metadata(sample, config)

            # Store metadata
            for key in metadata_lists:
                metadata_lists[key].append(metadata[key])

            # Build prompt (LLM will generate both instruction and response)
            user_content = build_prompt(metadata)
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
            "meta_speaking_rate": metadata_lists["speaking_rate"],
            "meta_accent": metadata_lists["accent"],
        }

    # Apply transformations
    print("  Preparing prompts...")
    # Explicitly define features to avoid schema inference issues with multiprocessing
    new_features = Features(
        {
            "prompt": Value("string"),
            "meta_text": Value("string"),
            "meta_emotion": Value("string"),
            "meta_gender": Value("string"),
            "meta_age": Value("string"),
            "meta_speaking_rate": Value("string"),
            "meta_accent": Value("string"),
        }
    )
    ds = ds.map(
        prepare_examples,
        batched=True,
        num_proc=num_proc,
        desc="Preparing",
        features=Features({**ds.features, **new_features}),
    )

    # Generate responses using pipeline with dataset (more efficient than .map())
    print("  Generating instructions and responses...")
    thinking_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
    )

    # Generate multiple responses per sample if requested
    if effective_num_responses > 1:
        # Repeat prompts for multiple generations per sample
        all_prompts = []
        sample_indices = []
        response_indices = []
        for idx, prompt in enumerate(ds["prompt"]):
            for resp_idx in range(effective_num_responses):
                all_prompts.append(prompt)
                sample_indices.append(idx)
                response_indices.append(resp_idx)

        # Create temporary dataset with repeated prompts
        from datasets import Dataset as HFDataset

        prompt_ds = HFDataset.from_dict({"prompt": all_prompts})

        instructions = []
        responses = []
        for out in tqdm(
            pipe(
                KeyDataset(prompt_ds, "prompt"),
                generation_config=generation_config,
                batch_size=batch_size,
                return_full_text=False,
            ),
            total=len(prompt_ds),
            desc=f"Generating ({effective_num_responses}x)",
        ):
            text = thinking_pattern.sub("", out[0]["generated_text"]).strip()
            instruction, response = parse_instruction_response(text)
            instructions.append(instruction)
            responses.append(response)

        # Expand dataset by selecting samples according to sample_indices
        ds = ds.select(sample_indices)
        ds = ds.add_column("sift_instruction", instructions)
        ds = ds.add_column("sift_response", responses)
        ds = ds.add_column("response_idx", response_indices)
    else:
        # Single response per sample (original behavior)
        instructions = []
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
            instruction, response = parse_instruction_response(text)
            instructions.append(instruction)
            responses.append(response)

        ds = ds.add_column("sift_instruction", instructions)
        ds = ds.add_column("sift_response", responses)
        ds = ds.add_column("response_idx", [0] * len(ds))

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

    # Remove original columns that would conflict with our renamed columns
    conflict_cols = [
        c
        for c in ["text", "emotion", "gender", "age", "speaking_rate", "accent"]
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
        "sift_instruction",
        "sift_response",
        "response_idx",
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
    ds = ds.cast_column("speaking_rate", Value("string"))
    ds = ds.cast_column("accent", Value("string"))
    ds = ds.cast_column("sift_instruction", Value("string"))
    ds = ds.cast_column("response_idx", Value("int32"))
    ds = ds.cast_column("duration", Value("float64"))
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
    batch_size: Annotated[int, typer.Option(help="Batch size for generation")] = 256,
    max_samples: Annotated[
        int | None, typer.Option(help="Max samples per dataset (for testing)")
    ] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Max new tokens for generation")] = 256,
    num_responses: Annotated[
        int, typer.Option(help="Number of instruction-response pairs per audio sample")
    ] = 3,
    datasets: Annotated[list[str] | None, typer.Option(help="Specific datasets to process")] = None,
    num_proc: Annotated[int, typer.Option(help="Number of CPU processes for preprocessing")] = 8,
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
    typer.echo(f"Responses per sample: {num_responses}")
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
                batch_size=batch_size,
                max_samples=max_samples,
                max_new_tokens=max_new_tokens,
                num_responses=num_responses,
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

        # Update dataset card
        typer.echo("Updating dataset card...")
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            info = api.dataset_info(output_repo)
            # Get all splits (existing + new)
            existing_splits = list(info.splits.keys()) if info.splits else []
            all_splits = list(set(existing_splits) | set(all_datasets.keys()))
            create_dataset_card(output_repo, all_splits)
        except Exception as e:
            typer.echo(f"Warning: Could not fetch existing splits: {e}")
            create_dataset_card(output_repo, list(all_datasets.keys()))

        typer.echo("Done!")
    else:
        typer.echo("No datasets were successfully processed.")


if __name__ == "__main__":
    app()
