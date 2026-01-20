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

import argparse
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# System message for paralinguistic-aware responses
SIFT_SYSTEM_MESSAGE = (
    "You are a powerful virtual human who is capable of perceiving both text and speech inputs "
    "and generate precise natural responses. "
    "Speech inputs will be wrapped by <audio> and </audio> tags, containing both the text transcription "
    "and paralinguistic information. "
    "You must always pretend that you can indeed hear the input audios. "
    "NEVER mention that any metadata is provided through texts, and only use them in your response when necessary."
)

# Instruction for SIT mode
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
    weight: float = 1.0
    max_samples: int | None = None  # Per-dataset sample limit (overrides --max-samples)


# HuggingFace datasets with paralinguistic labels (English only)
DATASET_CONFIGS = [
    # CREMA-D: emotion, gender, age, text (sentence)
    DatasetConfig(
        name="crema-d",
        hf_path="myleslinder/crema-d",
        split="train",
        audio_field="audio",
        text_field="sentence",  # Spoken sentence
        emotion_field="label",
        gender_field="Sex",
        age_field="Age",
        weight=2.0,
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
    # TESS: emotion, age, text
    DatasetConfig(
        name="tess",
        hf_path="myleslinder/tess",
        split="train",
        audio_field="audio",
        text_field="text",  # "Say the word X"
        emotion_field="label",
        gender_field=None,  # All female speakers
        age_field="speaker_age",  # Age in years (e.g., 64)
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


def age_to_group(age: str | int | None) -> str:
    """Convert numeric age to age group."""
    if age is None:
        return "?"
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
        return "?"
    except (ValueError, TypeError):
        # Already a string like "young adult"
        if isinstance(age, str) and age.lower() not in ("", "na", "null", "unk", "unknown", "nan"):
            return age.lower()
        return "?"


def normalize_label(value: str | None) -> str:
    """Normalize a label value, returning '?' for missing/invalid values."""
    if value is None:
        return "?"
    value_str = str(value).lower().strip()
    if value_str in ("", "na", "null", "unk", "unknown", "nan", "none"):
        return "?"
    return value_str


def extract_metadata(sample: dict, config: DatasetConfig) -> dict:
    """Extract paralinguistic metadata from a sample."""
    metadata = {
        "text": "",
        "emotion": "?",
        "gender": "?",
        "age": "?",
    }

    # Extract text transcription
    if config.text_field and config.text_field in sample:
        text = sample[config.text_field]
        if text:
            metadata["text"] = str(text).strip().lower()

    # Extract emotion
    if config.emotion_field and config.emotion_field in sample:
        metadata["emotion"] = normalize_label(sample[config.emotion_field])

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
        value = metadata.get(key, "?")
        if value != "?":
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


@torch.no_grad()
def generate_responses(
    samples: list[dict],
    configs: list[DatasetConfig],
    model,
    tokenizer,
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate SIFT responses for a batch of samples."""
    # Build prompts
    prompts = []
    for sample, config in zip(samples, configs):
        metadata = extract_metadata(sample, config)
        _, user_content = build_prompt(metadata)

        messages = [
            {"role": "system", "content": SIFT_SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only new tokens
    responses = []
    for input_ids, output_ids in zip(inputs.input_ids, outputs):
        new_tokens = output_ids[len(input_ids) :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Check for truncation
        eos_token_id = tokenizer.eos_token_id
        eos_ids = set(eos_token_id) if isinstance(eos_token_id, list) else {eos_token_id}

        if not any(tok_id in eos_ids for tok_id in new_tokens.tolist()):
            response += " <|truncated|>"

        responses.append(response.strip())

    return responses


def process_dataset(
    config: DatasetConfig,
    model,
    tokenizer,
    batch_size: int,
    max_samples: int | None,
    max_new_tokens: int,
) -> Dataset | None:
    """Process a single dataset and generate SIFT responses."""
    print(f"\nProcessing {config.name}...")

    # Load dataset
    if config.hf_config:
        ds = load_dataset(config.hf_path, config.hf_config, split=config.split, streaming=True)
    else:
        ds = load_dataset(config.hf_path, split=config.split, streaming=True)

    # Collect samples
    # Per-dataset max_samples takes priority over global max_samples
    effective_max = config.max_samples if config.max_samples is not None else max_samples
    samples = []
    for sample in tqdm(ds, desc=f"Loading {config.name}"):
        # Filter by duration if audio is decoded
        audio = sample.get(config.audio_field)
        if audio and isinstance(audio, dict) and "array" in audio:
            duration = len(audio["array"]) / audio.get("sampling_rate", 16000)
            if not (1.0 < duration <= 30.0):
                continue

        samples.append(sample)
        if effective_max and len(samples) >= effective_max:
            break

    if not samples:
        print(f"  No valid samples found in {config.name}")
        return None

    print(f"  Loaded {len(samples)} samples")

    # Generate responses in batches
    all_responses = []
    all_metadata = []

    for i in tqdm(range(0, len(samples), batch_size), desc=f"Generating {config.name}"):
        batch = samples[i : i + batch_size]
        batch_configs = [config] * len(batch)

        responses = generate_responses(
            batch,
            batch_configs,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
        )

        for sample, response in zip(batch, responses):
            metadata = extract_metadata(sample, config)
            all_responses.append(response)
            all_metadata.append(metadata)

        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build output dataset
    output_data = {
        "text": [m["text"] for m in all_metadata],
        "emotion": [m["emotion"] for m in all_metadata],
        "gender": [m["gender"] for m in all_metadata],
        "age": [m["age"] for m in all_metadata],
        "sift_response": all_responses,
        "source_dataset": [config.name] * len(all_responses),
    }

    # Include audio if available
    if config.audio_field:
        output_data["audio"] = [s.get(config.audio_field) for s in samples[: len(all_responses)]]

    return Dataset.from_dict(output_data)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SIFT datasets for paralinguistic training"
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        default="mazesmazes/sift-audio",
        help="HuggingFace repo ID for output",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="LLM model for response generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generation (32 recommended for A40 48GB)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (for testing)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to process (default: all)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster inference",
    )
    parser.add_argument(
        "--push-every",
        type=int,
        default=1,
        help="Push to hub every N datasets (default: 1)",
    )
    args = parser.parse_args()

    # Filter datasets if specified
    configs = DATASET_CONFIGS
    if args.datasets:
        configs = [c for c in configs if c.name in args.datasets]
        if not configs:
            print(f"No matching datasets found. Available: {[c.name for c in DATASET_CONFIGS]}")
            return

    print(f"Processing {len(configs)} datasets")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_repo}")

    # Load model
    print(f"\nLoading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Process each dataset
    all_datasets = {}
    datasets_processed = 0

    for config in configs:
        try:
            ds = process_dataset(
                config=config,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
            )

            if ds is not None:
                all_datasets[config.name] = ds
                datasets_processed += 1

                # Push periodically
                if datasets_processed % args.push_every == 0:
                    print(f"\nPushing {len(all_datasets)} splits to {args.output_repo}...")
                    dataset_dict = DatasetDict(all_datasets)
                    dataset_dict.push_to_hub(args.output_repo, private=False)

        except Exception as e:
            print(f"Error processing {config.name}: {e}")
            continue

    # Final push
    if all_datasets:
        print(f"\nFinal push: {len(all_datasets)} splits to {args.output_repo}...")
        dataset_dict = DatasetDict(all_datasets)
        dataset_dict.push_to_hub(args.output_repo, private=False)
        print("Done!")
    else:
        print("No datasets were successfully processed.")


if __name__ == "__main__":
    main()
