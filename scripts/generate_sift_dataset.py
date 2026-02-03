#!/usr/bin/env python3
"""Generate SIFT datasets for paralinguistic training.

This script processes audio datasets with paralinguistic metadata (emotion, gender, age)
and generates natural language responses using an LLM.

Supports three modes (following AZeroS approach):
- sift_s: Semantic only - conversational response to transcription
- sift_ssp: System + semantic + paralinguistic - empathetic response with tone awareness
- sit_ssp: System + instruction + semantic + paralinguistic - audio description/analysis

Now also generates NeuCodec codes for SIFT responses using Kokoro TTS, enabling
joint training of the audio head alongside the LLM.

Usage:
    python -m scripts.generate_sift_dataset --output-repo user/sift-dataset

    # With TTS + NeuCodec encoding:
    python -m scripts.generate_sift_dataset --output-repo user/sift-dataset --generate-codes

    # On RunPod via CLI:
    ta runpod sift <host> <port> --output-repo user/sift-dataset --generate-codes
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

import torch
import typer

# Enable fast HuggingFace transfers (must be set before importing HF libraries)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import Audio, DatasetDict, Value, concatenate_datasets, load_dataset
from huggingface_hub import DatasetCard, DatasetCardData
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, pipeline
from transformers.pipelines.pt_utils import KeyDataset


class TTSEncoder:
    """Generates speech from text using Parler-TTS Mini and encodes to NeuCodec."""

    NEUCODEC_SAMPLE_RATE = 16000
    PARLER_SAMPLE_RATE = 44100
    PARLER_MODEL_ID = "parler-tts/parler-tts-mini-v1"
    DEFAULT_VOICE_DESC = "Laura's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

    def __init__(self, device: str = "cuda", voice_description: str | None = None):
        self.device = device
        self.voice_description = voice_description or self.DEFAULT_VOICE_DESC
        self.tts_model = None
        self.tts_tokenizer = None
        self.codec_model = None
        self._resampler = None

    def _load_models(self):
        """Lazy load TTS and codec models."""
        if self.tts_model is None:
            print("Loading Parler-TTS Mini...")
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.PARLER_MODEL_ID
            ).to(self.device)
            self.tts_tokenizer = AutoTokenizer.from_pretrained(self.PARLER_MODEL_ID)
            print("Parler-TTS Mini loaded")

        if self.codec_model is None:
            print("Loading NeuCodec...")
            from neucodec import NeuCodec

            self.codec_model = NeuCodec.from_pretrained("neuphonic/neucodec")
            self.codec_model.eval()
            if self.device != "cpu" and torch.cuda.is_available():
                self.codec_model = self.codec_model.to(self.device)
            print(f"NeuCodec loaded on {self.device}")

        # Create resampler on GPU: 44.1kHz -> 16kHz
        if self._resampler is None:
            from torchaudio.transforms import Resample

            self._resampler = Resample(
                orig_freq=self.PARLER_SAMPLE_RATE,
                new_freq=self.NEUCODEC_SAMPLE_RATE,
            )
            if self.device != "cpu" and torch.cuda.is_available():
                self._resampler = self._resampler.to(self.device)

    def _generate_audio(self, text: str) -> torch.Tensor | None:
        """Generate audio for a single text using Parler-TTS.

        Returns:
            Audio tensor (1, samples) on GPU, or None on failure
        """
        if not text or not text.strip():
            return None

        try:
            # Tokenize description and text
            description_tokens = self.tts_tokenizer(self.voice_description, return_tensors="pt").to(
                self.device
            )
            text_tokens = self.tts_tokenizer(text, return_tensors="pt").to(self.device)

            # Generate audio - returns (1, samples)
            with torch.no_grad():
                return self.tts_model.generate(
                    input_ids=description_tokens.input_ids,
                    attention_mask=description_tokens.attention_mask,
                    prompt_input_ids=text_tokens.input_ids,
                    prompt_attention_mask=text_tokens.attention_mask,
                )

        except Exception as e:
            print(f"Warning: TTS failed for text '{text[:50]}...': {e}")
            return None

    def _encode_audio(self, audio: torch.Tensor) -> list[int]:
        """Encode audio tensor to NeuCodec codes on GPU.

        Args:
            audio: Audio tensor (1, samples) on GPU at 44.1kHz

        Returns:
            List of codec indices
        """
        # Resample on GPU: 44.1kHz -> 16kHz
        audio = self._resampler(audio)

        # NeuCodec expects (1, 1, samples) - batch, channels, time
        audio = audio.unsqueeze(0)  # (1, 1, samples)

        # Encode with NeuCodec on GPU
        with torch.no_grad():
            codes = self.codec_model.encode_code(audio_or_path=audio)
            # codes shape: (1, 1, seq_len) -> list
            return codes.squeeze(0).squeeze(0).cpu().tolist()

    def generate_codes(self, texts: list[str], batch_size: int = 64) -> list[list[int]]:
        """Generate NeuCodec codes for a list of texts.

        Args:
            texts: List of text strings to convert to speech codes
            batch_size: Batch size for progress reporting

        Returns:
            List of NeuCodec code lists (one per text)
        """
        self._load_models()

        all_codes: list[list[int]] = []

        for i in tqdm(range(0, len(texts), batch_size), desc="TTS + NeuCodec"):
            batch_texts = texts[i : i + batch_size]
            batch_codes: list[list[int]] = []

            for text in batch_texts:
                audio = self._generate_audio(text)
                if audio is None:
                    batch_codes.append([])
                    continue

                try:
                    codes = self._encode_audio(audio)
                    batch_codes.append(codes)
                except Exception as e:
                    print(f"Warning: Encoding failed: {e}")
                    batch_codes.append([])

            all_codes.extend(batch_codes)

        return all_codes

    def close(self):
        """Clean up resources."""
        if self.tts_model is not None:
            del self.tts_model
            self.tts_model = None
        if self.codec_model is not None:
            del self.codec_model
            self.codec_model = None
        torch.cuda.empty_cache()


class SiftMode(str, Enum):
    """SIFT generation modes following AZeroS approach."""

    SIFT_S = "sift_s"  # Semantic only - conversational response
    SIFT_SSP = "sift_ssp"  # System + semantic + paralinguistic - empathetic response
    SIT_SSP = "sit_ssp"  # System + instruction + semantic + paralinguistic - description


# System prompts for each mode (optimized for SmolLM3-3B)
SIFT_S_SYSTEM_PROMPT = (
    "You are the user's friend. Respond warmly and briefly to what they tell you."
)

SIFT_SSP_SYSTEM_PROMPT = (
    "You can hear audio. Respond empathetically to what the person says, "
    "being aware of their tone and emotion. Don't describe the audio - just respond to it naturally."
)

SIT_SSP_SYSTEM_PROMPT = (
    'Describe audio in one sentence starting with "Sounds like". '
    "Include emotion, gender, what they said (quoted), and voice quality. "
    "Example: \"Sounds like an angry man saying 'leave me alone' in a harsh voice.\""
)


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
    # CommonVoice: Large-scale multilingual dataset (English subset only)
    # Has age, gender, accent metadata (no emotion)
    DatasetConfig(
        name="commonvoice",
        hf_path="fixie-ai/common_voice_17_0",
        hf_config="en",  # English only
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


def volume_to_label(relative_db: float | None) -> str | None:
    """Convert relative_db to volume label.

    Thresholds derived from AbstractTTS dataset distributions:
    - Quiet: < -16.4 dB (below 25th percentile)
    - Normal: -16.4 to -10.0 dB (25th to 75th percentile)
    - Loud: > -10.0 dB (above 75th percentile)

    Returns None for normal volume (don't mention unremarkable features).
    """
    if relative_db is None:
        return None

    if relative_db < -16.4:
        return "quiet"
    if relative_db > -10.0:
        return "loud"
    return None  # Normal volume - don't mention


def pace_to_label(rate: str | float | None) -> str | None:
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


def normalize_meld_emotion(value: int | str | None) -> str | None:
    """Convert MELD emotion integer to string label."""
    if value is None:
        return None
    if isinstance(value, int):
        label = MELD_EMOTION_MAP.get(value)
        return normalize_emotion(label) if label else None
    return normalize_emotion(value)


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
        "pace": "",
        "accent": "",
        "volume": "",
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
    if config.pace_field and config.pace_field in sample:
        raw_rate = sample[config.pace_field]
        metadata["pace"] = pace_to_label(raw_rate)

    # Extract accent
    if config.accent_field and config.accent_field in sample:
        accent = normalize_label(sample[config.accent_field])
        metadata["accent"] = accent

    # Extract volume (relative dB)
    if config.volume_field and config.volume_field in sample:
        raw_volume = sample[config.volume_field]
        metadata["volume"] = volume_to_label(raw_volume)

    return metadata


def build_audio_context(metadata: dict) -> str:
    """Build audio context with metadata for the LLM.

    Args:
        metadata: Extracted metadata from the audio sample

    Returns:
        Audio context string with metadata in tags
    """
    # Build paralinguistic metadata
    # Order: demographics first, then voice characteristics, then content-related
    para_parts = []
    for key in [
        "age",
        "gender",
        "volume",
        "pace",
        "emotion",
        "accent",
    ]:
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


def build_prompt_for_mode(metadata: dict, mode: SiftMode, tokenizer) -> str:
    """Build the appropriate prompt for a given SIFT mode.

    Args:
        metadata: Extracted metadata from the audio sample
        mode: The SIFT mode to use
        tokenizer: The tokenizer to apply chat template

    Returns:
        Formatted prompt string ready for the LLM
    """
    if mode == SiftMode.SIFT_S:
        # Semantic only - conversational response to transcription
        messages = [
            {"role": "system", "content": SIFT_S_SYSTEM_PROMPT},
            {"role": "user", "content": metadata["text"] or ""},
        ]
    elif mode == SiftMode.SIFT_SSP:
        # System + semantic + paralinguistic - empathetic response
        audio_context = build_audio_context(metadata)
        messages = [
            {"role": "system", "content": SIFT_SSP_SYSTEM_PROMPT},
            {"role": "user", "content": audio_context},
        ]
    elif mode == SiftMode.SIT_SSP:
        # System + semantic + paralinguistic - audio description
        audio_context = build_audio_context(metadata)
        messages = [
            {"role": "system", "content": SIT_SSP_SYSTEM_PROMPT},
            {"role": "user", "content": audio_context},
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


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

This dataset contains audio samples paired with LLM-generated responses following the
AZeroS multi-mode approach. Each audio sample is processed in three different modes
to train models that can both respond conversationally AND describe/analyze audio.

## SIFT Modes

Each audio sample generates three training samples with different behaviors:

| Mode | Input Format | Expected Behavior |
|------|--------------|-------------------|
| `sift_s` | Just transcription | Conversational response (voice assistant) |
| `sift_ssp` | System + audio tags (no instruction) | Empathetic response with tone awareness |
| `sit_ssp` | System + audio tags + instruction | Audio description/analysis |

### Example

For audio of a happy woman saying "I got the job!":

- **sift_s**: "Congratulations! That's wonderful news!"
- **sift_ssp**: "That's amazing! I can hear how thrilled you are!"
- **sit_ssp**: "A young adult female excitedly announcing she got a job..."

## Splits

{splits_list}

## Usage

```python
from datasets import load_dataset

# Load a specific split
ds = load_dataset("{repo_id}", split="crema_d")

# Filter by mode
sift_s_only = ds.filter(lambda x: x["mode"] == "sift_s")
sit_ssp_only = ds.filter(lambda x: x["mode"] == "sit_ssp")

# Access a sample
sample = ds[0]
print(f"Mode: {{sample['mode']}}")
print(f"Response: {{sample['sift_response']}}")
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
| `mode` | string | SIFT mode: sift_s, sift_ssp, or sit_ssp |
| `sift_response` | string | Generated response for this mode |
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
    modes: list[SiftMode] | None = None,
    tts_encoder: TTSEncoder | None = None,
    tts_batch_size: int = 64,
):
    """Process a single dataset and generate SIFT responses for all modes.

    Generates one sample per mode for each audio clip, resulting in 3x the data
    when using all three modes (sift_s, sift_ssp, sit_ssp).

    If tts_encoder is provided, also generates NeuCodec codes for each response
    using Kokoro TTS.
    """
    if modes is None:
        modes = list(SiftMode)

    print(f"\nProcessing {config.name} with modes: {[m.value for m in modes]}...")

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
        # Shuffle before selecting to get random samples (especially important for CommonVoice)
        ds = ds.shuffle(seed=None)  # None = random seed each run
        ds = ds.select(range(effective_max))

    print(f"  Loaded {len(ds)} samples, generating {len(ds) * len(modes)} total samples")

    # Extract metadata for all samples first
    print("  Extracting metadata...")
    all_metadata = []
    for sample in tqdm(ds, desc="Extracting", total=len(ds)):
        all_metadata.append(extract_metadata(sample, config))

    # Process each mode
    mode_datasets = []
    thinking_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
    )

    for mode in modes:
        print(f"\n  Processing mode: {mode.value}")

        # Build prompts for this mode
        prompts = []
        for metadata in tqdm(all_metadata, desc=f"Building {mode.value} prompts"):
            prompts.append(build_prompt_for_mode(metadata, mode, tokenizer))

        # Create a copy of the dataset for this mode
        mode_ds = ds.add_column("prompt", prompts)
        mode_ds = mode_ds.add_column("mode", [mode.value] * len(ds))

        # Add metadata columns
        mode_ds = mode_ds.add_column("meta_text", [m["text"] or "" for m in all_metadata])
        mode_ds = mode_ds.add_column("meta_emotion", [m["emotion"] or "" for m in all_metadata])
        mode_ds = mode_ds.add_column("meta_gender", [m["gender"] or "" for m in all_metadata])
        mode_ds = mode_ds.add_column("meta_age", [m["age"] or "" for m in all_metadata])
        mode_ds = mode_ds.add_column("meta_pace", [m["pace"] or "" for m in all_metadata])
        mode_ds = mode_ds.add_column("meta_accent", [m["accent"] or "" for m in all_metadata])
        mode_ds = mode_ds.add_column("meta_volume", [m["volume"] or "" for m in all_metadata])

        # Generate responses
        print(f"  Generating {mode.value} responses...")
        responses = []
        for out in tqdm(
            pipe(
                KeyDataset(mode_ds, "prompt"),
                generation_config=generation_config,
                batch_size=batch_size,
                return_full_text=False,
            ),
            total=len(mode_ds),
            desc=f"Generating {mode.value}",
        ):
            text = thinking_pattern.sub("", out[0]["generated_text"]).strip()
            responses.append(text)

        mode_ds = mode_ds.add_column("sift_response", responses)
        mode_ds = mode_ds.add_column("source_dataset", [config.name] * len(ds))

        # Generate NeuCodec codes for responses if TTS encoder is provided
        if tts_encoder is not None:
            print(f"  Generating TTS + NeuCodec codes for {mode.value}...")
            codes = tts_encoder.generate_codes(responses, batch_size=tts_batch_size)
            mode_ds = mode_ds.add_column("response_codes", codes)

        mode_datasets.append(mode_ds)

    # Concatenate all mode datasets
    print("\n  Combining all modes...")
    combined_ds = concatenate_datasets(mode_datasets)

    # Remove original columns that would conflict with our renamed columns
    conflict_cols = [
        c
        for c in [
            "text",
            "emotion",
            "gender",
            "age",
            "pace",
            "accent",
            "volume",
        ]
        if c in combined_ds.column_names
    ]
    if conflict_cols:
        combined_ds = combined_ds.remove_columns(conflict_cols)

    # Rename meta columns to final names
    combined_ds = combined_ds.rename_columns(
        {
            "meta_text": "text",
            "meta_emotion": "emotion",
            "meta_gender": "gender",
            "meta_age": "age",
            "meta_pace": "pace",
            "meta_accent": "accent",
            "meta_volume": "volume",
        }
    )

    # Keep only needed columns
    keep_cols = [
        "audio",
        "text",
        "emotion",
        "gender",
        "age",
        "pace",
        "accent",
        "volume",
        "sift_response",
        "source_dataset",
        "mode",
        "response_codes",  # NeuCodec codes for TTS of sift_response
    ]
    remove_cols = [c for c in combined_ds.column_names if c not in keep_cols]
    if remove_cols:
        combined_ds = combined_ds.remove_columns(remove_cols)

    # Cast columns to consistent types
    combined_ds = combined_ds.cast_column("emotion", Value("string"))
    combined_ds = combined_ds.cast_column("gender", Value("string"))
    combined_ds = combined_ds.cast_column("age", Value("string"))
    combined_ds = combined_ds.cast_column("pace", Value("string"))
    combined_ds = combined_ds.cast_column("accent", Value("string"))
    combined_ds = combined_ds.cast_column("volume", Value("string"))
    combined_ds = combined_ds.cast_column("mode", Value("string"))

    print(f"  Final dataset: {len(combined_ds)} samples")
    return combined_ds.cast_column("audio", Audio(sampling_rate=16000))


app = typer.Typer(help="Generate SIFT datasets for paralinguistic training")


@app.command("add-codes")
def add_codes_to_existing(
    input_repo: Annotated[
        str, typer.Option(help="Existing SIFT dataset repo to update")
    ] = "mazesmazes/sift-audio-2",
    output_repo: Annotated[
        str, typer.Option(help="Output repo (can be same as input to update in place)")
    ] = "mazesmazes/sift-audio-2",
    splits: Annotated[
        list[str] | None, typer.Option(help="Specific splits to process (default: all)")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Batch size for TTS + NeuCodec encoding")] = 64,
    max_samples: Annotated[
        int | None, typer.Option(help="Max samples per split (for testing)")
    ] = None,
):
    """Add NeuCodec codes to an existing SIFT dataset.

    This is much faster than regenerating from scratch since it only adds
    the response_codes column using Kokoro TTS + NeuCodec encoding.

    Uses multiprocessing for parallel TTS generation.

    Usage:
        python -m scripts.generate_sift_dataset add-codes --input-repo mazesmazes/sift-audio-2
    """
    typer.echo(f"Adding NeuCodec codes to {input_repo}")
    typer.echo(f"Output: {output_repo}")
    # Initialize TTS encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_encoder = TTSEncoder(device=device)

    # Load existing dataset
    typer.echo(f"\nLoading {input_repo}...")
    from datasets import DatasetDict, load_dataset

    ds_dict = load_dataset(input_repo)

    # Filter splits if specified
    if splits:
        # Expand comma-separated values
        expanded = []
        for s in splits:
            expanded.extend(s.split(","))
        split_names = [s.strip() for s in expanded if s.strip()]
    else:
        split_names = list(ds_dict.keys())

    typer.echo(f"Processing splits: {split_names}")

    updated_splits = {}
    for split_name in split_names:
        if split_name not in ds_dict:
            typer.echo(f"  Warning: Split '{split_name}' not found, skipping")
            continue

        ds = ds_dict[split_name]
        typer.echo(f"\n  Processing {split_name}: {len(ds)} samples")

        # Limit samples if requested
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
            typer.echo(f"    Limited to {len(ds)} samples")

        # Check if response_codes already exists
        if "response_codes" in ds.column_names:
            typer.echo("    response_codes already exists, skipping")
            updated_splits[split_name] = ds
            continue

        # Get sift_response texts
        responses = ds["sift_response"]

        # Generate codes
        typer.echo("    Generating TTS + NeuCodec codes...")
        codes = tts_encoder.generate_codes(responses, batch_size=batch_size)

        # Add column
        ds = ds.add_column("response_codes", codes)
        updated_splits[split_name] = ds
        typer.echo(f"    Done: {len(ds)} samples with codes")

    # Clean up TTS workers
    tts_encoder.close()

    # Push to hub
    if updated_splits:
        typer.echo(f"\nPushing to {output_repo}...")
        # Merge with existing splits that weren't processed
        for split_name in ds_dict:
            if split_name not in updated_splits:
                updated_splits[split_name] = ds_dict[split_name]

        result_dict = DatasetDict(updated_splits)
        result_dict.push_to_hub(output_repo, private=False)
        typer.echo("Done!")
    else:
        typer.echo("No splits were processed.")


@app.command()
def main(
    output_repo: Annotated[
        str, typer.Option(help="HuggingFace repo ID for output")
    ] = "mazesmazes/sift-audio-2",
    model_name: Annotated[
        str, typer.Option(help="LLM model for response generation")
    ] = "HuggingFaceTB/SmolLM3-3B",
    batch_size: Annotated[int, typer.Option(help="Batch size for generation")] = 2048,
    max_samples: Annotated[
        int | None, typer.Option(help="Max samples per dataset (for testing)")
    ] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Max new tokens for generation")] = 256,
    datasets: Annotated[list[str] | None, typer.Option(help="Specific datasets to process")] = None,
    push_every: Annotated[int, typer.Option(help="Push to hub every N datasets")] = 1,
    modes: Annotated[
        list[str] | None,
        typer.Option(help="SIFT modes to generate (sift_s, sift_ssp, sit_ssp). Default: all three"),
    ] = None,
    generate_codes: Annotated[
        bool, typer.Option(help="Generate NeuCodec codes for responses using Parler-TTS")
    ] = False,
    tts_batch_size: Annotated[
        int, typer.Option(help="Batch size for TTS + NeuCodec encoding")
    ] = 64,
):
    """Generate SIFT datasets for paralinguistic training.

    Generates three types of training samples per audio clip:
    - sift_s: Conversational response to transcription (voice assistant behavior)
    - sift_ssp: Empathetic response with tone awareness
    - sit_ssp: Audio description/analysis (audio understanding behavior)

    With --generate-codes, also generates NeuCodec audio codes for each response
    using Parler-TTS Mini, enabling joint training of projector + audio head.
    """
    # Parse modes
    if modes:
        # Expand comma-separated values
        expanded_modes = []
        for m in modes:
            expanded_modes.extend(m.split(","))
        mode_names = [m.strip().lower() for m in expanded_modes if m.strip()]
        selected_modes = []
        for name in mode_names:
            try:
                selected_modes.append(SiftMode(name))
            except ValueError:
                typer.echo(f"Invalid mode: {name}. Valid modes: {[m.value for m in SiftMode]}")
                raise typer.Exit(1) from None
    else:
        selected_modes = list(SiftMode)

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
    typer.echo(f"Modes: {[m.value for m in selected_modes]}")
    typer.echo(f"Model: {model_name}")
    typer.echo(f"Output: {output_repo}")
    if generate_codes:
        typer.echo("TTS + NeuCodec encoding: ENABLED")

    # Initialize TTS encoder if requested
    tts_encoder = None
    if generate_codes:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_encoder = TTSEncoder(device=device)

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
        model_kwargs={"attn_implementation": "sdpa"},
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
                modes=selected_modes,
                tts_encoder=tts_encoder,
                tts_batch_size=tts_batch_size,
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
    # Clean up TTS workers
    if tts_encoder is not None:
        tts_encoder.close()

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
