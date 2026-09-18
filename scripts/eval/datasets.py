"""Dataset configuration and loading for ASR evaluation."""

from dataclasses import dataclass

from datasets import Audio, load_dataset


@dataclass
class DatasetConfig:
    """Unified configuration for all dataset types."""

    path: str
    audio_field: str
    text_field: str = "text"
    config: str | None = None
    default_split: str = "test"


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    # ASR datasets
    "loquacious": DatasetConfig(
        path="speechbrain/LoquaciousSet",
        config="small",
        audio_field="wav",
        text_field="text",
    ),
    "earnings22": DatasetConfig(
        path="sanchit-gandhi/earnings22_robust_split",
        config="default",
        audio_field="audio",
        text_field="sentence",
    ),
    "ami": DatasetConfig(
        path="edinburghcstr/ami",
        config="ihm",
        audio_field="audio",
        text_field="text",
    ),
    "ami-sdm": DatasetConfig(
        path="edinburghcstr/ami",
        config="sdm",
        audio_field="audio",
        text_field="text",
    ),
    "gigaspeech": DatasetConfig(
        path="fixie-ai/gigaspeech",
        config="dev",
        audio_field="audio",
        text_field="text",
        default_split="dev",
    ),
    "spgispeech": DatasetConfig(
        path="kensho/spgispeech",
        config="test",
        audio_field="audio",
        text_field="transcript",
        default_split="test",
    ),
    "tedlium": DatasetConfig(
        path="sanchit-gandhi/tedlium-data",
        config="default",
        audio_field="audio",
        text_field="text",
    ),
    "commonvoice": DatasetConfig(
        path="fixie-ai/common_voice_17_0",
        config="en",
        audio_field="audio",
        text_field="sentence",
    ),
    "peoples": DatasetConfig(
        path="fixie-ai/peoples_speech",
        config="clean",
        audio_field="audio",
        text_field="text",
    ),
    "voxpopuli": DatasetConfig(
        path="facebook/voxpopuli",
        config="en",
        audio_field="audio",
        text_field="normalized_text",
    ),
    "librispeech": DatasetConfig(
        path="openslr/librispeech_asr",
        config="clean",
        audio_field="audio",
        text_field="text",
    ),
    "librispeech-other": DatasetConfig(
        path="openslr/librispeech_asr",
        config="other",
        audio_field="audio",
        text_field="text",
    ),
    "expresso": DatasetConfig(
        path="ylacombe/expresso",
        audio_field="audio",
        text_field="text",
        default_split="train",  # Only split available
    ),
}


def load_eval_dataset(name: str, split: str, config_override: str | None = None):
    """Load any dataset by name with unified interface."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    cfg = DATASET_REGISTRY[name]
    config = config_override or cfg.config

    print(f"Loading {cfg.path} (config: {config}, split: {split})...")
    ds = (
        load_dataset(cfg.path, config, split=split, streaming=True)
        if config
        else load_dataset(cfg.path, split=split, streaming=True)
    )
    return ds.cast_column(cfg.audio_field, Audio(sampling_rate=16000))
