"""Dataset configuration and loading for ASR evaluation."""

from dataclasses import dataclass

from datasets import Audio, load_dataset


@dataclass
class DatasetConfig:
    """Unified configuration for all dataset types."""

    name: str
    path: str
    audio_field: str
    text_field: str = "text"
    config: str | None = None
    default_split: str = "test"
    weight: float = 1.0
    # Diarization-specific
    speakers_field: str | None = None
    timestamps_start_field: str | None = None
    timestamps_end_field: str | None = None
    # Alignment-specific
    words_field: str | None = None


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    # ASR datasets
    "loquacious": DatasetConfig(
        name="loquacious",
        path="speechbrain/LoquaciousSet",
        config="small",
        audio_field="wav",
        text_field="text",
    ),
    "earnings22": DatasetConfig(
        name="earnings22",
        path="sanchit-gandhi/earnings22_robust_split",
        config="default",
        audio_field="audio",
        text_field="sentence",
    ),
    "ami": DatasetConfig(
        name="ami",
        path="edinburghcstr/ami",
        config="ihm",
        audio_field="audio",
        text_field="text",
    ),
    "gigaspeech": DatasetConfig(
        name="gigaspeech",
        path="fixie-ai/gigaspeech",
        config="dev",
        audio_field="audio",
        text_field="text",
        default_split="dev",
    ),
    "tedlium": DatasetConfig(
        name="tedlium",
        path="sanchit-gandhi/tedlium-data",
        config="default",
        audio_field="audio",
        text_field="text",
    ),
    "commonvoice": DatasetConfig(
        name="commonvoice",
        path="fixie-ai/common_voice_17_0",
        config="en",
        audio_field="audio",
        text_field="sentence",
    ),
    "peoples": DatasetConfig(
        name="peoples",
        path="fixie-ai/peoples_speech",
        config="clean",
        audio_field="audio",
        text_field="text",
    ),
    "librispeech": DatasetConfig(
        name="librispeech",
        path="openslr/librispeech_asr",
        config="clean",
        audio_field="audio",
        text_field="text",
    ),
    "librispeech-other": DatasetConfig(
        name="librispeech-other",
        path="openslr/librispeech_asr",
        config="other",
        audio_field="audio",
        text_field="text",
    ),
    # British Isles dialects - tests accent robustness (Irish, Scottish, Welsh, etc.)
    "english-dialects-irish": DatasetConfig(
        name="english-dialects-irish",
        path="ylacombe/english_dialects",
        config="irish_male",
        audio_field="audio",
        text_field="text",
        default_split="train",
    ),
    "english-dialects-scottish": DatasetConfig(
        name="english-dialects-scottish",
        path="ylacombe/english_dialects",
        config="scottish_male",
        audio_field="audio",
        text_field="text",
        default_split="train",
    ),
    "english-dialects-welsh": DatasetConfig(
        name="english-dialects-welsh",
        path="ylacombe/english_dialects",
        config="welsh_male",
        audio_field="audio",
        text_field="text",
        default_split="train",
    ),
    "english-dialects-northern": DatasetConfig(
        name="english-dialects-northern",
        path="ylacombe/english_dialects",
        config="northern_male",
        audio_field="audio",
        text_field="text",
        default_split="train",
    ),
    # Edinburgh Accented Speech - 25 different accents
    "edacc": DatasetConfig(
        name="edacc",
        path="edinburghcstr/edacc",
        audio_field="audio",
        text_field="text",
        default_split="test",
    ),
    # Spontaneous conversational speech - disfluencies, hesitations, natural speech
    "switchboard": DatasetConfig(
        name="switchboard",
        path="hhoangphuoc/switchboard",
        audio_field="audio",
        text_field="transcript",
        default_split="test",
    ),
    # Diarization datasets
    "callhome": DatasetConfig(
        name="callhome",
        path="talkbank/callhome",
        config="eng",
        audio_field="audio",
        text_field="text",
        default_split="data",
        speakers_field="speakers",
        timestamps_start_field="timestamps_start",
        timestamps_end_field="timestamps_end",
    ),
    # Alignment datasets
    "librispeech-alignments": DatasetConfig(
        name="librispeech-alignments",
        path="gilkeyio/librispeech-alignments",
        audio_field="audio",
        text_field="transcript",
        default_split="dev_clean",
        words_field="words",
    ),
}

DIARIZATION_DATASETS = {"callhome"}
ALIGNMENT_DATASETS = {"librispeech-alignments"}


def load_eval_dataset(
    name: str, split: str, config_override: str | None = None, decode_audio: bool = True
):
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
    audio_opts = Audio(sampling_rate=16000) if decode_audio else Audio(decode=False)
    return ds.cast_column(cfg.audio_field, audio_opts)
