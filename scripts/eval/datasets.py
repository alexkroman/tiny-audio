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


# Test splits ship in corpus order — grouped by speaker, chapter, meeting or
# call — so taking the first N rows of a stream samples a handful of voices
# rather than the corpus. Measured on `openslr/librispeech_asr` clean/test
# (2,620 rows, 40 speakers): the first 100 streamed rows contain **2 unique
# speaker_ids and 4 chapter_ids**, with one speaker supplying 78 of them. So a
# reported `-n 100` LibriSpeech WER was a two-speaker number, and any change
# that happened to help or hurt those two voices moved it.
#
# A shuffle buffer fixes this cheaply. It is a reservoir over a streaming
# dataset, not a true permutation: rows are drawn from a window of
# SHUFFLE_BUFFER_SIZE, and shard order is permuted too. That is enough to break
# speaker contiguity without materialising the split. The seed is fixed, so
# selection stays deterministic and run-to-run comparisons remain valid.
#
# 10,000 exceeds the row count of most test splits here, in which case the
# buffer holds the whole split and the draw is a genuine uniform sample.
SHUFFLE_BUFFER_SIZE = 10_000
SHUFFLE_SEED = 42


def load_eval_dataset(
    name: str,
    split: str,
    config_override: str | None = None,
    shuffle: bool = True,
):
    """Load any dataset by name with unified interface.

    Args:
        shuffle: Draw from a fixed-seed shuffle buffer instead of taking rows in
            corpus order. On by default — see SHUFFLE_BUFFER_SIZE for why. Pass
            False only to reproduce a pre-2026-09-18 first-N number.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    cfg = DATASET_REGISTRY[name]
    config = config_override or cfg.config

    shuffle_note = f", shuffled seed={SHUFFLE_SEED}" if shuffle else ", UNSHUFFLED first-N"
    print(f"Loading {cfg.path} (config: {config}, split: {split}{shuffle_note})...")
    ds = (
        load_dataset(cfg.path, config, split=split, streaming=True)
        if config
        else load_dataset(cfg.path, split=split, streaming=True)
    )
    ds = ds.cast_column(cfg.audio_field, Audio(sampling_rate=16000))
    if shuffle:
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER_SIZE)
    return ds
