"""Dataset configuration and loading for ASR evaluation."""

import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from datasets import (
    Audio,
    Dataset,
    IterableDataset,
    load_dataset,
    load_from_disk,
)


@dataclass
class DatasetConfig:
    """Unified configuration for all dataset types."""

    path: str
    audio_field: str
    text_field: str = "text"
    config: str | None = None
    default_split: str = "test"
    # References are speaker-attributed `<SPK_1>text<SPK_2>text`: evaluators
    # label speakers where they can, and metrics add cpWER.
    speakers: bool = False


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
    # Multi-speaker AMI windows (headset segments re-mixed on the meeting
    # timeline) with AMI's human transcripts as `<SPK_n>` references (a
    # prebuilt Hub dataset).
    "ami-speakers": DatasetConfig(
        path="mazesmazes/ami-speaker-windows",
        audio_field="audio",
        text_field="text",
        speakers=True,
    ),
    # Whole AMI test meetings (~20-50 min each) re-mixed from headset
    # segments, references as `<SPK_n>` turns over the meeting: cpWER here
    # measures linking speakers across an entire recording (a prebuilt Hub
    # dataset).
    "ami-speakers-long": DatasetConfig(
        path="mazesmazes/ami-speaker-meetings",
        audio_field="audio",
        text_field="text",
        speakers=True,
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
    decode_audio: bool = True,
) -> IterableDataset:
    """Load any dataset by name with unified interface.

    Args:
        shuffle: Draw from a fixed-seed shuffle buffer instead of taking rows in
            corpus order. On by default — see SHUFFLE_BUFFER_SIZE for why. Pass
            False only to reproduce a pre-2026-09-18 first-N number.
        decode_audio: False leaves audio as its encoded `{"bytes", "path"}`
            (what `CachedSamples` stores); row selection is the same either way.
    """
    if name not in DATASET_REGISTRY:
        msg = f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}"
        raise ValueError(msg)

    cfg = DATASET_REGISTRY[name]
    config = config_override or cfg.config

    shuffle_note = f", shuffled seed={SHUFFLE_SEED}" if shuffle else ", UNSHUFFLED first-N"
    print(f"Loading {cfg.path} (config: {config}, split: {split}{shuffle_note})...")
    # streaming=True with a named split always yields an IterableDataset.
    ds = cast(
        IterableDataset,
        (
            load_dataset(cfg.path, config, split=split, streaming=True)
            if config
            else load_dataset(cfg.path, split=split, streaming=True)
        ),
    )
    ds = ds.cast_column(cfg.audio_field, Audio(sampling_rate=16000, decode=decode_audio))
    if shuffle:
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER_SIZE)
    return ds


# Where `CachedSamples` keeps the rows each eval stream has yielded (gitignored).
SAMPLE_CACHE_DIR = Path(__file__).resolve().parents[2] / "datasets_cache" / "eval_samples"


class CachedSamples:
    """An eval stream that saves the rows it yields, so the next run reads them from disk.

    The shuffle seed fixes which rows a run draws, but every run still streamed
    them from the Hub, and the shuffle buffer reads SHUFFLE_BUFFER_SIZE rows of
    audio before yielding the first -- 12 datasets of that per `-d all` run,
    identical each time. This records each row as it is read, in its original
    encoded form, and on the next run replays those rows first, decoded by the
    same `Audio(sampling_rate=16000)` feature the stream uses, so the samples
    and their audio are identical. Rows past the cached prefix stream as
    before (`skip` past the prefix) and extend the cache.

    The cache is keyed by dataset, config, split and shuffle settings; delete
    SAMPLE_CACHE_DIR to pick up an upstream dataset change.
    """

    def __init__(self, raw: IterableDataset, audio_field: str, cache_dir: Path) -> None:
        self.raw = raw
        self.audio_field = audio_field
        self.cache_dir = cache_dir
        self.decoder = Audio(sampling_rate=16000)

    def _cached_rows(self) -> list[dict[str, Any]]:
        if not self.cache_dir.exists():
            return []
        rows = cast(
            "list[dict[str, Any]]", list(cast(Dataset, load_from_disk(str(self.cache_dir))))
        )
        print(f"  {len(rows)} samples cached at {self.cache_dir}")
        return rows

    def _save(self, rows: list[dict[str, Any]]) -> None:
        self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target, then swap in, so a concurrent reader never
        # sees a half-written cache.
        staging = Path(tempfile.mkdtemp(dir=self.cache_dir.parent))
        Dataset.from_list(rows, features=self.raw.features).save_to_disk(str(staging / "rows"))
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        (staging / "rows").rename(self.cache_dir)
        shutil.rmtree(staging, ignore_errors=True)

    def _decoded(self, row: dict[str, Any]) -> dict[str, Any]:
        return {**row, self.audio_field: self.decoder.decode_example(row[self.audio_field])}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        cached = self._cached_rows()
        new: list[dict[str, Any]] = []
        try:
            for row in cached:
                yield self._decoded(row)
            for row in self.raw.skip(len(cached)):
                new.append(row)
                yield self._decoded(row)
        finally:
            # Runs when the evaluator stops reading (it breaks at max_samples).
            if new:
                self._save(cached + new)


def load_eval_samples(name: str, split: str, config_override: str | None = None) -> CachedSamples:
    """`load_eval_dataset`, shuffled, with its drawn rows cached on disk (`CachedSamples`)."""
    cfg = DATASET_REGISTRY[name]
    config = config_override or cfg.config
    key = "--".join(
        [name, config or "default", split, f"seed{SHUFFLE_SEED}", f"buf{SHUFFLE_BUFFER_SIZE}"]
    )
    raw = load_eval_dataset(name, split, config_override, decode_audio=False)
    return CachedSamples(raw, cfg.audio_field, SAMPLE_CACHE_DIR / key)
