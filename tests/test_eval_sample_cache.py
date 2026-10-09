"""Tests for CachedSamples: eval rows replay from disk exactly as streamed."""

import io
import itertools
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
import soundfile
from datasets import Audio, Dataset, IterableDataset

from scripts.eval.datasets import CachedSamples


def _stream(n: int) -> IterableDataset:
    """n rows of distinct encoded WAV audio, as `load_eval_dataset(decode_audio=False)` yields."""
    rows = []
    for i in range(n):
        buf = io.BytesIO()
        soundfile.write(buf, np.full(160, i / 10, dtype=np.float32), 16000, format="WAV")
        rows.append({"audio": {"bytes": buf.getvalue(), "path": None}, "text": f"row {i}"})
    ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000, decode=False))
    return ds.to_iterable_dataset()


def _read(samples: CachedSamples, n: int) -> list[dict[str, object]]:
    return list(itertools.islice(iter(samples), n))


def test_rows_are_cached_then_replayed_and_extended(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first = _read(CachedSamples(_stream(5), "audio", cache), 3)
    assert [r["text"] for r in first] == ["row 0", "row 1", "row 2"]
    assert cache.exists()

    # Next run: the 3 cached rows come from disk, rows 3-4 stream and extend the cache.
    second = _read(CachedSamples(_stream(5), "audio", cache), 5)
    assert [r["text"] for r in second] == [f"row {i}" for i in range(5)]
    assert len(Dataset.load_from_disk(str(cache))) == 5

    # Decoded audio is what the stream's own Audio feature produces.
    direct = Audio(sampling_rate=16000)
    for i, row in enumerate(second):
        expected = direct.decode_example(next(itertools.islice(iter(_stream(5)), i, None))["audio"])
        audio = row["audio"]
        assert isinstance(audio, dict)
        np.testing.assert_array_equal(audio["array"], expected["array"])
        assert audio["sampling_rate"] == 16000


def test_fully_cached_run_never_streams(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _read(CachedSamples(_stream(4), "audio", cache), 4)

    class Exploding:
        features = None

        def skip(self, n: int) -> NoReturn:
            msg = "streamed despite a full cache"
            raise AssertionError(msg)

    samples = CachedSamples(cast(IterableDataset, Exploding()), "audio", cache)
    assert [r["text"] for r in _read(samples, 4)] == [f"row {i}" for i in range(4)]
