"""NemotronDiarizer's post-processing: speaker selection and segments."""

from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pytest

from tiny_audio.asr_pipeline import chunk_bounds
from tiny_audio.diarization import NemotronDiarizer


def activity(
    spans: list[tuple[int, float, float, float]], seconds: float = 4.0
) -> npt.NDArray[np.float32]:
    act = np.zeros((int(seconds * 100), 8), dtype=np.float32)
    for col, start, end, p in spans:
        act[int(start * 100) : int(end * 100), col] = p
    return act


class TestTopSpeakers:
    def test_every_column_that_speaks_in_arrival_order(self) -> None:
        act = activity([(4, 0, 1, 0.9), (1, 1, 2, 0.9), (6, 2, 3, 0.3)])  # col 6 never > 0.5
        assert NemotronDiarizer.top_speakers(act) == [1, 4]

    @pytest.mark.parametrize("kw", [{"num_speakers": 1}, {"max_speakers": 1}])
    def test_cap_keeps_most_speech(self, kw: dict[str, int]) -> None:
        act = activity([(0, 0, 0.5, 0.9), (2, 0.5, 4, 0.9)])
        assert NemotronDiarizer.top_speakers(act, **kw) == [2]

    def test_silence_still_returns_one_speaker(self) -> None:
        assert NemotronDiarizer.top_speakers(activity([])) == [0]


class TestSegments:
    def test_overlap_produces_overlapping_segments(self) -> None:
        act = activity([(0, 0, 2, 0.9), (1, 1.5, 3, 0.9)])
        segs = NemotronDiarizer.segments(act, [0, 1])
        assert segs == [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": 2.0},
            {"speaker": "SPEAKER_1", "start": 1.5, "end": 3.0},
        ]


class TestChunkBounds:
    def test_short_audio_is_one_chunk(self) -> None:
        assert chunk_bounds(np.ones(10 * 16000, np.float32), 16000) == [(0, 160000)]

    def test_long_audio_chunks_stay_within_training_length(self) -> None:
        audio = np.random.default_rng(0).normal(0, 0.1, 120 * 16000).astype(np.float32)
        bounds = chunk_bounds(audio, 16000)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == len(audio)
        assert all(a[1] == b[0] for a, b in pairwise(bounds))
        assert all((e - s) / 16000 <= 18.0 for s, e in bounds)
        assert all((e - s) / 16000 >= 8.0 for s, e in bounds[:-1])
