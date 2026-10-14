"""Tests for scripts.build_notsofar_train: NOTSOFAR transcripts -> multi-speaker windows."""

import io
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile

from scripts import build_notsofar_train as b


def _seg(speaker: str, start: float, end: float, text: str, timing: Any = None) -> dict[str, Any]:
    return {
        "speaker_id": speaker,
        "start_time": start,
        "end_time": end,
        "text": text,
        "word_timing": timing or [],
    }


def test_segment_words_keep_case_and_take_times() -> None:
    seg = _seg(
        "A",
        1.0,
        3.0,
        "Um, hi <PName>Bob</PName> <FILL/> there ?",
        [["um", 1.0, 1.2], ["hi", 1.3, 1.5], ["bob", 1.6, 1.9], ["there", 2.0, 2.4]],
    )
    words = b.segment_words(seg, 7)
    assert [w.text for w in words] == ["Um,", "hi", "Bob", "there?"]
    assert [(w.start, w.end) for w in words] == [(1.0, 1.2), (1.3, 1.5), (1.6, 1.9), (2.0, 2.4)]
    assert all(w.timed and w.segment == 7 for w in words)


def test_segment_words_interpolate_unmatched() -> None:
    seg = _seg("A", 0.0, 2.0, "so Kaley said", [["so", 0.1, 0.3], ["said", 1.0, 1.4]])
    words = b.segment_words(seg, 0)
    assert [(w.text, w.timed) for w in words] == [("so", True), ("Kaley", False), ("said", True)]
    assert (words[1].start, words[1].end) == (0.3, 1.0)  # between its neighbours


def test_unknown_spans_timed_or_whole_segment() -> None:
    timed = _seg("A", 0.0, 3.0, "so <UNKNOWN/> there", [["<UNKNOWN/>", 1.0, 1.5]])
    untimed = _seg("A", 4.0, 6.0, "so <UNKNOWN/> there")
    assert b.unknown_spans(timed) == [(1.0, 1.5)]
    assert b.unknown_spans(untimed) == [(4.0, 6.0)]
    assert b.unknown_spans(_seg("A", 0.0, 1.0, "fine")) == []


def test_window_text_keeps_both_speakers_grouped_by_segment() -> None:
    transcript = [
        _seg("B", 1.0, 2.0, "yeah right", [["yeah", 1.0, 1.4], ["right", 1.5, 1.9]]),
        _seg(
            "A",
            0.0,
            3.0,
            "I think we should go",
            [
                ["i", 0.0, 0.2],
                ["think", 0.3, 0.6],
                ["we", 1.2, 1.3],
                ["should", 1.6, 2.0],
                ["go", 2.6, 3.0],
            ],
        ),
    ]
    words, _ = b.meeting_words(transcript)
    # A's segment starts first, so its words come first even though B talks inside it;
    # "go" is 40% inside [0, 2.8) and left out.
    text, kept, timed = b.window_text(words, 0.0, 2.8)
    assert text == "I think we should yeah right"
    assert (kept, timed) == (6, 6)


def test_window_rows_drop_unknown_and_empty_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = np.random.default_rng(0).uniform(-0.1, 0.1, 30 * 16000).astype(np.float32)
    transcript = [
        _seg("A", 1.0, 3.0, "hello there", [["hello", 1.0, 1.5], ["there", 2.0, 2.5]]),
        _seg("B", 11.0, 12.0, "<UNKNOWN/>", [["<UNKNOWN/>", 11.0, 12.0]]),
    ]
    words, blocked = b.meeting_words(transcript)

    def fixed_windows(_a: Any, _sr: int, _max: float, _min: float) -> list[tuple[int, int]]:
        return [(0, 10 * 16000), (10 * 16000, 20 * 16000), (20 * 16000, 30 * 16000)]

    monkeypatch.setattr(b, "chunk_bounds", fixed_windows)
    rows = b.window_rows(words, blocked, audio, "MTG_1", "sc_x", random.Random(0))
    # Window 2 touches <UNKNOWN/>; window 3 holds no words.
    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "hello there"
    assert (row["meeting_id"], row["device"], row["audio_duration"]) == ("MTG_1", "sc_x", 10.0)
    cut, sr = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    assert sr == 16000
    np.testing.assert_allclose(cut, audio[: 10 * 16000], atol=1e-4)


def test_window_lengths_follow_the_inference_chunker() -> None:
    audio = np.random.default_rng(1).uniform(-0.1, 0.1, 120 * 16000).astype(np.float32)
    words = [b.Word(f"w{k}", k + 0.1, k + 0.4, True, k) for k in range(120)]
    rows = b.window_rows(words, [], audio, "MTG_1", "sc_x", random.Random(3))
    durations = [r["audio_duration"] for r in rows]
    assert sum(durations) == pytest.approx(120.0)
    assert max(durations) <= 18.0
    assert min(durations[:-1]) >= 3.0  # the last window is whatever is left


def test_live_recordings_skip_silent_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, value in [("dead", 0.0), ("one", 0.1), ("two", 0.2)]:
        soundfile.write(tmp_path / f"{name}.wav", np.full(1600, value, np.float32), 16000)

    def fake_download(path: str) -> Path:
        return tmp_path / f"{path}.wav"

    monkeypatch.setattr(b, "download_file", fake_download)
    picked = b.live_recordings(["dead", "one", "two"], 2, random.Random(0))
    assert sorted(p for p, _ in picked) == ["one", "two"]
