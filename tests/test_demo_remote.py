"""Tests for the Space's client to `ta serve` (demo/app.py): compact uploads, query options."""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import numpy as np
import pytest
import soundfile

from demo import app


@pytest.fixture
def cd_wav(tmp_path: Path) -> Path:
    """Two seconds of 44.1 kHz stereo 16-bit WAV, the size of a typical recording."""
    path = tmp_path / "clip.wav"
    tone = np.sin(np.linspace(0, 2 * np.pi * 440 * 2, 88200)).astype(np.float32) * 0.3
    soundfile.write(path, np.stack([tone, tone], axis=1), 44100, subtype="PCM_16")
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_upload_is_16k_mono_flac_and_much_smaller(cd_wav: Path, tmp_path: Path) -> None:
    body, content_type = app.compact_audio(str(cd_wav))
    assert content_type == "audio/flac"
    assert len(body) * 5 < cd_wav.stat().st_size
    flac = tmp_path / "out.flac"
    flac.write_bytes(body)
    info = soundfile.info(flac)  # a real header, with the stream's length
    assert (info.samplerate, info.channels) == (16000, 1)
    assert info.duration == pytest.approx(2.0, abs=0.01)


def test_upload_falls_back_to_the_original_bytes(
    cd_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_ffmpeg(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", no_ffmpeg)
    assert app.compact_audio(str(cd_wav)) == (cd_wav.read_bytes(), "application/octet-stream")


def test_query_parameters_match_the_server_parser() -> None:
    params = app.query_parameters({"return_timestamps": True, "num_speakers": 3})
    assert params == {"return_timestamps": "true", "num_speakers": "3"}


class TestRemoteRunner:
    @staticmethod
    def _serve(monkeypatch: pytest.MonkeyPatch, status: int) -> list[dict[str, Any]]:
        sent: list[dict[str, Any]] = []

        def post(url: str, **kwargs: Any) -> httpx.Response:
            sent.append({"url": url, **kwargs})
            return httpx.Response(status, json={"text": "hi"}, request=httpx.Request("POST", url))

        monkeypatch.setattr(app.httpx, "post", post)

        def compact(_audio: str) -> tuple[bytes, str]:
            return b"flac", "audio/flac"

        monkeypatch.setattr(app, "compact_audio", compact)
        return sent

    def test_posts_raw_audio_with_query_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = self._serve(monkeypatch, 200)
        run = app.remote_runner("https://pod", "key")
        assert run("clip.wav", {"return_speakers": True}) == {"text": "hi"}
        assert sent[0]["content"] == b"flac"
        assert sent[0]["params"] == {"return_speakers": "true"}
        assert sent[0]["headers"] == {"Authorization": "Bearer key", "Content-Type": "audio/flac"}

    def test_payload_too_large_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._serve(monkeypatch, 413)
        with pytest.raises(gr.Error, match="too large"):
            app.remote_runner("https://pod", None)("clip.wav", {})
