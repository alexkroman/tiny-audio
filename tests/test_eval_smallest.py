"""Tests for SmallestEvaluator (Smallest.ai Pulse pre-recorded STT)."""

import io
import json
import urllib.error
from email.message import Message
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.eval.evaluators.asr import SmallestEvaluator


def _response(payload: dict[str, str]) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    return resp


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("url", code, "err", Message(), io.BytesIO(b"rate limited"))


@pytest.fixture
def audio() -> dict[str, object]:
    return {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # tenacity sleeps between attempts; skip the wait so retry tests stay fast.
    def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("tenacity.nap.time.sleep", no_sleep)


class TestSmallestEvaluator:
    def test_returns_transcription_and_sends_wav(
        self, monkeypatch: pytest.MonkeyPatch, audio: dict[str, object]
    ) -> None:
        urlopen = MagicMock(return_value=_response({"status": "success", "transcription": "hi"}))
        monkeypatch.setattr("urllib.request.urlopen", urlopen)

        text, elapsed, confidence = SmallestEvaluator(api_key="sk").transcribe(audio)

        assert (text, confidence) == ("hi", None)
        assert elapsed >= 0
        request = urlopen.call_args.args[0]
        assert "model=pulse" in request.full_url
        assert "language=en" in request.full_url
        assert request.get_header("Authorization") == "Bearer sk"
        assert request.get_header("Content-type") == "audio/wav"
        assert request.data[:4] == b"RIFF"

    def test_retries_rate_limit(
        self, monkeypatch: pytest.MonkeyPatch, audio: dict[str, object]
    ) -> None:
        ok = _response({"status": "success", "transcription": "ok"})
        urlopen = MagicMock(side_effect=[_http_error(429), _http_error(503), ok])
        monkeypatch.setattr("urllib.request.urlopen", urlopen)

        text, _, _ = SmallestEvaluator(api_key="sk").transcribe(audio)

        assert text == "ok"
        assert urlopen.call_count == 3

    def test_client_error_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, audio: dict[str, object]
    ) -> None:
        urlopen = MagicMock(side_effect=_http_error(401))
        monkeypatch.setattr("urllib.request.urlopen", urlopen)

        with pytest.raises(RuntimeError, match="HTTP 401"):
            SmallestEvaluator(api_key="sk").transcribe(audio)
        assert urlopen.call_count == 1

    def test_non_success_payload_raises(
        self, monkeypatch: pytest.MonkeyPatch, audio: dict[str, object]
    ) -> None:
        urlopen = MagicMock(return_value=_response({"status": "error", "message": "bad"}))
        monkeypatch.setattr("urllib.request.urlopen", urlopen)

        with pytest.raises(RuntimeError, match="non-success"):
            SmallestEvaluator(api_key="sk").transcribe(audio)
