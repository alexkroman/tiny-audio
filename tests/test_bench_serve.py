"""Tests for scripts/bench_serve.py against a fake `ta serve` (httpx mock transport)."""

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile
import typer

from scripts import bench_serve


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    soundfile.write(str(path), np.zeros(16000 * 2, dtype=np.float32), 16000)
    return path


def _fake_server(
    monkeypatch: pytest.MonkeyPatch, texts: list[str], fail_on: frozenset[int] = frozenset()
) -> list[dict[str, Any]]:
    """Route bench_serve's AsyncClient to a handler that answers like `ta serve`.

    The POSTs numbered in `fail_on` (1-based) get the RunPod proxy's 524.
    """
    posted: list[dict[str, Any]] = []
    stats = {"batches": 0, "chunks": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/stats":
            return httpx.Response(200, json=stats)
        posted.append(json.loads(request.content))
        if len(posted) in fail_on:
            return httpx.Response(524, text="A timeout occurred")
        stats["batches"] += 1
        stats["chunks"] += 2  # two chunks per request, one batch each
        return httpx.Response(200, json={"text": texts[(len(posted) - 1) % len(texts)]})

    real_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        assert kwargs.get("trust_env") is False  # never through a local HTTPS_PROXY
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(bench_serve.httpx, "AsyncClient", client)
    return posted


def test_reports_each_concurrency_level(
    monkeypatch: pytest.MonkeyPatch, wav: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    posted = _fake_server(monkeypatch, ["hello"])
    bench_serve.bench(
        url="http://server",
        audio=wav,
        concurrency=[1, 2],
        requests=3,
        timestamps=True,
        speakers=False,
    )
    out = capsys.readouterr().out
    assert len(posted) == 1 + 3 + 3  # warm-up, then each level
    assert posted[0]["parameters"] == {"return_timestamps": True}
    rows = [line.split() for line in out.splitlines() if line.strip()[:1].isdigit()]
    assert [(row[0], row[1], row[-1]) for row in rows] == [("1", "3", "2.0"), ("2", "3", "2.0")]
    assert "2.0 s of audio" in out
    assert "text: hello" in out
    assert "different transcripts" not in out


def test_flags_differing_transcripts(
    monkeypatch: pytest.MonkeyPatch, wav: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_server(monkeypatch, ["a", "b"])
    bench_serve.bench(
        url="http://server", audio=wav, concurrency=[2], requests=2, timestamps=False, speakers=True
    )
    assert "2 different transcripts for identical audio" in capsys.readouterr().out


def test_divergence_reports_wer_against_most_common() -> None:
    report = bench_serve.divergence(["a b c d", "a b c d", "a b c d", "a b x d"])
    assert "2 different transcripts" in report
    assert "mean WER 6.2%" in report  # one of four is 1/4 words off
    assert "max 25.0%" in report


def test_default_requests_saturate_each_level(monkeypatch: pytest.MonkeyPatch, wav: Path) -> None:
    posted = _fake_server(monkeypatch, ["hello"])
    bench_serve.bench(
        url="http://server",
        audio=wav,
        concurrency=[2, 16],
        requests=None,
        timestamps=False,
        speakers=False,
    )
    assert len(posted) == 1 + 8 + 32  # warm-up, max(8, 2*2), max(8, 2*16)


def test_failed_request_is_reported_not_fatal(
    monkeypatch: pytest.MonkeyPatch, wav: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One failure doesn't cancel the rest of the level (the server would see disconnects)."""
    posted = _fake_server(monkeypatch, ["hello"], fail_on=frozenset({3}))
    bench_serve.bench(
        url="http://server",
        audio=wav,
        concurrency=[2],
        requests=4,
        timestamps=False,
        speakers=False,
    )
    out = capsys.readouterr().out
    assert len(posted) == 1 + 4  # every request still sent
    [row] = [line.split() for line in out.splitlines() if line.strip()[:1].isdigit()]
    assert row[:3] == ["2", "4", "1"]  # conc, reqs, fail
    assert "HTTP 524 (RunPod proxy timeout" in out


def test_failed_warm_up_exits(monkeypatch: pytest.MonkeyPatch, wav: Path) -> None:
    _fake_server(monkeypatch, ["hello"], fail_on=frozenset({1}))
    with pytest.raises(typer.Exit):
        bench_serve.bench(
            url="http://server",
            audio=wav,
            concurrency=[1],
            requests=1,
            timestamps=False,
            speakers=False,
        )
