"""Tests for scripts/serve.py: the GPU batcher, request parsing and the HTTP app."""

import asyncio
import base64
import io
import threading
import time
from collections.abc import Callable
from typing import Any, cast

import httpx
import numpy as np
import numpy.typing as npt
import pytest
import soundfile
import torch
from starlette.applications import Starlette
from starlette.types import Message

from scripts import serve
from scripts.serve import ChunkBatcher, RequestError, _Job, decode_request, parse_parameters
from tiny_audio.asr_pipeline import ASRPipeline
from tiny_audio.asr_types import PreparedChunk
from tiny_audio.diarization import NemotronDiarizer


def _chunk(frames: int) -> PreparedChunk:
    """Build a zero-feature prepared chunk whose length is ``frames``."""
    return {
        "input_features": torch.zeros(1, frames, 4),
        "attention_mask": torch.ones(1, frames, dtype=torch.long),
    }


def _frames(prepared: list[PreparedChunk]) -> list[int]:
    """Return the frame length of each chunk, so a test can see what a batch held."""
    return [int(p["attention_mask"].shape[-1]) for p in prepared]


def _poll[T](fetch: Callable[[], T], done: Callable[[T], bool], what: str) -> T:
    """`fetch()` every 10 ms until `done` accepts it (up to 5 s), and return that value.

    It sleeps before each check rather than only when a check fails, so the same
    lines run whatever the thread timing: the per-file coverage floor for this
    file was flaky when a fast CI runner never reached the sleep.
    """
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        time.sleep(0.01)
        if done(value := fetch()):
            return value
    pytest.fail(f"never: {what}")


class TestChunkBatcher:
    """ChunkBatcher batches queued chunks, keeps per-request order and fans out errors."""

    def test_results_come_back_in_order(self) -> None:
        batcher = ChunkBatcher(lambda batch: [str(n) for n in _frames(batch)], max_batch_size=4)
        batcher.start()
        try:
            assert batcher.run([_chunk(3), _chunk(1), _chunk(2)]) == ["3", "1", "2"]
        finally:
            batcher.close()

    def test_queued_chunks_share_batches(self) -> None:
        """Chunks that queue while a batch runs go out together in the next one."""
        entered = threading.Event()
        release = threading.Event()
        sizes: list[int] = []

        def generate(batch: list[PreparedChunk]) -> list[str]:
            entered.set()
            release.wait(timeout=5)  # hold the first batch open while the rest queue
            sizes.append(len(batch))
            return ["x"] * len(batch)

        batcher = ChunkBatcher(generate, max_batch_size=8)
        batcher.start()
        results: list[list[str]] = []

        def request() -> None:
            results.append(batcher.run([_chunk(10)]))

        threads = [threading.Thread(target=request) for _ in range(5)]
        try:
            threads[0].start()
            assert entered.wait(timeout=5)  # the GPU thread is inside the first batch
            for t in threads[1:]:
                t.start()
            # The blocked GPU thread can't drain them, so all four stay queued.
            _poll(batcher._inbox.qsize, lambda queued: queued >= 4, "four requests queued")
            release.set()
            for t in threads:
                t.join(timeout=5)
        finally:
            release.set()
            batcher.close()
        assert sizes == [1, 4]
        assert len(results) == 5
        assert batcher.stats.batches == 2
        assert batcher.stats.chunks == 5
        assert batcher.stats.largest_batch == 4

    def test_batch_is_oldest_plus_nearest_lengths(self) -> None:
        batcher = ChunkBatcher(lambda _batch: [], max_batch_size=3)
        pending = [serve._Job(_chunk(n)) for n in (100, 10, 95, 300, 104)]
        batch = batcher._take_batch(pending)
        assert [job.frames for job in batch] == [100, 104, 95]  # oldest first, then nearest
        assert [job.frames for job in pending] == [10, 300]

    def test_generate_error_reaches_every_waiter(self) -> None:
        def boom(_batch: list[PreparedChunk]) -> list[str]:
            msg = "CUDA out of memory"
            raise RuntimeError(msg)

        batcher = ChunkBatcher(boom, max_batch_size=4)
        batcher.start()
        try:
            with pytest.raises(RuntimeError, match="out of memory"):
                batcher.run([_chunk(1), _chunk(2)])
        finally:
            batcher.close()

    def test_an_interrupt_reaches_every_waiter_and_keeps_unwinding(self) -> None:
        def interrupted(_batch: list[PreparedChunk]) -> list[str]:
            raise KeyboardInterrupt

        batcher = ChunkBatcher(interrupted, max_batch_size=4)
        jobs = [_Job(_chunk(1)), _Job(_chunk(2))]
        with pytest.raises(KeyboardInterrupt):
            batcher._execute(jobs)
        assert all(isinstance(job.future.exception(), KeyboardInterrupt) for job in jobs)


class TestParameters:
    """parse_parameters accepts typed JSON, coerces query strings and rejects the rest."""

    def test_json_types_pass(self) -> None:
        params = parse_parameters({"return_speakers": True, "num_speakers": 2}, from_query=False)
        assert params == {"return_speakers": True, "num_speakers": 2}

    def test_query_strings_are_coerced(self) -> None:
        params = parse_parameters(
            {"return_timestamps": "true", "max_speakers": "3"}, from_query=True
        )
        assert params == {"return_timestamps": True, "max_speakers": 3}

    @pytest.mark.parametrize(
        "raw",
        [
            {"user_prompt": "x"},  # server-side only
            {"num_speakers": True},  # bool is not a count
            {"return_speakers": "yes"},  # strings only coerce from a query string
        ],
    )
    def test_rejects(self, raw: dict[str, Any]) -> None:
        with pytest.raises(RequestError):
            parse_parameters(raw, from_query=False)


class TestDecodeRequest:
    """decode_request splits a JSON or raw-audio body into audio bytes and parameters."""

    def test_json_body(self) -> None:
        body = b'{"inputs": "UklGRg==", "parameters": {"return_timestamps": true}}'
        audio, params = decode_request(body, "application/json", {})
        assert audio == b"RIFF"
        assert params == {"return_timestamps": True}

    def test_raw_body_takes_query_parameters(self) -> None:
        audio, params = decode_request(b"RIFF", "audio/wav", {"return_speakers": "1"})
        assert audio == b"RIFF"
        assert params == {"return_speakers": True}

    @pytest.mark.parametrize(
        ("body", "content_type"),
        [
            (b"{nope", "application/json"),
            (b'{"inputs": "not base64!"}', "application/json"),
            (b'{"audio": "UklGRg=="}', "application/json"),
            (b"", "audio/wav"),
        ],
    )
    def test_rejects(self, body: bytes, content_type: str) -> None:
        with pytest.raises(RequestError):
            decode_request(body, content_type, {})


class FakePipeline:
    """Records calls; answers like ASRPipeline (numpy values included)."""

    def __init__(self, error: Exception | None = None) -> None:
        """Start with no calls; ``error``, if given, is raised by every call."""
        self.calls: list[tuple[bytes, dict[str, Any]]] = []
        self.error = error

    def __call__(self, audio: bytes, **params: Any) -> dict[str, Any]:
        """Record the call, then raise the configured error or return a fixed transcript."""
        self.calls.append((audio, params))
        if self.error is not None:
            raise self.error
        return {"text": "hello", "words": [{"word": "hello", "start": np.float32(0.5)}]}


class Client:
    """Requests straight into the ASGI app, in-process (httpx's ASGI transport)."""

    def __init__(self, app: Starlette) -> None:
        """Wrap ``app``; each request opens a fresh ASGI transport to it."""
        self.app = app

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send one request to the app on a private event loop and return the response."""

        async def send() -> httpx.Response:
            """Issue the request through an AsyncClient bound to the ASGI transport."""
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send())

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """POST to ``url``; ``kwargs`` go to httpx (json, content, headers)."""
        return self.request("POST", url, **kwargs)

    def get(self, url: str) -> httpx.Response:
        """GET ``url``."""
        return self.request("GET", url)


MakeClient = Callable[..., tuple[Client, FakePipeline]]


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch) -> MakeClient:
    """Provide a factory for an in-process client over the app and its FakePipeline.

    The factory sets or clears the API-key env var and stubs ``decode_audio`` to pass
    bytes straight through, both undone by monkeypatch at teardown.
    """

    def make(
        api_key: str | None = None, error: Exception | None = None
    ) -> tuple[Client, FakePipeline]:
        if api_key:
            monkeypatch.setenv(serve.API_KEY_ENV, api_key)
        else:
            monkeypatch.delenv(serve.API_KEY_ENV, raising=False)
        pipe = FakePipeline(error)

        def passthrough(audio: bytes) -> bytes:  # decoding has its own tests below
            return audio

        monkeypatch.setattr(serve, "decode_audio", passthrough)
        batcher = ChunkBatcher(lambda _batch: [], max_batch_size=4)
        app = serve.create_app(cast(ASRPipeline, pipe), batcher, "me/model")
        return Client(app), pipe

    return make


class TestApp:
    """The HTTP app routes requests to the pipeline, maps errors to status codes and auths."""

    def test_json_request(self, make_client: MakeClient) -> None:
        client, pipe = make_client()
        response = client.post(
            "/",
            json={
                "inputs": base64.b64encode(b"RIFF").decode(),
                "parameters": {"return_timestamps": True},
            },
        )
        assert response.status_code == 200
        assert response.json() == {"text": "hello", "words": [{"word": "hello", "start": 0.5}]}
        assert pipe.calls == [(b"RIFF", {"return_timestamps": True})]

    def test_raw_audio_request(self, make_client: MakeClient) -> None:
        client, pipe = make_client()
        response = client.post(
            "/transcribe?return_speakers=true",
            content=b"RIFF",
            headers={"content-type": "audio/wav"},
        )
        assert response.status_code == 200
        assert pipe.calls == [(b"RIFF", {"return_speakers": True})]

    def test_bad_request_is_400(self, make_client: MakeClient) -> None:
        client, pipe = make_client()
        response = client.post("/", json={"inputs": "UklGRg==", "parameters": {"x": 1}})
        assert response.status_code == 400
        assert "unknown parameters" in response.json()["error"]
        assert pipe.calls == []

    def test_undecodable_audio_is_400(self, make_client: MakeClient) -> None:
        client, _ = make_client(error=ValueError("ffmpeg could not decode"))
        response = client.post("/", content=b"junk", headers={"content-type": "audio/wav"})
        assert response.status_code == 400

    def test_server_error_is_500(self, make_client: MakeClient) -> None:
        client, _ = make_client(error=RuntimeError("CUDA out of memory"))
        response = client.post("/", content=b"RIFF", headers={"content-type": "audio/wav"})
        assert response.status_code == 500
        assert "out of memory" in response.json()["error"]

    def test_open_without_api_key(self, make_client: MakeClient) -> None:
        client, _ = make_client()
        assert client.post("/", content=b"RIFF").status_code == 200

    def test_api_key_enforced_when_set(self, make_client: MakeClient) -> None:
        client, pipe = make_client(api_key="s3cret")
        assert client.post("/", content=b"RIFF").status_code == 401
        ok = client.post("/", content=b"RIFF", headers={"authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        assert len(pipe.calls) == 1

    def test_health_and_stats(self, make_client: MakeClient) -> None:
        client, _ = make_client()
        assert client.get("/health").json() == {"status": "ok", "model": "me/model"}
        assert client.get("/stats").json()["max_batch_size"] == 4

    def test_stats_break_down_request_time(self, make_client: MakeClient) -> None:
        client, _ = make_client()
        client.post("/", content=b"RIFF", headers={"content-type": "audio/wav"})
        stats = client.get("/stats").json()
        assert stats["requests"] == 1
        assert stats["request_seconds"] >= stats["audio_decode_seconds"] >= 0
        assert {"gpu_seconds", "queue_wait_seconds", "gpu_wait_seconds", "other_seconds"} <= set(
            stats
        )


class TestJobs:
    """`POST /jobs` answers at once; `GET /jobs/<id>` returns the result when it is ready."""

    @staticmethod
    def _wait(client: Client, job_id: str) -> httpx.Response:
        """Poll the job until it stops answering 202 (up to ~5 s) and return that response."""

        def fetch() -> httpx.Response:
            return client.get(f"/jobs/{job_id}")

        return _poll(fetch, lambda response: response.status_code != 202, "job finished")

    def test_job_returns_what_a_request_would(self, make_client: MakeClient) -> None:
        client, pipe = make_client()
        submitted = client.post(
            "/jobs?return_speakers=true", content=b"RIFF", headers={"content-type": "audio/wav"}
        )
        assert submitted.status_code == 202
        result = self._wait(client, submitted.json()["id"])
        assert result.status_code == 200
        assert result.json() == {"text": "hello", "words": [{"word": "hello", "start": 0.5}]}
        assert pipe.calls == [(b"RIFF", {"return_speakers": True})]

    def test_failed_job_keeps_its_status(self, make_client: MakeClient) -> None:
        client, _ = make_client(error=RuntimeError("CUDA out of memory"))
        job_id = client.post("/jobs", content=b"RIFF").json()["id"]
        result = self._wait(client, job_id)
        assert result.status_code == 500
        assert "out of memory" in result.json()["error"]

    def test_unknown_job_is_404(self, make_client: MakeClient) -> None:
        client, _ = make_client()
        assert client.get("/jobs/nope").status_code == 404

    def test_api_key_enforced_on_jobs(self, make_client: MakeClient) -> None:
        client, pipe = make_client(api_key="s3cret")
        assert client.post("/jobs", content=b"RIFF").status_code == 401
        assert client.get("/jobs/anything").status_code == 401
        assert pipe.calls == []


def test_client_disconnect_mid_upload_is_quiet(make_client: MakeClient) -> None:
    """A client that hangs up while sending its body gets no traceback, just a 499."""
    client, pipe = make_client()
    sent: list[dict[str, Any]] = []

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(dict(message))

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(b"content-type", b"audio/wav")],
        "http_version": "1.1",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1234),
        "root_path": "",
    }
    asyncio.run(client.app(scope, receive, send))
    assert sent[0]["status"] == 499
    assert pipe.calls == []


class TestDecodeAudio:
    """Request audio decodes in-process (torchcodec) to mono float32 at 16 kHz."""

    @staticmethod
    def _wav(samples: npt.NDArray[np.float32], rate: int) -> bytes:
        """Encode ``samples`` as 16-bit PCM WAV bytes at ``rate``."""
        buf = io.BytesIO()
        soundfile.write(buf, samples, rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def test_16k_mono_wav_is_unchanged(self) -> None:
        samples = (np.sin(np.linspace(0, 100, 16000)) * 0.5).astype(np.float32)
        decoded = serve.decode_audio(self._wav(samples, 16000))
        assert decoded["sampling_rate"] == 16000
        assert decoded["array"].dtype == np.float32
        np.testing.assert_allclose(decoded["array"], samples, atol=1 / 32768)

    def test_stereo_is_downmixed(self) -> None:
        left = np.full(1600, 0.5, dtype=np.float32)
        decoded = serve.decode_audio(
            self._wav(np.stack([left, -left / 2], axis=1).astype(np.float32), 16000)
        )
        assert decoded["array"].ndim == 1
        assert len(decoded["array"]) == 1600

    def test_other_rates_are_resampled_to_16k(self) -> None:
        decoded = serve.decode_audio(self._wav(np.zeros(48000, dtype=np.float32), 48000))
        assert decoded["sampling_rate"] == 16000
        assert abs(len(decoded["array"]) - 16000) <= 16  # one second, give or take a frame

    def test_undecodable_audio_is_a_request_error(self) -> None:
        with pytest.raises(RequestError, match="could not decode audio"):
            serve.decode_audio(b"not audio at all")


class TestDiarizeOnly:
    """`diarize_only` answers Nemotron's activity and runs no ASR."""

    def test_returns_activity_without_asr(
        self, make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, pipe = make_client()
        activity = np.zeros((300, 8), dtype=np.float32)
        activity[:100, 0] = 0.9
        activity[100:, 3] = 0.75
        activity[50:60, 5] = 0.6  # a short third speaker, capped away below

        def decode(_audio: bytes) -> dict[str, Any]:
            return {"array": np.zeros(48000, dtype=np.float32), "sampling_rate": 16000}

        def nemotron(_array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            return activity

        monkeypatch.setattr(serve, "decode_audio", decode)
        monkeypatch.setattr(NemotronDiarizer, "activity", nemotron)
        response = client.post(
            "/?diarize_only=true&max_speakers=2",
            content=b"RIFF",
            headers={"content-type": "audio/wav"},
        )
        assert response.status_code == 200
        body = response.json()
        assert pipe.calls == []
        assert body["keep"] == [0, 3]
        assert body["shape"] == [300, 8]
        assert body["frame_s"] == 0.01
        decoded = np.frombuffer(base64.b64decode(body["activity"]), dtype="<f2").reshape(300, 8)
        np.testing.assert_allclose(decoded, activity, atol=1e-3)
        assert [s["speaker"] for s in body["speaker_segments"]] == ["SPEAKER_0", "SPEAKER_1"]
