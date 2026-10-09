"""Batched HTTP inference server for tiny_audio (`ta serve`).

Three kinds of thread, so nothing blocks the event loop and the GPU never
idles while there is work:

- The event loop (starlette on uvicorn) only accepts requests, checks the
  optional API key and awaits results. It never parses a body, decodes audio
  or touches a model.
- Request threads do each request's CPU work -- JSON/base64 parsing, ffmpeg
  decode, chunking, feature extraction, alignment, diarization -- and hand
  every chunk that needs the decoder to the batcher.
- One GPU thread (`ChunkBatcher`) owns the ASR model. It drains everything
  queued, builds a batch around the oldest chunk from the chunks nearest it in
  length, and runs one batched `generate`. Chunks queue up while a batch runs,
  so the next batch fills itself under load, and a lone request never waits.

Request format matches the HF Space client (demo/app.py):
`POST /` with `{"inputs": <base64 audio>, "parameters": {...}}`, or the raw
audio bytes as the body with parameters in the query string.
"""

import base64
import binascii
import hmac
import json
import logging
import os
import queue
import threading
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

import anyio
import anyio.to_thread
import numpy as np
import torch
import typer
import uvicorn
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from torchcodec.decoders import AudioDecoder

from scripts.inference import GraphedDecoder, load_serving_pipeline
from tiny_audio.alignment import get_device
from tiny_audio.asr_pipeline import ASRPipeline, PreparedChunk

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

# Request options a client may set, with their types. Everything else the
# pipeline accepts (generate kwargs, user_prompt) stays server-side: those
# change shared model state or batch shape.
PARAMETERS: dict[str, type] = {
    "return_timestamps": bool,
    "return_speakers": bool,
    "num_speakers": int,
    "max_speakers": int,
}
# Threads for in-flight requests. Each holds one while it waits on the GPU, so
# this is how many requests can feed the batcher at once; more than that wait
# (asynchronously) for a thread rather than being refused.
REQUEST_THREADS = 64
# Optional bearer key. Unset means the server is open.
API_KEY_ENV = "TINY_AUDIO_API_KEY"


class RequestError(ValueError):
    """A client error: answered 400 with its message."""


@dataclass
class _Job:
    """One chunk waiting for the decoder, and where its text goes."""

    prepared: PreparedChunk
    future: "Future[str]" = field(default_factory=Future)
    queued_at: float = field(default_factory=time.perf_counter)

    @property
    def frames(self) -> int:
        return int(self.prepared["attention_mask"].shape[-1])


@dataclass
class BatcherStats:
    """Counters for `GET /stats`: how full and how busy the GPU's batches run."""

    batches: int = 0
    chunks: int = 0
    largest_batch: int = 0
    gpu_seconds: float = 0.0  # inside generate
    queue_wait_seconds: float = 0.0  # summed over chunks: queued -> its batch starts


@dataclass
class RequestStats:
    """Where requests spend their time server-side, summed (written by request threads)."""

    requests: int = 0
    total_seconds: float = 0.0
    audio_decode_seconds: float = 0.0
    gpu_wait_seconds: float = 0.0  # chunks queued or decoding on the GPU
    other_seconds: float = 0.0  # features, chunking, alignment, diarization, JSON
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, total: float, audio_decode: float, gpu_wait: float) -> None:
        with self.lock:
            self.requests += 1
            self.total_seconds += total
            self.audio_decode_seconds += audio_decode
            self.gpu_wait_seconds += gpu_wait
            self.other_seconds += total - audio_decode - gpu_wait


class ChunkBatcher:
    """Runs chunks from every in-flight request through one model in shared batches."""

    def __init__(
        self, generate: Callable[[list[PreparedChunk]], list[str]], max_batch_size: int
    ) -> None:
        self._generate = generate
        self.max_batch_size = max(1, max_batch_size)
        self._inbox: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._loop, name="tiny-audio-gpu", daemon=True)
        self.stats = BatcherStats()
        self._waits = threading.local()  # per request thread: seconds blocked in run()

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        """Finish what is queued, then stop the GPU thread."""
        self._inbox.put(None)
        self._thread.join()

    def run(self, prepared: list[PreparedChunk]) -> list[str]:
        """Texts for `prepared` (blocks the calling request thread, not the event loop).

        Installed as `ASRPipeline.chunk_runner`, so every transcription path --
        plain, timestamps, per-speaker streams -- batches through here.
        """
        start = time.perf_counter()
        jobs = [_Job(p) for p in prepared]
        for job in jobs:
            self._inbox.put(job)
        texts = [job.future.result() for job in jobs]
        self._waits.seconds = self.thread_wait() + time.perf_counter() - start
        return texts

    def thread_wait(self, *, reset: bool = False) -> float:
        """Seconds the calling thread has spent in `run` (since the last reset)."""
        seconds = float(getattr(self._waits, "seconds", 0.0))
        if reset:
            self._waits.seconds = 0.0
        return seconds

    def _loop(self) -> None:
        pending: list[_Job] = []
        closing = False
        while pending or not closing:
            if not pending:
                job = self._inbox.get()
                if job is None:
                    return
                pending.append(job)
            # Everything that queued while the last batch ran: no waiting, so
            # a lone request goes straight through and load batches itself.
            while True:
                try:
                    job = self._inbox.get_nowait()
                except queue.Empty:
                    break
                if job is None:
                    closing = True
                else:
                    pending.append(job)
            batch = self._take_batch(pending)
            self._execute(batch)

    def _take_batch(self, pending: list[_Job]) -> list[_Job]:
        """The oldest job plus the pending jobs closest to it in length.

        Oldest-first keeps every request moving; filling by length keeps the
        padding (computed and thrown away) small.
        """
        oldest = pending[0]
        nearest = sorted(pending[1:], key=lambda j: abs(j.frames - oldest.frames))
        batch = [oldest, *nearest[: self.max_batch_size - 1]]
        taken = {id(j) for j in batch}
        pending[:] = [j for j in pending if id(j) not in taken]
        return batch

    def _execute(self, batch: list[_Job]) -> None:
        start = time.perf_counter()
        self.stats.queue_wait_seconds += sum(start - job.queued_at for job in batch)
        try:
            texts = self._generate([job.prepared for job in batch])
        except BaseException as e:  # every waiter must hear about it
            for job in batch:
                job.future.set_exception(e)
            return
        for job, text in zip(batch, texts, strict=True):
            job.future.set_result(text)
        self.stats.gpu_seconds += time.perf_counter() - start
        self.stats.batches += 1
        self.stats.chunks += len(batch)
        self.stats.largest_batch = max(self.stats.largest_batch, len(batch))


def parse_parameters(raw: Mapping[str, Any], *, from_query: bool) -> dict[str, Any]:
    """Validated pipeline options; query-string values arrive as text."""
    unknown = sorted(set(raw) - set(PARAMETERS))
    if unknown:
        msg = f"unknown parameters {unknown}; allowed: {sorted(PARAMETERS)}"
        raise RequestError(msg)
    params: dict[str, Any] = {}
    for name, given in raw.items():
        kind = PARAMETERS[name]
        value: Any = given
        if from_query and isinstance(given, str):
            if kind is bool and given.lower() in ("1", "true", "yes", "0", "false", "no"):
                value = given.lower() in ("1", "true", "yes")
            elif kind is int and given.lstrip("-").isdigit():
                value = int(given)
        # bool is an int subclass: reject True where a count is expected.
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            msg = f"parameter {name!r} must be {kind.__name__}, got {value!r}"
            raise RequestError(msg)
        params[name] = value
    return params


def decode_request(
    body: bytes, content_type: str, query: Mapping[str, str]
) -> tuple[bytes, dict[str, Any]]:
    """`(audio bytes, parameters)` from a JSON body or a raw audio body."""
    if content_type.startswith("application/json"):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            msg = f"invalid JSON: {e}"
            raise RequestError(msg) from e
        if not isinstance(payload, dict) or not isinstance(payload.get("inputs"), str):
            msg = 'JSON body must be {"inputs": <base64 audio>, "parameters": {...}}'
            raise RequestError(msg)
        try:
            audio = base64.b64decode(payload["inputs"], validate=True)
        except (binascii.Error, ValueError) as e:
            msg = "'inputs' is not valid base64"
            raise RequestError(msg) from e
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, dict):
            msg = "'parameters' must be an object"
            raise RequestError(msg)
        return audio, parse_parameters(parameters, from_query=False)
    if not body:
        msg = "empty body: send audio bytes, or JSON with base64 'inputs'"
        raise RequestError(msg)
    return body, parse_parameters(query, from_query=True)


def _jsonable(value: Any) -> Any:
    """numpy scalars and arrays (from alignment/diarization) as plain JSON values."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def decode_audio(audio: bytes) -> dict[str, Any]:
    """Decode request audio in-process to mono float32 at 16 kHz, with torchcodec.

    Replaces the ffmpeg process the pipeline spawns for bytes (~76 ms of a
    request thread per clip on the 4090 pod). torchcodec runs FFmpeg's decoders
    and resampler in-process, so any format FFmpeg reads works, other rates and
    stereo resample and downmix exactly as before, and 16 kHz WAV comes out
    bit-identical. Unlike soundfile it releases the GIL, so it doesn't stall the
    GPU thread's decode loop: 1,340 WAV / 1,012 MP3 clips/s across 32 threads
    against soundfile's 453 / 56. Undecodable audio is a 400.
    """
    try:
        samples = AudioDecoder(audio, sample_rate=16000, num_channels=1).get_all_samples()
    except (RuntimeError, ValueError) as e:
        msg = f"could not decode audio: {e}"
        raise RequestError(msg) from e
    return {"array": samples.data[0].numpy(), "sampling_rate": 16000}


def transcribe_request(
    pipe: ASRPipeline,
    body: bytes,
    content_type: str,
    query: Mapping[str, str],
    batcher: ChunkBatcher | None = None,
    stats: RequestStats | None = None,
) -> bytes:
    """One request's whole blocking path, run on a request thread: bytes in, JSON out."""
    start = time.perf_counter()
    audio, params = decode_request(body, content_type, query)
    decoded = decode_audio(audio)
    decode_seconds = time.perf_counter() - start
    if batcher is not None:
        batcher.thread_wait(reset=True)
    try:
        result = pipe(decoded, **params)
    except ValueError as e:  # undecodable audio, bad speaker hints
        raise RequestError(str(e)) from e
    content = json.dumps(_jsonable(result)).encode()
    if stats is not None:
        gpu_wait = batcher.thread_wait() if batcher is not None else 0.0
        stats.add(time.perf_counter() - start, decode_seconds, gpu_wait)
    return content


def _authorized(request: Request, api_key: str | None) -> bool:
    """True when no key is configured, or the request carries it as a bearer token."""
    if not api_key:
        return True
    given = request.headers.get("authorization", "").removeprefix("Bearer ")
    return hmac.compare_digest(given.encode(), api_key.encode())


async def _read_body(request: Request) -> bytes | None:
    """The request body, or None if the client disconnected mid-upload.

    A cancelled request or a proxy timeout closes the connection while the
    body streams in; that is routine, so it logs one line instead of a
    traceback.
    """
    try:
        return await request.body()
    except ClientDisconnect:
        logger.info("client disconnected before sending the full request")
        return None


def create_app(pipe: ASRPipeline, batcher: ChunkBatcher, model_id: str) -> Starlette:
    """The HTTP app over a loaded pipeline whose chunks run through `batcher`."""
    limiter = anyio.CapacityLimiter(REQUEST_THREADS)
    request_stats = RequestStats()
    started = time.perf_counter()
    api_key = os.environ.get(API_KEY_ENV)

    async def transcribe(request: Request) -> Response:
        if not _authorized(request, api_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await _read_body(request)
        if body is None:
            return Response(status_code=499)  # client closed request; nobody to answer
        try:
            content = await anyio.to_thread.run_sync(
                transcribe_request,
                pipe,
                body,
                request.headers.get("content-type", ""),
                dict(request.query_params),
                batcher,
                request_stats,
                limiter=limiter,
            )
        except RequestError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.exception("request failed")
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
        return Response(content, media_type="application/json")

    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok", "model": model_id})

    async def stats(_request: Request) -> Response:
        s, r = batcher.stats, request_stats
        return JSONResponse(
            {
                "uptime_seconds": round(time.perf_counter() - started, 3),
                "batches": s.batches,
                "chunks": s.chunks,
                "mean_batch_size": round(s.chunks / s.batches, 2) if s.batches else 0.0,
                "largest_batch": s.largest_batch,
                "max_batch_size": batcher.max_batch_size,
                "gpu_seconds": round(s.gpu_seconds, 3),
                "queue_wait_seconds": round(s.queue_wait_seconds, 3),
                "requests": r.requests,
                "request_seconds": round(r.total_seconds, 3),
                "audio_decode_seconds": round(r.audio_decode_seconds, 3),
                "gpu_wait_seconds": round(r.gpu_wait_seconds, 3),
                "other_seconds": round(r.other_seconds, 3),
            }
        )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        yield
        # Off the event loop: close() joins the GPU thread after its last batch.
        await anyio.to_thread.run_sync(batcher.close)

    return Starlette(
        routes=[
            Route("/", transcribe, methods=["POST"]),
            Route("/transcribe", transcribe, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Route("/stats", stats, methods=["GET"]),
        ],
        lifespan=lifespan,
    )


def warm_chunk(pipe: ASRPipeline) -> PreparedChunk:
    """A 10 s noise chunk to decode before serving."""
    noise = np.random.default_rng(0).normal(0, 0.05, 16000 * 10).astype(np.float32)
    return pipe.prepare_chunk(noise, 16000)


def warm_up(
    pipe: ASRPipeline,
    generate: Callable[[list[PreparedChunk]], list[str]],
    sizes: list[int],
) -> None:
    """Decode a batch of each size, so no request pays kernel autotuning."""
    prepared = warm_chunk(pipe)
    for size in sizes:
        typer.echo(f"  batch {size}")
        generate([prepared] * size)


@app.command()
def serve(
    model: Annotated[
        str, typer.Option("--model", "-m", help="HuggingFace Hub model ID or local checkpoint")
    ] = "mazesmazes/tiny-audio",
    host: Annotated[
        str, typer.Option("--host", help="Interface to bind (0.0.0.0 to accept remote clients)")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 8000,
    max_batch_size: Annotated[
        int | None,
        typer.Option(
            "--max-batch-size", help="Most chunks per GPU batch (default: 32 on CUDA, else 8)"
        ),
    ] = None,
    allow_slow_kernels: Annotated[
        bool,
        typer.Option(
            "--allow-slow-kernels",
            help="Serve on CUDA without the fast linear-attention kernels (several times slower)",
        ),
    ] = False,
    cuda_graphs: Annotated[
        bool,
        typer.Option(
            "--cuda-graphs/--no-cuda-graphs",
            help="On CUDA, decode with a static cache and CUDA graphs (compiles at startup)",
        ),
    ] = True,
) -> None:
    """Serve batched transcription over HTTP; runs on CUDA, Apple MPS or CPU."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    batch = max_batch_size or (32 if device.type == "cuda" else 8)
    typer.echo(f"Loading {model} on {device} (max batch {batch})...")
    pipe = load_serving_pipeline(
        model, device=device, max_batch_size=batch, require_fast_kernels=not allow_slow_kernels
    )
    # One CPU thread per torch op. Request threads already run in parallel, one
    # request each; with torch's default pool (one thread per core) every
    # request's feature extraction fanned out across all cores at once, and the
    # oversubscription starved the GPU thread's decode loop. Measured on the
    # pod, 32 request threads: 63.6 -> 115.7 clips/s of CPU-side work.
    torch.set_num_threads(1)
    generate: Callable[[list[PreparedChunk]], list[str]] = pipe.generate_prepared
    if device.type == "cuda" and cuda_graphs:
        decoder = GraphedDecoder(pipe, batch)
        generate = decoder
        typer.echo(f"CUDA graphs for batch sizes {decoder.buckets}; compiling (~10-50 s each)...")
        decoder.warm_up(warm_chunk(pipe))
    else:
        typer.echo("Warming up...")
        warm_up(pipe, generate, sorted({1, batch}))
    batcher = ChunkBatcher(generate, batch)
    pipe.chunk_runner = batcher.run
    batcher.start()
    auth = f"bearer key from ${API_KEY_ENV}" if os.environ.get(API_KEY_ENV) else "open, no key"
    typer.echo(f"Serving on http://{host}:{port} ({auth})")
    uvicorn.run(create_app(pipe, batcher, model), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
