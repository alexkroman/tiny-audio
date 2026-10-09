"""Load-test a running `ta serve` (`ta bench`): latency and throughput per concurrency.

    ta bench --url https://<pod-id>-8000.proxy.runpod.net        # concurrency 1, 8, 32
    ta bench --url http://127.0.0.1:8000 --concurrency 4 --requests 16 --speakers

Fires `--requests` POSTs of the same audio (default: twice the concurrency,
at least 8, so every level stays saturated) with at most `--concurrency` in
flight, and reports per-request latency, real-time factor (audio seconds
transcribed per wall second) and the server's mean GPU batch size over the
run, read from `GET /stats` before and after.
"""

import asyncio
import base64
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import httpx
import jiwer
import soundfile
import typer

app = typer.Typer(add_completion=False)


@dataclass
class LevelResult:
    """One concurrency level: successful latencies and texts, and every failure."""

    wall: float = 0.0
    latencies: list[float] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _describe_failure(response: httpx.Response) -> str:
    """Status plus the server's message; 524 is the RunPod proxy's ~100 s cutoff."""
    hint = (
        " (RunPod proxy timeout: request took over ~100 s)" if response.status_code == 524 else ""
    )
    return f"HTTP {response.status_code}{hint}: {response.text[:200]}"


async def _run_level(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    concurrency: int,
    requests: int,
) -> LevelResult:
    """`requests` POSTs, `concurrency` at a time; a failure is recorded, not raised.

    Raising would cancel every other in-flight request, which the server then
    sees as a wave of client disconnects.
    """
    gate = asyncio.Semaphore(concurrency)
    result = LevelResult()

    async def one() -> None:
        async with gate:
            start = time.perf_counter()
            try:
                response = await client.post(url, json=payload)
            except httpx.HTTPError as e:
                result.errors.append(f"{type(e).__name__}: {e}")
                return
            if response.is_error:
                result.errors.append(_describe_failure(response))
                return
            result.latencies.append(time.perf_counter() - start)
            result.texts.append(response.json().get("text", ""))

    start = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(requests)))
    result.wall = time.perf_counter() - start
    return result


async def _bench(
    url: str, audio: Path, levels: list[int], requests: int | None, parameters: dict[str, Any]
) -> None:
    seconds = soundfile.info(str(audio)).duration
    payload = {"inputs": base64.b64encode(audio.read_bytes()).decode(), "parameters": parameters}
    base = url.rstrip("/")
    # trust_env=False: connect straight to the server, ignoring HTTPS_PROXY.
    # A local proxy (e.g. Aikido safe-chain under `poetry run`) drops some of
    # many concurrent tunnels, which read as 502s from a healthy server.
    async with httpx.AsyncClient(timeout=1800, trust_env=False) as client:
        # One untimed request so model caches and allocator are warm.
        warm = await client.post(f"{base}/", json=payload)
        if warm.is_error:
            typer.echo(f"Warm-up request failed: {_describe_failure(warm)}", err=True)
            raise typer.Exit(1)
        typer.echo(f"{audio.name}: {seconds:.1f} s of audio, parameters {parameters or '{}'}")
        typer.echo(
            f"{'conc':>5} {'reqs':>5} {'fail':>5} {'p50 s':>8} {'p90 s':>8} {'max s':>8} "
            f"{'RTFx':>8} {'batch':>6} {'gpu%':>5} {'wait ms':>8} {'srv ms':>7}"
        )
        texts: list[str] = []
        for concurrency in levels:
            count = requests or max(8, 2 * concurrency)
            before = (await client.get(f"{base}/stats")).json()
            level = await _run_level(client, f"{base}/", payload, concurrency, count)
            after = (await client.get(f"{base}/stats")).json()
            delta = server_deltas(before, after)
            ordered = sorted(level.latencies) or [float("nan")]
            p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
            typer.echo(
                f"{concurrency:>5} {count:>5} {len(level.errors):>5} "
                f"{statistics.median(ordered):>8.2f} {p90:>8.2f} {ordered[-1]:>8.2f} "
                f"{len(level.latencies) * seconds / level.wall:>8.1f} "
                f"{delta['mean_batch']:>6.1f} {100 * delta['gpu_seconds'] / level.wall:>5.0f} "
                f"{1000 * delta['queue_wait_per_chunk']:>8.0f} {1000 * delta['request_mean']:>7.0f}"
            )
            if delta["requests"]:
                typer.echo(
                    "  server per request: "
                    f"audio decode {1000 * delta['audio_decode_mean']:.0f} ms, "
                    f"waiting on GPU {1000 * delta['gpu_wait_mean']:.0f} ms, "
                    f"other CPU {1000 * delta['other_mean']:.0f} ms; "
                    f"client p50 minus server mean = "
                    f"{1000 * (statistics.median(ordered) - delta['request_mean']):.0f} ms network"
                )
            if level.errors:
                typer.echo(f"  ! {len(level.errors)} failed, e.g. {level.errors[0]}")
            texts = level.texts or texts
            if len(set(level.texts)) > 1:
                typer.echo(f"  ! {divergence(level.texts)}")
        if texts:
            typer.echo(f"\ntext: {texts[0][:200]}")


def server_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    """What the server spent during one level, from two `GET /stats` snapshots.

    Missing keys (an older server) read as 0.
    """

    def d(key: str) -> float:
        return float(after.get(key, 0)) - float(before.get(key, 0))

    batches, chunks, requests = d("batches"), d("chunks"), d("requests")

    def per_request(key: str) -> float:
        return d(key) / requests if requests else 0.0

    return {
        "mean_batch": chunks / batches if batches else 0.0,
        "gpu_seconds": d("gpu_seconds"),
        "queue_wait_per_chunk": d("queue_wait_seconds") / chunks if chunks else 0.0,
        "requests": requests,
        "request_mean": per_request("request_seconds"),
        "audio_decode_mean": per_request("audio_decode_seconds"),
        "gpu_wait_mean": per_request("gpu_wait_seconds"),
        "other_mean": per_request("other_seconds"),
    }


def divergence(texts: list[str]) -> str:
    """How far identical requests' transcripts drift: WER of each vs the most common one.

    Batching and GPU kernels are not bit-exact, so near-tie tokens can flip
    between requests; the size of the drift, not its existence, is the signal.
    """
    reference = Counter(texts).most_common(1)[0][0]
    wers = [jiwer.wer(reference, t) if reference else float(bool(t)) for t in texts]
    return (
        f"{len(set(texts))} different transcripts for identical audio; vs the most common: "
        f"mean WER {100 * statistics.mean(wers):.1f}%, max {100 * max(wers):.1f}%"
    )


@app.command()
def bench(
    url: Annotated[str, typer.Option("--url", help="Server base URL")] = "http://127.0.0.1:8000",
    audio: Annotated[
        Path, typer.Option("--audio", exists=True, dir_okay=False, help="Audio file to send")
    ] = Path("demo/examples/ami_meeting.wav"),
    concurrency: Annotated[
        list[int] | None,
        typer.Option("--concurrency", help="In-flight requests (repeatable; default 1, 8, 32)"),
    ] = None,
    requests: Annotated[
        int | None,
        typer.Option("--requests", help="Requests per level (default: 2x concurrency, min 8)"),
    ] = None,
    timestamps: Annotated[
        bool, typer.Option("--timestamps", help="Request word timestamps")
    ] = False,
    speakers: Annotated[
        bool, typer.Option("--speakers", help="Request speaker diarization")
    ] = False,
) -> None:
    """Measure latency and throughput of a running `ta serve`."""
    parameters: dict[str, Any] = {}
    if timestamps:
        parameters["return_timestamps"] = True
    if speakers:
        parameters["return_speakers"] = True
    asyncio.run(_bench(url, audio, concurrency or [1, 8, 32], requests, parameters))


if __name__ == "__main__":
    app()
