"""ASR evaluator implementations."""

import contextlib
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.eval.audio import as_16k_array, prepare_wav_bytes

from .base import Evaluator, console, setup_assemblyai

try:
    from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode
    from Foundation import NSURL, NSLocale
    from Speech import (
        SFSpeechRecognizer,
        SFSpeechRecognizerAuthorizationStatusAuthorized,
        SFSpeechURLRecognitionRequest,
    )

    _APPLE_SPEECH_AVAILABLE = True
except ImportError:
    _APPLE_SPEECH_AVAILABLE = False


def print_generation_config(model, model_path: str):
    """Print generation config in a visible format."""
    gen_config = model.generation_config
    console.print(
        "\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]"
    )
    console.print(f"[bold]Model:[/bold] {model_path}")
    console.print("[bold cyan]Generation Config:[/bold cyan]")
    console.print(f"  max_new_tokens:      {gen_config.max_new_tokens}")
    console.print(f"  min_new_tokens:      {gen_config.min_new_tokens}")
    console.print(f"  num_beams:           {gen_config.num_beams}")
    console.print(f"  do_sample:           {gen_config.do_sample}")
    console.print(f"  repetition_penalty:  {gen_config.repetition_penalty}")
    console.print(f"  length_penalty:      {gen_config.length_penalty}")
    console.print(f"  no_repeat_ngram_size: {gen_config.no_repeat_ngram_size}")
    console.print(
        "[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n"
    )


def _resolve_local_runtime() -> tuple[int | str, str]:
    """Pick the device and weight dtype for a locally loaded ASRModel pipeline.

    Returns `(device, model_dtype)` where model_dtype is the STRING name
    `ASRConfig.model_dtype` expects, not a `torch.dtype`. That distinction is
    the whole point of this function.

    `pipeline(dtype=...)` and `from_pretrained(dtype=...)` do nothing to this
    model class. ASRModel.__init__ does `target_dtype = getattr(torch,
    config.model_dtype)` and then casts both submodules with an explicit
    `.to(dtype=...)` after load (deliberately -- see the comment in
    _load_audio_encoder about loader paths honoring `dtype=` inconsistently),
    so a dtype kwarg is silently overwritten by the saved config. Verified
    against the published granite-qwen config: `dtype=torch.float16` leaves
    `model_dtype` at "float32", while `model_dtype="float16"` takes. Routing it
    through model_kwargs is the only path that lands.

    Why it matters: granite-qwen's config.json carries `model_dtype: float32`,
    which is load-bearing for TRAINING only -- fp32 master weights exist so
    AdamW's 2e-5 decoder step doesn't round away under bf16's ULP (see
    configs/experiments/granite_qwen.yaml). Inference has no optimizer, so it
    was holding 2.27B params in fp32 and reading 9.1 GB of weights per decoded
    token instead of 4.5 GB, on a decode loop that is entirely
    memory-bandwidth bound.

    bfloat16 rather than float16 on MPS. Measured on torch 2.8 / Metal, the
    two are the same speed (0.78 vs 0.79 ms/iter on a LayerNorm + 2048x6144
    matmul + SiLU), so fp16 buys nothing -- while costing the exponent range
    that matters here, since fp16 tops out at 65504 and both the Granite
    encoder and Qwen3.5 were pretrained in bf16. Equal speed, strictly more
    headroom.

    CPU stays fp32: reduced precision there is emulated rather than
    accelerated without AMX, so it is a slowdown, not a win.

    attn_implementation is deliberately NOT set. ASRModel routes whatever the
    config asks for through _resolve_attn_implementation, which already coerces
    FA2 -> sdpa off CUDA and anything -> eager on MPS (Metal's sdpa kernel
    returns wrong results for cached single-token decode against a
    sliding-window mask). Passing it here would be a second, silently diverging
    copy of that policy -- which is what it had become: the streaming evaluator
    asked for "sdpa" on MPS and got eager anyway.
    """
    if torch.cuda.is_available():
        return 0, "bfloat16"
    if torch.backends.mps.is_available():
        return "mps", "bfloat16"
    return -1, "float32"


def _build_local_pipeline(model_path: str):
    """Build the ASR pipeline both local evaluators run on."""
    from transformers import pipeline

    device, model_dtype = _resolve_local_runtime()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        trust_remote_code=True,
        device=device,
        model_kwargs={"model_dtype": model_dtype},
    )
    console.print(f"[dim]Using device: {device}, model_dtype: {model_dtype}[/dim]")
    return pipe


class LocalEvaluator(Evaluator):
    """Evaluator for local models."""

    def __init__(self, model_path: str, user_prompt: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.pipe = _build_local_pipeline(model_path)
        # Explicit, not incidental. LocalStreamingEvaluator has always done
        # this; this class never did, and got away with it only because
        # nothing in the stack was train/eval-sensitive. A partially unfrozen
        # Granite encoder is: its 16 BatchNorm1d modules normalise with batch
        # statistics in train mode, which cost 25 WER points on Earnings22.
        self.pipe.model.eval()
        self.user_prompt = user_prompt

        print_generation_config(self.pipe.model, model_path)

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        start = time.time()
        # Request per-step top-1/top-2 logprobs. The Hub-loaded pipeline may be
        # an older version without scores support, in which case the kwarg is
        # absorbed by generate() but the result dict won't include the new
        # `top1_logprob` / `top2_logprob` fields and we return confidence=None.
        result = self.pipe(audio, user_prompt=self.user_prompt, output_scores=True)
        elapsed = time.time() - start

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        confidence: dict | None = None
        if isinstance(result, dict):
            top1 = result.get("top1_logprob")
            top2 = result.get("top2_logprob")
            if top1 and top2 and len(top1) == len(top2):
                n = len(top1)
                mean_top1 = sum(top1) / n
                mean_margin = sum(a - b for a, b in zip(top1, top2)) / n
                confidence = {
                    "mean_top1_logprob": mean_top1,
                    "mean_margin": mean_margin,
                    "num_tokens": n,
                }
        return text, elapsed, confidence


class LocalStreamingEvaluator(Evaluator):
    """Evaluator for local models with streaming metrics (TTFB, processing time)."""

    def __init__(self, model_path: str, user_prompt: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.pipe = _build_local_pipeline(model_path)
        self.model = self.pipe.model
        self.model.eval()
        self.processor = self.model.get_processor()
        self.user_prompt = user_prompt

        # Track timing stats
        self.ttfb_times: list[float] = []
        self.processing_times: list[float] = []

        # Print generation config
        print_generation_config(self.model, model_path)

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        from transformers import TextIteratorStreamer

        audio_array = as_16k_array(audio)

        # Process audio (ASRProcessor handles sampling_rate internally)
        inputs = self.processor(
            audio_array,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(
            device=self.model.device, dtype=self.model.dtype
        )
        audio_attention_mask = inputs["audio_attention_mask"].to(self.model.device)

        # Set up streamer to capture first token time
        streamer = TextIteratorStreamer(
            self.model.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        first_token_time = [None]
        generation_start = [None]

        def generate():
            generation_start[0] = time.time()
            self.model.generate(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
                streamer=streamer,
            )

        # Start generation in background thread
        thread = threading.Thread(target=generate)
        thread.start()

        # Collect tokens and measure TTFB
        tokens = []
        for text in streamer:
            if first_token_time[0] is None and text:
                first_token_time[0] = time.time()
            tokens.append(text)

        thread.join()
        processing_end = time.time()

        # Calculate timing metrics
        processing_time = processing_end - generation_start[0] if generation_start[0] else 0
        ttfb = (
            (first_token_time[0] - generation_start[0])
            if first_token_time[0] and generation_start[0]
            else None
        )

        # Store for aggregation
        self.processing_times.append(processing_time)
        if ttfb is not None:
            self.ttfb_times.append(ttfb)

        # Print timing info
        ttfb_str = f"{ttfb * 1000:.0f}ms" if ttfb else "N/A"
        console.print(
            f"  [dim][Streaming] TTFB: {ttfb_str}, Processing: {processing_time * 1000:.0f}ms[/dim]"
        )

        full_text = "".join(tokens).strip()
        return full_text, processing_time, None

    def compute_metrics(self) -> dict:
        """Compute final metrics including streaming-specific timing."""
        metrics = super().compute_metrics()
        if self.ttfb_times:
            metrics["avg_ttfb"] = sum(self.ttfb_times) / len(self.ttfb_times)
            metrics["min_ttfb"] = min(self.ttfb_times)
            metrics["max_ttfb"] = max(self.ttfb_times)
        if self.processing_times:
            metrics["avg_processing"] = sum(self.processing_times) / len(self.processing_times)
        return metrics


class EndpointEvaluator(Evaluator):
    """Evaluator for HuggingFace Inference Endpoints."""

    def __init__(self, endpoint_url: str, **kwargs):
        super().__init__(**kwargs)
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(base_url=endpoint_url)

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        wav_bytes = prepare_wav_bytes(audio)

        start = time.time()
        result = self.client.automatic_speech_recognition(wav_bytes)
        elapsed = time.time() - start

        if isinstance(result, dict):
            text = result.get("text", result.get("transcription", ""))
        elif hasattr(result, "text"):
            text = result.text
        else:
            text = str(result)
        return text, elapsed, None


class AssemblyAIEvaluator(Evaluator):
    """Evaluator for AssemblyAI API."""

    def __init__(
        self, api_key: str, model: str = "universal-3-pro", base_url: str | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.transcriber = setup_assemblyai(api_key, model, base_url=base_url)

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()
        transcript = self.transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start
        return transcript.text or "", elapsed, None


class AssemblyAIStreamingEvaluator(Evaluator):
    """Evaluator for AssemblyAI Streaming API (Universal-Streaming model).

    Opens a fresh websocket per call so parallel workers don't share state.
    A class-level semaphore caps real concurrent streams below the account's
    server-side limit; override via ASSEMBLYAI_STREAM_CONCURRENCY.
    """

    # WebSocket close codes the server uses for transient backpressure
    # (concurrent-stream cap, overload). These deserve a retry rather than
    # bubbling up as a permanent failure.
    _RETRYABLE_CODES = frozenset({1008, 1011, 1013, 4029, 4102})
    _MAX_RETRIES = 6
    _BACKOFF_CAP = 30.0
    _DEFAULT_CONCURRENCY = 8

    _stream_semaphore: threading.Semaphore | None = None
    _semaphore_lock = threading.Lock()

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self._ensure_semaphore()

    @classmethod
    def _ensure_semaphore(cls) -> None:
        if cls._stream_semaphore is not None:
            return
        with cls._semaphore_lock:
            if cls._stream_semaphore is None:
                cap = int(os.environ.get("ASSEMBLYAI_STREAM_CONCURRENCY", cls._DEFAULT_CONCURRENCY))
                cls._stream_semaphore = threading.Semaphore(cap)

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        import random

        pcm_data = self._prepare_pcm(audio)

        last_err = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                assert self._stream_semaphore is not None
                with self._stream_semaphore:
                    text, elapsed = self._run_session(pcm_data)
                    return text, elapsed, None
            except RuntimeError as e:
                last_err = e
                code = getattr(e, "streaming_code", None)
                if code not in self._RETRYABLE_CODES or attempt == self._MAX_RETRIES:
                    raise
                # Exponential backoff with jitter, capped to keep latency bounded.
                delay = min(self._BACKOFF_CAP, 0.5 * (2**attempt)) * (1 + random.random())
                time.sleep(delay)
        raise last_err  # unreachable; appeases type checker

    def _prepare_pcm(self, audio) -> bytes:
        audio_array = as_16k_array(audio)

        if isinstance(audio_array, np.ndarray):
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / np.abs(audio_array).max()
            return (audio_array * 32767).astype(np.int16).tobytes()
        return audio_array

    def _run_session(self, pcm_data: bytes) -> tuple[str, float]:
        from assemblyai.streaming.v3 import (
            SpeechModel,
            StreamingClient,
            StreamingClientOptions,
            StreamingEvents,
            StreamingParameters,
        )

        # Per-call state — closures capture these, so threads cannot stomp on
        # each other's connection or callback buckets.
        transcripts: dict[int, str] = {}
        error_box: list[object] = [None]
        turn_done = threading.Event()

        client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
            )
        )

        def on_turn(_client, event):
            if event.transcript and event.end_of_turn and event.turn_is_formatted:
                transcripts[event.turn_order] = event.transcript
                turn_done.set()

        def on_error(_client, err):
            error_box[0] = err
            turn_done.set()

        def on_terminated(_client, _event):
            turn_done.set()

        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Error, on_error)
        client.on(StreamingEvents.Termination, on_terminated)
        client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
                speech_model=SpeechModel.universal_streaming_english,
            )
        )

        start_time = time.time()

        chunk_size = 3200  # 100ms of 16kHz 16-bit audio
        for i in range(0, len(pcm_data), chunk_size):
            client.stream(pcm_data[i : i + chunk_size])
            time.sleep(0.02)

        client.disconnect(terminate=True)
        turn_done.wait(timeout=30)

        elapsed = time.time() - start_time

        if error_box[0] is not None:
            err = error_box[0]
            code = getattr(err, "code", None)
            msg = str(err) or "no message"
            exc = RuntimeError(f"Streaming error (code={code}): {msg}")
            exc.streaming_code = code  # type: ignore[attr-defined]
            raise exc

        full_transcript = " ".join(transcripts[k] for k in sorted(transcripts.keys()))
        return full_transcript, elapsed


class DeepgramEvaluator(Evaluator):
    """Evaluator for Deepgram Nova 3 API."""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        from deepgram import DeepgramClient

        self.client = DeepgramClient(api_key=api_key)

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()

        response = self.client.listen.v1.media.transcribe_file(
            request=wav_bytes,
            model="nova-3",
        )
        elapsed = time.time() - start

        text = response.results.channels[0].alternatives[0].transcript
        return text, elapsed, None


class ElevenLabsEvaluator(Evaluator):
    """Evaluator for ElevenLabs Scribe API."""

    def __init__(self, api_key: str, model: str = "scribe_v2", **kwargs):
        super().__init__(**kwargs)
        from elevenlabs.client import ElevenLabs

        self.client = ElevenLabs(api_key=api_key)
        self.model = model

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()

        transcription = self.client.speech_to_text.convert(
            file=io.BytesIO(wav_bytes),
            model_id=self.model,
        )
        elapsed = time.time() - start

        # Extract text from transcription response
        text = transcription.text if hasattr(transcription, "text") else str(transcription)
        return text or "", elapsed, None


def _pump_run_loop_until(event: threading.Event, timeout_seconds: float) -> bool:
    """Pump the main CF run loop in 50ms slices until event is set or timeout.

    Speech.framework delivers callbacks via the main run loop; a plain
    threading.Event.wait() starves the framework's XPC delivery and the
    callback never fires.
    """
    deadline = time.time() + timeout_seconds
    while not event.is_set():
        if time.time() >= deadline:
            return False
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, True)
    return True


class AppleSpeechEvaluator(Evaluator):
    """Evaluator for Apple SFSpeechRecognizer (on-device, macOS only)."""

    AUTH_TIMEOUT_SECONDS = 300.0
    TRANSCRIBE_TIMEOUT_SECONDS = 60.0

    def __init__(self, locale: str = "en-US", **kwargs):
        if not _APPLE_SPEECH_AVAILABLE:
            raise ImportError(
                "Apple SFSpeechRecognizer backend requires PyObjC on macOS. "
                "Install with: pip install pyobjc-framework-Speech"
            )

        if kwargs.get("num_workers", 1) > 1:
            console.print(
                "[yellow]Warning: AppleSpeechEvaluator forces num_workers=1 "
                "(SFSpeechRecognizer is single-task)[/yellow]"
            )
            kwargs["num_workers"] = 1
        super().__init__(**kwargs)

        self.locale = locale
        self.temp_dir = tempfile.mkdtemp(prefix="apple-speech-")
        self._authorize()
        self.recognizer = self._build_recognizer(locale)

    def _authorize(self) -> None:
        auth_event = threading.Event()
        status_box = [None]

        def handler(status):
            status_box[0] = status
            auth_event.set()

        SFSpeechRecognizer.requestAuthorization_(handler)
        if not _pump_run_loop_until(auth_event, self.AUTH_TIMEOUT_SECONDS):
            raise TimeoutError("Speech recognition authorization request timed out")
        if status_box[0] != SFSpeechRecognizerAuthorizationStatusAuthorized:
            raise RuntimeError(
                f"Speech recognition not authorized (status={status_box[0]}). "
                "Approve at System Settings > Privacy & Security > Speech Recognition."
            )

    def _build_recognizer(self, locale: str):
        ns_locale = NSLocale.alloc().initWithLocaleIdentifier_(locale)
        recognizer = SFSpeechRecognizer.alloc().initWithLocale_(ns_locale)
        if recognizer is None:
            raise ValueError(f"Unsupported locale: {locale}")
        if not recognizer.supportsOnDeviceRecognition():
            raise RuntimeError(f"On-device recognition unavailable for locale {locale}")
        if not recognizer.isAvailable():
            raise RuntimeError("SFSpeechRecognizer not available right now")
        return recognizer

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        wav_bytes = prepare_wav_bytes(audio)
        fd, temp_path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(wav_bytes)

            url = NSURL.fileURLWithPath_(temp_path)
            request = SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
            request.setRequiresOnDeviceRecognition_(True)
            request.setShouldReportPartialResults_(False)

            done_event = threading.Event()
            text_box = [""]
            error_box = [None]

            def handler(result, error):
                if error is not None:
                    error_box[0] = str(error)
                    done_event.set()
                    return
                if result is None:
                    return
                if result.isFinal():
                    text_box[0] = str(result.bestTranscription().formattedString())
                    done_event.set()

            start = time.time()
            task = self.recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)

            if not _pump_run_loop_until(done_event, self.TRANSCRIBE_TIMEOUT_SECONDS):
                task.cancel()
                raise RuntimeError(
                    f"Recognition timed out after {self.TRANSCRIBE_TIMEOUT_SECONDS}s"
                )
            elapsed = time.time() - start

            if error_box[0]:
                raise RuntimeError(f"SFSpeechRecognizer error: {error_box[0]}")
            return text_box[0], elapsed, None
        finally:
            with contextlib.suppress(OSError):
                Path(temp_path).unlink()

    def close(self) -> None:
        if getattr(self, "temp_dir", None):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


class SwiftSDKEvaluator(Evaluator):
    """Evaluator for the TinyAudio Swift SDK on Apple Silicon.

    Triggered by `ta eval -m swift://<repo-id>` or `ta eval -m swift://<path>`.
    Builds the SDK in release mode, then subprocesses to a persistent Swift
    binary (`tiny-audio-swift-eval`) that loads the SDK's Transcriber once
    and processes audio files from stdin.

    By default the binary downloads the bundled HF weights at first run.
    Pass ``model_dir`` to evaluate against a locally-built bundle (sets
    ``TINY_AUDIO_LOCAL_MODEL_DIR`` in the subprocess env — see
    ``Transcriber.load()`` in tiny-audio-swift). The ``repo_id`` arg is
    informational only.

    The Swift package now lives in a sibling repo. Override its location
    via the ``TINY_AUDIO_SWIFT_DIR`` env var (path to the ``swift/``
    package root). Default: ``~/Code/ios/tiny-audio-swift/swift``.
    """

    def __init__(
        self,
        repo_id: str = "mazesmazes/tiny-audio-mlx",
        model_dir: Path | None = None,
        **kwargs,
    ):
        if kwargs.get("num_workers", 1) > 1:
            console.print(
                "[yellow]Warning: SwiftSDKEvaluator forces num_workers=1 "
                "(single Swift subprocess)[/yellow]"
            )
            kwargs["num_workers"] = 1
        super().__init__(**kwargs)
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else None
        if self.model_dir is not None:
            console.print(
                f"[bold green]Swift SDK using local model dir:[/bold green] {self.model_dir}"
            )
        else:
            console.print(
                f"[dim]Swift SDK ignores repo_id={repo_id!r} "
                "(loads bundled HF weights — pass swift://<path> to override)[/dim]"
            )

        swift_dir = Path(
            os.environ.get(
                "TINY_AUDIO_SWIFT_DIR",
                Path.home() / "Code" / "ios" / "tiny-audio-swift" / "swift",
            )
        ).expanduser()
        if not (swift_dir / "Package.swift").exists():
            raise RuntimeError(
                f"Swift package not found at {swift_dir}. Set TINY_AUDIO_SWIFT_DIR "
                f"to the path containing Package.swift (the tiny-audio-swift "
                f"checkout's swift/ directory)."
            )
        swift_build = swift_dir / ".build"

        # Build the debug test bundle first: SwiftPM only emits `mlx.metallib`
        # for XCTest targets, not standalone executables. Cheap when up-to-date.
        console.print("[bold cyan]Building Swift tests (for mlx.metallib)...[/bold cyan]")
        test_build_result = subprocess.run(
            ["swift", "build", "--package-path", str(swift_dir), "--build-tests"],
            check=False,
            capture_output=True,
            text=True,
        )
        if test_build_result.returncode != 0:
            raise RuntimeError(
                "swift build --build-tests failed:\n"
                f"stdout:\n{test_build_result.stdout}\n"
                f"stderr:\n{test_build_result.stderr}"
            )

        # Always rebuild release before running. Cheap when up-to-date.
        console.print("[bold cyan]Building tiny-audio-swift-eval (release)...[/bold cyan]")
        build_result = subprocess.run(
            check=False,
            args=[
                "swift",
                "build",
                "--package-path",
                str(swift_dir),
                "-c",
                "release",
                "--product",
                "tiny-audio-swift-eval",
            ],
            capture_output=True,
            text=True,
        )
        if build_result.returncode != 0:
            raise RuntimeError(
                "swift build failed:\n"
                f"stdout:\n{build_result.stdout}\n"
                f"stderr:\n{build_result.stderr}"
            )

        binary = swift_build / "release" / "tiny-audio-swift-eval"
        if not binary.exists():
            raise RuntimeError(f"tiny-audio-swift-eval binary missing after build: {binary}")

        # Copy mlx.metallib next to the release binary so MLX can find it at
        # runtime. The metallib should land in the debug test bundle after
        # `swift build --build-tests`, but mlx-swift's build process is flaky
        # about generating it in fresh checkouts. Fall back to scanning other
        # tiny-audio-swift checkouts on disk for a same-version metallib.
        binary_dir = binary.parent
        metallib_dst = binary_dir / "mlx.metallib"
        if not metallib_dst.exists():
            arch = platform.machine()  # "arm64" on Apple Silicon, "x86_64" on Intel
            primary_src = (
                swift_build
                / f"{arch}-apple-macosx"
                / "debug"
                / "TinyAudioPackageTests.xctest"
                / "Contents"
                / "MacOS"
                / "mlx.metallib"
            )
            metallib_src: Path | None = primary_src if primary_src.exists() else None
            if metallib_src is None:
                xctest_subpath = (
                    f"swift/.build/{arch}-apple-macosx/debug/"
                    "TinyAudioPackageTests.xctest/Contents/MacOS/mlx.metallib"
                )
                # Search siblings + worktrees, e.g. ~/Code/tiny-audio*/swift/.build/...
                for candidate in (Path.home() / "Code").glob(f"*/{xctest_subpath}"):
                    metallib_src = candidate
                    break
                if metallib_src is None:
                    for candidate in (Path.home() / "Code").glob(
                        f"*/.claude/worktrees/*/{xctest_subpath}"
                    ):
                        metallib_src = candidate
                        break
            if metallib_src is None:
                raise RuntimeError(
                    f"mlx.metallib not found at {primary_src} and no fallback "
                    "metallib located on disk. mlx-swift's build process did not "
                    "emit one. Workaround: copy a working `mlx.metallib` from "
                    "another tiny-audio-swift checkout's debug test bundle into "
                    f"{primary_src}, then re-run."
                )
            shutil.copy2(str(metallib_src), str(metallib_dst))
            console.print(f"[dim]Copied mlx.metallib from {metallib_src} to binary directory[/dim]")

        cmd = [str(binary)]
        console.print(f"[bold cyan]Spawning Swift SDK eval subprocess:[/bold cyan] {' '.join(cmd)}")

        env = os.environ.copy()
        if self.model_dir is not None:
            env["TINY_AUDIO_LOCAL_MODEL_DIR"] = str(self.model_dir)

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # captured for diagnostics on crash
            text=True,
            bufsize=1,  # line-buffered
            env=env,
        )

        # Wait for the binary to emit `{"ready": true}` once load + warmup is done.
        ready_line = self.proc.stdout.readline()
        if not ready_line:
            err = self.proc.stderr.read()
            raise RuntimeError(f"Swift binary failed to start. stderr:\n{err}")
        try:
            ready_msg = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Swift binary emitted non-JSON startup line: {ready_line!r}"
            ) from exc
        if "error" in ready_msg:
            raise RuntimeError(f"Swift binary load failed: {ready_msg['error']}")
        if not ready_msg.get("ready"):
            raise RuntimeError(f"Swift binary unexpected startup line: {ready_msg}")
        console.print("[bold green]Swift SDK ready[/bold green]")

    def transcribe(self, audio) -> tuple[str, float, dict | None]:
        # The eval framework passes either a path string or a dict-like with an
        # 'array' field. The Swift binary takes file paths only — write to a
        # temp wav if we got an in-memory array.
        path, is_temp = self._resolve_audio_path(audio)
        try:
            self.proc.stdin.write(path + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read()
                raise RuntimeError(f"Swift binary closed unexpectedly. stderr:\n{err}")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Swift binary emitted non-JSON line: {line!r}") from exc
            if "error" in msg:
                raise RuntimeError(f"Swift transcribe failed: {msg['error']}")
            text = msg.get("text", "")
            elapsed_ms = msg.get("elapsed_ms", 0)
            return text, elapsed_ms / 1000.0, None
        finally:
            if is_temp:
                Path(path).unlink(missing_ok=True)

    def _resolve_audio_path(self, audio) -> tuple[str, bool]:
        """Return (filesystem_path, is_temp) for the Swift binary to read.

        The eval framework's `audio` argument can be:
        - a string path (HF datasets sometimes give the source path)
        - a dict with 'path' (HF datasets style)
        - a dict with 'array' + 'sampling_rate' (decoded numpy)
        - a torchcodec AudioDecoder
        - an AudioSamples-like object with `.data` and `.sample_rate`
        For arrays/decoders without a path, materialise to a temp wav.
        """
        import numpy as np

        if isinstance(audio, str):
            return audio, False
        if isinstance(audio, dict):
            # HF datasets Audio feature: dict with 'path' (absolute) + optionally 'array'/'bytes'.
            path = audio.get("path")
            if path and Path(path).is_absolute() and Path(path).exists():
                return path, False
            # 'bytes' field: raw file bytes (wav, mp3, etc.) — write to temp file.
            if audio.get("bytes"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio["bytes"])
                    tmp_name = tmp.name
                return tmp_name, True
            # 'array' + 'sampling_rate': decoded numpy array — encode to wav.
            array_val = audio.get("array")
            if array_val is None:
                raise ValueError("audio dict has no usable 'path', 'bytes', or 'array' field")
            arr = np.asarray(array_val, dtype=np.float32)
            sr = int(audio.get("sampling_rate", 16000))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
            sf.write(tmp_name, arr, sr, subtype="PCM_16")
            return tmp_name, True
        # AudioDecoder / AudioSamples fallback for torchcodec / SDK-style inputs.
        if hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            data = samples.data.detach().cpu().numpy()
            sr = int(samples.sample_rate)
            if data.ndim > 1:
                data = data.mean(axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
            sf.write(tmp_name, data, sr, subtype="PCM_16")
            return tmp_name, True
        raise ValueError(f"unsupported audio input type for Swift eval: {type(audio)}")

    def __del__(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                with contextlib.suppress(Exception):
                    self.proc.wait(timeout=1)
