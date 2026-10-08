"""Evaluator for the TinyAudio Swift SDK, driven through a persistent subprocess."""

import contextlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import IO, Any, Unpack

import numpy as np
import soundfile as sf

from scripts.eval.audio import LazyAudioDecoder, is_str_dict

from .base import Evaluator, EvaluatorOptions, Transcription, console


def _swift_package_dir() -> Path:
    """Locate the tiny-audio-swift package root (TINY_AUDIO_SWIFT_DIR or the default)."""
    swift_dir = Path(
        os.environ.get(
            "TINY_AUDIO_SWIFT_DIR",
            Path.home() / "Code" / "ios" / "tiny-audio-swift" / "swift",
        )
    ).expanduser()
    if not (swift_dir / "Package.swift").exists():
        msg = (
            f"Swift package not found at {swift_dir}. Set TINY_AUDIO_SWIFT_DIR "
            f"to the path containing Package.swift (the tiny-audio-swift "
            f"checkout's swift/ directory)."
        )
        raise RuntimeError(msg)
    return swift_dir


def _run_swift_build(args: list[str], failure: str) -> None:
    """Run a `swift build` invocation, raising with its output if it fails."""
    result = subprocess.run(
        check=False,
        args=args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = f"{failure}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        raise RuntimeError(msg)


def _build_eval_binary(swift_dir: Path) -> Path:
    """Build the test bundle (for mlx.metallib) and the release eval binary; return it."""
    # Build the debug test bundle first: SwiftPM only emits `mlx.metallib`
    # for XCTest targets, not standalone executables. Cheap when up-to-date.
    console.print("[bold cyan]Building Swift tests (for mlx.metallib)...[/bold cyan]")
    _run_swift_build(
        ["swift", "build", "--package-path", str(swift_dir), "--build-tests"],
        "swift build --build-tests failed",
    )

    # Always rebuild release before running. Cheap when up-to-date.
    console.print("[bold cyan]Building tiny-audio-swift-eval (release)...[/bold cyan]")
    _run_swift_build(
        [
            "swift",
            "build",
            "--package-path",
            str(swift_dir),
            "-c",
            "release",
            "--product",
            "tiny-audio-swift-eval",
        ],
        "swift build failed",
    )

    binary = swift_dir / ".build" / "release" / "tiny-audio-swift-eval"
    if not binary.exists():
        msg = f"tiny-audio-swift-eval binary missing after build: {binary}"
        raise RuntimeError(msg)
    return binary


def _fallback_metallib(arch: str) -> Path | None:
    """Find a same-layout mlx.metallib in another tiny-audio-swift checkout on disk."""
    xctest_subpath = (
        f"swift/.build/{arch}-apple-macosx/debug/"
        "TinyAudioPackageTests.xctest/Contents/MacOS/mlx.metallib"
    )
    # Search siblings + worktrees, e.g. ~/Code/tiny-audio*/swift/.build/...
    for pattern in (f"*/{xctest_subpath}", f"*/.claude/worktrees/*/{xctest_subpath}"):
        for candidate in (Path.home() / "Code").glob(pattern):
            return candidate
    return None


def _ensure_metallib(binary: Path, swift_dir: Path) -> None:
    """Copy mlx.metallib next to the release binary if it is not already there."""
    # Copy mlx.metallib next to the release binary so MLX can find it at
    # runtime. The metallib should land in the debug test bundle after
    # `swift build --build-tests`, but mlx-swift's build process is flaky
    # about generating it in fresh checkouts. Fall back to scanning other
    # tiny-audio-swift checkouts on disk for a same-version metallib.
    metallib_dst = binary.parent / "mlx.metallib"
    if metallib_dst.exists():
        return
    arch = platform.machine()  # "arm64" on Apple Silicon, "x86_64" on Intel
    primary_src = (
        swift_dir
        / ".build"
        / f"{arch}-apple-macosx"
        / "debug"
        / "TinyAudioPackageTests.xctest"
        / "Contents"
        / "MacOS"
        / "mlx.metallib"
    )
    metallib_src = primary_src if primary_src.exists() else _fallback_metallib(arch)
    if metallib_src is None:
        msg = (
            f"mlx.metallib not found at {primary_src} and no fallback "
            "metallib located on disk. mlx-swift's build process did not "
            "emit one. Workaround: copy a working `mlx.metallib` from "
            "another tiny-audio-swift checkout's debug test bundle into "
            f"{primary_src}, then re-run."
        )
        raise RuntimeError(msg)
    shutil.copy2(str(metallib_src), str(metallib_dst))
    console.print(f"[dim]Copied mlx.metallib from {metallib_src} to binary directory[/dim]")


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
        **kwargs: Unpack[EvaluatorOptions],
    ) -> None:
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

        swift_dir = _swift_package_dir()
        binary = _build_eval_binary(swift_dir)
        _ensure_metallib(binary, swift_dir)
        self._spawn(binary)

    def _spawn(self, binary: Path) -> None:
        """Start the persistent eval binary and wait for its ready handshake."""
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
        _, stdout, stderr = self._pipes()
        ready_line = stdout.readline()
        if not ready_line:
            err = stderr.read()
            msg = f"Swift binary failed to start. stderr:\n{err}"
            raise RuntimeError(msg)
        try:
            ready_msg: dict[str, Any] = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            msg = f"Swift binary emitted non-JSON startup line: {ready_line!r}"
            raise RuntimeError(msg) from exc
        if "error" in ready_msg:
            msg = f"Swift binary load failed: {ready_msg['error']}"
            raise RuntimeError(msg)
        if not ready_msg.get("ready"):
            msg = f"Swift binary unexpected startup line: {ready_msg}"
            raise RuntimeError(msg)
        console.print("[bold green]Swift SDK ready[/bold green]")

    def transcribe(self, audio: object) -> Transcription:
        # The eval framework passes either a path string or a dict-like with an
        # 'array' field. The Swift binary takes file paths only — write to a
        # temp wav if we got an in-memory array.
        path, is_temp = self._resolve_audio_path(audio)
        try:
            stdin, stdout, stderr = self._pipes()
            stdin.write(path + "\n")
            stdin.flush()
            line = stdout.readline()
            if not line:
                err = stderr.read()
                msg = f"Swift binary closed unexpectedly. stderr:\n{err}"
                raise RuntimeError(msg)
            try:
                reply: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Swift binary emitted non-JSON line: {line!r}"
                raise RuntimeError(msg) from exc
            if "error" in reply:
                msg = f"Swift transcribe failed: {reply['error']}"
                raise RuntimeError(msg)
            text: str = reply.get("text", "")
            elapsed_ms: float = reply.get("elapsed_ms", 0)
            return text, elapsed_ms / 1000.0, None
        finally:
            if is_temp:
                Path(path).unlink(missing_ok=True)

    def _pipes(self) -> tuple[IO[str], IO[str], IO[str]]:
        """The binary's stdin, stdout and stderr; all three are opened as PIPEs."""
        proc = self.proc
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None
        return proc.stdin, proc.stdout, proc.stderr

    def _resolve_audio_path(self, audio: object) -> tuple[str, bool]:
        """Return (filesystem_path, is_temp) for the Swift binary to read.

        The eval framework's `audio` argument can be:
        - a string path (HF datasets sometimes give the source path)
        - a dict with 'path' (HF datasets style)
        - a dict with 'array' + 'sampling_rate' (decoded numpy)
        - a torchcodec AudioDecoder
        - an AudioSamples-like object with `.data` and `.sample_rate`
        For arrays/decoders without a path, materialise to a temp wav.
        """
        if isinstance(audio, str):
            return audio, False
        if is_str_dict(audio):
            # HF datasets Audio feature: dict with 'path' (absolute) + optionally 'array'/'bytes'.
            path: str | None = audio.get("path")
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
                msg = "audio dict has no usable 'path', 'bytes', or 'array' field"
                raise ValueError(msg)
            arr = np.asarray(array_val, dtype=np.float32)
            sr = int(audio.get("sampling_rate", 16000))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
            sf.write(tmp_name, arr, sr, subtype="PCM_16")
            return tmp_name, True
        # AudioDecoder / AudioSamples fallback for torchcodec / SDK-style inputs.
        if isinstance(audio, LazyAudioDecoder):
            samples = audio.get_all_samples()
            data: np.ndarray[Any, np.dtype[np.floating[Any]]] = samples.data.detach().cpu().numpy()
            sr = int(samples.sample_rate)
            if data.ndim > 1:
                data = data.mean(axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
            sf.write(tmp_name, data, sr, subtype="PCM_16")
            return tmp_name, True
        msg = f"unsupported audio input type for Swift eval: {type(audio)}"
        raise ValueError(msg)

    def __del__(self) -> None:
        if hasattr(self, "proc") and self.proc.poll() is None:
            try:
                self._pipes()[0].close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                with contextlib.suppress(Exception):
                    self.proc.wait(timeout=1)
