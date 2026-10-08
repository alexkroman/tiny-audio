"""Tests for scripts.eval.evaluators.swift_sdk with the Swift toolchain faked out."""

import io
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.eval.evaluators import base, swift_sdk
from scripts.eval.evaluators.swift_sdk import (
    SwiftSDKEvaluator,
    _build_eval_binary,
    _ensure_metallib,
    _fallback_metallib,
    _run_swift_build,
    _swift_package_dir,
)

_XCTEST_METALLIB = Path("debug/TinyAudioPackageTests.xctest/Contents/MacOS/mlx.metallib")


class FakeCompleted:
    """The subset of `subprocess.CompletedProcess` that `_run_swift_build` reads."""

    def __init__(self, returncode: int, stdout: str = "out", stderr: str = "err") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeProc:
    """A `subprocess.Popen` stand-in whose stdout replays canned JSON lines."""

    instances: list["FakeProc"] = []
    stdout_text = '{"ready": true}\n'

    def __init__(self, cmd: list[str], **kwargs: Any) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(FakeProc.stdout_text)
        self.stderr = io.StringIO("boom on stderr")
        self.returncode: int | None = None
        self.killed = False
        self.wait_timeouts: list[float] = []
        self.fail_wait = False
        FakeProc.instances.append(self)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        self.wait_timeouts.append(timeout)
        if self.fail_wait and not self.killed:
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        self.returncode = -9 if self.killed else 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True


@pytest.fixture(autouse=True)
def _offline_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # pyright: ignore[reportUnusedFunction]
    """The base Evaluator builds a Whisper normalizer, which downloads from the Hub."""
    monkeypatch.setattr(base, "TextNormalizer", lambda: None)


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> type[FakeProc]:
    FakeProc.instances = []
    FakeProc.stdout_text = '{"ready": true}\n'
    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    return FakeProc


@pytest.fixture
def swift_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake Swift package with a built release binary and its metallib in place."""
    pkg = tmp_path / "swift"
    release = pkg / ".build" / "release"
    release.mkdir(parents=True)
    (pkg / "Package.swift").write_text("// swift-tools-version")
    (release / "tiny-audio-swift-eval").write_text("binary")
    (release / "mlx.metallib").write_text("metallib")
    monkeypatch.setenv("TINY_AUDIO_SWIFT_DIR", str(pkg))
    return pkg


@pytest.fixture
def build_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(*, args: list[str], **_: Any) -> FakeCompleted:
        calls.append(args)
        return FakeCompleted(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _evaluator(fake: type[FakeProc]) -> tuple[SwiftSDKEvaluator, FakeProc]:
    evaluator = SwiftSDKEvaluator()
    return evaluator, fake.instances[-1]


# ------------------------------------------------------------ module helpers


def test_swift_package_dir_uses_env(swift_dir: Path) -> None:
    assert _swift_package_dir() == swift_dir


def test_swift_package_dir_defaults_under_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TINY_AUDIO_SWIFT_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / "Code" / "ios" / "tiny-audio-swift" / "swift"
    with pytest.raises(RuntimeError, match="Swift package not found") as exc:
        _swift_package_dir()
    assert str(default) in str(exc.value)
    default.mkdir(parents=True)
    (default / "Package.swift").write_text("")
    assert _swift_package_dir() == default


def test_run_swift_build_success_passes_args(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> FakeCompleted:
        seen.update(kwargs)
        return FakeCompleted(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    _run_swift_build(["swift", "build"], "never shown")
    assert seen == {
        "check": False,
        "args": ["swift", "build"],
        "capture_output": True,
        "text": True,
    }


def test_run_swift_build_failure_includes_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda **_: FakeCompleted(1, stdout="compiling", stderr="error: x")
    )
    with pytest.raises(RuntimeError, match="build broke") as exc:
        _run_swift_build(["swift", "build"], "build broke")
    assert "compiling" in str(exc.value)
    assert "error: x" in str(exc.value)


def test_build_eval_binary_runs_tests_then_release(
    swift_dir: Path, build_calls: list[list[str]]
) -> None:
    binary = _build_eval_binary(swift_dir)
    assert binary == swift_dir / ".build" / "release" / "tiny-audio-swift-eval"
    assert build_calls == [
        ["swift", "build", "--package-path", str(swift_dir), "--build-tests"],
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
    ]


def test_build_eval_binary_missing_binary(swift_dir: Path, build_calls: list[list[str]]) -> None:
    (swift_dir / ".build" / "release" / "tiny-audio-swift-eval").unlink()
    with pytest.raises(RuntimeError, match="binary missing after build"):
        _build_eval_binary(swift_dir)
    assert len(build_calls) == 2


def test_build_eval_binary_stops_on_test_build_failure(
    swift_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(*, args: list[str], **_: Any) -> FakeCompleted:
        calls.append(args)
        return FakeCompleted(1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="--build-tests failed"):
        _build_eval_binary(swift_dir)
    assert len(calls) == 1


def _metallib_at(root: Path, arch: str, prefix: str) -> Path:
    path = root / prefix / "swift" / ".build" / f"{arch}-apple-macosx" / _XCTEST_METALLIB
    path.parent.mkdir(parents=True)
    path.write_text(f"from:{prefix}")
    return path


def test_fallback_metallib_finds_sibling_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert _fallback_metallib("arm64") is None
    found = _metallib_at(tmp_path / "Code", "arm64", "tiny-audio-other")
    assert _fallback_metallib("arm64") == found
    assert _fallback_metallib("x86_64") is None


def test_fallback_metallib_finds_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    found = _metallib_at(tmp_path / "Code" / "repo" / ".claude" / "worktrees", "arm64", "wt")
    assert _fallback_metallib("arm64") == found


def test_ensure_metallib_noop_when_present(swift_dir: Path) -> None:
    binary = swift_dir / ".build" / "release" / "tiny-audio-swift-eval"
    _ensure_metallib(binary, swift_dir)
    assert (binary.parent / "mlx.metallib").read_text() == "metallib"


def test_ensure_metallib_copies_primary(swift_dir: Path) -> None:
    binary = swift_dir / ".build" / "release" / "tiny-audio-swift-eval"
    (binary.parent / "mlx.metallib").unlink()
    _metallib_at(swift_dir.parent, platform.machine(), "")
    _ensure_metallib(binary, swift_dir)
    assert (binary.parent / "mlx.metallib").read_text() == "from:"


def test_ensure_metallib_uses_fallback(swift_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = swift_dir / ".build" / "release" / "tiny-audio-swift-eval"
    (binary.parent / "mlx.metallib").unlink()
    elsewhere = swift_dir.parent / "elsewhere.metallib"
    elsewhere.write_text("fallback")
    archs: list[str] = []

    def fake_fallback(arch: str) -> Path:
        archs.append(arch)
        return elsewhere

    monkeypatch.setattr(swift_sdk, "_fallback_metallib", fake_fallback)
    _ensure_metallib(binary, swift_dir)
    assert archs == [platform.machine()]
    assert (binary.parent / "mlx.metallib").read_text() == "fallback"


def test_ensure_metallib_raises_without_any_source(
    swift_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = swift_dir / ".build" / "release" / "tiny-audio-swift-eval"
    (binary.parent / "mlx.metallib").unlink()
    monkeypatch.setattr(swift_sdk, "_fallback_metallib", lambda arch: None)
    with pytest.raises(RuntimeError, match="mlx.metallib not found"):
        _ensure_metallib(binary, swift_dir)


# ---------------------------------------------------------------- evaluator


@pytest.mark.usefixtures("swift_dir", "build_calls")
def test_init_spawns_binary_and_forces_single_worker(fake_popen: type[FakeProc]) -> None:
    evaluator = SwiftSDKEvaluator(num_workers=4)
    proc = fake_popen.instances[-1]
    assert evaluator.num_workers == 1
    assert evaluator.model_dir is None
    assert proc.cmd[0].endswith("tiny-audio-swift-eval")
    assert proc.kwargs["bufsize"] == 1
    assert "TINY_AUDIO_LOCAL_MODEL_DIR" not in proc.kwargs["env"]


@pytest.mark.usefixtures("swift_dir", "build_calls")
def test_init_with_model_dir_sets_env(fake_popen: type[FakeProc], tmp_path: Path) -> None:
    evaluator = SwiftSDKEvaluator(model_dir=tmp_path / "bundle")
    proc = fake_popen.instances[-1]
    assert evaluator.model_dir == (tmp_path / "bundle").resolve()
    assert proc.kwargs["env"]["TINY_AUDIO_LOCAL_MODEL_DIR"] == str(evaluator.model_dir)


@pytest.mark.usefixtures("swift_dir", "build_calls")
@pytest.mark.parametrize(
    ("startup", "message"),
    [
        ("", "failed to start. stderr:\nboom on stderr"),
        ("not json\n", "non-JSON startup line"),
        ('{"error": "no weights"}\n', "load failed: no weights"),
        ('{"ready": false}\n', "unexpected startup line"),
    ],
)
def test_spawn_rejects_bad_handshake(
    fake_popen: type[FakeProc], startup: str, message: str
) -> None:
    fake_popen.stdout_text = startup
    with pytest.raises(RuntimeError) as exc:
        SwiftSDKEvaluator()
    assert message in str(exc.value)


def _queue_replies(proc: FakeProc, *lines: str) -> None:
    proc.stdout = io.StringIO("".join(f"{line}\n" for line in lines))


@pytest.mark.usefixtures("swift_dir", "build_calls")
def test_transcribe_path_round_trip(fake_popen: type[FakeProc]) -> None:
    evaluator, proc = _evaluator(fake_popen)
    _queue_replies(proc, json.dumps({"text": "hello world", "elapsed_ms": 250}))
    assert evaluator.transcribe("/data/clip.wav") == ("hello world", 0.25, None)
    assert proc.stdin.getvalue() == "/data/clip.wav\n"


@pytest.mark.usefixtures("swift_dir", "build_calls")
def test_transcribe_defaults_missing_fields(fake_popen: type[FakeProc]) -> None:
    evaluator, proc = _evaluator(fake_popen)
    _queue_replies(proc, "{}")
    assert evaluator.transcribe("/a.wav") == ("", 0.0, None)


@pytest.mark.usefixtures("swift_dir", "build_calls")
@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (None, "closed unexpectedly. stderr:\nboom on stderr"),
        ("garbage", "non-JSON line"),
        ('{"error": "decode failed"}', "transcribe failed: decode failed"),
    ],
)
def test_transcribe_errors(fake_popen: type[FakeProc], reply: str | None, message: str) -> None:
    evaluator, proc = _evaluator(fake_popen)
    if reply is None:
        proc.stdout = io.StringIO("")
    else:
        _queue_replies(proc, reply)
    with pytest.raises(RuntimeError) as exc:
        evaluator.transcribe("/a.wav")
    assert message in str(exc.value)


@pytest.mark.usefixtures("swift_dir", "build_calls")
def test_transcribe_array_writes_and_removes_temp_wav(
    fake_popen: type[FakeProc], monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator, proc = _evaluator(fake_popen)
    _queue_replies(proc, '{"text": "ok"}')
    written: list[tuple[int, int]] = []
    real_write = sf.write

    def capture(name: str, data: np.ndarray[Any, Any], sr: int, **kwargs: Any) -> None:
        written.append((sr, len(data)))
        real_write(name, data, sr, **kwargs)
        assert Path(name).exists()

    monkeypatch.setattr(swift_sdk.sf, "write", capture)
    audio = {"array": np.zeros(800, dtype=np.float32), "sampling_rate": 8000}
    assert evaluator.transcribe(audio)[0] == "ok"
    assert written == [(8000, 800)]
    assert not Path(proc.stdin.getvalue().strip()).exists()


@pytest.mark.usefixtures("swift_dir", "build_calls")
def test_resolve_audio_path_variants(fake_popen: type[FakeProc], tmp_path: Path) -> None:
    evaluator, _ = _evaluator(fake_popen)
    existing = tmp_path / "real.wav"
    existing.write_bytes(b"RIFF")
    assert evaluator._resolve_audio_path(str(existing)) == (str(existing), False)
    assert evaluator._resolve_audio_path({"path": str(existing)}) == (str(existing), False)

    name, is_temp = evaluator._resolve_audio_path({"path": "rel.wav", "bytes": b"abc"})
    assert is_temp
    assert Path(name).read_bytes() == b"abc"
    Path(name).unlink()

    with pytest.raises(ValueError, match="no usable"):
        evaluator._resolve_audio_path({"path": None})
    with pytest.raises(ValueError, match="unsupported audio input type"):
        evaluator._resolve_audio_path(42)


class _Samples:
    def __init__(self, data: torch.Tensor, sample_rate: float) -> None:
        self.data = data
        self.sample_rate = sample_rate


class _Decoder:
    def __init__(self, samples: _Samples) -> None:
        self._samples = samples

    def get_all_samples(self) -> _Samples:
        return self._samples


@pytest.mark.usefixtures("swift_dir", "build_calls")
def test_resolve_audio_path_downmixes_decoder(fake_popen: type[FakeProc]) -> None:
    evaluator, _ = _evaluator(fake_popen)
    stereo = torch.full((2, 1600), 0.25)
    name, is_temp = evaluator._resolve_audio_path(_Decoder(_Samples(stereo, 16000.0)))
    try:
        data, sr = sf.read(name)
        assert is_temp
        assert sr == 16000
        assert data.shape == (1600,)
        assert np.allclose(data, 0.25, atol=1e-3)
    finally:
        Path(name).unlink()


@pytest.mark.usefixtures("swift_dir", "build_calls")
def test_del_closes_stdin_and_waits(fake_popen: type[FakeProc]) -> None:
    evaluator, proc = _evaluator(fake_popen)
    evaluator.__del__()
    assert proc.stdin.closed
    assert proc.wait_timeouts == [5]
    assert not proc.killed
    evaluator.__del__()  # already exited: nothing more happens
    assert proc.wait_timeouts == [5]


@pytest.mark.usefixtures("swift_dir", "build_calls")
def test_del_kills_hung_process(fake_popen: type[FakeProc]) -> None:
    evaluator, proc = _evaluator(fake_popen)
    proc.fail_wait = True
    evaluator.__del__()
    assert proc.killed
    assert proc.wait_timeouts == [5, 1]
    assert proc.returncode == -9
