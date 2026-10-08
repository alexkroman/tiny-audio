"""Tests for scripts/deploy/handler_local.py (the handler itself is faked)."""

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest
import typer
from typer.testing import CliRunner

from scripts.deploy import handler_local

runner = CliRunner()


class FakeHandler:
    """Records requests and returns a canned result (or raises)."""

    def __init__(self, path: str, result: Any = None, error: Exception | None = None) -> None:
        self.path = path
        self.result = {"text": "hello world"} if result is None else result
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def __call__(self, data: dict[str, Any]) -> Any:
        self.requests.append(data)
        if self.error is not None:
            raise self.error
        return self.result


def _install_handler_module(
    monkeypatch: pytest.MonkeyPatch, **handler_kwargs: Any
) -> list[FakeHandler]:
    """Make `importlib.import_module("tiny_audio.handler")` return a fake module."""
    created: list[FakeHandler] = []

    def factory(path: str) -> FakeHandler:
        handler = FakeHandler(path, **handler_kwargs)
        created.append(handler)
        return handler

    module = ModuleType("tiny_audio.handler")
    module.EndpointHandler = factory  # type: ignore[attr-defined]
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any) -> ModuleType:
        return module if name == "tiny_audio.handler" else real_import(name, *args)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    return created


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF")
    return path


class TestFindLatestModel:
    def test_missing_dir(self, tmp_path: Path) -> None:
        assert handler_local.find_latest_model(str(tmp_path / "nope")) is None

    def test_no_checkpoints(self, tmp_path: Path) -> None:
        assert handler_local.find_latest_model(str(tmp_path)) is None


class TestFindTestAudio:
    def test_prefers_gradio_sample(self) -> None:
        found = handler_local.find_test_audio()
        assert found is not None
        assert Path(found).is_file()

    def test_falls_back_to_project_audio(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(handler_local, "get_project_root", lambda: tmp_path)

        def files(_package: str) -> MagicMock:
            package = MagicMock()
            package.joinpath.return_value.is_file.return_value = False
            return package

        monkeypatch.setattr(handler_local.resources, "files", files)
        assert handler_local.find_test_audio() is None

        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x.flac").write_bytes(b"")
        assert handler_local.find_test_audio() == str(tmp_path / "sub" / "x.flac")

        (tmp_path / "demo").mkdir()
        (tmp_path / "demo" / "sample.wav").write_bytes(b"")
        assert handler_local.find_test_audio() == str(tmp_path / "demo" / "sample.wav")

    def test_no_gradio(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def no_module(_: str) -> None:
            raise ModuleNotFoundError("gradio")

        monkeypatch.setattr(handler_local, "get_project_root", lambda: tmp_path)
        monkeypatch.setattr(handler_local.resources, "files", no_module)
        assert handler_local.find_test_audio() is None


class TestResolveAudioPath:
    def test_explicit_path_wins(self, audio_file: Path) -> None:
        assert handler_local._resolve_audio_path(audio_file) == str(audio_file)

    def test_autodetect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(handler_local, "find_test_audio", lambda: "/a.wav")
        assert handler_local._resolve_audio_path(None) == "/a.wav"

    def test_none_found_is_bad_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(handler_local, "find_test_audio", lambda: None)
        with pytest.raises(typer.BadParameter):
            handler_local._resolve_audio_path(None)


class TestLoadHandler:
    def test_import_error_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(name: str, *_: Any) -> None:
            msg = "no torch"
            raise ImportError(msg)

        monkeypatch.setattr(importlib, "import_module", fail)
        with pytest.raises(typer.Exit):
            handler_local._load_handler("m")

    def test_constructor_error_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = ModuleType("tiny_audio.handler")

        def broken(path: str) -> None:
            msg = "bad weights"
            raise RuntimeError(msg)

        module.EndpointHandler = broken  # type: ignore[attr-defined]

        def import_module(_name: str) -> ModuleType:
            return module

        monkeypatch.setattr(importlib, "import_module", import_module)
        with pytest.raises(typer.Exit):
            handler_local._load_handler("m")


class TestRunHandlerCommand:
    def test_single_greedy(self, monkeypatch: pytest.MonkeyPatch, audio_file: Path) -> None:
        created = _install_handler_module(monkeypatch)
        result = runner.invoke(
            handler_local.app, ["--model", "my/model", "--audio", str(audio_file)]
        )
        assert result.exit_code == 0, result.output
        assert "hello world" in result.output
        assert "Test completed!" in result.output
        (handler,) = created
        assert handler.path == "my/model"
        assert handler.requests == [
            {
                "inputs": str(audio_file),
                "parameters": {"max_new_tokens": 200, "num_beams": 1, "do_sample": False},
            }
        ]

    def test_sampling_passes_temperature(
        self, monkeypatch: pytest.MonkeyPatch, audio_file: Path
    ) -> None:
        created = _install_handler_module(monkeypatch, result=["not", "a", "dict"])
        result = runner.invoke(
            handler_local.app,
            ["--audio", str(audio_file), "--do-sample", "--temperature", "0.7"],
        )
        assert result.exit_code == 0, result.output
        assert created[0].requests[0]["parameters"]["temperature"] == 0.7
        assert '"not"' in result.output  # non-dict results are dumped as JSON

    def test_dict_without_text_is_dumped(
        self, monkeypatch: pytest.MonkeyPatch, audio_file: Path
    ) -> None:
        _install_handler_module(monkeypatch, result={"words": []})
        result = runner.invoke(handler_local.app, ["--audio", str(audio_file)])
        assert '"words": []' in result.output

    def test_inference_failure_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, audio_file: Path
    ) -> None:
        _install_handler_module(monkeypatch, error=RuntimeError("OOM"))
        result = runner.invoke(handler_local.app, ["--audio", str(audio_file)])
        assert result.exit_code == 0
        assert "Inference failed: OOM" in result.output

    def test_batch(self, monkeypatch: pytest.MonkeyPatch, audio_file: Path) -> None:
        created = _install_handler_module(monkeypatch, result={"texts": ["a", "b", "c"]})
        result = runner.invoke(handler_local.app, ["--audio", str(audio_file), "--batch-test"])
        assert result.exit_code == 0, result.output
        request = created[0].requests[0]
        assert request["inputs"] == [str(audio_file)] * 3
        assert request["parameters"]["batch_size"] == 3
        assert "Sample 3: c" in result.output

    def test_batch_unexpected_shape_is_dumped(
        self, monkeypatch: pytest.MonkeyPatch, audio_file: Path
    ) -> None:
        _install_handler_module(monkeypatch, result={"text": "x"})
        result = runner.invoke(handler_local.app, ["--audio", str(audio_file), "--batch-test"])
        assert '"text": "x"' in result.output

    def test_batch_failure_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, audio_file: Path
    ) -> None:
        _install_handler_module(monkeypatch, error=ValueError("bad batch"))
        result = runner.invoke(handler_local.app, ["--audio", str(audio_file), "--batch-test"])
        assert result.exit_code == 0
        assert "Batch inference failed: bad batch" in result.output
