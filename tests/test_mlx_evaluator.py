"""Tests for scripts/eval/evaluators/mlx.py and `ta eval --mlx` (no MLX needed: faked)."""

import sys
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
import pytest
import typer

from scripts.eval.cli import _build_tiny_audio_evaluator
from scripts.eval.evaluators.mlx import MLXEvaluator


class _FakeTranscriber:
    """Stands in for MLXTranscriber: records what it was asked to load and transcribe."""

    loaded: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.revision = "0123456789abcdef"

    @classmethod
    def from_pretrained(cls, model_id: str) -> "_FakeTranscriber":
        cls.loaded.append(model_id)
        return cls()

    def transcribe(self, audio: np.ndarray[Any, Any]) -> str:
        return f"{audio.dtype} {audio.shape[0]}"


@pytest.fixture
def fake_mlx(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTranscriber]:
    """`scripts.mlx_inference` replaced by a module exposing the fake transcriber."""
    module = ModuleType("scripts.mlx_inference")
    module.__dict__["MLXTranscriber"] = _FakeTranscriber
    monkeypatch.setitem(sys.modules, "scripts.mlx_inference", module)
    _FakeTranscriber.loaded = []
    return _FakeTranscriber


class TestMLXEvaluator:
    """One worker, 16 kHz float32 audio in, the transcriber's text out."""

    def test_transcribes_with_the_loaded_model(self, fake_mlx: type[_FakeTranscriber]) -> None:
        evaluator = MLXEvaluator(model_path="org/model", num_workers=4)
        assert fake_mlx.loaded == ["org/model"]
        assert evaluator.num_workers == 1
        text, elapsed, confidence = evaluator.transcribe(
            {"array": np.ones(800, dtype=np.float64), "sampling_rate": 16000}
        )
        assert text == "float32 800"
        assert elapsed >= 0
        assert confidence is None

    @pytest.mark.usefixtures("fake_mlx")
    def test_speaker_datasets_are_refused(self) -> None:
        evaluator = MLXEvaluator(model_path="org/model")
        evaluator.speakers = True
        with pytest.raises(NotImplementedError, match="speakers"):
            evaluator.transcribe({"array": np.ones(10), "sampling_rate": 16000})


class TestEvalWiring:
    """`--mlx` picks the MLX evaluator under its own name and rejects online modes."""

    def _build(self, **overrides: Any) -> tuple[str, Any]:
        args: dict[str, Any] = {
            "model": "mazesmazes/tiny-audio-granite-qwen-500",
            "endpoint": False,
            "streaming": False,
            "num_workers": 1,
            "user_prompt": None,
            "local_code": False,
            "mlx": True,
        }
        return _build_tiny_audio_evaluator(**{**args, **overrides})

    @pytest.mark.usefixtures("fake_mlx")
    def test_runs_under_its_own_name(self) -> None:
        model_id, evaluator = self._build()
        assert model_id == "tiny-audio-granite-qwen-500-mlx"
        assert isinstance(evaluator, MLXEvaluator)

    @pytest.mark.parametrize(
        "override", [{"endpoint": True}, {"streaming": True}, {"user_prompt": "x"}]
    )
    @pytest.mark.usefixtures("fake_mlx")
    def test_online_modes_are_rejected(self, override: dict[str, Any]) -> None:
        with pytest.raises(typer.BadParameter, match="--mlx"):
            self._build(**override)
