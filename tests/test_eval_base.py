"""Tests for scripts/eval/evaluators/base.py - base evaluator and result types."""

from typing import Unpack
from unittest.mock import MagicMock

import pytest

import scripts.eval.cli as cli
import scripts.eval.evaluators.asr as asr
from scripts.eval.constants import ASSEMBLYAI_MODELS, AssemblyAIModel
from scripts.eval.evaluators.base import (
    EvalResult,
    Evaluator,
    EvaluatorOptions,
    Transcription,
)


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating an EvalResult."""
        result = EvalResult(
            prediction="hello world",
            reference="hello world",
            wer=0.0,
            time=1.5,
        )
        assert result.prediction == "hello world"
        assert result.reference == "hello world"
        assert result.wer == 0.0
        assert result.time == 1.5

    def test_result_with_high_wer(self) -> None:
        """Test result with high WER."""
        result = EvalResult(
            prediction="completely wrong",
            reference="hello world",
            wer=100.0,
            time=0.5,
        )
        assert result.wer == 100.0


class TestEvaluator:
    """Tests for base Evaluator class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        evaluator = Evaluator()
        assert evaluator.audio_field == "audio"
        assert evaluator.text_field == "text"
        assert evaluator.num_workers == 1
        assert evaluator.results == []

    def test_init_custom_fields(self) -> None:
        """Test custom field initialization."""
        evaluator = Evaluator(
            audio_field="wav",
            text_field="transcript",
            num_workers=4,
        )
        assert evaluator.audio_field == "wav"
        assert evaluator.text_field == "transcript"
        assert evaluator.num_workers == 4

    def test_transcribe_not_implemented(self) -> None:
        """Test that transcribe raises NotImplementedError."""
        evaluator = Evaluator()
        with pytest.raises(NotImplementedError):
            evaluator.transcribe(None)

    def test_compute_metrics_empty(self) -> None:
        """Test compute_metrics with no results."""
        evaluator = Evaluator()
        metrics = evaluator.compute_metrics()
        assert metrics["wer"] == 0.0
        assert metrics["avg_time"] == 0.0
        assert metrics["num_samples"] == 0

    def test_compute_metrics_with_results(self) -> None:
        """Test compute_metrics with results."""
        evaluator = Evaluator()
        evaluator.results = [
            EvalResult("hello", "hello", 0.0, 1.0),
            EvalResult("world", "world", 0.0, 2.0),
        ]
        metrics = evaluator.compute_metrics()
        assert metrics["wer"] == 0.0
        assert metrics["avg_time"] == 1.5
        assert metrics["num_samples"] == 2


class TestMockEvaluator:
    """Tests using a mock evaluator implementation."""

    class MockEvaluator(Evaluator):
        """Mock evaluator for testing."""

        def __init__(self, responses: list[str], **kwargs: Unpack[EvaluatorOptions]) -> None:
            super().__init__(**kwargs)
            self.responses = responses
            self.call_count = 0

        def transcribe(self, audio: object) -> Transcription:
            response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            return response, 0.1, None

    def test_sequential_evaluation(self) -> None:
        """Test sequential evaluation."""
        evaluator = self.MockEvaluator(
            responses=["hello world", "test response"],
            num_workers=1,
        )
        # Create mock dataset
        dataset = [
            {"audio": b"audio1", "text": "hello world"},
            {"audio": b"audio2", "text": "test response"},
        ]
        results = evaluator.evaluate(dataset)
        assert len(results) == 2
        assert results[0].wer == 0.0  # Perfect match
        assert results[1].wer == 0.0  # Perfect match

    def test_parallel_evaluation(self) -> None:
        """Test parallel evaluation."""
        evaluator = self.MockEvaluator(
            responses=["response"],
            num_workers=2,
        )
        dataset = [
            {"audio": b"audio1", "text": "response"},
            {"audio": b"audio2", "text": "response"},
            {"audio": b"audio3", "text": "response"},
            {"audio": b"audio4", "text": "response"},
        ]
        results = evaluator.evaluate(dataset)
        assert len(results) == 4

    def test_max_samples_limit(self) -> None:
        """Test max_samples parameter limits evaluation."""
        evaluator = self.MockEvaluator(responses=["test"])
        dataset = [{"audio": b"a", "text": "test"} for _ in range(10)]
        results = evaluator.evaluate(dataset, max_samples=3)
        assert len(results) == 3

    def test_skip_tedlium_ignore_markers(self) -> None:
        """Test that TEDLIUM ignore markers are skipped."""
        evaluator = self.MockEvaluator(responses=["test"])
        dataset = [
            {"audio": b"a", "text": "ignore_time_segment_in_scoring"},
            {"audio": b"b", "text": "valid text"},
        ]
        results = evaluator.evaluate(dataset)
        assert len(results) == 1
        assert results[0].reference == "valid text"

    def test_skip_inaudible_samples(self) -> None:
        """Test that inaudible samples are skipped."""
        evaluator = self.MockEvaluator(responses=["test"])
        dataset = [
            {"audio": b"a", "text": "[inaudible]"},
            {"audio": b"b", "text": "This is INAUDIBLE content"},
            {"audio": b"c", "text": "valid text"},
        ]
        results = evaluator.evaluate(dataset)
        assert len(results) == 1
        assert results[0].reference == "valid text"


class TestAssemblyAIModels:
    """Tests for AssemblyAI model constants."""

    def test_available_models(self) -> None:
        """Test that expected models are available."""
        assert "best" in ASSEMBLYAI_MODELS
        assert "universal" in ASSEMBLYAI_MODELS
        assert "universal-3-pro" in ASSEMBLYAI_MODELS
        assert "universal-3-5-pro" in ASSEMBLYAI_MODELS

    def test_model_count(self) -> None:
        """Test number of available models."""
        assert len(ASSEMBLYAI_MODELS) == 4


class TestEvaluatorReuseAcrossDatasets:
    """One evaluator instance must serve a whole `ta eval -d a -d b` sweep.

    `scripts/eval/cli._build_evaluator` now runs once before the dataset
    loop, so the model / Swift subprocess / API client is set up a single
    time. That only works if per-dataset state travels through `evaluate`.
    """

    class MockEvaluator(Evaluator):
        def __init__(self, **kwargs: Unpack[EvaluatorOptions]) -> None:
            super().__init__(**kwargs)
            self.setup_count = 1  # stands in for a model load

        def transcribe(self, audio: object) -> Transcription:
            return str(audio), 0.1, None

    def test_field_overrides_apply_per_call(self) -> None:
        """Differently-shaped corpora work without rebuilding the evaluator."""
        evaluator = self.MockEvaluator()

        loquacious_like = [{"wav": "hello", "text": "hello"}]
        earnings_like = [{"audio": "world", "sentence": "world"}]

        first = evaluator.evaluate(loquacious_like, audio_field="wav", text_field="text")
        second = evaluator.evaluate(earnings_like, audio_field="audio", text_field="sentence")

        assert [r.reference for r in first] == ["hello"]
        assert [r.reference for r in second] == ["world"]
        assert evaluator.setup_count == 1

    def test_omitted_overrides_keep_constructor_fields(self) -> None:
        evaluator = self.MockEvaluator(audio_field="wav", text_field="transcript")
        evaluator.evaluate([{"wav": "a", "transcript": "a"}])
        assert evaluator.audio_field == "wav"
        assert evaluator.text_field == "transcript"

    def test_results_do_not_pool_across_datasets(self) -> None:
        """Second dataset's metrics must not include the first dataset's rows."""
        evaluator = self.MockEvaluator()
        evaluator.evaluate([{"audio": "a", "text": "a"} for _ in range(3)])
        evaluator.evaluate([{"audio": "b", "text": "b"}])
        assert evaluator.compute_metrics()["num_samples"] == 1

    def test_subclass_accumulators_are_reset(self) -> None:
        """`_reset_run_state` is the hook subclasses extend; verify it fires."""

        class TimingEvaluator(TestEvaluatorReuseAcrossDatasets.MockEvaluator):
            def __init__(self, **kwargs: Unpack[EvaluatorOptions]) -> None:
                super().__init__(**kwargs)
                self.ttfb_times: list[float] = []

            def _reset_run_state(self) -> None:
                super()._reset_run_state()
                self.ttfb_times = []

            def transcribe(self, audio: object) -> Transcription:
                self.ttfb_times.append(0.2)
                return str(audio), 0.1, None

        evaluator = TimingEvaluator()
        evaluator.evaluate([{"audio": "a", "text": "a"} for _ in range(3)])
        evaluator.evaluate([{"audio": "b", "text": "b"}])
        assert evaluator.ttfb_times == [0.2]


class TestLocalEvaluatorNumWorkers:
    """`-w` must reach LocalEvaluator; `_build_evaluator` used to drop it silently."""

    def test_num_workers_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        class StubLocalEvaluator:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(cli, "LocalEvaluator", StubLocalEvaluator)
        cli._build_evaluator(
            model="some/checkpoint",
            endpoint=False,
            streaming=False,
            assemblyai_model=AssemblyAIModel("universal-3-pro"),
            assemblyai_api_key=None,
            deepgram_api_key=None,
            elevenlabs_api_key=None,
            smallest_api_key=None,
            base_url=None,
            locale="en-US",
            num_workers=2,
            user_prompt=None,
        )
        assert captured["num_workers"] == 2

    @pytest.mark.parametrize(("device", "expected"), [("mps", 1), (0, 2)])
    def test_mps_clamps_to_one_worker(
        self, monkeypatch: pytest.MonkeyPatch, device: int | str, expected: int
    ) -> None:
        """MPS segfaults under concurrent Metal encoding, so -w must not thread there."""

        def resolve_local_runtime() -> tuple[int | str, str]:
            return device, "bfloat16"

        def build_local_pipeline(*a: object, **k: object) -> MagicMock:
            return MagicMock()

        def print_generation_config(*a: object, **k: object) -> None:
            return None

        monkeypatch.setattr(asr, "_resolve_local_runtime", resolve_local_runtime)
        monkeypatch.setattr(asr, "_build_local_pipeline", build_local_pipeline)
        monkeypatch.setattr(asr, "print_generation_config", print_generation_config)
        evaluator = asr.LocalEvaluator(model_path="some/checkpoint", num_workers=2)
        assert evaluator.num_workers == expected
