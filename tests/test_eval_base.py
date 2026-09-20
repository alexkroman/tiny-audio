"""Tests for scripts/eval/evaluators/base.py - base evaluator and result types."""

import pytest

from scripts.eval.evaluators.base import (
    ASSEMBLYAI_MODELS,
    EvalResult,
    Evaluator,
)


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_create_result(self):
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

    def test_result_with_high_wer(self):
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

    def test_init_defaults(self):
        """Test default initialization."""
        evaluator = Evaluator()
        assert evaluator.audio_field == "audio"
        assert evaluator.text_field == "text"
        assert evaluator.num_workers == 1
        assert evaluator.results == []

    def test_init_custom_fields(self):
        """Test custom field initialization."""
        evaluator = Evaluator(
            audio_field="wav",
            text_field="transcript",
            num_workers=4,
        )
        assert evaluator.audio_field == "wav"
        assert evaluator.text_field == "transcript"
        assert evaluator.num_workers == 4

    def test_transcribe_not_implemented(self):
        """Test that transcribe raises NotImplementedError."""
        evaluator = Evaluator()
        with pytest.raises(NotImplementedError):
            evaluator.transcribe(None)

    def test_compute_metrics_empty(self):
        """Test compute_metrics with no results."""
        evaluator = Evaluator()
        metrics = evaluator.compute_metrics()
        assert metrics["wer"] == 0.0
        assert metrics["avg_time"] == 0.0
        assert metrics["num_samples"] == 0

    def test_compute_metrics_with_results(self):
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

        def __init__(self, responses: list[str], **kwargs):
            super().__init__(**kwargs)
            self.responses = responses
            self.call_count = 0

        def transcribe(self, audio):
            response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            return response, 0.1, None

    def test_sequential_evaluation(self):
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

    def test_parallel_evaluation(self):
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

    def test_max_samples_limit(self):
        """Test max_samples parameter limits evaluation."""
        evaluator = self.MockEvaluator(responses=["test"])
        dataset = [{"audio": b"a", "text": "test"} for _ in range(10)]
        results = evaluator.evaluate(dataset, max_samples=3)
        assert len(results) == 3

    def test_skip_tedlium_ignore_markers(self):
        """Test that TEDLIUM ignore markers are skipped."""
        evaluator = self.MockEvaluator(responses=["test"])
        dataset = [
            {"audio": b"a", "text": "ignore_time_segment_in_scoring"},
            {"audio": b"b", "text": "valid text"},
        ]
        results = evaluator.evaluate(dataset)
        assert len(results) == 1
        assert results[0].reference == "valid text"

    def test_skip_inaudible_samples(self):
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

    def test_available_models(self):
        """Test that expected models are available."""
        assert "best" in ASSEMBLYAI_MODELS
        assert "universal" in ASSEMBLYAI_MODELS
        assert "universal-3-pro" in ASSEMBLYAI_MODELS
        assert "universal-3-5-pro" in ASSEMBLYAI_MODELS

    def test_model_count(self):
        """Test number of available models."""
        assert len(ASSEMBLYAI_MODELS) == 4


class TestEvaluatorReuseAcrossDatasets:
    """One evaluator instance must serve a whole `ta eval -d a -d b` sweep.

    `scripts/eval/cli._build_evaluator` now runs once before the dataset
    loop, so the model / Swift subprocess / API client is set up a single
    time. That only works if per-dataset state travels through `evaluate`.
    """

    class MockEvaluator(Evaluator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.setup_count = 1  # stands in for a model load

        def transcribe(self, audio):
            return str(audio), 0.1, None

    def test_field_overrides_apply_per_call(self):
        """Differently-shaped corpora work without rebuilding the evaluator."""
        evaluator = self.MockEvaluator()

        loquacious_like = [{"wav": "hello", "text": "hello"}]
        earnings_like = [{"audio": "world", "sentence": "world"}]

        first = evaluator.evaluate(loquacious_like, audio_field="wav", text_field="text")
        second = evaluator.evaluate(earnings_like, audio_field="audio", text_field="sentence")

        assert [r.reference for r in first] == ["hello"]
        assert [r.reference for r in second] == ["world"]
        assert evaluator.setup_count == 1

    def test_omitted_overrides_keep_constructor_fields(self):
        evaluator = self.MockEvaluator(audio_field="wav", text_field="transcript")
        evaluator.evaluate([{"wav": "a", "transcript": "a"}])
        assert evaluator.audio_field == "wav"
        assert evaluator.text_field == "transcript"

    def test_results_do_not_pool_across_datasets(self):
        """Second dataset's metrics must not include the first dataset's rows."""
        evaluator = self.MockEvaluator()
        evaluator.evaluate([{"audio": "a", "text": "a"} for _ in range(3)])
        evaluator.evaluate([{"audio": "b", "text": "b"}])
        assert evaluator.compute_metrics()["num_samples"] == 1

    def test_subclass_accumulators_are_reset(self):
        """`_reset_run_state` is the hook subclasses extend; verify it fires."""

        class TimingEvaluator(TestEvaluatorReuseAcrossDatasets.MockEvaluator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.ttfb_times: list[float] = []

            def _reset_run_state(self) -> None:
                super()._reset_run_state()
                self.ttfb_times = []

            def transcribe(self, audio):
                self.ttfb_times.append(0.2)
                return str(audio), 0.1, None

        evaluator = TimingEvaluator()
        evaluator.evaluate([{"audio": "a", "text": "a"} for _ in range(3)])
        evaluator.evaluate([{"audio": "b", "text": "b"}])
        assert evaluator.ttfb_times == [0.2]
