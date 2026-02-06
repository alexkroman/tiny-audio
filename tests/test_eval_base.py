"""Tests for scripts/eval/evaluators/base.py - base evaluator and result types."""

import pytest

from scripts.eval.evaluators.base import (
    AlignmentResult,
    DiarizationResult,
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


class TestDiarizationResult:
    """Tests for DiarizationResult dataclass."""

    def test_create_diarization_result(self):
        """Test creating a DiarizationResult."""
        result = DiarizationResult(
            der=15.5,
            confusion=5.0,
            missed=8.0,
            false_alarm=2.5,
            time=3.0,
            num_speakers_ref=3,
            num_speakers_hyp=3,
        )
        assert result.der == 15.5
        assert result.confusion == 5.0
        assert result.missed == 8.0
        assert result.false_alarm == 2.5
        assert result.num_speakers_ref == 3
        assert result.num_speakers_hyp == 3

    def test_default_values(self):
        """Test default values for optional fields."""
        result = DiarizationResult(
            der=10.0,
            confusion=3.0,
            missed=5.0,
            false_alarm=2.0,
            time=1.0,
            num_speakers_ref=2,
            num_speakers_hyp=2,
        )
        assert result.total == 0.0
        assert result.confusion_raw == 0.0
        assert result.missed_raw == 0.0
        assert result.false_alarm_raw == 0.0


class TestDiarizationEvaluator:
    """Tests for DiarizationEvaluator class."""

    def test_init_num_workers_default(self):
        """Test that num_workers defaults to 1."""
        from scripts.eval.evaluators.diarization import DiarizationEvaluator

        evaluator = DiarizationEvaluator()
        assert evaluator.num_workers == 1

    def test_init_num_workers_custom(self):
        """Test that custom num_workers is accepted."""
        from scripts.eval.evaluators.diarization import DiarizationEvaluator

        evaluator = DiarizationEvaluator(num_workers=4)
        assert evaluator.num_workers == 4


class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    def test_create_alignment_result(self):
        """Test creating an AlignmentResult."""
        result = AlignmentResult(
            pred_starts=[0.0, 0.5, 1.0],
            pred_ends=[0.4, 0.9, 1.4],
            ref_starts=[0.0, 0.5, 1.0],
            ref_ends=[0.4, 0.9, 1.4],
            num_aligned_words=3,
            num_ref_words=3,
            num_pred_words=3,
            time=0.5,
            reference_text="hello world test",
            predicted_text="hello world test",
        )
        assert len(result.pred_starts) == 3
        assert result.num_aligned_words == 3


class TestBaseAlignmentEvaluator:
    """Tests for BaseAlignmentEvaluator class."""

    def test_init_num_workers_default(self):
        """Test that num_workers defaults to 1."""
        from scripts.eval.evaluators.alignment import BaseAlignmentEvaluator

        evaluator = BaseAlignmentEvaluator()
        assert evaluator.num_workers == 1

    def test_init_num_workers_custom(self):
        """Test that custom num_workers is accepted."""
        from scripts.eval.evaluators.alignment import BaseAlignmentEvaluator

        evaluator = BaseAlignmentEvaluator(num_workers=4)
        assert evaluator.num_workers == 4


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
            return response, 0.1

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
