"""Tests for scripts/eval package.

Note: Audio utility tests (audio_to_wav_bytes, prepare_wav_bytes, TextNormalizer)
are in test_eval_audio.py to avoid duplication.
"""

import pytest

from scripts.eval.audio import TextNormalizer
from scripts.eval.datasets import (
    ALIGNMENT_DATASETS,
    DATASET_REGISTRY,
    DIARIZATION_DATASETS,
    DatasetConfig,
)
from scripts.eval.evaluators import (
    AlignmentResult,
    DiarizationResult,
    EvalResult,
    Evaluator,
)
from scripts.eval.evaluators.alignment import align_words_to_reference


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_basic_config(self):
        """Test creating a basic dataset config."""
        config = DatasetConfig(
            name="test",
            path="test/dataset",
            audio_field="audio",
        )

        assert config.name == "test"
        assert config.path == "test/dataset"
        assert config.audio_field == "audio"
        assert config.text_field == "text"  # default
        assert config.default_split == "test"  # default

    def test_diarization_config(self):
        """Test diarization-specific fields."""
        config = DatasetConfig(
            name="diarization_test",
            path="test/diarization",
            audio_field="audio",
            speakers_field="speakers",
            timestamps_start_field="start",
            timestamps_end_field="end",
        )

        assert config.speakers_field == "speakers"
        assert config.timestamps_start_field == "start"
        assert config.timestamps_end_field == "end"

    def test_alignment_config(self):
        """Test alignment-specific fields."""
        config = DatasetConfig(
            name="alignment_test",
            path="test/alignment",
            audio_field="audio",
            words_field="words",
        )

        assert config.words_field == "words"


class TestDatasetRegistry:
    """Tests for DATASET_REGISTRY."""

    def test_registry_not_empty(self):
        """Test that the registry contains datasets."""
        assert len(DATASET_REGISTRY) > 0

    def test_loquacious_exists(self):
        """Test that loquacious dataset is in registry."""
        assert "loquacious" in DATASET_REGISTRY
        cfg = DATASET_REGISTRY["loquacious"]
        assert cfg.audio_field == "wav"
        assert cfg.text_field == "text"

    def test_all_configs_have_required_fields(self):
        """Test that all configs have required fields."""
        for name, cfg in DATASET_REGISTRY.items():
            assert cfg.name == name, f"Config name mismatch for {name}"
            assert cfg.path, f"Missing path for {name}"
            assert cfg.audio_field, f"Missing audio_field for {name}"

    def test_diarization_datasets_have_speaker_fields(self):
        """Test that diarization datasets have speaker fields."""
        for name in DIARIZATION_DATASETS:
            cfg = DATASET_REGISTRY[name]
            assert cfg.speakers_field, f"Missing speakers_field for {name}"
            assert cfg.timestamps_start_field, f"Missing timestamps_start_field for {name}"
            assert cfg.timestamps_end_field, f"Missing timestamps_end_field for {name}"

    def test_alignment_datasets_have_words_field(self):
        """Test that alignment datasets have words field."""
        for name in ALIGNMENT_DATASETS:
            cfg = DATASET_REGISTRY[name]
            assert cfg.words_field, f"Missing words_field for {name}"


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


class TestDiarizationResult:
    """Tests for DiarizationResult dataclass."""

    def test_create_result(self):
        """Test creating a DiarizationResult."""
        result = DiarizationResult(
            der=10.5,
            confusion=3.0,
            missed=5.0,
            false_alarm=2.5,
            time=2.0,
            num_speakers_ref=2,
            num_speakers_hyp=2,
        )

        assert result.der == 10.5
        assert result.confusion == 3.0
        assert result.missed == 5.0
        assert result.false_alarm == 2.5
        assert result.num_speakers_ref == 2


class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    def test_create_result(self):
        """Test creating an AlignmentResult."""
        result = AlignmentResult(
            pred_starts=[0.0, 0.5, 1.0],
            pred_ends=[0.4, 0.9, 1.4],
            ref_starts=[0.0, 0.5, 1.0],
            ref_ends=[0.5, 1.0, 1.5],
            num_aligned_words=3,
            num_ref_words=3,
            num_pred_words=3,
            time=1.0,
            reference_text="hello there friend",
            predicted_text="hello there friend",
        )

        assert result.num_aligned_words == 3
        assert len(result.pred_starts) == 3


class TestAlignWordsToReference:
    """Tests for align_words_to_reference function."""

    @pytest.fixture
    def normalizer(self):
        """Create a TextNormalizer instance."""
        return TextNormalizer()

    def test_perfect_alignment(self, normalizer):
        """Test alignment when prediction matches reference."""
        pred_words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        ref_words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]

        aligned = align_words_to_reference(pred_words, ref_words, normalizer)

        assert len(aligned) == 2

    def test_partial_alignment(self, normalizer):
        """Test alignment when only some words match."""
        pred_words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "friend", "start": 0.5, "end": 1.0},
        ]
        ref_words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]

        aligned = align_words_to_reference(pred_words, ref_words, normalizer)

        assert len(aligned) == 1
        assert aligned[0][0]["word"] == "hello"

    def test_empty_prediction(self, normalizer):
        """Test alignment with empty prediction."""
        pred_words = []
        ref_words = [{"word": "hello", "start": 0.0, "end": 0.5}]

        aligned = align_words_to_reference(pred_words, ref_words, normalizer)

        assert len(aligned) == 0

    def test_skip_unk_tokens(self, normalizer):
        """Test that <unk> tokens in reference are skipped."""
        pred_words = [{"word": "hello", "start": 0.0, "end": 0.5}]
        ref_words = [
            {"word": "<unk>", "start": 0.0, "end": 0.2},
            {"word": "hello", "start": 0.2, "end": 0.5},
        ]

        aligned = align_words_to_reference(pred_words, ref_words, normalizer)

        assert len(aligned) == 1


class TestEvaluatorBase:
    """Tests for base Evaluator class."""

    def test_compute_metrics_empty(self):
        """Test compute_metrics with no results."""
        evaluator = Evaluator()
        metrics = evaluator.compute_metrics()

        assert metrics["wer"] == 0.0
        assert metrics["avg_time"] == 0.0
        assert metrics["num_samples"] == 0

    def test_evaluator_initialization(self):
        """Test Evaluator initialization."""
        evaluator = Evaluator(audio_field="audio", text_field="text")

        assert evaluator.audio_field == "audio"
        assert evaluator.text_field == "text"
        assert evaluator.results == []

    def test_transcribe_not_implemented(self):
        """Test that base transcribe raises NotImplementedError."""
        evaluator = Evaluator()

        with pytest.raises(NotImplementedError):
            evaluator.transcribe(None)
