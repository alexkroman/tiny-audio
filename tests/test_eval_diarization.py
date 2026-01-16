"""Tests for LocalDiarizationEvaluator and clustering algorithms."""

import numpy as np
import pytest


class TestSpeakerClusterer:
    """Tests for SpeakerClusterer."""

    def test_empty_embeddings(self):
        """Test handling of empty embeddings."""
        from tiny_audio.diarization import SpeakerClusterer

        clusterer = SpeakerClusterer()
        labels = clusterer(np.array([]).reshape(0, 192))
        assert len(labels) == 0

    def test_single_embedding(self):
        """Test single embedding returns label 0."""
        from tiny_audio.diarization import SpeakerClusterer

        clusterer = SpeakerClusterer()
        labels = clusterer(np.random.randn(1, 192))
        assert len(labels) == 1
        assert labels[0] == 0

    def test_few_embeddings_returns_zeros(self):
        """Test fewer than 6 embeddings returns all zeros."""
        from tiny_audio.diarization import SpeakerClusterer

        clusterer = SpeakerClusterer()
        labels = clusterer(np.random.randn(5, 192))
        assert len(labels) == 5
        assert np.all(labels == 0)

    def test_oracle_num_speakers(self):
        """Test with oracle number of speakers."""
        from tiny_audio.diarization import SpeakerClusterer

        clusterer = SpeakerClusterer()
        # Create 2 distinct clusters
        emb1 = np.random.randn(15, 192) + np.array([1.0] * 192)
        emb2 = np.random.randn(15, 192) + np.array([-1.0] * 192)
        embeddings = np.vstack([emb1, emb2])

        labels = clusterer(embeddings, num_speakers=2)
        assert len(labels) == 30
        assert len(np.unique(labels)) == 2


class TestLocalDiarizationEvaluator:
    """Tests for LocalDiarizationEvaluator class."""

    @pytest.fixture
    def mock_vad_model(self, mocker):
        """Set up VAD mock."""
        mock_vad = mocker.patch("tiny_audio.diarization.LocalSpeakerDiarizer._get_ten_vad_model")
        mock_model = mocker.MagicMock()
        mock_model.process.return_value = (None, True)  # TEN-VAD returns (frame, is_speech)
        mock_vad.return_value = mock_model
        return mock_model

    @pytest.fixture
    def mock_speaker_model(self, mocker):
        """Set up speaker model mock."""
        mock_speaker = mocker.patch(
            "tiny_audio.diarization.LocalSpeakerDiarizer._get_eres2netv2_model"
        )
        mock_model = mocker.MagicMock()
        # ERes2NetV2 returns a tensor from forward pass
        import torch

        mock_model.return_value = torch.randn(1, 192)
        mock_speaker.return_value = mock_model
        return mock_model

    @pytest.fixture
    def mock_clusterer(self, mocker):
        """Set up clusterer mock - mock the entire SpeakerClusterer class."""
        mock_cluster = mocker.patch("tiny_audio.diarization.SpeakerClusterer")
        mock_obj = mocker.MagicMock()
        mock_obj.return_value = np.array([0, 0])
        mock_cluster.return_value = mock_obj
        return mock_obj

    @pytest.fixture
    def evaluator(self):
        """Create evaluator."""
        from scripts.eval.evaluators.diarization import LocalDiarizationEvaluator

        return LocalDiarizationEvaluator()

    def test_init_default_params(self):
        """Test default parameters."""
        from scripts.eval.evaluators.diarization import LocalDiarizationEvaluator

        evaluator = LocalDiarizationEvaluator()
        assert evaluator.num_speakers is None
        assert evaluator.min_speakers == 2
        assert evaluator.max_speakers == 10

    def test_init_custom_params(self):
        """Test custom parameters."""
        from scripts.eval.evaluators.diarization import LocalDiarizationEvaluator

        evaluator = LocalDiarizationEvaluator(num_speakers=2, min_speakers=2, max_speakers=4)
        assert evaluator.num_speakers == 2
        assert evaluator.min_speakers == 2
        assert evaluator.max_speakers == 4

    def test_init_removes_hf_token(self):
        """Test that hf_token is removed from kwargs."""
        from scripts.eval.evaluators.diarization import LocalDiarizationEvaluator

        evaluator = LocalDiarizationEvaluator(hf_token="ignored")
        assert not hasattr(evaluator, "hf_token") or evaluator.hf_token is None

    def test_diarize_returns_segments(
        self, evaluator, mock_vad_model, mock_speaker_model, mock_clusterer
    ):
        """Test that diarize returns speaker segments."""
        audio = {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}

        segments, elapsed = evaluator.diarize(audio)

        assert isinstance(segments, list)
        assert elapsed > 0
        for seg in segments:
            assert "speaker" in seg
            assert "start" in seg
            assert "end" in seg

    def test_diarize_unsupported_format_raises(
        self, evaluator, mock_vad_model, mock_speaker_model, mock_clusterer
    ):
        """Test that unsupported audio format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported audio"):
            evaluator.diarize("not_a_valid_format")

    def test_diarize_unsupported_dict_format_raises(
        self, evaluator, mock_vad_model, mock_speaker_model, mock_clusterer
    ):
        """Test that unsupported dict format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported audio dict format"):
            evaluator.diarize({"unknown_key": "value"})


class TestLocalDiarizationEvaluatorExport:
    """Tests for module exports."""

    def test_evaluator_importable(self):
        """Test LocalDiarizationEvaluator can be imported from evaluators."""
        from scripts.eval.evaluators import LocalDiarizationEvaluator

        assert LocalDiarizationEvaluator is not None

    def test_inherits_base(self):
        """Test LocalDiarizationEvaluator inherits from DiarizationEvaluator."""
        from scripts.eval.evaluators import (
            DiarizationEvaluator,
            LocalDiarizationEvaluator,
        )

        assert issubclass(LocalDiarizationEvaluator, DiarizationEvaluator)
