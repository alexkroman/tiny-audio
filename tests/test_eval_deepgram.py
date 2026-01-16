"""Tests for Deepgram evaluator implementations.

Uses pytest-mock and factory functions from conftest.py for cleaner mocking.
"""

import pytest
from conftest import (
    build_deepgram_diarization_response,
    build_deepgram_transcription_response,
)


class TestDeepgramDiarizationEvaluator:
    """Tests for DeepgramDiarizationEvaluator class."""

    @pytest.fixture
    def mock_deepgram(self, mocker):
        """Set up Deepgram client mock."""
        mock_client_class = mocker.patch("deepgram.DeepgramClient")
        mock_client = mocker.MagicMock()
        mock_client_class.return_value = mock_client
        mocker.patch("scripts.eval.evaluators.diarization.time.sleep")
        return mock_client

    @pytest.fixture
    def evaluator(self, mock_deepgram):
        """Create evaluator with mocked Deepgram client."""
        from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

        return DeepgramDiarizationEvaluator(api_key="test-key")

    def test_init_creates_client(self, mocker):
        """Test that initialization creates Deepgram client."""
        mock_client = mocker.patch("deepgram.DeepgramClient")
        from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

        evaluator = DeepgramDiarizationEvaluator(api_key="test-key")

        mock_client.assert_called_once_with(api_key="test-key")
        assert evaluator.client is not None

    def test_init_removes_hf_token(self, mock_deepgram):
        """Test that hf_token is removed from kwargs."""
        from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

        evaluator = DeepgramDiarizationEvaluator(api_key="test-key", hf_token="ignored")
        assert not hasattr(evaluator, "hf_token") or evaluator.hf_token is None

    def test_init_accepts_custom_fields(self, mock_deepgram):
        """Test that custom field names are accepted."""
        from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

        evaluator = DeepgramDiarizationEvaluator(
            api_key="test-key",
            audio_field="wav",
            speakers_field="spk",
        )
        assert evaluator.audio_field == "wav"
        assert evaluator.speakers_field == "spk"

    @pytest.mark.parametrize("num_workers", [1, 4])
    def test_init_num_workers(self, mock_deepgram, num_workers):
        """Test that num_workers is configurable."""
        from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

        evaluator = DeepgramDiarizationEvaluator(api_key="test-key", num_workers=num_workers)
        assert evaluator.num_workers == num_workers

    def test_diarize_returns_segments(self, evaluator, mocker):
        """Test that diarize returns speaker segments."""
        mocker.patch(
            "scripts.eval.evaluators.diarization.prepare_wav_bytes",
            return_value=b"wav_bytes",
        )
        evaluator.client.listen.v1.media.transcribe_file.return_value = (
            build_deepgram_diarization_response(
                [
                    {"speaker": 0, "start": 0.0, "end": 1.5},
                    {"speaker": 1, "start": 1.5, "end": 3.0},
                ]
            )
        )

        segments, elapsed = evaluator.diarize({"array": [0.0], "sampling_rate": 16000})

        assert len(segments) == 2
        assert segments[0]["speaker"] == "SPEAKER_0"
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 1.5
        assert segments[1]["speaker"] == "SPEAKER_1"
        assert elapsed > 0

        # Verify API was called with correct params
        evaluator.client.listen.v1.media.transcribe_file.assert_called_once_with(
            request=b"wav_bytes",
            model="nova-3",
            diarize=True,
            utterances=True,
        )

    def test_diarize_empty_utterances(self, evaluator, mocker):
        """Test diarize handles empty utterances."""
        mocker.patch(
            "scripts.eval.evaluators.diarization.prepare_wav_bytes",
            return_value=b"wav_bytes",
        )
        evaluator.client.listen.v1.media.transcribe_file.return_value = (
            build_deepgram_diarization_response(None)
        )

        segments, _ = evaluator.diarize({"array": [0.0], "sampling_rate": 16000})

        assert segments == []


class TestDeepgramAlignmentEvaluator:
    """Tests for DeepgramAlignmentEvaluator class."""

    @pytest.fixture
    def mock_deepgram(self, mocker):
        """Set up Deepgram client mock."""
        mock_client_class = mocker.patch("deepgram.DeepgramClient")
        mock_client = mocker.MagicMock()
        mock_client_class.return_value = mock_client
        mocker.patch("scripts.eval.evaluators.alignment.time.sleep")
        return mock_client

    @pytest.fixture
    def evaluator(self, mock_deepgram):
        """Create evaluator with mocked Deepgram client."""
        from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

        return DeepgramAlignmentEvaluator(api_key="test-key")

    def test_init_creates_client(self, mocker):
        """Test that initialization creates Deepgram client."""
        mock_client = mocker.patch("deepgram.DeepgramClient")
        from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

        evaluator = DeepgramAlignmentEvaluator(api_key="test-key")

        mock_client.assert_called_once_with(api_key="test-key")
        assert evaluator.client is not None

    def test_init_accepts_custom_fields(self, mock_deepgram):
        """Test that custom field names are accepted."""
        from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

        evaluator = DeepgramAlignmentEvaluator(
            api_key="test-key",
            audio_field="wav",
            text_field="transcript",
            words_field="word_timestamps",
        )
        assert evaluator.audio_field == "wav"
        assert evaluator.text_field == "transcript"
        assert evaluator.words_field == "word_timestamps"

    @pytest.mark.parametrize("num_workers", [1, 4])
    def test_init_num_workers(self, mock_deepgram, num_workers):
        """Test that num_workers is configurable."""
        from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

        evaluator = DeepgramAlignmentEvaluator(api_key="test-key", num_workers=num_workers)
        assert evaluator.num_workers == num_workers

    def test_transcribe_with_timestamps_returns_words(self, evaluator, mocker):
        """Test that transcribe_with_timestamps returns word timestamps."""
        mocker.patch(
            "scripts.eval.evaluators.alignment.prepare_wav_bytes",
            return_value=b"wav_bytes",
        )
        evaluator.client.listen.v1.media.transcribe_file.return_value = (
            build_deepgram_transcription_response(
                transcript="hello world",
                words=[
                    {"word": "hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                ],
            )
        )

        text, words, elapsed = evaluator.transcribe_with_timestamps(
            {"array": [0.0], "sampling_rate": 16000}
        )

        assert text == "hello world"
        assert len(words) == 2
        assert words[0]["word"] == "hello"
        assert words[0]["start"] == 0.0
        assert words[0]["end"] == 0.5
        assert words[1]["word"] == "world"
        assert elapsed > 0

        # Verify API was called with correct params
        evaluator.client.listen.v1.media.transcribe_file.assert_called_once_with(
            request=b"wav_bytes",
            model="nova-3",
            smart_format=True,
        )

    def test_transcribe_with_timestamps_empty_words(self, evaluator, mocker):
        """Test transcribe_with_timestamps handles empty words."""
        mocker.patch(
            "scripts.eval.evaluators.alignment.prepare_wav_bytes",
            return_value=b"wav_bytes",
        )
        evaluator.client.listen.v1.media.transcribe_file.return_value = (
            build_deepgram_transcription_response(transcript="", words=None)
        )

        text, words, _ = evaluator.transcribe_with_timestamps(
            {"array": [0.0], "sampling_rate": 16000}
        )

        assert text == ""
        assert words == []

    def test_transcribe_with_timestamps_no_alternatives(self, evaluator, mocker):
        """Test transcribe_with_timestamps handles no alternatives."""
        mocker.patch(
            "scripts.eval.evaluators.alignment.prepare_wav_bytes",
            return_value=b"wav_bytes",
        )
        # Create response with empty alternatives
        response = mocker.MagicMock()
        channel = mocker.MagicMock()
        channel.alternatives = []
        response.results.channels = [channel]
        evaluator.client.listen.v1.media.transcribe_file.return_value = response

        text, words, _ = evaluator.transcribe_with_timestamps(
            {"array": [0.0], "sampling_rate": 16000}
        )

        assert text == ""
        assert words == []


class TestDeepgramEvaluatorsExport:
    """Tests for module exports."""

    def test_diarization_evaluator_importable(self):
        """Test DeepgramDiarizationEvaluator can be imported from evaluators."""
        from scripts.eval.evaluators import DeepgramDiarizationEvaluator

        assert DeepgramDiarizationEvaluator is not None

    def test_alignment_evaluator_importable(self):
        """Test DeepgramAlignmentEvaluator can be imported from evaluators."""
        from scripts.eval.evaluators import DeepgramAlignmentEvaluator

        assert DeepgramAlignmentEvaluator is not None

    def test_diarization_inherits_base(self):
        """Test DeepgramDiarizationEvaluator inherits from DiarizationEvaluator."""
        from scripts.eval.evaluators import (
            DeepgramDiarizationEvaluator,
            DiarizationEvaluator,
        )

        assert issubclass(DeepgramDiarizationEvaluator, DiarizationEvaluator)

    def test_alignment_inherits_base(self):
        """Test DeepgramAlignmentEvaluator inherits from BaseAlignmentEvaluator."""
        from scripts.eval.evaluators import (
            BaseAlignmentEvaluator,
            DeepgramAlignmentEvaluator,
        )

        assert issubclass(DeepgramAlignmentEvaluator, BaseAlignmentEvaluator)
