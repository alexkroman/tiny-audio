"""Tests for Deepgram evaluator implementations."""

from unittest.mock import MagicMock, patch


class TestDeepgramDiarizationEvaluator:
    """Tests for DeepgramDiarizationEvaluator class."""

    def test_init_creates_client(self):
        """Test that initialization creates Deepgram client."""
        with patch("deepgram.DeepgramClient") as mock_client:
            from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

            evaluator = DeepgramDiarizationEvaluator(api_key="test-key")

            mock_client.assert_called_once_with(api_key="test-key")
            assert evaluator.client is not None

    def test_init_removes_hf_token(self):
        """Test that hf_token is removed from kwargs."""
        with patch("deepgram.DeepgramClient"):
            from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

            # Should not raise even with hf_token
            evaluator = DeepgramDiarizationEvaluator(api_key="test-key", hf_token="ignored")
            assert not hasattr(evaluator, "hf_token") or evaluator.hf_token is None

    def test_init_accepts_custom_fields(self):
        """Test that custom field names are accepted."""
        with patch("deepgram.DeepgramClient"):
            from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

            evaluator = DeepgramDiarizationEvaluator(
                api_key="test-key",
                audio_field="wav",
                speakers_field="spk",
            )
            assert evaluator.audio_field == "wav"
            assert evaluator.speakers_field == "spk"

    def test_init_num_workers_default(self):
        """Test that num_workers defaults to 1."""
        with patch("deepgram.DeepgramClient"):
            from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

            evaluator = DeepgramDiarizationEvaluator(api_key="test-key")
            assert evaluator.num_workers == 1

    def test_init_num_workers_custom(self):
        """Test that custom num_workers is accepted."""
        with patch("deepgram.DeepgramClient"):
            from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

            evaluator = DeepgramDiarizationEvaluator(api_key="test-key", num_workers=4)
            assert evaluator.num_workers == 4

    def test_diarize_returns_segments(self):
        """Test that diarize returns speaker segments."""
        with patch("deepgram.DeepgramClient") as mock_client_class, patch(
            "scripts.eval.evaluators.diarization.prepare_wav_bytes"
        ) as mock_prepare, patch("scripts.eval.evaluators.diarization.time.sleep"):
            from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

            # Setup mock response
            mock_utterance1 = MagicMock()
            mock_utterance1.speaker = 0
            mock_utterance1.start = 0.0
            mock_utterance1.end = 1.5

            mock_utterance2 = MagicMock()
            mock_utterance2.speaker = 1
            mock_utterance2.start = 1.5
            mock_utterance2.end = 3.0

            mock_response = MagicMock()
            mock_response.results.utterances = [mock_utterance1, mock_utterance2]

            mock_client = MagicMock()
            mock_client.listen.v1.media.transcribe_file.return_value = mock_response
            mock_client_class.return_value = mock_client
            mock_prepare.return_value = b"wav_bytes"

            evaluator = DeepgramDiarizationEvaluator(api_key="test-key")
            segments, elapsed = evaluator.diarize({"array": [0.0], "sampling_rate": 16000})

            assert len(segments) == 2
            assert segments[0]["speaker"] == "SPEAKER_0"
            assert segments[0]["start"] == 0.0
            assert segments[0]["end"] == 1.5
            assert segments[1]["speaker"] == "SPEAKER_1"
            assert elapsed > 0

            # Verify API was called with correct params
            mock_client.listen.v1.media.transcribe_file.assert_called_once_with(
                request=b"wav_bytes",
                model="nova-3",
                diarize=True,
                utterances=True,
            )

    def test_diarize_empty_utterances(self):
        """Test diarize handles empty utterances."""
        with patch("deepgram.DeepgramClient") as mock_client_class, patch(
            "scripts.eval.evaluators.diarization.prepare_wav_bytes"
        ) as mock_prepare, patch("scripts.eval.evaluators.diarization.time.sleep"):
            from scripts.eval.evaluators.diarization import DeepgramDiarizationEvaluator

            mock_response = MagicMock()
            mock_response.results.utterances = None

            mock_client = MagicMock()
            mock_client.listen.v1.media.transcribe_file.return_value = mock_response
            mock_client_class.return_value = mock_client
            mock_prepare.return_value = b"wav_bytes"

            evaluator = DeepgramDiarizationEvaluator(api_key="test-key")
            segments, _ = evaluator.diarize({"array": [0.0], "sampling_rate": 16000})

            assert segments == []


class TestDeepgramAlignmentEvaluator:
    """Tests for DeepgramAlignmentEvaluator class."""

    def test_init_creates_client(self):
        """Test that initialization creates Deepgram client."""
        with patch("deepgram.DeepgramClient") as mock_client:
            from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

            evaluator = DeepgramAlignmentEvaluator(api_key="test-key")

            mock_client.assert_called_once_with(api_key="test-key")
            assert evaluator.client is not None

    def test_init_accepts_custom_fields(self):
        """Test that custom field names are accepted."""
        with patch("deepgram.DeepgramClient"):
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

    def test_init_num_workers_default(self):
        """Test that num_workers defaults to 1."""
        with patch("deepgram.DeepgramClient"):
            from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

            evaluator = DeepgramAlignmentEvaluator(api_key="test-key")
            assert evaluator.num_workers == 1

    def test_init_num_workers_custom(self):
        """Test that custom num_workers is accepted."""
        with patch("deepgram.DeepgramClient"):
            from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

            evaluator = DeepgramAlignmentEvaluator(api_key="test-key", num_workers=4)
            assert evaluator.num_workers == 4

    def test_transcribe_with_timestamps_returns_words(self):
        """Test that transcribe_with_timestamps returns word timestamps."""
        with patch("deepgram.DeepgramClient") as mock_client_class, patch(
            "scripts.eval.evaluators.alignment.prepare_wav_bytes"
        ) as mock_prepare, patch("scripts.eval.evaluators.alignment.time.sleep"):
            from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

            # Setup mock response
            mock_word1 = MagicMock()
            mock_word1.word = "hello"
            mock_word1.start = 0.0
            mock_word1.end = 0.5

            mock_word2 = MagicMock()
            mock_word2.word = "world"
            mock_word2.start = 0.5
            mock_word2.end = 1.0

            mock_alternative = MagicMock()
            mock_alternative.transcript = "hello world"
            mock_alternative.words = [mock_word1, mock_word2]

            mock_channel = MagicMock()
            mock_channel.alternatives = [mock_alternative]

            mock_response = MagicMock()
            mock_response.results.channels = [mock_channel]

            mock_client = MagicMock()
            mock_client.listen.v1.media.transcribe_file.return_value = mock_response
            mock_client_class.return_value = mock_client
            mock_prepare.return_value = b"wav_bytes"

            evaluator = DeepgramAlignmentEvaluator(api_key="test-key")
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
            mock_client.listen.v1.media.transcribe_file.assert_called_once_with(
                request=b"wav_bytes",
                model="nova-3",
                smart_format=True,
            )

    def test_transcribe_with_timestamps_empty_words(self):
        """Test transcribe_with_timestamps handles empty words."""
        with patch("deepgram.DeepgramClient") as mock_client_class, patch(
            "scripts.eval.evaluators.alignment.prepare_wav_bytes"
        ) as mock_prepare, patch("scripts.eval.evaluators.alignment.time.sleep"):
            from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

            mock_alternative = MagicMock()
            mock_alternative.transcript = ""
            mock_alternative.words = None

            mock_channel = MagicMock()
            mock_channel.alternatives = [mock_alternative]

            mock_response = MagicMock()
            mock_response.results.channels = [mock_channel]

            mock_client = MagicMock()
            mock_client.listen.v1.media.transcribe_file.return_value = mock_response
            mock_client_class.return_value = mock_client
            mock_prepare.return_value = b"wav_bytes"

            evaluator = DeepgramAlignmentEvaluator(api_key="test-key")
            text, words, _ = evaluator.transcribe_with_timestamps(
                {"array": [0.0], "sampling_rate": 16000}
            )

            assert text == ""
            assert words == []

    def test_transcribe_with_timestamps_no_alternatives(self):
        """Test transcribe_with_timestamps handles no alternatives."""
        with patch("deepgram.DeepgramClient") as mock_client_class, patch(
            "scripts.eval.evaluators.alignment.prepare_wav_bytes"
        ) as mock_prepare, patch("scripts.eval.evaluators.alignment.time.sleep"):
            from scripts.eval.evaluators.alignment import DeepgramAlignmentEvaluator

            mock_channel = MagicMock()
            mock_channel.alternatives = []

            mock_response = MagicMock()
            mock_response.results.channels = [mock_channel]

            mock_client = MagicMock()
            mock_client.listen.v1.media.transcribe_file.return_value = mock_response
            mock_client_class.return_value = mock_client
            mock_prepare.return_value = b"wav_bytes"

            evaluator = DeepgramAlignmentEvaluator(api_key="test-key")
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
        from scripts.eval.evaluators import DeepgramDiarizationEvaluator, DiarizationEvaluator

        assert issubclass(DeepgramDiarizationEvaluator, DiarizationEvaluator)

    def test_alignment_inherits_base(self):
        """Test DeepgramAlignmentEvaluator inherits from BaseAlignmentEvaluator."""
        from scripts.eval.evaluators import BaseAlignmentEvaluator, DeepgramAlignmentEvaluator

        assert issubclass(DeepgramAlignmentEvaluator, BaseAlignmentEvaluator)
