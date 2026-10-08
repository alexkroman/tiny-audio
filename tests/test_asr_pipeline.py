"""Tests for ASRPipeline and helper classes."""

import numpy as np
import pytest


class TestExtractAudio:
    """Tests for ASRPipeline._extract_audio method."""

    @pytest.fixture
    def extract_audio(self):
        """Get the extract_audio function without loading a model."""
        from tiny_audio.asr_pipeline import ASRPipeline

        class MockPipeline:
            _extract_audio = ASRPipeline._extract_audio

        return MockPipeline()

    def test_dict_with_array(self, extract_audio):
        """Dict with 'array' key should extract audio."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_audio._extract_audio({"array": audio, "sampling_rate": 16000})
        assert result is not None
        assert "array" in result
        assert result["sampling_rate"] == 16000

    def test_dict_with_raw(self, extract_audio):
        """Dict with 'raw' key should extract audio."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_audio._extract_audio({"raw": audio, "sampling_rate": 16000})
        assert result is not None
        assert "array" in result

    def test_numpy_array(self, extract_audio):
        """Numpy array should be extracted with default sample rate."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_audio._extract_audio(audio)
        assert result is not None
        assert result["sampling_rate"] == 16000

    def test_unsupported_returns_none(self, extract_audio):
        """Unsupported input type should return None."""
        result = extract_audio._extract_audio(12345)
        assert result is None


class TestSpeakerAssignment:
    """Tests for SpeakerDiarizer.assign_speakers_to_words."""

    def test_exact_overlap(self):
        """Words within speaker segments get assigned correctly."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]

        result = SpeakerDiarizer.assign_speakers_to_words(words, segments)

        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_00"

    def test_multiple_speakers(self):
        """Words should be assigned to correct speakers."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "hi", "start": 2.0, "end": 2.5},
        ]
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 1.5, "end": 3.0},
        ]

        result = SpeakerDiarizer.assign_speakers_to_words(words, segments)

        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_closest_segment_fallback(self):
        """Words outside segments should be assigned to closest speaker."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [{"word": "hello", "start": 1.0, "end": 1.5}]  # Between segments
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0},
        ]

        result = SpeakerDiarizer.assign_speakers_to_words(words, segments)
        # Midpoint is 1.25, closer to SPEAKER_01 (midpoint 2.5) than SPEAKER_00 (midpoint 0.25)
        # Actually: |1.25 - 0.25| = 1.0, |1.25 - 2.5| = 1.25, so SPEAKER_00 is closer
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_empty_segments(self):
        """Empty segments should result in None speaker."""
        from tiny_audio.asr_pipeline import SpeakerDiarizer

        words = [{"word": "hello", "start": 0.0, "end": 0.5}]
        segments = []

        result = SpeakerDiarizer.assign_speakers_to_words(words, segments)

        assert result[0]["speaker"] is None


class TestSanitizeParameters:
    """Tests for ASRPipeline._sanitize_parameters."""

    def test_removes_custom_params(self):
        """Custom params should be removed before parent validation."""
        from unittest.mock import patch

        from tiny_audio.asr_pipeline import ASRPipeline

        # Create a mock pipeline that won't call parent __init__
        with patch.object(ASRPipeline, "__init__", return_value=None):
            pipeline = ASRPipeline.__new__(ASRPipeline)

        # Mock the parent's _sanitize_parameters
        with patch(
            "transformers.AutomaticSpeechRecognitionPipeline._sanitize_parameters"
        ) as mock_parent:
            mock_parent.return_value = ({}, {}, {})

            # Call with custom params
            pipeline._sanitize_parameters(
                return_timestamps=True,
                return_speakers=True,
                num_speakers=2,
                min_speakers=1,
                max_speakers=3,
                hf_token="test",
                other_param="value",
            )

            # Parent should be called without our custom params
            call_kwargs = mock_parent.call_args[1]
            assert "return_timestamps" not in call_kwargs
            assert "return_speakers" not in call_kwargs
            assert "num_speakers" not in call_kwargs
            assert "hf_token" not in call_kwargs
            # But other params should remain
            assert call_kwargs.get("other_param") == "value"


class TestPostprocess:
    """Tests for ASRPipeline.postprocess method."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with postprocess method."""
        from unittest.mock import MagicMock

        from tiny_audio.asr_pipeline import ASRPipeline

        # Create instance without calling __init__
        pipeline = object.__new__(ASRPipeline)

        # Create mock tokenizer
        pipeline.tokenizer = MagicMock()
        pipeline.tokenizer.decode.return_value = "hello world"

        # postprocess reads the model's stop ids; a real pipeline always has one.
        pipeline.model = MagicMock()
        pipeline.model.generation_config.eos_token_id = None

        return pipeline

    def test_handles_list_outputs(self, mock_pipeline):
        """Should handle list of outputs from chunking."""
        import torch

        result = mock_pipeline.postprocess([{"tokens": torch.tensor([1, 2, 3])}])
        assert "text" in result

    def test_handles_tensor_tokens(self, mock_pipeline):
        """Should handle tensor tokens."""
        import torch

        result = mock_pipeline.postprocess({"tokens": torch.tensor([[1, 2, 3]])})
        assert "text" in result
        mock_pipeline.tokenizer.decode.assert_called()

    def test_strips_think_tags(self, mock_pipeline):
        """Should strip <think>...</think> tags from output."""
        mock_pipeline.tokenizer.decode.return_value = "<think>reasoning</think> hello world"
        import torch

        result = mock_pipeline.postprocess({"tokens": torch.tensor([1, 2, 3])})
        assert "<think>" not in result["text"]
        assert "reasoning" not in result["text"]
        assert "hello world" in result["text"]


class TestExtractAudioFileAndBytes:
    """_extract_audio handles file paths and bytes inputs (uses ffmpeg_read)."""

    def test_extract_audio_from_bytes(self):
        """Bytes input should be parsed by ffmpeg_read."""
        from unittest.mock import patch

        from tiny_audio.asr_pipeline import ASRPipeline

        class MockPipeline:
            _extract_audio = ASRPipeline._extract_audio

        with patch("tiny_audio.asr_pipeline.ffmpeg_read") as mock_read:
            mock_read.return_value = np.zeros(16000, dtype=np.float32)
            result = MockPipeline()._extract_audio(b"some-audio-bytes")

        assert result is not None
        assert result["sampling_rate"] == 16000
        mock_read.assert_called_once()

    def test_extract_audio_from_path(self, tmp_path):
        """File path input should be opened and read via ffmpeg_read."""
        from unittest.mock import patch

        from tiny_audio.asr_pipeline import ASRPipeline

        class MockPipeline:
            _extract_audio = ASRPipeline._extract_audio

        # Create a dummy file
        f = tmp_path / "audio.wav"
        f.write_bytes(b"fake-wav-bytes")

        with patch("tiny_audio.asr_pipeline.ffmpeg_read") as mock_read:
            mock_read.return_value = np.zeros(16000, dtype=np.float32)
            result = MockPipeline()._extract_audio(str(f))

        assert result is not None
        assert result["sampling_rate"] == 16000


class TestPipelineCall:
    """ASRPipeline.__call__ orchestrates transcription, alignment, diarization."""

    @pytest.fixture
    def pipeline(self, base_asr_model):
        from tiny_audio.asr_pipeline import ASRPipeline

        return ASRPipeline(
            model=base_asr_model,
            feature_extractor=base_asr_model.feature_extractor,
            tokenizer=base_asr_model.tokenizer,
        )

    def test_call_basic_transcription(self, pipeline):
        """Plain call returns dict with 'text' key."""
        audio = np.zeros(16000, dtype=np.float32)  # 1s silence
        result = pipeline({"array": audio, "sampling_rate": 16000})
        assert "text" in result

    @pytest.fixture
    def chunked(self, monkeypatch):
        """Stub per-chunk transcription: chunk k transcribes as "w{k}a w{k}b"."""
        import transformers

        calls = []

        def fake_call(self, inputs, **kwargs):
            calls.append(len(inputs.get("raw", inputs.get("array"))))
            k = len(calls) - 1
            return {"text": f"w{k}a w{k}b"}

        monkeypatch.setattr(transformers.AutomaticSpeechRecognitionPipeline, "__call__", fake_call)
        return calls

    @staticmethod
    def _speech(seconds: float, quiet_at: float) -> np.ndarray:
        """Noise with one silent 200 ms gap at `quiet_at`, so the chunk cut is predictable."""
        audio = np.random.default_rng(0).normal(0, 0.1, int(seconds * 16000)).astype(np.float32)
        audio[int(quiet_at * 16000) : int((quiet_at + 0.2) * 16000)] = 0.0
        return audio

    def test_timestamps_chunk_align_and_offset(self, pipeline, chunked, monkeypatch):
        """Long audio is transcribed per chunk; word times land on the recording timeline."""
        from tiny_audio.asr_pipeline import QwenForcedAligner

        def fake_align(chunks, sample_rate=16000):
            return [
                [{"word": w, "start": 1.0, "end": 1.5} for w in text.split()] for _, text in chunks
            ]

        monkeypatch.setattr(QwenForcedAligner, "align_chunks", fake_align)
        audio = self._speech(30.0, quiet_at=12.0)
        result = pipeline({"array": audio, "sampling_rate": 16000}, return_timestamps=True)

        assert len(chunked) == 2  # 30 s > 18 s: one cut, in the 8-18 s window
        cut_s = chunked[0] / 16000
        assert 12.0 <= cut_s <= 12.2
        assert result["text"] == "w0a w0b w1a w1b"
        assert [w["start"] for w in result["words"]] == pytest.approx(
            [1.0, 1.0, 1.0 + cut_s, 1.0 + cut_s]
        )

    def test_plain_call_chunks_without_aligning(self, pipeline, chunked, monkeypatch):
        """Long audio chunks even without timestamps; one call would stop at max_new_tokens."""
        from tiny_audio.asr_pipeline import QwenForcedAligner

        monkeypatch.setattr(
            QwenForcedAligner, "align_chunks", lambda *a, **k: pytest.fail("aligner called")
        )
        result = pipeline({"array": self._speech(30.0, 12.0), "sampling_rate": 16000})
        assert len(chunked) == 2
        assert 12.0 <= chunked[0] / 16000 <= 12.2
        assert result == {"text": "w0a w0b w1a w1b"}

    def test_plain_short_clip_is_one_call(self, pipeline, chunked):
        result = pipeline({"array": self._speech(5.0, 2.0), "sampling_rate": 16000})
        assert chunked == [5 * 16000]
        assert result == {"text": "w0a w0b"}

    def test_digital_silence_is_not_transcribed(self, pipeline, chunked):
        """All-zero chunks give "" without a model call: the model hallucinates on them."""
        audio = self._speech(20.0, 15.0)
        audio[: 12 * 16000] = 0.0  # the cut lands at 8 s: chunk 1 is exact zeros
        result = pipeline({"array": audio, "sampling_rate": 16000})
        assert len(chunked) == 1  # only the second chunk reached the model
        assert result == {"text": "w0a w0b"}

    def test_silent_short_clip_skips_the_model(self, pipeline, chunked):
        assert pipeline({"array": np.zeros(16000, dtype=np.float32)}) == {"text": ""}
        assert chunked == []

    def test_plain_chunks_concatenate_logprobs(self, pipeline, monkeypatch):
        import transformers

        def fake_call(self, inputs, **kwargs):
            return {"text": "x", "top1_logprob": [-0.1], "top2_logprob": [-2.0]}

        monkeypatch.setattr(transformers.AutomaticSpeechRecognitionPipeline, "__call__", fake_call)
        result = pipeline({"array": self._speech(30.0, 12.0), "sampling_rate": 16000})
        assert result == {"text": "x x", "top1_logprob": [-0.1, -0.1], "top2_logprob": [-2.0, -2.0]}

    def test_alignment_failure_recorded(self, pipeline, chunked, monkeypatch):
        from tiny_audio.asr_pipeline import QwenForcedAligner

        def boom(*args, **kwargs):
            raise RuntimeError("model not loadable")

        monkeypatch.setattr(QwenForcedAligner, "align_chunks", boom)
        result = pipeline(
            {"array": self._speech(5.0, 2.0), "sampling_rate": 16000}, return_timestamps=True
        )
        assert result["text"] == "w0a w0b"
        assert result["words"] == []
        assert "model not loadable" in result["timestamp_error"]

    def test_speakers_from_one_nemotron_pass(self, pipeline, chunked, monkeypatch):
        """Words take the Nemotron speaker active at their time; segments come back too."""
        from tiny_audio.asr_pipeline import NemotronDiarizer, QwenForcedAligner

        monkeypatch.setattr(
            QwenForcedAligner,
            "align_chunks",
            lambda chunks, sample_rate=16000: [
                [
                    {"word": "w0a", "start": 0.5, "end": 0.9},
                    {"word": "w0b", "start": 3.1, "end": 3.5},
                ]
            ],
        )
        activity = np.zeros((500, 8), dtype=np.float32)
        activity[0:250, 3] = 0.9  # arrives first -> SPEAKER_0
        activity[250:500, 5] = 0.9
        monkeypatch.setattr(NemotronDiarizer, "activity", lambda audio, sample_rate=16000: activity)

        result = pipeline(
            {"array": self._speech(5.0, 2.0), "sampling_rate": 16000}, return_speakers=True
        )
        assert [w["speaker"] for w in result["words"]] == ["SPEAKER_0", "SPEAKER_1"]
        assert result["speaker_segments"] == [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": 2.5},
            {"speaker": "SPEAKER_1", "start": 2.5, "end": 5.0},
        ]

    def test_min_speakers_rejected(self, pipeline):
        with pytest.raises(ValueError, match="min_speakers"):
            pipeline(
                {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                return_speakers=True,
                min_speakers=2,
            )

    def test_call_user_prompt_overrides_default(self, pipeline):
        """user_prompt kwarg temporarily replaces TRANSCRIBE_PROMPT."""
        original_prompt = pipeline.model.TRANSCRIBE_PROMPT
        audio = np.zeros(16000, dtype=np.float32)

        pipeline(
            {"array": audio, "sampling_rate": 16000},
            user_prompt="Custom prompt:",
        )

        # Should restore original prompt after the call
        assert original_prompt == pipeline.model.TRANSCRIBE_PROMPT
