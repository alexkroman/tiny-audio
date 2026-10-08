"""Tests for ASRPipeline and helper classes."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
import torch
import transformers

from tiny_audio.alignment import AlignedWord
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.asr_pipeline import ASRPipeline, NemotronDiarizer, QwenForcedAligner


class TestExtractAudio:
    """Tests for ASRPipeline._extract_audio method."""

    @pytest.fixture
    def extract_audio(self) -> ASRPipeline:
        """Get the extract_audio function without loading a model."""
        # Instance without calling __init__: _extract_audio touches no state.
        return object.__new__(ASRPipeline)

    def test_dict_with_array(self, extract_audio: ASRPipeline) -> None:
        """Dict with 'array' key should extract audio."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_audio._extract_audio({"array": audio, "sampling_rate": 16000})
        assert result is not None
        assert "array" in result
        assert result["sampling_rate"] == 16000

    def test_dict_with_raw(self, extract_audio: ASRPipeline) -> None:
        """Dict with 'raw' key should extract audio."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_audio._extract_audio({"raw": audio, "sampling_rate": 16000})
        assert result is not None
        assert "array" in result

    def test_numpy_array(self, extract_audio: ASRPipeline) -> None:
        """Numpy array should be extracted with default sample rate."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_audio._extract_audio(audio)
        assert result is not None
        assert result["sampling_rate"] == 16000

    def test_unsupported_returns_none(self, extract_audio: ASRPipeline) -> None:
        """Unsupported input type should return None."""
        result = extract_audio._extract_audio(12345)
        assert result is None


class TestSanitizeParameters:
    """Tests for ASRPipeline._sanitize_parameters."""

    def test_removes_custom_params(self) -> None:
        """Custom params should be removed before parent validation."""
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
    def mock_pipeline(self) -> ASRPipeline:
        """Create a mock pipeline with postprocess method."""
        # Create instance without calling __init__
        pipeline = object.__new__(ASRPipeline)

        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "hello world"
        pipeline.tokenizer = tokenizer

        # postprocess reads the model's stop ids; a real pipeline always has one.
        model = MagicMock()
        model.generation_config.eos_token_id = None
        pipeline.model = model

        return pipeline

    def test_handles_list_outputs(self, mock_pipeline: ASRPipeline) -> None:
        """Should handle list of outputs from chunking."""
        result = mock_pipeline.postprocess([{"tokens": torch.tensor([1, 2, 3])}])
        assert "text" in result

    def test_handles_tensor_tokens(self, mock_pipeline: ASRPipeline) -> None:
        """Should handle tensor tokens."""
        result = mock_pipeline.postprocess({"tokens": torch.tensor([[1, 2, 3]])})
        assert "text" in result
        cast(MagicMock, mock_pipeline.tokenizer).decode.assert_called()

    def test_strips_think_tags(self, mock_pipeline: ASRPipeline) -> None:
        """Should strip <think>...</think> tags from output."""
        cast(
            MagicMock, mock_pipeline.tokenizer
        ).decode.return_value = "<think>reasoning</think> hello world"

        result = mock_pipeline.postprocess({"tokens": torch.tensor([1, 2, 3])})
        assert "<think>" not in result["text"]
        assert "reasoning" not in result["text"]
        assert "hello world" in result["text"]


class TestExtractAudioFileAndBytes:
    """_extract_audio handles file paths and bytes inputs (uses ffmpeg_read)."""

    def test_extract_audio_from_bytes(self) -> None:
        """Bytes input should be parsed by ffmpeg_read."""

        pipeline = object.__new__(ASRPipeline)

        with patch("tiny_audio.asr_pipeline.ffmpeg_read") as mock_read:
            mock_read.return_value = np.zeros(16000, dtype=np.float32)
            result = pipeline._extract_audio(b"some-audio-bytes")

        assert result is not None
        assert result["sampling_rate"] == 16000
        mock_read.assert_called_once()

    def test_extract_audio_from_path(self, tmp_path: Path) -> None:
        """File path input should be opened and read via ffmpeg_read."""

        pipeline = object.__new__(ASRPipeline)

        # Create a dummy file
        f = tmp_path / "audio.wav"
        f.write_bytes(b"fake-wav-bytes")

        with patch("tiny_audio.asr_pipeline.ffmpeg_read") as mock_read:
            mock_read.return_value = np.zeros(16000, dtype=np.float32)
            result = pipeline._extract_audio(str(f))

        assert result is not None
        assert result["sampling_rate"] == 16000


class TestPipelineCall:
    """ASRPipeline.__call__ orchestrates transcription, alignment, diarization."""

    @pytest.fixture
    def pipeline(self, base_asr_model: ASRModel) -> ASRPipeline:
        return ASRPipeline(
            model=base_asr_model,
            feature_extractor=base_asr_model.feature_extractor,
            tokenizer=base_asr_model.tokenizer,
        )

    def test_call_basic_transcription(self, pipeline: ASRPipeline) -> None:
        """Plain call returns dict with 'text' key."""
        audio = np.zeros(16000, dtype=np.float32)  # 1s silence
        result = pipeline({"array": audio, "sampling_rate": 16000})
        assert "text" in result

    @pytest.fixture
    def chunked(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Stub per-chunk transcription: chunk k transcribes as "w{k}a w{k}b"."""
        calls: list[int] = []

        def fake_call(self: object, inputs: dict[str, Any], **kwargs: Any) -> dict[str, str]:
            calls.append(len(inputs["raw"] if "raw" in inputs else inputs["array"]))
            k = len(calls) - 1
            return {"text": f"w{k}a w{k}b"}

        monkeypatch.setattr(transformers.AutomaticSpeechRecognitionPipeline, "__call__", fake_call)
        return calls

    @staticmethod
    def _speech(seconds: float, quiet_at: float) -> npt.NDArray[np.float32]:
        """Noise with one silent 200 ms gap at `quiet_at`, so the chunk cut is predictable."""
        audio = np.random.default_rng(0).normal(0, 0.1, int(seconds * 16000)).astype(np.float32)
        audio[int(quiet_at * 16000) : int((quiet_at + 0.2) * 16000)] = 0.0
        return audio

    def test_timestamps_chunk_align_and_offset(
        self, pipeline: ASRPipeline, chunked: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Long audio is transcribed per chunk; word times land on the recording timeline."""

        def fake_align(
            chunks: Sequence[tuple[object, str]], sample_rate: int = 16000
        ) -> list[list[AlignedWord]]:
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

    def test_plain_call_chunks_without_aligning(
        self, pipeline: ASRPipeline, chunked: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Long audio chunks even without timestamps; one call would stop at max_new_tokens."""

        def no_align(*args: object, **kwargs: object) -> NoReturn:
            pytest.fail("aligner called")

        monkeypatch.setattr(QwenForcedAligner, "align_chunks", no_align)
        result = pipeline({"array": self._speech(30.0, 12.0), "sampling_rate": 16000})
        assert len(chunked) == 2
        assert 12.0 <= chunked[0] / 16000 <= 12.2
        assert result == {"text": "w0a w0b w1a w1b"}

    def test_plain_short_clip_is_one_call(self, pipeline: ASRPipeline, chunked: list[int]) -> None:
        result = pipeline({"array": self._speech(5.0, 2.0), "sampling_rate": 16000})
        assert chunked == [5 * 16000]
        assert result == {"text": "w0a w0b"}

    def test_digital_silence_is_not_transcribed(
        self, pipeline: ASRPipeline, chunked: list[int]
    ) -> None:
        """All-zero chunks give "" without a model call: the model hallucinates on them."""
        audio = self._speech(20.0, 15.0)
        audio[: 12 * 16000] = 0.0  # the cut lands at 8 s: chunk 1 is exact zeros
        result = pipeline({"array": audio, "sampling_rate": 16000})
        assert len(chunked) == 1  # only the second chunk reached the model
        assert result == {"text": "w0a w0b"}

    def test_silent_short_clip_skips_the_model(
        self, pipeline: ASRPipeline, chunked: list[int]
    ) -> None:
        assert pipeline({"array": np.zeros(16000, dtype=np.float32)}) == {"text": ""}
        assert chunked == []

    def test_plain_chunks_concatenate_logprobs(
        self, pipeline: ASRPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_call(self: object, inputs: object, **kwargs: Any) -> dict[str, Any]:
            return {"text": "x", "top1_logprob": [-0.1], "top2_logprob": [-2.0]}

        monkeypatch.setattr(transformers.AutomaticSpeechRecognitionPipeline, "__call__", fake_call)
        result = pipeline({"array": self._speech(30.0, 12.0), "sampling_rate": 16000})
        assert result == {"text": "x x", "top1_logprob": [-0.1, -0.1], "top2_logprob": [-2.0, -2.0]}

    def test_alignment_failure_recorded(
        self, pipeline: ASRPipeline, chunked: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: object, **kwargs: object) -> NoReturn:
            msg = "model not loadable"
            raise RuntimeError(msg)

        monkeypatch.setattr(QwenForcedAligner, "align_chunks", boom)
        result = pipeline(
            {"array": self._speech(5.0, 2.0), "sampling_rate": 16000}, return_timestamps=True
        )
        assert result["text"] == "w0a w0b"
        assert result["words"] == []
        assert "model not loadable" in result["timestamp_error"]

    def test_speakers_from_one_nemotron_pass(
        self, pipeline: ASRPipeline, chunked: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Words take the Nemotron speaker active at their time; segments come back too."""

        def fake_align(
            chunks: Sequence[tuple[object, str]], sample_rate: int = 16000
        ) -> list[list[AlignedWord]]:
            return [
                [
                    {"word": "w0a", "start": 0.5, "end": 0.9},
                    {"word": "w0b", "start": 3.1, "end": 3.5},
                ]
            ]

        monkeypatch.setattr(QwenForcedAligner, "align_chunks", fake_align)
        activity = np.zeros((500, 8), dtype=np.float32)
        activity[0:250, 3] = 0.9  # arrives first -> SPEAKER_0
        activity[250:500, 5] = 0.9

        def fake_activity(
            audio: npt.ArrayLike, sample_rate: int = 16000
        ) -> npt.NDArray[np.float32]:
            return activity

        monkeypatch.setattr(NemotronDiarizer, "activity", fake_activity)

        result = pipeline(
            {"array": self._speech(5.0, 2.0), "sampling_rate": 16000}, return_speakers=True
        )
        assert [w["speaker"] for w in result["words"]] == ["SPEAKER_0", "SPEAKER_1"]
        assert result["speaker_segments"] == [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": 2.5},
            {"speaker": "SPEAKER_1", "start": 2.5, "end": 5.0},
        ]

    def test_min_speakers_rejected(self, pipeline: ASRPipeline) -> None:
        with pytest.raises(ValueError, match="min_speakers"):
            pipeline(
                {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                return_speakers=True,
                min_speakers=2,
            )

    def test_call_user_prompt_overrides_default(self, pipeline: ASRPipeline) -> None:
        """user_prompt kwarg temporarily replaces TRANSCRIBE_PROMPT."""
        original_prompt = pipeline.model.TRANSCRIBE_PROMPT
        audio = np.zeros(16000, dtype=np.float32)

        pipeline(
            {"array": audio, "sampling_rate": 16000},
            user_prompt="Custom prompt:",
        )

        # Should restore original prompt after the call
        assert original_prompt == pipeline.model.TRANSCRIBE_PROMPT
