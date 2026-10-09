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
from tiny_audio.asr_pipeline import (
    ASRPipeline,
    NemotronDiarizer,
    PreparedChunk,
    QwenForcedAligner,
    collate_chunks,
)


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
        tokenizer = cast(MagicMock, mock_pipeline.tokenizer)
        tokenizer.decode.return_value = "<think>reasoning</think> hello world"

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
        # device="cpu": with no device, transformers' Pipeline moves the model to
        # the default accelerator IN PLACE. base_asr_model is session-scoped, so
        # on a Mac every later test got an MPS model fed CPU tensors.
        return ASRPipeline(
            model=base_asr_model,
            feature_extractor=base_asr_model.feature_extractor,
            tokenizer=base_asr_model.tokenizer,
            device="cpu",
        )

    def test_shared_model_stays_on_cpu(
        self, pipeline: ASRPipeline, base_asr_model: ASRModel
    ) -> None:
        """Wrapping the session model must not move it for the tests that follow."""
        assert pipeline.device == torch.device("cpu")
        assert {p.device.type for p in base_asr_model.parameters()} == {"cpu"}

    def test_call_basic_transcription(self, pipeline: ASRPipeline) -> None:
        """Plain call returns dict with 'text' key."""
        audio = np.zeros(16000, dtype=np.float32)  # 1s silence
        result = pipeline({"array": audio, "sampling_rate": 16000})
        assert "text" in result

    @pytest.fixture
    def chunked(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Stub chunk transcription: the k-th chunk (by sample count) reads "w{k}a w{k}b".

        Stubs both routes to the model: the batched one (`prepare_chunk` +
        `generate_prepared`) and the one-call-per-chunk fallback that generate
        kwargs take (the parent pipeline's `__call__`).
        """
        calls: list[int] = []

        def transcribe(samples: int) -> str:
            calls.append(samples)
            k = len(calls) - 1
            return f"w{k}a w{k}b"

        def fake_prepare(self: object, chunk: npt.NDArray[np.float32], sample_rate: int) -> Any:
            return {"input_features": torch.zeros(len(chunk)), "attention_mask": torch.zeros(1)}

        def fake_generate(self: object, prepared: Sequence[Any]) -> list[str]:
            return [transcribe(int(p["input_features"].shape[0])) for p in prepared]

        def fake_call(self: object, inputs: dict[str, Any], **kwargs: Any) -> dict[str, str]:
            return {"text": transcribe(len(inputs["raw"] if "raw" in inputs else inputs["array"]))}

        monkeypatch.setattr(ASRPipeline, "prepare_chunk", fake_prepare)
        monkeypatch.setattr(ASRPipeline, "generate_prepared", fake_generate)
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
        result = pipeline(
            {"array": self._speech(30.0, 12.0), "sampling_rate": 16000}, output_scores=True
        )
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

    def test_speakers_come_from_per_speaker_streams(
        self, pipeline: ASRPipeline, chunked: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each Nemotron speaker is transcribed on its own turn; segments come back too."""

        def fake_align(
            chunks: Sequence[tuple[object, str]], sample_rate: int = 16000
        ) -> list[list[AlignedWord]]:
            return [
                [{"word": w, "start": 0.5, "end": 0.9} for w in text.split()[:1]]
                for _, text in chunks
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
        assert chunked == [int(2.5 * 16000), int(2.5 * 16000)]  # one call per speaker turn
        assert [(w["word"], w["speaker"]) for w in result["words"]] == [
            ("w0a", "SPEAKER_0"),
            ("w1a", "SPEAKER_1"),
        ]
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


class TestBatchedChunks:
    """Chunks are padded into one batch and run through the model together."""

    @staticmethod
    def _prepared(frames: int, dim: int = 4, time_major: bool = True) -> PreparedChunk:
        shape = (1, frames, dim) if time_major else (1, dim, frames)
        return {
            "input_features": torch.ones(shape),
            "attention_mask": torch.ones(1, frames, dtype=torch.long),
        }

    @pytest.mark.parametrize("time_major", [True, False])
    def test_collate_pads_time_axis_and_mask(self, time_major: bool) -> None:
        batch = collate_chunks(
            [self._prepared(3, time_major=time_major), self._prepared(5, time_major=time_major)]
        )
        expected = (2, 5, 4) if time_major else (2, 4, 5)
        assert tuple(batch["input_features"].shape) == expected
        assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        short = batch["input_features"][0]
        padded = short[3:] if time_major else short[:, 3:]
        assert float(padded.abs().sum()) == 0.0  # padding is zeros, not copies

    def test_batches_split_at_max_batch_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sizes: list[int] = []

        def fake_generate(self: object, prepared: Sequence[PreparedChunk]) -> list[str]:
            sizes.append(len(prepared))
            return [str(int(p["attention_mask"].shape[-1])) for p in prepared]

        monkeypatch.setattr(ASRPipeline, "generate_prepared", fake_generate)
        pipeline = object.__new__(ASRPipeline)
        pipeline.max_batch_size = 2
        texts = pipeline._generate_in_batches([self._prepared(n) for n in (1, 2, 3, 4, 5)])
        assert sizes == [2, 2, 1]
        assert texts == ["1", "2", "3", "4", "5"]

    def test_chunk_runner_replaces_local_batching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A server's runner sees every non-silent chunk of the request at once."""

        def fake_prepare(
            _self: object, chunk: npt.NDArray[np.float32], sample_rate: int
        ) -> PreparedChunk:
            return TestBatchedChunks._prepared(len(chunk))

        monkeypatch.setattr(ASRPipeline, "prepare_chunk", fake_prepare)
        seen: list[int] = []

        def runner(prepared: list[PreparedChunk]) -> list[str]:
            seen.extend(int(p["attention_mask"].shape[-1]) for p in prepared)
            return [f"t{i}" for i in range(len(prepared))]

        pipeline = object.__new__(ASRPipeline)
        pipeline.chunk_runner = runner
        noise = np.full(100, 0.1, dtype=np.float32)
        silence = np.zeros(50, dtype=np.float32)
        texts = pipeline._transcribe_chunks([noise, silence, noise[:70]], 16000)
        assert seen == [100, 70]  # silence never reaches the runner
        assert texts == ["t0", "", "t1"]


class TestBatchedGenerateParity:
    """A batch decodes like its rows one at a time (real model, CPU)."""

    def test_batched_matches_one_at_a_time(
        self, base_asr_model: ASRModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rows don't leak into each other: a ragged batch gives each row its own text.

        Short on purpose. The test model is untrained, so its logits are near
        ties, and padding's different float rounding flips one eventually (seen
        ~120 tokens in). Batching is not bit-exact; it only agrees wherever the
        model is decisive, which 16 tokens of a real mixing bug would not be.
        """
        monkeypatch.setattr(base_asr_model.generation_config, "max_new_tokens", 16)
        pipeline = ASRPipeline(
            model=base_asr_model,
            feature_extractor=base_asr_model.feature_extractor,
            tokenizer=base_asr_model.tokenizer,
            device="cpu",
        )
        rng = np.random.default_rng(0)
        chunks = [rng.normal(0, 0.1, n).astype(np.float32) for n in (16000, 24000, 8000)]
        prepared = [pipeline.prepare_chunk(c, 16000) for c in chunks]
        solo = [pipeline.generate_prepared([p])[0] for p in prepared]
        assert pipeline.generate_prepared(prepared) == solo
