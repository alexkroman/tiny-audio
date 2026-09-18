"""Tests for ASRPipeline._forward / postprocess / __call__ plumbing.

The pipeline is built with `object.__new__` and a MagicMock model, so nothing
here loads weights or touches the Hub.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import transformers

from tiny_audio.asr_pipeline import ASRPipeline


@pytest.fixture
def pipeline():
    """ASRPipeline whose model is a MagicMock on CPU."""
    pipe = object.__new__(ASRPipeline)
    pipe.model = MagicMock()
    pipe.model.device = torch.device("cpu")
    pipe.model.generation_config.eos_token_id = None
    pipe.model.TRANSCRIBE_PROMPT = "default prompt"
    pipe.tokenizer = MagicMock()
    pipe.tokenizer.decode.return_value = "hello world"
    pipe.feature_extractor = MagicMock()
    pipe.feature_extractor.sampling_rate = 16000
    return pipe


def model_inputs() -> dict:
    return {
        "input_features": torch.zeros(1, 80, 10),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
        "is_last": False,
    }


class TestForward:
    """Token-only and score-carrying generate outputs."""

    def test_plain_generate_returns_tokens_and_is_last(self, pipeline):
        tokens = torch.tensor([[5, 6, 7]])
        pipeline.model.generate.return_value = tokens
        out = pipeline._forward(model_inputs())
        assert out == {"tokens": tokens, "is_last": False}
        kwargs = pipeline.model.generate.call_args.kwargs
        assert kwargs["input_features"].shape == (1, 80, 10)
        assert kwargs["audio_attention_mask"].shape == (1, 10)

    def test_is_last_defaults_to_true(self, pipeline):
        pipeline.model.generate.return_value = torch.tensor([[1]])
        inputs = model_inputs()
        del inputs["is_last"]
        assert pipeline._forward(inputs)["is_last"] is True

    def test_output_scores_yields_top_two_logprobs(self, pipeline):
        step_scores = (
            torch.tensor([[0.0, 2.0, 1.0, -1.0]]),
            torch.tensor([[3.0, 0.0, 0.0, 0.0]]),
        )
        pipeline.model.generate.return_value = SimpleNamespace(
            sequences=torch.tensor([[1, 0]]), scores=step_scores
        )
        out = pipeline._forward(model_inputs(), output_scores=True)

        assert pipeline.model.generate.call_args.kwargs["return_dict_in_generate"] is True
        assert len(out["top1_logprob"]) == 2
        assert len(out["top2_logprob"]) == 2
        for step, (top1, top2) in enumerate(zip(out["top1_logprob"], out["top2_logprob"])):
            expected = torch.topk(torch.log_softmax(step_scores[step][0], dim=-1), k=2).values
            assert top1 == pytest.approx(expected[0].item())
            assert top2 == pytest.approx(expected[1].item())
            assert top1 >= top2

    def test_output_scores_with_no_steps(self, pipeline):
        pipeline.model.generate.return_value = SimpleNamespace(
            sequences=torch.tensor([[1]]), scores=()
        )
        out = pipeline._forward(model_inputs(), output_scores=True)
        assert out["top1_logprob"] == []
        assert out["top2_logprob"] == []

    def test_batched_scores_are_refused(self, pipeline):
        pipeline.model.generate.return_value = SimpleNamespace(
            sequences=torch.tensor([[1], [2]]), scores=(torch.zeros(2, 4),)
        )
        with pytest.raises(ValueError, match="batch of 2"):
            pipeline._forward(model_inputs(), output_scores=True)


class TestPostprocess:
    """Token filtering, think-tag stripping, repetition truncation, logprobs."""

    def test_filters_every_configured_eos_id(self, pipeline):
        pipeline.model.generation_config.eos_token_id = [2, 9]
        pipeline.postprocess({"tokens": torch.tensor([[4, 2, 5, 9]])})
        decoded = pipeline.tokenizer.decode.call_args.args[0]
        assert decoded == [4, 5]

    def test_scalar_eos_id(self, pipeline):
        pipeline.model.generation_config.eos_token_id = 2
        pipeline.postprocess({"tokens": torch.tensor([[2, 4]])})
        assert pipeline.tokenizer.decode.call_args.args[0] == [4]

    def test_trailing_repetitions_are_truncated(self, pipeline):
        pipeline.tokenizer.decode.return_value = "so the the the the"
        assert pipeline.postprocess({"tokens": torch.tensor([[1]])})["text"] == "so the"

    def test_logprobs_pass_through(self, pipeline):
        out = pipeline.postprocess(
            {"tokens": torch.tensor([[1]]), "top1_logprob": [-0.1], "top2_logprob": [-2.0]}
        )
        assert out["top1_logprob"] == [-0.1]
        assert out["top2_logprob"] == [-2.0]

    def test_no_logprob_keys_when_not_captured(self, pipeline):
        out = pipeline.postprocess({"tokens": torch.tensor([[1]])})
        assert set(out) == {"text"}

    def test_empty_list_falls_back_to_parent(self, pipeline, monkeypatch):
        monkeypatch.setattr(
            transformers.AutomaticSpeechRecognitionPipeline,
            "postprocess",
            lambda self, outputs, **kw: {"text": "parent", "outputs": outputs},
        )
        out = pipeline.postprocess([])
        assert out == {"text": "parent", "outputs": {}}


class TestPreprocess:
    """Dataset-style dicts are translated for the parent pipeline."""

    def test_array_dict_becomes_raw_and_is_last_is_added(self, pipeline, monkeypatch):
        seen = {}

        def fake_preprocess(self, inputs, **params):
            seen["inputs"] = inputs
            yield {"input_features": torch.zeros(1)}
            yield {"input_features": torch.zeros(1), "is_last": False}

        monkeypatch.setattr(
            transformers.AutomaticSpeechRecognitionPipeline, "preprocess", fake_preprocess
        )
        audio = np.zeros(160, dtype=np.float32)
        items = list(pipeline.preprocess({"array": audio}))

        assert seen["inputs"]["raw"] is audio
        assert seen["inputs"]["sampling_rate"] == 16000
        assert [item["is_last"] for item in items] == [True, False]


class TestCallPromptHandling:
    """`user_prompt` swaps the model prompt for exactly one call."""

    def test_prompt_is_restored_after_an_exception(self, pipeline, monkeypatch):
        seen = {}

        def failing_transcribe(inputs, **kwargs):
            seen["prompt_during_call"] = pipeline.model.TRANSCRIBE_PROMPT
            raise RuntimeError("boom")

        monkeypatch.setattr(pipeline, "_transcribe", failing_transcribe)
        with pytest.raises(RuntimeError, match="boom"):
            pipeline({"array": np.zeros(16)}, user_prompt="custom")

        assert seen["prompt_during_call"] == "custom"
        assert pipeline.model.TRANSCRIBE_PROMPT == "default prompt"

    def test_no_user_prompt_leaves_model_untouched(self, pipeline, monkeypatch):
        monkeypatch.setattr(pipeline, "_transcribe", lambda inputs, **kw: {"text": "x"})
        pipeline({"array": np.zeros(16)})
        assert pipeline.model.TRANSCRIBE_PROMPT == "default prompt"

    def test_return_speakers_implies_timestamps(self, pipeline, monkeypatch):
        seen = {}

        def fake_transcribe(inputs, **kwargs):
            seen.update(kwargs)
            return {"text": "x"}

        monkeypatch.setattr(pipeline, "_transcribe", fake_transcribe)
        pipeline({"array": np.zeros(16)}, return_speakers=True, num_speakers=2)
        assert seen["return_timestamps"] is True
        assert seen["return_speakers"] is True
        assert seen["diarization_params"] == {
            "num_speakers": 2,
            "min_speakers": None,
            "max_speakers": None,
        }


class TestTranscribeOrchestration:
    """Alignment and diarization are attached to the parent's transcript."""

    @pytest.fixture(autouse=True)
    def parent_call(self, monkeypatch):
        seen = {}

        def fake_call(self, inputs, **kwargs):
            seen["inputs"] = inputs
            return {"text": "hello world"}

        monkeypatch.setattr(transformers.AutomaticSpeechRecognitionPipeline, "__call__", fake_call)
        return seen

    def test_decoded_audio_is_handed_to_parent_once(self, pipeline, parent_call, monkeypatch):
        monkeypatch.setattr(
            "tiny_audio.asr_pipeline.ForcedAligner.align",
            lambda audio, text, sample_rate: [{"word": "hello", "start": 0.0, "end": 0.5}],
        )
        audio = np.zeros(16000, dtype=np.float32)
        result = pipeline._transcribe(
            {"array": audio, "sampling_rate": 16000},
            return_timestamps=True,
            return_speakers=False,
            diarization_params={},
        )
        assert parent_call["inputs"]["array"] is audio
        assert result["words"] == [{"word": "hello", "start": 0.0, "end": 0.5}]

    def test_diarization_failure_is_recorded_not_raised(self, pipeline, monkeypatch):
        monkeypatch.setattr(
            "tiny_audio.asr_pipeline.ForcedAligner.align",
            lambda audio, text, sample_rate: [{"word": "hello", "start": 0.0, "end": 0.5}],
        )

        def failing_diarize(audio, sample_rate, **kw):
            raise RuntimeError("no ecapa")

        monkeypatch.setattr("tiny_audio.asr_pipeline.SpeakerDiarizer.diarize", failing_diarize)
        result = pipeline._transcribe(
            {"array": np.zeros(16000, dtype=np.float32)},
            return_timestamps=True,
            return_speakers=True,
            diarization_params={"num_speakers": None},
        )
        assert result["speaker_segments"] == []
        assert result["diarization_error"] == "no ecapa"
        # Alignment still succeeded and is kept.
        assert result["words"][0]["word"] == "hello"

    def test_speakers_are_assigned_to_words(self, pipeline, monkeypatch):
        monkeypatch.setattr(
            "tiny_audio.asr_pipeline.ForcedAligner.align",
            lambda audio, text, sample_rate: [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 2.0, "end": 2.5},
            ],
        )
        seen = {}

        def fake_diarize(audio, sample_rate, **kw):
            seen.update(kw)
            return [
                {"speaker": "SPEAKER_0", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_1", "start": 1.5, "end": 3.0},
            ]

        monkeypatch.setattr("tiny_audio.asr_pipeline.SpeakerDiarizer.diarize", fake_diarize)
        result = pipeline._transcribe(
            {"array": np.zeros(48000, dtype=np.float32)},
            return_timestamps=True,
            return_speakers=True,
            diarization_params={"num_speakers": 2, "min_speakers": None, "max_speakers": None},
        )
        # Only the non-None diarization params are forwarded.
        assert seen == {"num_speakers": 2}
        assert [w["speaker"] for w in result["words"]] == ["SPEAKER_0", "SPEAKER_1"]

    def test_empty_transcript_skips_alignment(self, pipeline, monkeypatch):
        monkeypatch.setattr(
            transformers.AutomaticSpeechRecognitionPipeline,
            "__call__",
            lambda self, inputs, **kw: {"text": ""},
        )

        def must_not_run(*a, **kw):
            raise AssertionError("aligner should not run on empty text")

        monkeypatch.setattr("tiny_audio.asr_pipeline.ForcedAligner.align", must_not_run)
        result = pipeline._transcribe(
            {"array": np.zeros(160, dtype=np.float32)},
            return_timestamps=True,
            return_speakers=False,
            diarization_params={},
        )
        assert result["words"] == []
