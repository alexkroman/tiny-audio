"""Deepgram and ElevenLabs on speaker datasets: diarize, then `<SPK_n>` turns."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from elevenlabs.types import SpeechToTextChunkResponseModel

from scripts.eval.evaluators.asr import DeepgramEvaluator, ElevenLabsEvaluator


@pytest.fixture
def audio() -> dict[str, object]:
    return {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}


def _deepgram(evaluator: DeepgramEvaluator) -> MagicMock:
    words = [
        SimpleNamespace(word="hi", speaker=3),
        SimpleNamespace(word="there", speaker=3),
        SimpleNamespace(word="hello", speaker=0),
        SimpleNamespace(word="again", speaker=3),
    ]
    alternative = SimpleNamespace(transcript="hi there hello again", words=words)
    response = SimpleNamespace(
        results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alternative])])
    )
    transcribe = MagicMock(return_value=response)
    evaluator.client = MagicMock()
    evaluator.client.listen.v1.media.transcribe_file = transcribe
    return transcribe


def _word(text: str, kind: str, speaker: str) -> dict[str, object]:
    return {"text": text, "type": kind, "speaker_id": speaker, "logprob": 0.0}


def _elevenlabs(evaluator: ElevenLabsEvaluator) -> MagicMock:
    response = SpeechToTextChunkResponseModel.model_validate(
        {
            "language_code": "en",
            "language_probability": 1.0,
            "text": "Hi there (laughs) Hello",
            "words": [
                _word("Hi", "word", "speaker_1"),
                _word(" ", "spacing", "speaker_1"),
                _word("there", "word", "speaker_1"),
                _word("(laughs)", "audio_event", "speaker_0"),
                _word("Hello", "word", "speaker_0"),
            ],
        }
    )
    convert = MagicMock(return_value=response)
    evaluator.client = MagicMock()
    evaluator.client.speech_to_text.convert = convert
    return convert


class TestDeepgramSpeakers:
    def test_speaker_dataset_diarizes_into_turns(self, audio: dict[str, object]) -> None:
        evaluator = DeepgramEvaluator(api_key="k")
        evaluator.speakers = True
        transcribe = _deepgram(evaluator)

        text, _, _ = evaluator.transcribe(audio)

        assert text == "<SPK_1>hi there<SPK_2>hello<SPK_1>again"
        assert transcribe.call_args.kwargs["diarize"] is True

    def test_plain_dataset_returns_transcript(self, audio: dict[str, object]) -> None:
        evaluator = DeepgramEvaluator(api_key="k")
        transcribe = _deepgram(evaluator)

        text, _, _ = evaluator.transcribe(audio)

        assert text == "hi there hello again"
        assert transcribe.call_args.kwargs["diarize"] is False


class TestElevenLabsSpeakers:
    def test_speaker_dataset_diarizes_into_turns(self, audio: dict[str, object]) -> None:
        evaluator = ElevenLabsEvaluator(api_key="k")
        evaluator.speakers = True
        convert = _elevenlabs(evaluator)

        text, _, _ = evaluator.transcribe(audio)

        assert text == "<SPK_1>Hi there<SPK_2>Hello"
        assert convert.call_args.kwargs["diarize"] is True

    def test_plain_dataset_returns_text(self, audio: dict[str, object]) -> None:
        evaluator = ElevenLabsEvaluator(api_key="k")
        convert = _elevenlabs(evaluator)

        text, _, _ = evaluator.transcribe(audio)

        assert text == "Hi there (laughs) Hello"
        assert convert.call_args.kwargs["diarize"] is False
