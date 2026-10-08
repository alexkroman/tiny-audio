"""Tests for AppleSpeechEvaluator."""

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TypedDict
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from scripts.eval.evaluators import apple_speech

_AUTHORIZED = 3
_DENIED = 1


class _FakeFrameworks(TypedDict):
    Speech: MagicMock
    recognizer: MagicMock
    apple_speech: ModuleType


def _grant(cb: Callable[[int], None]) -> None:
    cb(_AUTHORIZED)


def _deny(cb: Callable[[int], None]) -> None:
    cb(_DENIED)


@pytest.fixture
def fake_speech_frameworks(monkeypatch: pytest.MonkeyPatch) -> _FakeFrameworks:
    """Inject fake Speech / Foundation / CoreFoundation modules so tests run on
    Linux CI without pyobjc-framework-Speech installed."""
    fake_speech = MagicMock(name="Speech")
    fake_speech.SFSpeechRecognizerAuthorizationStatusAuthorized = _AUTHORIZED
    fake_speech.SFSpeechRecognizer.requestAuthorization_.side_effect = _grant

    recognizer = MagicMock(name="SFSpeechRecognizer-instance")
    recognizer.supportsOnDeviceRecognition.return_value = True
    recognizer.isAvailable.return_value = True
    fake_speech.SFSpeechRecognizer.alloc.return_value.initWithLocale_.return_value = recognizer

    fake_corefoundation = MagicMock(name="CoreFoundation")
    fake_corefoundation.CFRunLoopRunInMode.return_value = 0
    fake_corefoundation.kCFRunLoopDefaultMode = "kCFRunLoopDefaultMode"

    monkeypatch.setitem(sys.modules, "Speech", fake_speech)
    monkeypatch.setitem(sys.modules, "Foundation", MagicMock(name="Foundation"))
    monkeypatch.setitem(sys.modules, "CoreFoundation", fake_corefoundation)

    # Reload apple_speech.py so module-level imports pick up the fakes.
    importlib.reload(apple_speech)

    return {"Speech": fake_speech, "recognizer": recognizer, "apple_speech": apple_speech}


def _stage_transcription(
    recognizer: MagicMock, *, text: str = "hello world", error: str | None = None
) -> None:
    def fake_task(
        _request: object, handler: Callable[[MagicMock | None, str | None], None]
    ) -> MagicMock:
        if error is not None:
            handler(None, error)
        else:
            result = MagicMock()
            result.isFinal.return_value = True
            transcription = MagicMock()
            transcription.formattedString.return_value = text
            result.bestTranscription.return_value = transcription
            handler(result, None)
        return MagicMock()

    recognizer.recognitionTaskWithRequest_resultHandler_.side_effect = fake_task


class TestAppleSpeechEvaluator:
    def test_init_authorizes_and_builds_recognizer(
        self, fake_speech_frameworks: _FakeFrameworks
    ) -> None:
        ev = fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator(locale="en-US")
        assert ev.locale == "en-US"
        assert ev.recognizer is fake_speech_frameworks["recognizer"]

    def test_authorization_denied_raises(self, fake_speech_frameworks: _FakeFrameworks) -> None:
        speech = fake_speech_frameworks["Speech"]
        speech.SFSpeechRecognizer.requestAuthorization_.side_effect = _deny
        with pytest.raises(RuntimeError, match="not authorized"):
            fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator()

    def test_unsupported_locale_raises(self, fake_speech_frameworks: _FakeFrameworks) -> None:
        fake_speech_frameworks[
            "Speech"
        ].SFSpeechRecognizer.alloc.return_value.initWithLocale_.return_value = None
        with pytest.raises(ValueError, match="Unsupported locale"):
            fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator(locale="zz-ZZ")

    def test_on_device_unsupported_raises(self, fake_speech_frameworks: _FakeFrameworks) -> None:
        fake_speech_frameworks["recognizer"].supportsOnDeviceRecognition.return_value = False
        with pytest.raises(RuntimeError, match="On-device recognition unavailable"):
            fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator()

    def test_num_workers_gt_1_warns_and_downgrades(
        self, fake_speech_frameworks: _FakeFrameworks
    ) -> None:
        ev = fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator(num_workers=4)
        assert ev.num_workers == 1

    def test_transcribe_returns_text_elapsed_and_confidence(
        self, fake_speech_frameworks: _FakeFrameworks, mocker: MockerFixture
    ) -> None:
        """`transcribe` returns (text, time, confidence) for every evaluator.

        Confidence is None here: the Speech framework exposes no per-token
        logits. The uniform arity is what lets `Evaluator._process_sample`
        unpack the result without length-sniffing it.
        """
        mocker.patch.object(
            fake_speech_frameworks["apple_speech"], "prepare_wav_bytes", return_value=b"WAVDATA"
        )
        ev = fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator()
        _stage_transcription(ev.recognizer, text="hello world")

        text, elapsed, confidence = ev.transcribe(audio={"array": [], "sampling_rate": 16000})

        assert text == "hello world"
        assert elapsed >= 0
        assert confidence is None

    def test_transcribe_propagates_error(
        self, fake_speech_frameworks: _FakeFrameworks, mocker: MockerFixture
    ) -> None:
        mocker.patch.object(
            fake_speech_frameworks["apple_speech"], "prepare_wav_bytes", return_value=b"WAVDATA"
        )
        ev = fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator()
        _stage_transcription(ev.recognizer, error="audio too long")

        with pytest.raises(RuntimeError, match="audio too long"):
            ev.transcribe(audio={"array": [], "sampling_rate": 16000})

    def test_transcribe_cleans_up_temp_wav(
        self, fake_speech_frameworks: _FakeFrameworks, mocker: MockerFixture
    ) -> None:
        mocker.patch.object(
            fake_speech_frameworks["apple_speech"], "prepare_wav_bytes", return_value=b"WAVDATA"
        )
        ev = fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator()
        _stage_transcription(ev.recognizer)

        before = set(Path(ev.temp_dir).iterdir())
        ev.transcribe(audio={"array": [], "sampling_rate": 16000})
        after = set(Path(ev.temp_dir).iterdir())

        assert before == after, f"temp wav not cleaned up: {after - before}"

    def test_close_removes_temp_dir(self, fake_speech_frameworks: _FakeFrameworks) -> None:
        ev = fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator()
        temp_dir = ev.temp_dir
        assert Path(temp_dir).is_dir()

        ev.close()

        assert ev.temp_dir is None
        assert not Path(temp_dir).exists()

    def test_close_idempotent(self, fake_speech_frameworks: _FakeFrameworks) -> None:
        ev = fake_speech_frameworks["apple_speech"].AppleSpeechEvaluator()
        ev.close()
        ev.close()

        assert ev.temp_dir is None
