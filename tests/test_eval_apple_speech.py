"""Tests for AppleSpeechEvaluator."""

import sys
from unittest.mock import MagicMock

import pytest

_AUTHORIZED = 3
_DENIED = 1


@pytest.fixture
def fake_speech_frameworks(monkeypatch):
    """Inject fake Speech / Foundation / CoreFoundation modules so tests run on
    Linux CI without pyobjc-framework-Speech installed."""
    fake_speech = MagicMock(name="Speech")
    fake_speech.SFSpeechRecognizerAuthorizationStatusAuthorized = _AUTHORIZED
    fake_speech.SFSpeechRecognizer.requestAuthorization_.side_effect = lambda cb: cb(_AUTHORIZED)

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

    # Reload asr.py so module-level imports pick up the fakes.
    import importlib

    from scripts.eval.evaluators import asr

    importlib.reload(asr)

    return {"Speech": fake_speech, "recognizer": recognizer, "asr": asr}


def _stage_transcription(recognizer, *, text="hello world", error=None):
    def fake_task(_request, handler):
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
    def test_init_authorizes_and_builds_recognizer(self, fake_speech_frameworks):
        ev = fake_speech_frameworks["asr"].AppleSpeechEvaluator(locale="en-US")
        assert ev.locale == "en-US"
        assert ev.recognizer is fake_speech_frameworks["recognizer"]

    def test_authorization_denied_raises(self, fake_speech_frameworks):
        fake_speech_frameworks["Speech"].SFSpeechRecognizer.requestAuthorization_.side_effect = (
            lambda cb: cb(_DENIED)
        )
        with pytest.raises(RuntimeError, match="not authorized"):
            fake_speech_frameworks["asr"].AppleSpeechEvaluator()

    def test_unsupported_locale_raises(self, fake_speech_frameworks):
        fake_speech_frameworks[
            "Speech"
        ].SFSpeechRecognizer.alloc.return_value.initWithLocale_.return_value = None
        with pytest.raises(ValueError, match="Unsupported locale"):
            fake_speech_frameworks["asr"].AppleSpeechEvaluator(locale="zz-ZZ")

    def test_on_device_unsupported_raises(self, fake_speech_frameworks):
        fake_speech_frameworks["recognizer"].supportsOnDeviceRecognition.return_value = False
        with pytest.raises(RuntimeError, match="On-device recognition unavailable"):
            fake_speech_frameworks["asr"].AppleSpeechEvaluator()

    def test_num_workers_gt_1_warns_and_downgrades(self, fake_speech_frameworks):
        ev = fake_speech_frameworks["asr"].AppleSpeechEvaluator(num_workers=4)
        assert ev.num_workers == 1

    def test_transcribe_returns_text_and_elapsed(self, fake_speech_frameworks, mocker):
        mocker.patch.object(
            fake_speech_frameworks["asr"], "prepare_wav_bytes", return_value=b"WAVDATA"
        )
        ev = fake_speech_frameworks["asr"].AppleSpeechEvaluator()
        _stage_transcription(ev.recognizer, text="hello world")

        text, elapsed = ev.transcribe(audio={"array": [], "sampling_rate": 16000})

        assert text == "hello world"
        assert elapsed >= 0

    def test_transcribe_propagates_error(self, fake_speech_frameworks, mocker):
        mocker.patch.object(
            fake_speech_frameworks["asr"], "prepare_wav_bytes", return_value=b"WAVDATA"
        )
        ev = fake_speech_frameworks["asr"].AppleSpeechEvaluator()
        _stage_transcription(ev.recognizer, error="audio too long")

        with pytest.raises(RuntimeError, match="audio too long"):
            ev.transcribe(audio={"array": [], "sampling_rate": 16000})

    def test_transcribe_cleans_up_temp_wav(self, fake_speech_frameworks, mocker):
        from pathlib import Path

        mocker.patch.object(
            fake_speech_frameworks["asr"], "prepare_wav_bytes", return_value=b"WAVDATA"
        )
        ev = fake_speech_frameworks["asr"].AppleSpeechEvaluator()
        _stage_transcription(ev.recognizer)

        before = set(Path(ev.temp_dir).iterdir())
        ev.transcribe(audio={"array": [], "sampling_rate": 16000})
        after = set(Path(ev.temp_dir).iterdir())

        assert before == after, f"temp wav not cleaned up: {after - before}"

    def test_close_removes_temp_dir(self, fake_speech_frameworks):
        from pathlib import Path

        ev = fake_speech_frameworks["asr"].AppleSpeechEvaluator()
        temp_dir = ev.temp_dir
        assert Path(temp_dir).is_dir()

        ev.close()

        assert ev.temp_dir is None
        assert not Path(temp_dir).exists()

    def test_close_idempotent(self, fake_speech_frameworks):
        ev = fake_speech_frameworks["asr"].AppleSpeechEvaluator()
        ev.close()
        ev.close()
