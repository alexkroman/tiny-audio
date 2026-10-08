"""Apple SFSpeechRecognizer evaluator (on-device, macOS only)."""

import contextlib
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Unpack

from scripts.eval.audio import prepare_wav_bytes

from .base import Evaluator, EvaluatorOptions, Transcription, console

if TYPE_CHECKING:
    from Foundation import NSError
    from Speech import SFSpeechRecognitionResult, SFSpeechRecognizer


# PyObjC's frameworks exist only on macOS with pyobjc-framework-Speech. They
# are dynamic bridges (every attribute is resolved at runtime), so they are
# kept as modules and their members looked up at the call site; the objects
# they hand back are annotated with the stub classes in typings/.
@dataclass(frozen=True)
class _AppleFrameworks:
    core_foundation: ModuleType
    foundation: ModuleType
    speech: ModuleType


def _load_apple_frameworks() -> _AppleFrameworks | None:
    try:
        import CoreFoundation
        import Foundation
        import Speech
    except ImportError:
        return None
    return _AppleFrameworks(CoreFoundation, Foundation, Speech)


_APPLE_FRAMEWORKS = _load_apple_frameworks()


def _pump_run_loop_until(
    core_foundation: ModuleType, event: threading.Event, timeout_seconds: float
) -> bool:
    """Pump the main CF run loop in 50ms slices until event is set or timeout.

    Speech.framework delivers callbacks via the main run loop; a plain
    threading.Event.wait() starves the framework's XPC delivery and the
    callback never fires.
    """
    deadline = time.time() + timeout_seconds
    while not event.is_set():
        if time.time() >= deadline:
            return False
        core_foundation.CFRunLoopRunInMode(core_foundation.kCFRunLoopDefaultMode, 0.05, True)
    return True


class AppleSpeechEvaluator(Evaluator):
    """Evaluator for Apple SFSpeechRecognizer (on-device, macOS only)."""

    AUTH_TIMEOUT_SECONDS = 300.0
    TRANSCRIBE_TIMEOUT_SECONDS = 60.0

    def __init__(self, locale: str = "en-US", **kwargs: Unpack[EvaluatorOptions]) -> None:
        if _APPLE_FRAMEWORKS is None:
            msg = (
                "Apple SFSpeechRecognizer backend requires PyObjC on macOS. "
                "Install with: pip install pyobjc-framework-Speech"
            )
            raise ImportError(msg)
        self._apple = _APPLE_FRAMEWORKS

        if kwargs.get("num_workers", 1) > 1:
            console.print(
                "[yellow]Warning: AppleSpeechEvaluator forces num_workers=1 "
                "(SFSpeechRecognizer is single-task)[/yellow]"
            )
            kwargs["num_workers"] = 1
        super().__init__(**kwargs)

        self.locale = locale
        self.temp_dir: str | None = tempfile.mkdtemp(prefix="apple-speech-")
        self._authorize()
        self.recognizer = self._build_recognizer(locale)

    def _authorize(self) -> None:
        auth_event = threading.Event()
        status_box: list[int | None] = [None]

        def handler(status: int) -> None:
            status_box[0] = status
            auth_event.set()

        self._apple.speech.SFSpeechRecognizer.requestAuthorization_(handler)
        if not _pump_run_loop_until(
            self._apple.core_foundation, auth_event, self.AUTH_TIMEOUT_SECONDS
        ):
            msg = "Speech recognition authorization request timed out"
            raise TimeoutError(msg)
        if status_box[0] != self._apple.speech.SFSpeechRecognizerAuthorizationStatusAuthorized:
            msg = (
                f"Speech recognition not authorized (status={status_box[0]}). "
                "Approve at System Settings > Privacy & Security > Speech Recognition."
            )
            raise RuntimeError(msg)

    def _build_recognizer(self, locale: str) -> "SFSpeechRecognizer":
        ns_locale = self._apple.foundation.NSLocale.alloc().initWithLocaleIdentifier_(locale)
        recognizer: SFSpeechRecognizer | None = (
            self._apple.speech.SFSpeechRecognizer.alloc().initWithLocale_(ns_locale)
        )
        if recognizer is None:
            msg = f"Unsupported locale: {locale}"
            raise ValueError(msg)
        if not recognizer.supportsOnDeviceRecognition():
            msg = f"On-device recognition unavailable for locale {locale}"
            raise RuntimeError(msg)
        if not recognizer.isAvailable():
            msg = "SFSpeechRecognizer not available right now"
            raise RuntimeError(msg)
        return recognizer

    def transcribe(self, audio: object) -> Transcription:
        wav_bytes = prepare_wav_bytes(audio)
        fd, temp_path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(wav_bytes)

            url = self._apple.foundation.NSURL.fileURLWithPath_(temp_path)
            request = self._apple.speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
            request.setRequiresOnDeviceRecognition_(True)
            request.setShouldReportPartialResults_(False)

            done_event = threading.Event()
            text_box = [""]
            error_box: list[str | None] = [None]

            def handler(
                result: "SFSpeechRecognitionResult | None", error: "NSError | None"
            ) -> None:
                if error is not None:
                    error_box[0] = str(error)
                    done_event.set()
                    return
                if result is None:
                    return
                if result.isFinal():
                    text_box[0] = str(result.bestTranscription().formattedString())
                    done_event.set()

            start = time.time()
            task = self.recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)

            if not _pump_run_loop_until(
                self._apple.core_foundation, done_event, self.TRANSCRIBE_TIMEOUT_SECONDS
            ):
                task.cancel()
                msg = f"Recognition timed out after {self.TRANSCRIBE_TIMEOUT_SECONDS}s"
                raise RuntimeError(msg)
            elapsed = time.time() - start

            if error_box[0]:
                msg = f"SFSpeechRecognizer error: {error_box[0]}"
                raise RuntimeError(msg)
            return text_box[0], elapsed, None
        finally:
            with contextlib.suppress(OSError):
                Path(temp_path).unlink()

    def close(self) -> None:
        temp_dir = getattr(self, "temp_dir", None)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.temp_dir = None
