"""Stub for PyObjC's `Speech` bridge (macOS-only, so not installable on Linux).

PyObjC generates its bindings at runtime; this declares only the names the
repo uses (scripts/eval/evaluators/apple_speech.py). Method names are
Objective-C selectors, hence the N802 exemptions.
"""

from collections.abc import Callable
from typing import Self

from Foundation import NSURL, NSError, NSLocale, NSObject

SFSpeechRecognizerAuthorizationStatusAuthorized: int

class SFTranscription(NSObject):
    def formattedString(self) -> str: ...  # noqa: N802

class SFSpeechRecognitionResult(NSObject):
    def isFinal(self) -> bool: ...  # noqa: N802
    def bestTranscription(self) -> SFTranscription: ...  # noqa: N802

class SFSpeechRecognitionTask(NSObject):
    def cancel(self) -> None: ...

class SFSpeechURLRecognitionRequest(NSObject):
    def initWithURL_(self, url: NSURL, /) -> Self: ...  # noqa: N802
    def setRequiresOnDeviceRecognition_(self, value: bool, /) -> None: ...  # noqa: N802
    def setShouldReportPartialResults_(self, value: bool, /) -> None: ...  # noqa: N802

class SFSpeechRecognizer(NSObject):
    @staticmethod
    def requestAuthorization_(handler: Callable[[int], None], /) -> None: ...  # noqa: N802
    def initWithLocale_(self, locale: NSLocale, /) -> Self | None: ...  # noqa: N802
    def supportsOnDeviceRecognition(self) -> bool: ...  # noqa: N802
    def isAvailable(self) -> bool: ...  # noqa: N802
    def recognitionTaskWithRequest_resultHandler_(  # noqa: N802
        self,
        request: SFSpeechURLRecognitionRequest,
        handler: Callable[[SFSpeechRecognitionResult | None, NSError | None], None],
        /,
    ) -> SFSpeechRecognitionTask: ...
