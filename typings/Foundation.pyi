"""Stub for PyObjC's `Foundation` bridge (macOS-only, so not installable on Linux).

PyObjC generates its bindings at runtime; this declares only the names the
repo uses (scripts/eval/evaluators/apple_speech.py). Method names are
Objective-C selectors, hence the N802 exemptions.
"""

from typing import Self

class NSObject:
    @classmethod
    def alloc(cls) -> Self: ...

class NSError(NSObject): ...

class NSLocale(NSObject):
    def initWithLocaleIdentifier_(self, identifier: str, /) -> Self: ...  # noqa: N802

class NSURL(NSObject):
    @classmethod
    def fileURLWithPath_(cls, path: str, /) -> Self: ...  # noqa: N802
