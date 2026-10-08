"""Stub for PyObjC's `CoreFoundation` bridge (macOS-only, so not installable on Linux).

PyObjC generates its bindings at runtime; this declares only the names the
repo uses (scripts/eval/evaluators/apple_speech.py). Names are CoreFoundation's
own, hence the pep8-naming exemptions.
"""

kCFRunLoopDefaultMode: str  # noqa: N816

def CFRunLoopRunInMode(  # noqa: N802
    mode: str, seconds: float, return_after_source_handled: bool, /
) -> int: ...
