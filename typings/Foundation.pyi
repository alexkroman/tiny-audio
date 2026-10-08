"""Stub for PyObjC's `Foundation` bridge (macOS-only, so not installable on Linux).

PyObjC generates its bindings at runtime; every attribute is an Objective-C
object or function, typed here as Any.
"""

from typing import Any

def __getattr__(name: str) -> Any: ...
