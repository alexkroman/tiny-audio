"""Stub for the Hugging Face Spaces runtime's `spaces` package.

The package exists only inside a Space (see demo/requirements.txt); this covers
the one decorator demo/app.py uses.
"""

from collections.abc import Callable
from datetime import timedelta
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

class GPU:
    """`@spaces.GPU(duration=...)`: run the decorated function on a ZeroGPU slot."""

    def __init__(
        self,
        *,
        duration: int | timedelta | Callable[..., int | timedelta] | None = ...,
    ) -> None: ...
    def __call__(self, fn: _F, /) -> _F: ...
