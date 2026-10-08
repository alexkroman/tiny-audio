"""Stub for the parts of fabric's `Connection` that scripts/deploy/runpod.py uses.

fabric ships no type information; analysed from source, every call is untyped.
"""

from typing import IO, Any, Protocol

from invoke.runners import Result

__all__ = ["Connection", "Result"]

class _SFTPClient(Protocol):
    def chmod(self, path: str, mode: int) -> None: ...

class Connection:
    host: str
    user: str
    port: int
    connect_kwargs: dict[str, Any]
    connect_timeout: int | None
    def __init__(
        self,
        host: str,
        user: str | None = None,
        port: int | None = None,
        config: Any = None,
        gateway: Connection | str | None = None,
        forward_agent: bool | None = None,
        connect_timeout: int | None = None,
        connect_kwargs: dict[str, Any] | None = None,
        inline_ssh_env: bool | None = None,
    ) -> None: ...
    def run(self, command: str, **kwargs: Any) -> Result: ...
    def sudo(self, command: str, **kwargs: Any) -> Result: ...
    def put(
        self, local: str | IO[bytes], remote: str | None = None, preserve_mode: bool = True
    ) -> object: ...
    def get(
        self, remote: str, local: str | IO[bytes] | None = None, preserve_mode: bool = True
    ) -> object: ...
    def sftp(self) -> _SFTPClient: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
