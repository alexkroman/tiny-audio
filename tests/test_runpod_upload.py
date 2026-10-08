"""Tests for SFTP script upload (scripts/deploy/runpod.py)."""

from typing import BinaryIO, cast
from unittest.mock import MagicMock

from fabric import Connection

from scripts.deploy.runpod import _put_text, install_dependencies

# An em dash in a shell comment is the realistic trigger: 1 character, 3 UTF-8
# bytes. Handing paramiko a text stream makes it size the transfer in
# characters while the remote file is sized in bytes, and its confirm-by-stat
# check then raises "size mismatch in put!" on a transfer that succeeded.
NON_ASCII = "#!/bin/bash\n# Verify torch — we never pin it.\necho hi\n"


def _captured_upload(text: str) -> dict[str, object]:
    """Run `_put_text` against a mock connection and return what it uploaded."""
    conn = MagicMock()
    uploaded: dict[str, object] = {}

    def put(fl: BinaryIO, **kwargs: object) -> None:
        uploaded.update({"remote": kwargs.get("remote"), "body": fl.read()})

    conn.put.side_effect = put
    _put_text(cast(Connection, conn), text, "/tmp/script.sh")
    return uploaded


class TestPutText:
    """Tests for the text-to-SFTP upload helper."""

    def test_uploads_bytes_not_text(self) -> None:
        """Paramiko sizes the transfer by len() of what it reads; it must be bytes."""
        assert isinstance(_captured_upload(NON_ASCII)["body"], bytes)

    def test_uploaded_length_matches_remote_byte_size(self) -> None:
        """The regression: char count != byte count once any non-ASCII is present."""
        body = cast(bytes, _captured_upload(NON_ASCII)["body"])
        assert len(body) == len(NON_ASCII.encode("utf-8"))
        assert len(body) != len(NON_ASCII), "test string must exercise multi-byte chars"

    def test_round_trips_content_verbatim(self) -> None:
        body = cast(bytes, _captured_upload(NON_ASCII)["body"])
        assert body.decode("utf-8") == NON_ASCII

    def test_passes_remote_path_through(self) -> None:
        assert _captured_upload(NON_ASCII)["remote"] == "/tmp/script.sh"


class TestInstallDependenciesUpload:
    """Guards the real bootstrap script, whose comments do contain em dashes."""

    def test_setup_script_uploads_as_bytes(self) -> None:
        conn = MagicMock()
        uploaded: dict[str, object] = {}

        def put(fl: BinaryIO, **_: object) -> None:
            uploaded.update({"body": fl.read()})

        conn.put.side_effect = put
        conn.run.return_value = MagicMock(ok=True, stdout="")

        install_dependencies(cast(Connection, conn))

        body = uploaded["body"]
        assert isinstance(body, bytes)
        # Byte length is what the remote stat reports; character length is what
        # a StringIO upload would have claimed. They differ here, which is
        # precisely why this script cannot be uploaded as text.
        assert len(body) > len(body.decode("utf-8"))
