"""scripts/claude_hooks.py: the PreToolUse guard and the PostToolUse linter."""

import io
import json
from pathlib import Path

import pytest

from scripts import claude_hooks


def payload(path: str) -> dict[str, dict[str, str]]:
    return {"tool_input": {"file_path": path}}


class TestEditedPath:
    def test_absolute_and_relative_paths_resolve_to_the_repo(self, tmp_path: Path) -> None:
        assert claude_hooks.edited_path(payload(str(tmp_path / "a/b.py")), tmp_path) == "a/b.py"
        assert claude_hooks.edited_path(payload("a/b.py"), tmp_path) == "a/b.py"

    def test_notebook_paths_count(self, tmp_path: Path) -> None:
        notebook = {"tool_input": {"notebook_path": "n.ipynb"}}
        assert claude_hooks.edited_path(notebook, tmp_path) == "n.ipynb"

    def test_outside_the_repo_or_missing_is_none(self, tmp_path: Path) -> None:
        assert claude_hooks.edited_path(payload("/etc/hosts"), tmp_path) is None
        assert claude_hooks.edited_path({}, tmp_path) is None


class TestProtect:
    @pytest.mark.parametrize("rel", ["poetry.lock", "quality/file_length.json"])
    def test_generated_files_are_blocked(self, tmp_path: Path, rel: str) -> None:
        code, message = claude_hooks.protect(payload(rel), tmp_path)
        assert code == claude_hooks.BLOCK
        assert rel in message

    @pytest.mark.parametrize("rel", ["pyproject.toml", "quality/vulture_whitelist.py"])
    def test_other_files_pass(self, tmp_path: Path, rel: str) -> None:
        assert claude_hooks.protect(payload(rel), tmp_path) == (0, "")


class TestLintEdit:
    def test_formats_fixes_and_passes_clean_code(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = [1,2]\n")
        assert claude_hooks.lint_edit(payload("a.py"), tmp_path) == (0, "")
        assert (tmp_path / "a.py").read_text() == "x = [1, 2]\n"

    def test_reports_what_cannot_be_fixed(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def f():\n    return undefined_name\n")
        code, message = claude_hooks.lint_edit(payload("a.py"), tmp_path)
        assert code == claude_hooks.BLOCK
        assert "F821" in message

    def test_non_python_and_missing_files_are_skipped(self, tmp_path: Path) -> None:
        assert claude_hooks.lint_edit(payload("README.md"), tmp_path) == (0, "")
        assert claude_hooks.lint_edit(payload("gone.py"), tmp_path) == (0, "")


class TestMain:
    def test_runs_the_named_hook_on_stdin(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload("poetry.lock"))))
        assert claude_hooks.main(["protect"]) == claude_hooks.BLOCK
        assert "poetry lock" in capsys.readouterr().err

    def test_bad_payload_is_not_judged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        assert claude_hooks.main(["protect"]) == 0

    def test_unknown_hook_is_a_usage_error(self) -> None:
        assert claude_hooks.main(["nope"]) == 1
