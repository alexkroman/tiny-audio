"""Tests for scripts.hub.push with the Hub calls faked out."""

from pathlib import Path
from typing import Any

import pytest
import typer
from huggingface_hub import CommitOperationAdd

from scripts.hub import push


class FakeHfApi:
    """Records the token and `create_commit` calls instead of hitting the Hub."""

    tokens: list[str | None] = []
    commits: list[dict[str, Any]] = []

    def __init__(self, token: str | None = None) -> None:
        FakeHfApi.tokens.append(token)

    def create_commit(self, **kwargs: Any) -> None:
        FakeHfApi.commits.append(kwargs)


@pytest.fixture
def repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory holding a partial set of the files `push` publishes."""
    (tmp_path / "tiny_audio").mkdir()
    for name in ("asr_config.py", "asr_modeling.py", "handler.py"):
        (tmp_path / "tiny_audio" / name).write_text(f"# {name}")
    (tmp_path / "MODEL_CARD.md").write_text("# card")
    (tmp_path / "requirements.txt").write_text("torch\n")
    monkeypatch.chdir(tmp_path)
    FakeHfApi.tokens = []
    FakeHfApi.commits = []
    monkeypatch.setattr(push, "HfApi", FakeHfApi)
    return tmp_path


def _uploaded(commit: dict[str, Any]) -> dict[str, str | bytes]:
    ops: list[CommitOperationAdd] = commit["operations"]
    out: dict[str, str | bytes] = {}
    for op in ops:
        assert isinstance(op.path_or_fileobj, str | bytes)
        out[op.path_in_repo] = op.path_or_fileobj
    return out


@pytest.mark.usefixtures("repo_root")
def test_push_without_token_or_login_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(push, "get_token", lambda: None)
    with pytest.raises(typer.BadParameter, match="not logged in"):
        push.main(hf_token=None)
    assert FakeHfApi.commits == []


@pytest.mark.usefixtures("repo_root")
def test_push_uses_login_cache_and_skips_missing_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(push, "get_token", lambda: "cached")
    push.main(repo_id="me/model", branch="dev", checkpoint_dir=None, hf_token=None)

    assert FakeHfApi.tokens == [None]
    [commit] = FakeHfApi.commits
    assert commit["repo_id"] == "me/model"
    assert commit["repo_type"] == "model"
    assert commit["revision"] == "dev"
    files = _uploaded(commit)
    assert list(files) == [
        ".gitattributes",
        "asr_config.py",
        "asr_modeling.py",
        "handler.py",
        "README.md",
        "requirements.txt",
    ]
    gitattributes = files[".gitattributes"]
    assert isinstance(gitattributes, bytes)
    assert b"tokenizer_config.json -filter" in gitattributes
    assert files["README.md"] == "MODEL_CARD.md"
    assert files["handler.py"] == str(Path("tiny_audio") / "handler.py")


def test_push_includes_checkpoint_tokenizer_files(repo_root: Path) -> None:
    ckpt = repo_root / "ckpt"
    ckpt.mkdir()
    for name in ("tokenizer_config.json", "tokenizer.json"):
        (ckpt / name).write_text("{}")
    push.main(repo_id="me/model", branch="main", checkpoint_dir=ckpt, hf_token="tok")

    assert FakeHfApi.tokens == ["tok"]
    files = _uploaded(FakeHfApi.commits[0])
    assert files["tokenizer_config.json"] == str(ckpt / "tokenizer_config.json")
    assert files["tokenizer.json"] == str(ckpt / "tokenizer.json")
    assert "special_tokens_map.json" not in files
