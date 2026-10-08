"""Tests for scripts.deploy.hf_space with the Hub calls faked out."""

from pathlib import Path

import pytest
import typer

from scripts.deploy import hf_space
from scripts.deploy.hf_space import extract_repo_id


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("user/space", "user/space"),
        ("https://huggingface.co/spaces/user/space", "user/space"),
        ("https://huggingface.co/spaces/user/space/", "user/space"),
        ("https://huggingface.co/spaces/user/space/tree/main", "user/space"),
    ],
)
def test_extract_repo_id(given: str, expected: str) -> None:
    assert extract_repo_id(given) == expected


def test_extract_repo_id_rejects_non_space_urls() -> None:
    with pytest.raises(typer.BadParameter, match="Not a Space"):
        extract_repo_id("https://huggingface.co/user/model")


class FakeHfApi:
    """Records `create_repo` calls instead of hitting the Hub."""

    create_repo_calls: list[dict[str, object]] = []

    def create_repo(self, **kwargs: object) -> None:
        FakeHfApi.create_repo_calls.append(kwargs)


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    for name in ("app.py", "requirements.txt", "README.md"):
        (tmp_path / name).write_text(name)
    return tmp_path


@pytest.fixture
def uploads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    FakeHfApi.create_repo_calls = []
    calls: list[dict[str, object]] = []

    def fake_upload_folder(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(hf_space, "HfApi", FakeHfApi)
    monkeypatch.setattr(hf_space, "upload_folder", fake_upload_folder)
    return calls


@pytest.mark.parametrize(("delete_existing", "delete_patterns"), [(False, None), (True, ["*"])])
def test_deploy_creates_space_and_uploads(
    demo_dir: Path,
    uploads: list[dict[str, object]],
    capsys: pytest.CaptureFixture[str],
    delete_existing: bool,
    delete_patterns: list[str] | None,
) -> None:
    hf_space.deploy(
        repo_id="https://huggingface.co/spaces/me/demo",
        demo_dir=demo_dir,
        delete_existing=delete_existing,
        private=True,
    )
    assert FakeHfApi.create_repo_calls == [
        {
            "repo_id": "me/demo",
            "repo_type": "space",
            "space_sdk": "gradio",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert uploads == [
        {
            "folder_path": str(demo_dir),
            "repo_id": "me/demo",
            "repo_type": "space",
            "delete_patterns": delete_patterns,
            "commit_message": "Deploy demo to HF Space",
        }
    ]
    assert "https://huggingface.co/spaces/me/demo" in capsys.readouterr().out


def test_deploy_rejects_incomplete_demo_dir(
    demo_dir: Path, uploads: list[dict[str, object]]
) -> None:
    (demo_dir / "app.py").unlink()
    (demo_dir / "README.md").unlink()
    with pytest.raises(typer.BadParameter, match="app.py, README.md"):
        hf_space.deploy(repo_id="me/demo", demo_dir=demo_dir)
    assert FakeHfApi.create_repo_calls == []
    assert uploads == []
