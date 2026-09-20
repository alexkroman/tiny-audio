"""Tests for the pure helpers in scripts.deploy.hf_space."""

import pytest
import typer

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
def test_extract_repo_id(given, expected):
    assert extract_repo_id(given) == expected


def test_extract_repo_id_rejects_non_space_urls():
    with pytest.raises(typer.BadParameter, match="Not a Space"):
        extract_repo_id("https://huggingface.co/user/model")
