"""Tests for the pure helpers in scripts.deploy.hf_space."""

import pytest

from scripts.deploy.hf_space import extract_repo_id


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("user/space", "user/space"),
        ("https://huggingface.co/spaces/user/space", "user/space"),
        ("https://huggingface.co/spaces/user/space/", "user/space"),
        ("https://huggingface.co/user/model", "https://huggingface.co/user/model"),
    ],
)
def test_extract_repo_id(given, expected):
    assert extract_repo_id(given) == expected
