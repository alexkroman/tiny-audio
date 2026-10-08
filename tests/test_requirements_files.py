"""The hand-maintained requirements files must stay installable by pip.

requirements.txt is uploaded next to the model on the Hub (scripts/hub/push.py)
and demo/requirements.txt is what the Space installs; neither is generated from
poetry.lock, so nothing else checks them before a user's `pip install` does.
"""

from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement

REQUIREMENTS_FILES = [Path("requirements.txt"), Path("demo/requirements.txt")]


def requirement_lines(path: Path) -> list[str]:
    """Non-blank, non-comment lines, with trailing `# ...` comments removed."""
    lines = (line.split(" #", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


@pytest.mark.parametrize("path", REQUIREMENTS_FILES, ids=str)
def test_every_line_is_a_valid_requirement(path: Path) -> None:
    lines = requirement_lines(path)
    assert lines, f"{path} lists no requirements"
    invalid: list[str] = []
    for line in lines:
        try:
            Requirement(line)
        except InvalidRequirement as exc:
            invalid.append(f"{line!r}: {exc}")
    assert not invalid, f"{path} has lines pip cannot parse:\n" + "\n".join(invalid)


@pytest.mark.parametrize("path", REQUIREMENTS_FILES, ids=str)
def test_no_requirement_is_listed_twice(path: Path) -> None:
    names = [Requirement(line).name.lower() for line in requirement_lines(path)]
    assert len(names) == len(set(names)), f"{path} repeats a requirement: {sorted(names)}"
