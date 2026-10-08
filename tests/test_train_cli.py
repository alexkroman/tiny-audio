"""`ta train`: overrides reach the recipe's Hydra entry point untouched."""

import sys

import pytest
from typer.testing import CliRunner

from scripts.cli import app

runner = CliRunner()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    seen: list[list[str]] = []

    def fake_call(cmd: list[str]) -> int:
        seen.append(cmd)
        return 0

    monkeypatch.setattr("scripts.train_cli.subprocess.call", fake_call)
    return seen


@pytest.mark.parametrize(
    ("recipe", "module", "preset"),
    [
        ("asr", "scripts.train", "+experiments=stage_1"),
    ],
)
def test_experiment_flag_spells_each_trees_preset_key(calls: list[list[str]], recipe: str, module: str, preset: str) -> None:
    result = runner.invoke(app, ["train", recipe, "-e", "stage_1", "training.max_steps=5"])
    assert result.exit_code == 0, result.output
    assert calls == [[sys.executable, "-m", module, preset, "training.max_steps=5"]]


def test_hydra_flags_and_plus_overrides_pass_through(calls: list[list[str]]) -> None:
    args = ["+experiments=mps_smoke", "--cfg", "job", "hub_model_id=null"]
    result = runner.invoke(app, ["train", "asr", *args])
    assert result.exit_code == 0, result.output
    assert calls == [[sys.executable, "-m", "scripts.train", *args]]


def test_exit_code_is_the_trainers(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_call(cmd: list[str]) -> int:
        return 3

    monkeypatch.setattr("scripts.train_cli.subprocess.call", failing_call)
    assert runner.invoke(app, ["train", "asr"]).exit_code == 3
