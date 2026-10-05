"""`ta train`: overrides reach the recipe's Hydra entry point untouched."""

import sys

import pytest
from typer.testing import CliRunner

from scripts import train_cli
from scripts.cli import app

runner = CliRunner()


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(train_cli.subprocess, "call", lambda cmd: seen.append(cmd) or 0)
    return seen


@pytest.mark.parametrize(
    ("recipe", "module", "preset"),
    [
        ("asr", "scripts.train", "+experiments=stage_1"),
        ("turn-aware", "scripts.turn_aware.train", "+experiment=stage_1"),
        ("speaker-asr", "scripts.speaker_asr.train", "+experiment=stage_1"),
    ],
)
def test_experiment_flag_spells_each_trees_preset_key(calls, recipe, module, preset):
    result = runner.invoke(app, ["train", recipe, "-e", "stage_1", "training.max_steps=5"])
    assert result.exit_code == 0, result.output
    assert calls == [[sys.executable, "-m", module, preset, "training.max_steps=5"]]


def test_hydra_flags_and_plus_overrides_pass_through(calls):
    args = ["+experiment=context", "--cfg", "job", "hub_model_id=null"]
    result = runner.invoke(app, ["train", "speaker-asr", *args])
    assert result.exit_code == 0, result.output
    assert calls == [[sys.executable, "-m", "scripts.speaker_asr.train", *args]]


def test_exit_code_is_the_trainers(monkeypatch):
    monkeypatch.setattr(train_cli.subprocess, "call", lambda cmd: 3)
    assert runner.invoke(app, ["train", "asr"]).exit_code == 3
