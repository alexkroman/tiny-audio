#!/usr/bin/env python3
"""`ta train`: run a training recipe locally.

Each recipe is a `@hydra.main` script that parses its own argv, so these
commands do not re-declare its config: everything after the recipe name is
handed to `python -m <module>` untouched, exactly as `ta runpod` runs it on a
pod. `--experiment/-e` is the one convenience, spelling the preset override
each config tree expects (`+experiments=` for ASR, `+experiment=` for the
standalone turn-aware and speaker-ASR trees).
"""

import subprocess
import sys
from typing import Annotated

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Train a recipe locally (Hydra overrides pass through).",
)

# Forward Hydra overrides (`key=value`, `+experiment=x`) and Hydra's own flags
# (`--cfg job`, `--multirun`) instead of rejecting them as unknown options.
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}

EXPERIMENT_HELP = "Preset from the recipe's experiment directory (adds the +experiment override)"


def _run(module: str, preset_key: str, experiment: str | None, overrides: list[str]) -> None:
    args = [f"+{preset_key}={experiment}"] if experiment else []
    cmd = [sys.executable, "-m", module, *args, *overrides]
    typer.echo(" ".join(cmd[1:]), err=True)
    raise typer.Exit(subprocess.call(cmd))


@app.command("asr", context_settings=_PASSTHROUGH)
def asr_cmd(
    ctx: typer.Context,
    experiment: Annotated[
        str | None, typer.Option("--experiment", "-e", help=EXPERIMENT_HELP)
    ] = None,
) -> None:
    """Encoder + projector + LLM ASR (scripts/train.py, configs/experiments/)."""
    _run("scripts.train", "experiments", experiment, ctx.args)


@app.command("turn-aware", context_settings=_PASSTHROUGH)
def turn_aware_cmd(
    ctx: typer.Context,
    experiment: Annotated[
        str | None, typer.Option("--experiment", "-e", help=EXPERIMENT_HELP)
    ] = None,
) -> None:
    """Turn-aware Qwen3-ASR LoRA (configs/turn_aware/experiment/)."""
    _run("scripts.turn_aware.train", "experiment", experiment, ctx.args)


@app.command("speaker-asr", context_settings=_PASSTHROUGH)
def speaker_asr_cmd(
    ctx: typer.Context,
    experiment: Annotated[
        str | None, typer.Option("--experiment", "-e", help=EXPERIMENT_HELP)
    ] = None,
) -> None:
    """Speaker-attributed Qwen3-ASR LoRA (configs/speaker_asr/experiment/).

    Build the pool first with the same overrides: `ta speaker-asr build-pool`.
    """
    _run("scripts.speaker_asr.train", "experiment", experiment, ctx.args)
