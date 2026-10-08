#!/usr/bin/env python3
"""`ta train`: run a training recipe locally.

Each recipe is a `@hydra.main` script that parses its own argv, so these
commands do not re-declare its config: everything after the recipe name is
handed to `python -m <module>` untouched, exactly as `ta runpod` runs it on a
pod. `--experiment/-e` is the one convenience, spelling the preset override
(`+experiments=<name>` from configs/experiments/).
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

EXPERIMENT_HELP = "Preset from configs/experiments/ (adds the +experiments override)"


@app.callback()
def main() -> None:
    """Train a recipe locally (Hydra overrides pass through)."""
    # A callback keeps `ta train` a group with `asr` as a subcommand; with one
    # command and no callback, Typer would collapse it into `ta train` itself.


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
