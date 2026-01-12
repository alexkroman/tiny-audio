#!/usr/bin/env python3
"""Unified deployment CLI.

Commands:
    hf        Deploy demo to HuggingFace Space
    handler   Test inference endpoint handler locally
    runpod    Remote training operations (deploy, train, attach, checkpoint)
"""

import typer

from scripts.deploy.handler_local import test as handler_test
from scripts.deploy.hf_space import deploy as hf_deploy
from scripts.deploy.runpod import app as runpod_app

app = typer.Typer(
    name="deploy",
    help="Deployment tools for HF Space, RunPod, and handler testing",
    no_args_is_help=True,
)

app.command(name="hf", help="Deploy demo to HuggingFace Space")(hf_deploy)
app.command(name="handler", help="Test inference endpoint handler locally")(handler_test)
app.add_typer(runpod_app, name="runpod", help="Remote training on RunPod")

if __name__ == "__main__":
    app()
