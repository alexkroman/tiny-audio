#!/usr/bin/env python3
"""Unified CLI for Tiny Audio."""

import typer

app = typer.Typer(
    name="tiny-audio",
    help="Tiny Audio - ASR model training, evaluation, and deployment",
    no_args_is_help=True,
)


def register_subcommands():
    """Register all subcommand groups."""
    from demo.app import app as demo_app
    from scripts.analysis import app as analysis_app
    from scripts.debug.cli import app as debug_app
    from scripts.deploy.hf_space import deploy as hf_deploy
    from scripts.deploy.runpod import app as runpod_app
    from scripts.dev import app as dev_app
    from scripts.eval.cli import app as eval_app
    from scripts.hub.push import main as push_command
    from scripts.mlx.cli import app as mlx_app

    app.add_typer(eval_app, name="eval", help="Evaluate ASR models on datasets")
    app.add_typer(analysis_app, name="analysis", help="WER analysis and comparison tools")
    app.add_typer(runpod_app, name="runpod", help="Remote training on RunPod")
    app.command(name="deploy", help="Deploy demo to HuggingFace Space")(hf_deploy)
    app.command(name="push", help="Push model to HuggingFace Hub")(push_command)
    app.add_typer(debug_app, name="debug", help="Debug and analysis tools")
    app.add_typer(demo_app, name="demo", help="Launch Gradio demo")
    app.add_typer(dev_app, name="dev", help="Development commands")
    app.add_typer(mlx_app, name="mlx", help="MLX bundle build utilities")


register_subcommands()

if __name__ == "__main__":
    app()
