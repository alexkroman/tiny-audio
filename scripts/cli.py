#!/usr/bin/env python3
"""Unified CLI for Tiny Audio.

Usage:
    tiny-audio <command> [options]
    ta <command> [options]  # Short alias

Commands:
    eval        Evaluate ASR models on datasets
    analysis    WER analysis and comparison tools
    runpod      Remote training on RunPod
    deploy      Deploy demo to HuggingFace Space
    push        Push model to HuggingFace Hub
    debug       Debug and analysis tools
    demo        Launch Gradio demo
    dev         Development commands (lint, test, format, etc.)
"""

import typer

app = typer.Typer(
    name="tiny-audio",
    help="Tiny Audio - ASR model training, evaluation, and deployment",
    no_args_is_help=True,
)


def register_subcommands():
    """Register all subcommand groups."""
    # Eval
    from scripts.eval.cli import app as eval_app

    app.add_typer(eval_app, name="eval", help="Evaluate ASR models on datasets")

    # Analysis
    from scripts.analysis import app as analysis_app

    app.add_typer(analysis_app, name="analysis", help="WER analysis and comparison tools")

    # RunPod (top-level for remote training)
    from scripts.deploy.runpod import app as runpod_app

    app.add_typer(runpod_app, name="runpod", help="Remote training on RunPod")

    # Deploy (direct command for HF Space)
    from scripts.deploy.hf_space import deploy as hf_deploy

    app.command(name="deploy", help="Deploy demo to HuggingFace Space")(hf_deploy)

    # Push (direct command for HF Hub)
    from scripts.hub.push import main as push_command

    app.command(name="push", help="Push model to HuggingFace Hub")(push_command)

    # Debug
    from scripts.debug.cli import app as debug_app

    app.add_typer(debug_app, name="debug", help="Debug and analysis tools")

    # Demo
    from demo.app import app as demo_app

    app.add_typer(demo_app, name="demo", help="Launch Gradio demo")

    # Dev
    from scripts.dev import app as dev_app

    app.add_typer(dev_app, name="dev", help="Development commands")


# Register subcommands at import time
register_subcommands()

if __name__ == "__main__":
    app()
