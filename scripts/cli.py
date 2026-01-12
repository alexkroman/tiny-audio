#!/usr/bin/env python3
"""Unified CLI for Tiny Audio.

Usage:
    tiny-audio <command> [options]
    ta <command> [options]  # Short alias

Commands:
    eval        Evaluate ASR models on datasets
    analysis    WER analysis and comparison tools
    deploy      Deploy to HF Space, test handlers, RunPod training
    hub         HuggingFace Hub operations
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

    # Deploy
    from scripts.deploy.cli import app as deploy_app

    app.add_typer(deploy_app, name="deploy", help="Deployment tools (HF Space, RunPod, handler)")

    # Hub
    from scripts.hub.cli import app as hub_app

    app.add_typer(hub_app, name="hub", help="HuggingFace Hub operations")

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
