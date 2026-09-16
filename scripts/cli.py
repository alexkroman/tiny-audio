#!/usr/bin/env python3
"""Unified CLI for Tiny Audio."""

import importlib
import sys

import typer

app = typer.Typer(
    name="tiny-audio",
    help="Tiny Audio - ASR model training, evaluation, and deployment",
    no_args_is_help=True,
)

# Subcommand groups, as (module, attribute, help). Importing every one of these
# eagerly costs ~4.5s per invocation -- demo.app alone pulls in gradio, torch,
# transformers, torchvision and sklearn -- which pure-shell commands like
# `ta dev lint` would otherwise pay in full. Registration is therefore deferred
# to whichever group the command line actually names (see _register_for_argv).
_GROUPS: dict[str, tuple[str, str, str]] = {
    "eval": ("scripts.eval.cli", "app", "Evaluate ASR models on datasets"),
    "analysis": ("scripts.analysis", "app", "WER analysis and comparison tools"),
    "runpod": ("scripts.deploy.runpod", "app", "Remote training on RunPod"),
    "debug": ("scripts.debug.cli", "app", "Debug and analysis tools"),
    "demo": ("demo.app", "app", "Launch Gradio demo"),
    "dev": ("scripts.dev", "app", "Development commands"),
}

# Single commands rather than groups.
_COMMANDS: dict[str, tuple[str, str, str]] = {
    "deploy": ("scripts.deploy.hf_space", "deploy", "Deploy demo to HuggingFace Space"),
    "push": ("scripts.hub.push", "main", "Push model to HuggingFace Hub"),
}


def _register(name: str) -> None:
    """Import and attach one subcommand group or command."""
    if name in _GROUPS:
        module, attr, help_text = _GROUPS[name]
        sub_app = getattr(importlib.import_module(module), attr)
        app.add_typer(sub_app, name=name, help=help_text)
    else:
        module, attr, help_text = _COMMANDS[name]
        command = getattr(importlib.import_module(module), attr)
        app.command(name=name, help=help_text)(command)


def register_subcommands(only: str | None = None) -> None:
    """Register subcommand groups, or just `only` when given."""
    names = [only] if only is not None else [*_GROUPS, *_COMMANDS]
    for name in names:
        _register(name)


def _register_for_argv(argv: list[str]) -> None:
    """Register only the group named on the command line, else everything.

    `ta --help` and any programmatic use (tests import `app` directly) fall
    through to registering everything, so the full command list stays visible.
    """
    requested = next((arg for arg in argv if not arg.startswith("-")), None)
    if requested in _GROUPS or requested in _COMMANDS:
        register_subcommands(only=requested)
    else:
        register_subcommands()


_register_for_argv(sys.argv[1:])

if __name__ == "__main__":
    app()
