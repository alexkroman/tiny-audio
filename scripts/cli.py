#!/usr/bin/env python3
"""Unified CLI for Tiny Audio."""

import importlib

import typer
import typer.core
import typer.main

# Subcommands, as name -> module. Each module exposes a Typer named `app`: a
# multi-command module (analysis, runpod, debug, dev) mounts as a group, and a
# module with one `@app.command()` (eval, demo, deploy, push) mounts as a plain
# command. Help text comes from the module itself, so it is written once.
#
# Importing every one of these eagerly costs ~4.5s per invocation -- demo.app
# alone pulls in gradio, torch, transformers, torchvision and sklearn -- which
# pure-shell commands like `ta dev lint` would otherwise pay in full. LazyGroup
# therefore imports a module only when its command is looked up.
SUBCOMMANDS: dict[str, str] = {
    "eval": "scripts.eval.cli",
    "analysis": "scripts.analysis",
    "deploy": "scripts.deploy.hf_space",
    "push": "scripts.hub.push",
    "runpod": "scripts.deploy.runpod",
    "debug": "scripts.debug.cli",
    "demo": "demo.app",
    "dev": "scripts.dev",
}


class LazyGroup(typer.core.TyperGroup):
    """Click's lazy-loading group pattern, applied to Typer sub-apps.

    `list_commands` answers from `SUBCOMMANDS` without importing anything;
    `get_command` imports a module the first time its name is resolved and
    caches the resulting Click command. `ta --help` still loads every module,
    because rendering the command table needs each one's help text.
    """

    def list_commands(self, _ctx: typer.Context) -> list[str]:
        """Registration order, without importing any module."""
        return [*self.commands, *(name for name in SUBCOMMANDS if name not in self.commands)]

    def get_command(self, _ctx: typer.Context, cmd_name: str):
        """Import the module behind `cmd_name` on first use and cache its command."""
        if cmd_name not in self.commands and cmd_name in SUBCOMMANDS:
            sub_app = importlib.import_module(SUBCOMMANDS[cmd_name]).app
            command = typer.main.get_command(sub_app)
            command.name = cmd_name
            self.add_command(command, cmd_name)
        return self.commands.get(cmd_name)


app = typer.Typer(name="tiny-audio", cls=LazyGroup, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Tiny Audio - ASR model training, evaluation, and deployment."""
    # Typer builds a Group (rather than a single Command) only when the app has
    # a callback or eagerly registered sub-apps; ours are all lazy.


if __name__ == "__main__":
    app()
