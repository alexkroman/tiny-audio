"""Conventions every `ta` command follows, checked against the built Click tree.

The rules (also documented in README's "CLI conventions"):

* every option declares an explicit `--long-name` and a help string;
* a short flag (`-x`) means one thing across the whole CLI;
* a long flag has the same short alias (or none) everywhere it appears;
* pod-facing `ta runpod` commands take `HOST PORT` as their first positionals;
* analysis and debug commands take the model as their first positional.
"""

from collections import defaultdict

import pytest
import typer
import typer.main

from scripts.cli import app

# Options Click/Typer add on their own; not subject to the rules above.
_BUILTIN_OPTIONS = {"--help", "--install-completion", "--show-completion"}

# The one meaning each short flag has. New short flags must be added here so
# a reviewer sees the global namespace they are claiming.
RESERVED_SHORT_FLAGS = {
    "-a": "--audio",
    "-b": "--branch",
    "-c": "--config",
    "-d": "--datasets",
    "-e": "--experiment",
    "-f": "--force",
    "-k": "--top-k",
    "-l": "--list",
    "-m": "--model",
    "-n": "--max-samples",
    "-o": "--output-dir",
    "-p": "--port",
    "-r": "--repo-id",
    "-s": "--streaming",
    "-t": "--threshold",
    "-v": "--verbose",
    "-w": "--num-workers",
}


def _walk(command, path=()):
    """Yield (path, click.Command) for every leaf command under `command`."""
    ctx = typer.Context(command)
    if hasattr(command, "list_commands"):
        for name in command.list_commands(ctx):
            sub = command.get_command(ctx, name)
            if sub is not None:
                yield from _walk(sub, (*path, name))
    else:
        yield path, command


_LEAVES = list(_walk(typer.main.get_command(app)))
_LEAF_IDS = [" ".join(path) for path, _ in _LEAVES]


def _options(command):
    for param in command.params:
        if param.param_type_name == "option" and not set(param.opts) & _BUILTIN_OPTIONS:
            yield param


def _arguments(command):
    return [p for p in command.params if p.param_type_name == "argument"]


def test_every_subcommand_is_reachable():
    assert {path[0] for path, _ in _LEAVES} == {
        "eval",
        "analysis",
        "deploy",
        "push",
        "runpod",
        "debug",
        "demo",
        "dev",
    }


@pytest.mark.parametrize(("path", "command"), _LEAVES, ids=_LEAF_IDS)
def test_options_declare_long_name_and_help(path, command):
    for option in _options(command):
        longs = [o for o in option.opts if o.startswith("--")]
        assert longs, f"{' '.join(path)}: option {option.opts} has no --long-name"
        assert option.help, f"{' '.join(path)}: option {longs[0]} has no help text"


@pytest.mark.parametrize(("path", "command"), _LEAVES, ids=_LEAF_IDS)
def test_arguments_have_help(path, command):
    for argument in _arguments(command):
        assert argument.help, f"{' '.join(path)}: argument {argument.name} has no help text"


def test_short_flags_mean_one_thing():
    seen: dict[str, set[str]] = defaultdict(set)
    for path, command in _LEAVES:
        for option in _options(command):
            longs = [o for o in option.opts if o.startswith("--")]
            for short in (o for o in option.opts if not o.startswith("--")):
                where = " ".join(path)
                unknown = f"{where}: {short} is not in RESERVED_SHORT_FLAGS"
                assert short in RESERVED_SHORT_FLAGS, unknown
                wrong = f"{where}: {short} means {RESERVED_SHORT_FLAGS[short]}, not {longs}"
                assert RESERVED_SHORT_FLAGS[short] in longs, wrong
                seen[short].update(longs)
    assert all(len(longs) == 1 for longs in seen.values()), dict(seen)


def test_long_flags_keep_the_same_short_alias_everywhere():
    shorts_by_long: dict[str, set[frozenset[str]]] = defaultdict(set)
    for _path, command in _LEAVES:
        for option in _options(command):
            longs = [o for o in option.opts if o.startswith("--")]
            shorts = frozenset(o for o in option.opts if not o.startswith("--"))
            shorts_by_long[longs[0]].add(shorts)
    inconsistent = {k: v for k, v in shorts_by_long.items() if len(v) > 1}
    assert not inconsistent, inconsistent


def test_runpod_pod_commands_take_host_and_port_first():
    pod_commands = {"deploy", "train", "attach", "eval", "checkpoint"}
    for path, command in _LEAVES:
        if path[0] == "runpod" and path[1] in pod_commands:
            names = [a.name for a in _arguments(command)][:2]
            assert names == ["host", "port"], f"{' '.join(path)}: {names}"
            assert all(a.required for a in _arguments(command)[:2]), " ".join(path)


def test_analysis_and_debug_commands_take_the_model_positionally():
    for path, command in _LEAVES:
        if path[0] in {"analysis", "debug"}:
            where = " ".join(path)
            first = _arguments(command)[0].name
            assert first in {"model", "models"}, f"{where}: first argument is {first}"
            has_model_option = any("--model" in o.opts for o in _options(command))
            assert not has_model_option, f"{where}: --model duplicates the positional"
