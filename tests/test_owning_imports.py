"""scripts/owning_imports.py: repo names are reached through the module that defines them."""

import pytest
from typer.testing import CliRunner

from scripts import owning_imports, quality

# tiny_audio.asr_pipeline defines ASRPipeline and only imports PreparedChunk,
# collate_chunks and `re`; scripts.dev only imports subprocess.
# Built at runtime: as one literal, the gate would flag this file itself.
DEV_SUBPROCESS = "scripts.dev.subprocess"
DEFINED_HERE = "from tiny_audio.asr_pipeline import ASRPipeline\n"


class TestBindings:
    """Which top-level names a module defines and which it only imports."""

    def test_imports_in_type_checking_and_fallback_blocks_count(self) -> None:
        source = (
            "import os\n"
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from a import B\n"
            "try:\n    from .c import D\nexcept ImportError:\n    from c import D\n"
            "def f(): ...\n"
            "class K: ...\n"
            "x: int = 1\n"
        )
        defined, imported = owning_imports.bindings(source)
        assert defined == {"f", "K", "x"}
        assert imported == {"os", "TYPE_CHECKING", "B", "D"}

    def test_a_reassigned_import_counts_as_defined(self) -> None:
        defined, imported = owning_imports.bindings("from a import b\nb = wrap(b)\n")
        assert "b" in defined
        assert "b" not in imported


class TestViolations:
    """Each shape of reaching a name through a module that only imports it."""

    def test_names_from_their_owner_pass(self) -> None:
        source = DEFINED_HERE + "from tiny_audio.asr_types import PreparedChunk\n"
        assert owning_imports.violations(source, "t.py") == []

    def test_from_import_through_a_non_owner_fails(self) -> None:
        source = "from tiny_audio.asr_pipeline import ASRPipeline, PreparedChunk\n"
        (problem,) = owning_imports.violations(source, "t.py")
        assert problem.startswith("t.py:1:")
        assert "PreparedChunk" in problem

    def test_package_facades_are_not_exempt(self) -> None:
        source = "from scripts.eval.evaluators import Evaluator\n"
        assert owning_imports.violations(source, "t.py")

    def test_attribute_read_through_a_module_fails(self) -> None:
        source = "from scripts import dev\nmonkeypatch.setattr(dev.subprocess, 'run', f)\n"
        (problem,) = owning_imports.violations(source, "t.py")
        assert "`dev.subprocess`" in problem

    def test_assigning_a_module_attribute_is_allowed(self) -> None:
        assert (
            owning_imports.violations("from scripts import dev\ndev.subprocess = None\n", "t") == []
        )

    @pytest.mark.parametrize(
        ("target", "fails"),
        [
            (f"{DEV_SUBPROCESS}.call", True),  # through dev's import: patch subprocess
            ("tiny_audio.asr_pipeline.re", False),  # rebinding the consumer's own name
            ("tiny_audio.asr_attention._has_sliding_window_attention", False),  # defined there
            ("scripts.dev", False),
            ("scripts are dev.subprocess.call", False),  # prose, not a dotted path
        ],
    )
    def test_dotted_strings(self, target: str, fails: bool) -> None:
        source = f"monkeypatch.setattr({target!r}, f)\n"
        assert bool(owning_imports.violations(source, "t.py")) is fails


def test_module_name() -> None:
    assert owning_imports.module_name("scripts/eval/cli.py") == "scripts.eval.cli"
    assert owning_imports.module_name("scripts/eval/__init__.py") == "scripts.eval"


def test_the_repository_passes() -> None:
    result = CliRunner().invoke(quality.app, ["owning-imports"])
    assert result.exit_code == 0, result.output


def test_a_violation_fails_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_bad(_source: str, where: str) -> list[str]:
        return [f"{where}: bad"]

    monkeypatch.setattr(owning_imports, "violations", always_bad)
    result = CliRunner().invoke(quality.app, ["owning-imports"])
    assert result.exit_code == 1
    assert ": bad" in result.output
