"""scripts/mutate.py: mutant enumeration, application, and the run loop."""

from collections.abc import Callable
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from scripts import mutate


def always(result: bool) -> Callable[[list[Path]], bool]:
    """A stand-in for `mutate.tests_pass` with a fixed verdict."""

    def fake(_tests: list[Path]) -> bool:
        return result

    return fake


SOURCE = """\
def f(a, b):
    if a < b and not a:
        return a + 1
    return True


if __name__ == "__main__":
    f(1 == 2, 3)
"""


class TestMutants:
    """Which mutants a source admits, and applying one at a time."""

    def test_enumerates_every_site_outside_main(self) -> None:
        descriptions = sorted(m.description for m in mutate.mutants(SOURCE))
        assert descriptions == sorted(
            ["Lt -> GtE", "And -> Or", "drop `not`", "Add -> Sub", "1 -> 2", "True -> False"]
        )

    def test_each_mutant_changes_exactly_one_thing(self) -> None:
        found = mutate.mutants(SOURCE)
        sources = {mutate.mutate(SOURCE, m.index) for m in found}
        assert len(sources) == len(found)
        assert any("a >= b" in source for source in sources)

    def test_unknown_index_raises(self) -> None:
        with pytest.raises(IndexError):
            mutate.mutate(SOURCE, 99)


class TestRun:
    """The run loop records survivors and always restores the target file."""

    def test_restores_the_file_and_reports_survivors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "m.py"
        target.write_text(SOURCE)
        seen: list[str] = []

        def fake_tests_pass(_tests: list[Path]) -> bool:
            seen.append(target.read_text())
            return "a - 1" in seen[-1]  # only the Add -> Sub mutant survives

        monkeypatch.setattr(mutate, "tests_pass", fake_tests_pass)
        (survivor,) = mutate.run(target, [], None)
        assert survivor.description == "Add -> Sub"
        assert target.read_text() == SOURCE
        assert len(seen) == len(mutate.mutants(SOURCE))

    def test_max_limits_the_run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        target = tmp_path / "m.py"
        target.write_text(SOURCE)
        monkeypatch.setattr(mutate, "tests_pass", always(False))
        assert mutate.run(target, [], 2) == []
        assert target.read_text() == SOURCE

    def test_restores_the_file_on_interrupt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "m.py"
        target.write_text(SOURCE)

        def interrupted(_tests: list[Path]) -> bool:
            raise KeyboardInterrupt

        monkeypatch.setattr(mutate, "tests_pass", interrupted)
        with pytest.raises(KeyboardInterrupt):
            mutate.run(target, [], None)
        assert target.read_text() == SOURCE


class TestCommand:
    """`ta dev mutate`: refuses a failing baseline, fails on survivors."""

    runner = CliRunner()

    def app(self) -> typer.Typer:
        """A one-command Typer app around `mutate.main`."""
        app = typer.Typer()
        app.command()(mutate.main)
        return app

    def test_failing_baseline_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mutate, "tests_pass", always(False))
        result = self.runner.invoke(
            self.app(), ["scripts/mutate.py", "--tests", "tests/test_mutate.py"]
        )
        assert result.exit_code == 2
        assert "fail before any mutation" in result.output

    def test_survivors_fail_the_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mutate, "tests_pass", always(True))

        def run(*_args: object) -> list[mutate.Mutant]:
            return [mutate.Mutant(0, 3, "Add -> Sub")]

        monkeypatch.setattr(mutate, "run", run)
        result = self.runner.invoke(self.app(), ["scripts/mutate.py"])
        assert result.exit_code == 1
        assert "scripts/mutate.py:3: survived: Add -> Sub" in result.output

    def test_default_tests_follow_the_module_name(self) -> None:
        assert mutate.default_tests(Path("scripts/mutate.py")) == [
            mutate.ROOT / "tests" / "test_mutate.py"
        ]

    def test_tests_pass_runs_pytest(self) -> None:
        passing = mutate.ROOT / "tests" / "test_scripts_utils.py"
        assert mutate.tests_pass([passing]) is True
