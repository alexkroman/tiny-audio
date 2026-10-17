"""The course's code examples, checked by docs/site/course_code.py."""

from pathlib import Path

import pytest

from docs.site import course_code


def _block(lang: str, code: str) -> course_code.CodeBlock:
    page = course_code.ROOT / "docs" / "course" / "example.md"
    return course_code.CodeBlock(page, 1, lang, code)


class TestCourseExamples:
    BLOCKS = course_code.course_blocks()

    def test_course_has_examples_of_every_checked_language(self) -> None:
        assert {"python", "yaml", "bash"} <= {block.lang for block in self.BLOCKS}

    @pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.where)
    def test_block_compiles(self, block: course_code.CodeBlock) -> None:
        assert course_code.check_block(block) == []


class TestExtractBlocks:
    def test_strips_list_indent_and_blockquote_markers(self, tmp_path: Path) -> None:
        page = tmp_path / "page.md"
        page.write_text(
            "Text\n\n```text\ndiagram\n```\n\n"
            "1. Step\n\n   ```bash\n   echo one\n   ```\n\n"
            "> ```python\n> x = 1\n> ```\n"
        )
        blocks = course_code.extract_blocks(page)
        assert [(b.lang, b.code, b.line) for b in blocks] == [
            ("text", "diagram\n", 3),
            ("bash", "echo one\n", 9),
            ("python", "x = 1\n", 13),
        ]


class TestShellCommands:
    def test_joins_continuations_fills_placeholders_and_drops_comments(self) -> None:
        code = "# note\nta runpod attach <HOST> <PORT> \\\n    --logs  # recent\n"
        assert course_code.shell_commands(code) == [
            ["ta", "runpod", "attach", "HOST", "22", "--logs"]
        ]


class TestCheckBlock:
    @pytest.mark.parametrize(
        ("lang", "code", "problem"),
        [
            ("python", "print(1\n", "Python does not compile"),
            ("yaml", "a: [1\n", "YAML does not parse"),
            ("bash", "if true; then\n", "bash syntax error"),
            ("bash", "poetry run python scripts/missing.py\n", "no such script"),
            (
                "bash",
                "poetry run python scripts/train.py +experiments=nope\n",
                "no such experiment",
            ),
            ("bash", "poetry run ta runpod plann\n", "no such command 'plann'"),
            ("bash", "poetry run ta eval --bogus\n", "No such option: --bogus"),
            ("bash", "poetry run ta runpod deploy host notaport\n", "is not a valid int"),
        ],
    )
    def test_reports_broken_example(self, lang: str, code: str, problem: str) -> None:
        (error,) = course_code.check_block(_block(lang, code))
        assert problem in error
        assert "example.md:1" in error

    def test_course_created_experiment_is_allowed(self) -> None:
        code = "poetry run python scripts/train.py +experiments=my_run\n"
        assert course_code.check_block(_block("bash", code)) == []

    def test_help_passes(self) -> None:
        code = "poetry run ta --help\npoetry run ta runpod --help\npoetry run ta eval --help\n"
        assert course_code.check_block(_block("bash", code)) == []

    def test_syntax_only_skips_the_cli(self) -> None:
        block = _block("bash", "poetry run ta eval --bogus\n")
        assert course_code.check_block(block, syntax_only=True) == []

    def test_untagged_languages_are_skipped(self) -> None:
        assert course_code.check_block(_block("text", "Audio → (")) == []
