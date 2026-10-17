# CLAUDE.md

Tiny Audio is a speech-recognition model (audio encoder + projector + language-model decoder) with
its training, evaluation, serving and deployment tooling. `CONTRIBUTING.md` is the reference for the
CLI, configuration and project layout; this file is what an agent needs before changing code.

## Commands

```bash
poetry run ta dev format     # rewrite: black, ruff, mdformat, taplo, pretty-format-json
poetry run ta dev check      # every read-only gate; reports all failures together
poetry run ta dev test       # pytest + coverage floor + per-file coverage floors
poetry run ta dev precommit  # format, check, test, build: what the git hook runs
poetry run pytest tests/test_projectors.py -v
```

Run `ta dev check` and `ta dev test` before every push. The suite runs offline from fixtures
(`tests/offline_assets.py`); it never needs Hugging Face Hub access.

## Quality gates

Every threshold is a ratchet. Raise it when the code clears the next rung; never lower one, and
never edit a baseline under `quality/` by hand to get a red build green. Each `--update` flag only
records an improvement:

- `ta dev quality file-length`: 600 code lines per Python file.
- `ta dev quality coverage-floors`: per-file line coverage (runs after the tests).
- `ta dev quality test-assertions`: every test asserts something.
- `ta dev quality escape-hatches`: per-file budgets for `# noqa`, `# pyright: ignore`,
  `# type: ignore`, `# nosec` and `# pragma: no cover`. Fix the finding instead.
- `ta dev quality module-tests`: every module under `tiny_audio/` and `scripts/` is imported or
  named by a test of its own.
- `ta dev quality type-completeness`: share of `tiny_audio`'s and `scripts`' public API with known
  types.
- `ta dev quality invariants`: restated facts agree and settled decisions hold
  (`scripts/invariants.py`; the backticked paths in this file must exist).
- `ta dev quality owning-imports`: import every repo name from the module that defines it, and patch
  `subprocess.run`, not `scripts.dev.subprocess.run` (`scripts/owning_imports.py`).

`tiny_audio/`, `scripts/` and `tests/` go through every gate alike, including 95% docstring coverage
per tree (nested and `test_*` functions exempt).

## Settled decisions

Load the `project-guardrails` skill (`.claude/skills/project-guardrails/SKILL.md`) before changing
dependencies, training configs or the model code. Several of its rules are enforced by
`scripts/invariants.py`; if a task seems to need breaking one, stop and ask.

## Conventions

- Python 3.12, pyright strict, ruff with a broad rule set, line length 100.
- Comments explain why, often with the measurement or incident behind a choice. Keep them when
  editing nearby code, and update them when the fact they record changes.
- Each module has its own test file; tests may call private helpers directly.
- CLI commands follow the conventions in `CONTRIBUTING.md` and `tests/test_cli_conventions.py`.
