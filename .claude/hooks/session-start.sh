#!/usr/bin/env bash
# Claude Code on the web: install the project so `poetry run pytest` and
# `ta dev check` work in a fresh sandbox. A no-op for local sessions.
#
# The sandbox cannot reach huggingface.co; the test suite does not need it
# (tests/offline_assets.py serves the models and NLTK data from fixtures).
set -euo pipefail

[[ "${CLAUDE_CODE_REMOTE:-}" == "true" ]] || exit 0
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

POETRY_VERSION="2.3.1"  # keep in step with .github/workflows/ci.yml

if ! command -v poetry >/dev/null 2>&1; then
  python3 -m pip install --quiet --user "poetry==${POETRY_VERSION}"
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# pyproject requires Python 3.12 exactly; fall back to uv when the image's
# default interpreter is a different minor version.
if command -v python3.12 >/dev/null 2>&1; then
  poetry env use python3.12
else
  command -v uv >/dev/null 2>&1 || python3 -m pip install --quiet --user uv
  uv python install 3.12
  poetry env use "$(uv python find 3.12)"
fi

poetry install --with dev --no-interaction

if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
  echo "export PATH=\"${HOME}/.local/bin:\${PATH}\"" >>"${CLAUDE_ENV_FILE}"
fi
