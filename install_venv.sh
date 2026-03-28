#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v poetry >/dev/null 2>&1; then
  echo "Error: poetry is not installed or not in PATH." >&2
  echo "Install guide: https://python-poetry.org/docs/#installation" >&2
  exit 1
fi

echo "Configuring Poetry to create .venv inside project..."
poetry config virtualenvs.in-project true --local

echo "Installing dependencies into .venv..."
poetry install --no-interaction

VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: expected interpreter not found at $VENV_PY" >&2
  exit 1
fi

echo "Verifying PyQt6 import in .venv..."
"$VENV_PY" -c "import PyQt6; print('PyQt6 OK')"

echo "Done. Viewer runtime is ready at: $VENV_PY"
