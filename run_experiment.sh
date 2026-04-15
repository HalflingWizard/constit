#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
trap 'deactivate 2>/dev/null || true' EXIT

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/run_constitutional_ai_experiment.py "$@"

deactivate 2>/dev/null || true
