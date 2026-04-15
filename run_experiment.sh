#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
SMOKE_TEST_ROWS="${SMOKE_TEST_ROWS:-2}"

RUN_ARGS=()
SKIP_PREFLIGHT=0

for arg in "$@"; do
  case "$arg" in
    --skip-preflight)
      SKIP_PREFLIGHT=1
      ;;
    *)
      RUN_ARGS+=("$arg")
      ;;
  esac
done

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
trap 'deactivate 2>/dev/null || true' EXIT

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
  echo "Running preflight check on ${SMOKE_TEST_ROWS} row(s) across requested case(s)..."
  python scripts/run_constitutional_ai_experiment.py \
    --start 0 \
    --limit "$SMOKE_TEST_ROWS" \
    --overwrite \
    --output-root preflight_runs \
    "${RUN_ARGS[@]}"
  echo "Preflight check passed. Starting full run..."
fi

python scripts/run_constitutional_ai_experiment.py "${RUN_ARGS[@]}"

deactivate 2>/dev/null || true
