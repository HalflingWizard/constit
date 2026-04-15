#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
SMOKE_TEST_ROWS="${SMOKE_TEST_ROWS:-2}"
AUTO_WORKER_MAX="${AUTO_WORKER_MAX:-}"
AUTO_START_OLLAMA="${AUTO_START_OLLAMA:-1}"
AUTO_RESTART_OLLAMA="${AUTO_RESTART_OLLAMA:-1}"
OLLAMA_RUNTIME_ENV_FILE="${OLLAMA_RUNTIME_ENV_FILE:-ollama_runtime.env}"

RUN_ARGS=()
SKIP_PREFLIGHT=0
HAS_EXPLICIT_PARALLEL_WORKERS=0
CASES_SPECIFIED=0
REQUESTS_PARALLEL=1
EXPECT_CASE_VALUES=0

for arg in "$@"; do
  if [[ "$EXPECT_CASE_VALUES" -eq 1 ]]; then
    if [[ "$arg" == --* ]]; then
      EXPECT_CASE_VALUES=0
    else
      if [[ "$arg" == "constitutional_ai_parallel" ]]; then
        REQUESTS_PARALLEL=1
      fi
      RUN_ARGS+=("$arg")
      continue
    fi
  fi

  case "$arg" in
    --skip-preflight)
      SKIP_PREFLIGHT=1
      ;;
    --parallel-rule-workers|--parallel-rule-workers=*)
      HAS_EXPLICIT_PARALLEL_WORKERS=1
      RUN_ARGS+=("$arg")
      ;;
    --cases)
      CASES_SPECIFIED=1
      REQUESTS_PARALLEL=0
      EXPECT_CASE_VALUES=1
      RUN_ARGS+=("$arg")
      ;;
    *)
      RUN_ARGS+=("$arg")
      ;;
  esac
done

if [[ "$EXPECT_CASE_VALUES" -eq 1 ]]; then
  echo "Missing values after --cases" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
trap 'deactivate 2>/dev/null || true' EXIT

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Detecting hardware and generating Ollama runtime recommendations..."
if [[ -n "$AUTO_WORKER_MAX" ]]; then
  python scripts/tune_ollama_runtime.py --output-env "$OLLAMA_RUNTIME_ENV_FILE" --max-workers "$AUTO_WORKER_MAX"
else
  python scripts/tune_ollama_runtime.py --output-env "$OLLAMA_RUNTIME_ENV_FILE"
fi

if [[ -f "$OLLAMA_RUNTIME_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$OLLAMA_RUNTIME_ENV_FILE"
fi

restart_ollama_server() {
  local matched_pids=""
  matched_pids="$(pgrep -f "ollama serve" || true)"
  if [[ -n "$matched_pids" ]]; then
    echo "Restarting existing ollama serve process..."
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      kill "$pid" 2>/dev/null || true
    done <<< "$matched_pids"
    for _ in $(seq 1 15); do
      sleep 1
      if ! curl -fsS "${OLLAMA_API_BASE:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
        break
      fi
    done
  fi

  echo "Starting tuned ollama serve..."
  nohup ollama serve > ollama_serve.log 2>&1 &
  for _ in $(seq 1 30); do
    sleep 1
    if curl -fsS "${OLLAMA_API_BASE:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
      echo "Ollama server is ready."
      return 0
    fi
  done
  echo "Failed to start Ollama server. Check ollama_serve.log." >&2
  return 1
}

if ! curl -fsS "${OLLAMA_API_BASE:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
  if [[ "$AUTO_START_OLLAMA" -eq 1 ]] && command -v ollama >/dev/null 2>&1; then
    echo "Ollama server not reachable. Starting a local tuned ollama serve..."
    restart_ollama_server || exit 1
  else
    echo "Ollama server not reachable at ${OLLAMA_API_BASE:-http://localhost:11434}." >&2
    echo "Start it manually or leave AUTO_START_OLLAMA=1 with ollama installed." >&2
    exit 1
  fi
else
  if [[ "$AUTO_RESTART_OLLAMA" -eq 1 ]] && command -v ollama >/dev/null 2>&1; then
    restart_ollama_server || exit 1
  else
    echo "Ollama server already running. Tuned env was written to ${OLLAMA_RUNTIME_ENV_FILE}, but daemon-level changes only apply after restart."
  fi
fi

if [[ "$REQUESTS_PARALLEL" -eq 1 && "$HAS_EXPLICIT_PARALLEL_WORKERS" -eq 0 ]]; then
  echo "Auto-detecting a stable worker count for constitutional_ai_parallel..."
  if [[ -n "$AUTO_WORKER_MAX" ]]; then
    DETECT_OUTPUT="$(python scripts/detect_parallel_workers.py --max-workers "$AUTO_WORKER_MAX" "${RUN_ARGS[@]}")"
  else
    DETECT_OUTPUT="$(python scripts/detect_parallel_workers.py "${RUN_ARGS[@]}")"
  fi
  DETECTED_WORKERS="$(printf '%s\n' "$DETECT_OUTPUT" | tail -n 1)"
  if [[ ! "$DETECTED_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "Worker auto-detection failed. Output was:" >&2
    printf '%s\n' "$DETECT_OUTPUT" >&2
    exit 1
  fi
  echo "Using parallel-rule-workers=${DETECTED_WORKERS}"
  RUN_ARGS+=(--parallel-rule-workers "$DETECTED_WORKERS")
fi

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
