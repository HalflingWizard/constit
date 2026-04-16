# Ollama Constitutional AI Experiment Bundle

This folder is a minimal portable bundle for running the three experiment conditions and writing per-row JSON results:

- `default_prompt`
- `prompt_embedded_constitution`
- `constitutional_ai_parallel`

It uses:

- the filtered CounselChat dataset in `counsel_chat_mental_health.csv`
- the constitution rules in `constitution.txt`
- the bundled Constitutional AI package in `constitutional-ai-kit/`
- Ollama for writer, judge, and critic

## Model defaults

By default, all model roles use:

- provider: `ollama`
- model: `gemma3:1b`
- API base: `http://localhost:11434`

You can override these with environment variables or CLI flags.

## What the runner produces

Running the bundle creates:

- `experiment_runs/default_prompt/row_*.json`
- `experiment_runs/prompt_embedded_constitution/row_*.json`
- `experiment_runs/constitutional_ai_parallel/row_*.json`
- `preflight_runs/...` JSONs for the smoke test
- `worker_probe_runs/...` JSONs for automatic parallel worker detection
- `ollama_runtime.env` and `ollama_runtime.json` with detected hardware and recommended Ollama server settings
- one `progress.json` file per case for resumability

If the run is interrupted, rerunning the same command resumes from the completed JSON files unless you pass `--overwrite`.

## Prerequisites on the Linux server

1. Install Python 3.10+.
2. Install and run Ollama.
3. Make sure the model is available locally:

```bash
ollama pull gemma3:1b
ollama list
```

4. Confirm the Ollama server is reachable:

```bash
curl http://localhost:11434/api/tags
```

5. Confirm the exact model name exists on that machine:

```bash
ollama list
ollama pull gemma3:1b
```

## Run

From this folder:

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

That will:

- create a local virtual environment in `.venv` if needed
- activate the virtual environment
- install or update the required Python dependencies
- detect CPU/GPU resources and write recommended `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, and `OLLAMA_MAX_QUEUE`
- auto-start `ollama serve` with the tuned environment if no Ollama server is already running
- restart an already running local `ollama serve` process by default so the tuned daemon settings take effect
- if the configured model is missing from the running Ollama server after restart/start, automatically run `ollama pull` for that model
- if `constitutional_ai_parallel` is requested and you did not explicitly set `--parallel-rule-workers`, probe the highest stable worker count automatically
- the worker probe uses a smaller constitution subset by default, with `rules_for_probe = 3 × workers`, capped by the full constitution size
- if Ollama later starts returning transient `500`/`EOF` errors during the full parallel run, the runner automatically retries the row and steps `parallel-rule-workers` down until the run stabilizes
- run a short preflight check on the first 2 rows for the requested cases
- if the preflight succeeds, run the requested experiment conditions on the main output folder
- deactivate the virtual environment before exiting

## Useful commands

Run only one condition:

```bash
./run_experiment.sh --cases constitutional_ai_parallel
```

Skip the preflight check:

```bash
./run_experiment.sh --skip-preflight
```

Change the number of rows used by the preflight check:

```bash
SMOKE_TEST_ROWS=1 ./run_experiment.sh
```

Override the automatic worker probe upper bound:

```bash
AUTO_WORKER_MAX=6 ./run_experiment.sh
```

Disable automatic Ollama server startup:

```bash
AUTO_START_OLLAMA=0 ./run_experiment.sh
```

Disable automatic restart of an already running Ollama server:

```bash
AUTO_RESTART_OLLAMA=0 ./run_experiment.sh
```

Disable automatic model pull if the running Ollama server cannot see the configured model:

```bash
AUTO_PULL_OLLAMA_MODEL=0 ./run_experiment.sh
```

Run a small test slice:

```bash
./run_experiment.sh --start 0 --limit 5
```

Rerun everything from scratch:

```bash
./run_experiment.sh --overwrite
```

Use a different Ollama base URL or model:

```bash
OLLAMA_API_BASE=http://127.0.0.1:11434 \
OLLAMA_MODEL=gemma3:1b \
./run_experiment.sh
```

Or via flags:

```bash
./run_experiment.sh \
  --writer-provider ollama \
  --writer-model gemma3:1b \
  --writer-api-base http://localhost:11434 \
  --judge-provider ollama \
  --judge-model gemma3:1b \
  --judge-api-base http://localhost:11434
```

## Experiment design in this bundle

The dataset query is the `questionText` column from `counsel_chat_mental_health.csv`.

The three conditions are:

1. `default_prompt`
   Uses the default writer system prompt with no constitution rules in the engine.
2. `prompt_embedded_constitution`
   Appends the constitution rules to the writer system prompt and uses no engine rules.
3. `constitutional_ai_parallel`
   Uses the default writer system prompt and passes the constitution rules into the Constitutional AI engine in parallel mode.

The parallel condition is fixed to:

- `parallel_max_iterations = 1`
- `parallel_max_workers` is chosen automatically for the device when the parallel case is requested, unless you explicitly pass `--parallel-rule-workers`
- the chosen worker count is treated as a starting point, not a guarantee; long runs can still hit Ollama instability on harder rows, so the runner may automatically step down to fewer workers while retrying failed rows

## Important note about Ollama daemon settings

`OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, and `OLLAMA_MAX_QUEUE` affect the Ollama server process, not just the experiment script.

- By default, `run_experiment.sh` restarts a running local `ollama serve` process so the tuned settings take effect.
- By default, if the restarted or newly started Ollama daemon does not expose the configured model, the wrapper runs `ollama pull <model>` automatically and then rechecks availability.
- If you set `AUTO_RESTART_OLLAMA=0`, the bundle will keep using the existing daemon without applying new daemon-level settings.
- If you set `AUTO_PULL_OLLAMA_MODEL=0`, the bundle will not attempt automatic recovery when the model is missing from the running daemon.
- If your Ollama instance is managed by `systemd`, Docker, or another supervisor, you may prefer disabling the automatic restart and managing the service yourself.

## Data notes

This bundle includes the already prepared filtered dataset from the main project. The preparation already happened before this bundle was created:

- questions were deduplicated
- excluded topics were removed in the main project pipeline

This bundle does not redo dataset preparation. It only runs inference over the included CSV.

## Constitution rule count

The runner uses every non-empty line in `constitution.txt` as one rule. If you want exactly 31 rules, edit `constitution.txt` before running or publishing the repo.
