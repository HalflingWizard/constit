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
- run all three experiment conditions
- deactivate the virtual environment before exiting

## Useful commands

Run only one condition:

```bash
./run_experiment.sh --cases constitutional_ai_parallel
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

## Data notes

This bundle includes the already prepared filtered dataset from the main project. The preparation already happened before this bundle was created:

- questions were deduplicated
- excluded topics were removed in the main project pipeline

This bundle does not redo dataset preparation. It only runs inference over the included CSV.

## Constitution rule count

The runner uses every non-empty line in `constitution.txt` as one rule. If you want exactly 31 rules, edit `constitution.txt` before running or publishing the repo.
