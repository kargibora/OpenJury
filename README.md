## Branch Status (WIP)

This branch is a work in progress and intentionally diverges from the original codebase.
It introduces a more scalable, class-based architecture to make future integrations easier and improve readability and maintainability for both standalone use and OpenEuroLLM integration.
Expect breaking changes and incomplete documentation while the refactor is in progress.
The branch is intended to track and incorporate future relevant changes from `master`.

# OpenJury: LLM Evaluation with a Swappable Judge

OpenJury is a toolkit for running head-to-head and K-model LLM evaluations with configurable judge models, rubrics, and datasets.

## Installation

```bash
git clone https://github.com/OpenEuroLLM/OpenJury
cd OpenJury
uv sync
uv sync --extra vllm       # optional: local vLLM backend
uv sync --extra llamacpp   # optional: local llama.cpp backend
uv sync --extra yaml       # optional: YAML arena config files
```

## Environment Setup

For API-based models/judges, set the relevant credentials before running the pipeline.

Examples:

```bash
export OPENROUTER_API_KEY=...
export OPENAI_API_KEY=...
```

## Quick Start

### Run an arena from a config file (recommended)

```bash
uv run python -m openjury.cli.arena --config configs/arena/arena_2model.json
```

### Run a quick 2-model battle from CLI

```bash
uv run python -m openjury.cli.arena \
  --models VLLM/Qwen/Qwen2.5-0.5B-Instruct VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --dataset alpaca-eval \
  --n_instructions 20 \
  --rubric default \
  --output_dir results/arena/demo
```

This command:
- generates completions for the candidate models (if not already provided in config)
- judges them with the specified local/API judge model
- computes arena ratings
- writes results to `output_dir`

## Arena Config (JSON/YAML)

Minimal config example:

```json
{
  "dataset": "alpaca-eval",
  "n_instructions": 100,
  "models": [
    "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
    "VLLM/Qwen/Qwen2.5-1.5B-Instruct"
  ],
  "judge": {
    "model": "VLLM/Qwen/Qwen3-32B",
    "gpus": 1,
    "mode": "samplewise",
    "max_tokens": 2048,
    "temperature": 0.0
  },
  "rubric": "default",
  "matchmaker": {
    "strategy": "round_robin"
  },
  "generation": {
    "max_tokens": 4096,
    "truncate_input_chars": 8192
  },
  "ratings": {
    "bt_regularization": 0.01,
    "elo_k": 32.0
  },
  "output_dir": "results/arena/demo"
}
```

## Pipeline Usage Patterns

### 1) End-to-end arena (generate + judge in one command)

Use `openjury.cli.arena` when you want OpenJury to generate completions and run evaluation in one pipeline.

Candidate models and judge models can be mixed across local and API backends (for example, local `VLLM/...` candidates with an API judge such as `OpenRouter/...`).

### 2) Staged workflow (generate first, evaluate later)

Use `openjury.cli.generate` to create completions separately, then reference the parquet file in an arena config via `completions`.

This is useful for:
- running generation and judging on different machines
- reusing a fixed completion set across multiple judges/rubrics
- SLURM/job-based workflows

## SLURM Script Helper (Optional)

Use `openjury-slurm` to generate job scripts for staged execution on SLURM clusters.
It supports mixed execution targets (GPU jobs via `sbatch`, API models on the login node).

Quick smoke test:

```bash
uv run openjury-slurm --help
```

Minimal dry-run example (generate scripts only, do not submit):

```bash
uv run openjury-slurm \
  --dataset alpaca-eval \
  --models VLLM/Qwen/Qwen2.5-0.5B-Instruct VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --mode arena \
  --output_dir slurm_scripts
```

This writes a per-run folder with:
- generated job scripts (`*.sh`)
- `arena_config.json` for the arena step
- `submit_all.sh` orchestration script

Submit later:

```bash
bash slurm_scripts/<run_name>/submit_all.sh
```

Notes:
- `agreement` mode is temporarily disabled in this branch until a dedicated CLI entrypoint is added.
- Cluster settings (`ACCOUNT`, `PARTITION`, `USER_WORK_DIR`, `TIME_LIMIT`) are read from environment variables or `.env`.

### Per-model entries (optional overrides)

A model entry can be a string or an object.

```json
{
  "name": "VLLM/meta-llama/Llama-3.1-8B-Instruct",
  "gpus": 2,
  "max_tokens": 4096
}
```

You can also point to a pre-generated completions parquet file:

```json
{
  "name": "my-model",
  "completions": "/path/to/completions.parquet"
}
```

## Judge Modes

- `samplewise` (default): scores each model independently on the rubric
- `pairwise`: compares completions side-by-side. 

Set in config:

```json
"judge": {
  "model": "VLLM/Qwen/Qwen3-32B",
  "mode": "pairwise"
}
```

Or via CLI:

```bash
--judge_mode pairwise
```

## Judge Model Configuration (Local or API)

### API judge example (OpenRouter)

```json
"judge": {
  "model": "OpenRouter/deepseek/deepseek-chat-v3.1",
  "mode": "samplewise",
  "max_tokens": 2048,
  "temperature": 0.0
}
```

### Local judge example (vLLM)

```json
"judge": {
  "model": "VLLM/Qwen/Qwen3-32B",
  "gpus": 1,
  "tp": 1,
  "mode": "samplewise",
  "max_tokens": 2048,
  "temperature": 0.0
}
```

## Model Specification

Use the format:

```text
Provider/model-name-or-path
```

Supported provider prefixes:
- `VLLM/...`
- `ChatOpenAI/...`
- `OpenRouter/...`
- `LiteLLM/...`
- `LlamaCpp/...`
- `Dummy/...`

Examples:

```bash
VLLM/Qwen/Qwen2.5-7B-Instruct
ChatOpenAI/gpt-4o-mini
OpenRouter/deepseek/deepseek-chat-v3.1
LiteLLM/anthropic/claude-3-5-sonnet
LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf
```

## Rubrics

Built-in rubrics:
- `default`
- `coding`
- `translation`
- `overall` (single-dimension rubric for single-score evaluation)

There is no separate legacy "single-score without rubric" mode in the current pipeline.
Use `overall` when you want a single scalar quality score per completion.

Use a built-in rubric:

```bash
uv run python -m openjury.cli.arena --config my_arena.json --rubric coding
```

Use a custom rubric by setting `rubric` to a JSON file path in the config.

Custom rubric JSON shape:

```json
{
  "name": "custom",
  "description": "My rubric",
  "dimensions": [
    {
      "name": "helpfulness",
      "description": "How useful is the answer?",
      "scale_min": 1,
      "scale_max": 10,
      "weight": 1.0,
      "score_references": {
        "10": "Excellent performance on this dimension.",
        "7": "Good performance with minor issues.",
        "4": "Weak performance with clear deficiencies.",
        "1": "Very poor performance on this dimension."
      }
    }
  ]
}
```

## Outputs

The arena pipeline writes:
- `arena.json` (matches, rubric scores, ratings, metadata)
- `arena_config.json` (config snapshot)

Typical run command:

```bash
uv run python -m openjury.cli.arena --config my_arena.json
```

Results are written to `output_dir` from the config (or `--output_dir`).

## Standalone Generation Step (Optional)

Generate completions first, then reference them in an arena config:

```bash
uv run python -m openjury.cli.generate \
  --model VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --dataset alpaca-eval \
  --output /tmp/qwen05.parquet \
  --n_instructions 100 \
  --tensor_parallel_size 1
```

### Generate completions with an API model

```bash
uv run python -m openjury.cli.generate \
  --model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --dataset alpaca-eval \
  --output /tmp/deepseek.parquet \
  --n_instructions 100
```

### vLLM chat template override (optional)

For models without a tokenizer-defined chat template (or when you want to override it), you can provide a template string or file.

CLI (generation):

```bash
uv run python -m openjury.cli.generate \
  --model VLLM/my/base-model \
  --dataset alpaca-eval \
  --output /tmp/base.parquet \
  --chat_template_file ./templates/chatml.jinja
```

Arena config (judge or model entry):

```json
"judge": {
  "model": "VLLM/my/base-model",
  "mode": "samplewise",
  "chat_template_file": "./templates/chatml.jinja"
}
```

Then use:

```json
{
  "name": "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
  "completions": "/tmp/qwen05.parquet"
}
```

## Supported Datasets

| Dataset | Description |
|---|---|
| `alpaca-eval` | General instruction-following benchmark |
| `arena-hard` | More challenging evaluation suite |
| `m-arena-hard` | Multilingual Arena-Hard benchmark |
| `m-arena-hard-{lang}` | Language-specific variants |
| `m-arena-hard-EU` | Combined EU-language subset |
| `fluency-{lang}` | Fluency evaluation for pretrained/base models |
| `lmsys` / `comparia` | Human-preference style datasets |
