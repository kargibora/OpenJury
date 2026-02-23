## Branch Status (WIP)

This branch is a work in progress and intentionally diverges from the original codebase.
It introduces a more scalable, class-based architecture to make future integrations easier and improve readability/maintainability - which is required as a standalone open source project as well as to be used by OpenEuroLLM team.
Expect breaking changes and incomplete documentation while the refactor is in progress.
It will try to integrate any future changes to be done to the master branch. 

# OpenJury: LLM Evaluation with a Swappable Judge

OpenJury is a toolkit for running head-to-head and K-model LLM evaluations with configurable judge models, rubrics, and datasets.

## Installation

```bash
git clone https://github.com/OpenEuroLLM/OpenJury
cd OpenJury
uv sync
uv sync --extra vllm       # optional: local vLLM backend
uv sync --extra llamacpp   # optional: local llama.cpp backend
```

## Quick Start

### Run an arena from a config file (recommended)

```bash
uv run python -m openjury.steps.arena_step --config configs/arena/arena_2model.json
```

### Run a quick 2-model battle from CLI

```bash
uv run python -m openjury.steps.arena_step \
  --models VLLM/Qwen/Qwen2.5-0.5B-Instruct VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --dataset alpaca-eval \
  --n_instructions 20 \
  --rubric default \
  --output_dir results/arena/demo
```

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
- `overall` (a single rubric, can be used to simulate a single-score evaluation)

Use a built-in rubric:

```bash
uv run python -m openjury.steps.arena_step --config my_arena.json --rubric coding
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
      "weight": 1.0
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
uv run python -m openjury.steps.arena_step --config my_arena.json
```

Results are written to `output_dir` from the config (or `--output_dir`).

## Standalone Generation Step (Optional)

Generate completions first, then reference them in an arena config:

```bash
uv run python -m openjury.steps.generate_step \
  --model VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --dataset alpaca-eval \
  --output /tmp/qwen05.parquet \
  --n_instructions 100 \
  --tensor_parallel_size 1
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
