## Branch Status (WIP)

This branch is a work in progress and intentionally diverges from the original codebase.
It introduces a more scalable, class-based architecture to make future integrations easier and improve readability and maintainability for both standalone use and OpenEuroLLM integration.
Expect breaking changes and incomplete documentation while the refactor is in progress.
The branch is intended to track and incorporate future relevant changes from `master`.

# OpenJury: LLM Evaluation with a Swappable Judge

OpenJury is a toolkit for running head-to-head and K-model LLM evaluations with configurable judge models, criteria, and datasets.

Primary command model:
- `openjury-generate` for completion generation
- `openjury-evaluate annotate` for reusable judge-annotation artifacts
- `openjury-evaluate arena` as a preset/orchestration workflow over `annotate(matchmaker)` plus rating analysis
- `openjury-evaluate agreement` as a preset/orchestration workflow over `annotate(dataset_pairs)` plus human-agreement analysis

Additional command aliases:
- `openjury-arena` (alias of `evaluate arena`)
- `openjury-agreement` (alias of `evaluate agreement`)
- `openjury-annotate` (alias of `evaluate annotate`)

## Installation

Prerequisites:
- Python 3.12
- `uv`
- Linux for local GPU backends such as `vllm` and `sglang`

Notes:
- `uv sync` installs the base project and API/backoffice dependencies.
- Local GPU backends are optional extras.
- In this branch, the `vllm` extra is intended for Linux and Python `<3.14`.

```bash
git clone https://github.com/OpenEuroLLM/OpenJury
cd OpenJury
uv sync
uv sync --extra vllm       # optional: local vLLM backend
uv sync --extra sglang     # optional: local SGLang backend
uv sync --extra llamacpp   # optional: local llama.cpp backend
uv sync --extra yaml       # optional: YAML arena config files
```

Quick smoke test after install:

```bash
uv run openjury-evaluate --help
uv run openjury-generate --help
uv run openjury-annotate --help
```

### SGLang Quick Start

Install the optional dependency:

```bash
uv sync --extra sglang
```

Then use the `SGLang/...` provider prefix like any other local backend:

```bash
uv run openjury-generate \
  --model SGLang/Qwen/Qwen3-8B \
  --dataset alpaca-eval \
  --n_instructions 20 \
  --output /tmp/qwen3_sglang_smoke.parquet
```

Example agreement run with a local SGLang judge:

```bash
uv run openjury-evaluate agreement \
  --dataset lmsys \
  --judge_model SGLang/Qwen/Qwen3-8B \
  --judge_gpus 1 \
  --judge_mode samplewise \
  --criteria default \
  --n_instructions 50 \
  --output_dir results/agreement/sglang_smoke
```

Notes:
- `SGLang/...` is treated as a local GPU backend, like `VLLM/...`.
- `--tensor_parallel_size`, `--gpu_devices`, `--quantization`, and chat-template overrides are supported.
- If you prefer running an external SGLang server with an OpenAI-compatible endpoint, you can also use the existing `ChatOpenAI/...` or `LiteLLM/...` backends with `--api_base_url`.

## Environment Setup

For API-based models/judges, set the relevant credentials before running the pipeline.

Examples:

```bash
export OPENROUTER_API_KEY=...
export OPENAI_API_KEY=...
```

If you use a local virtual environment outside the default `.venv`, set:

```bash
export OPENJURY_VENV=/path/to/venv
```

Generated SLURM scripts use `OPENJURY_VENV` if present and otherwise fall back to `.venv`.

For containerized `VLLM/...` SLURM jobs, set:

```bash
export OPENJURY_CONTAINER_RUNTIME=apptainer
export OPENJURY_CONTAINER_IMAGE=/path/to/vllm-openai.sif
export OPENJURY_CONTAINER_HOME=/path/to/openjury_container_home
```

If `OPENJURY_CONTAINER_HOME` is not set, OpenJury falls back to
`$HF_HOME/openjury_container_home`.

Then prepare the persistent container home once on the login node:

```bash
openjury-container-prepare
```

For a fuller cluster-oriented walkthrough, see [docs/apptainer.md](docs/apptainer.md).

Pre-cache models and datasets into `HF_HOME` on the login node, then keep using the normal
`--slurm` workflow. If needed, you can override the `.env` defaults per run with
`--slurm_container_runtime`, `--slurm_container_image`, and `--slurm_container_home`.

## Execution Modes

OpenJury supports:
- local execution (e.g. `VLLM/...`, `SGLang/...`, `LlamaCpp/...`)
- API execution (e.g. `OpenRouter/...`, `ChatOpenAI/...`)
- SLURM script generation / submission via `--slurm` or `openjury-slurm`

Examples:

```bash
# Local arena (local candidates + local judge)
uv run openjury-evaluate arena --config configs/arena/arena_2model.json

# API-based agreement (login node / no SLURM required)
uv run openjury-evaluate agreement \
  --dataset lmsys \
  --judge_model OpenRouter/qwen/qwen3-32b \
  --n_instructions 100

# SLURM arena script generation
uv run openjury-evaluate arena --config configs/arena/arena_2model.json --slurm

# SLURM agreement script generation
uv run openjury-evaluate agreement \
  --dataset lmsys \
  --judge_model OpenRouter/qwen/qwen3-32b \
  --n_instructions 100 \
  --slurm
```

## Generic Annotation

`openjury-evaluate annotate` is the generic task for:

1. generating or loading unified challenger completions
2. aligning them by `instruction_id`
3. judging them against opponent completions on the same instructions

This task saves reusable judge annotations. It does not assume you want a specific
analysis mode such as human-agreement metrics.

Current scope:
- challenger completion generation is generic and uses the same unified machinery as
  `openjury-generate`
- annotation pairs can currently come from:
  - `challenger_vs_dataset_inline`
  - `dataset_pairs`
  - `matchmaker`
- this works well for datasets such as `lmsys`, `lmsys-140k`, and `comparia`

Example config:
- [configs/annotate/lmsys_gpt-oss20b_qwen3-32b.yaml](configs/annotate/lmsys_gpt-oss20b_qwen3-32b.yaml)
- [configs/annotate/lmsys_dataset_pairs_qwen3-32b.yaml](configs/annotate/lmsys_dataset_pairs_qwen3-32b.yaml)

Example run:

```bash
uv run openjury-evaluate annotate \
  --config configs/annotate/lmsys_gpt-oss20b_qwen3-32b.yaml
```

Equivalent CLI form:

```bash
uv run openjury-evaluate annotate \
  --dataset lmsys \
  --challenger_model VLLM/openai/gpt-oss-20b \
  --challenger_gpus 4 \
  --judge_model VLLM/Qwen/Qwen3-32B \
  --judge_gpus 4 \
  --judge_mode pairwise \
  --criteria default \
  --n_instructions 500 \
  --balance_by models \
  --pairing_source challenger_vs_dataset_inline \
  --pairing_strategy random_single \
  --max_model_len 16384 \
  --enforce_eager \
  --output_dir results/annotate/lmsys_gpt-oss20b_qwen3-32b
```

Direct matchmaker example:

```bash
uv run openjury-evaluate annotate \
  --dataset alpaca-eval \
  --pairing_source matchmaker \
  --pairing_strategy balanced_random \
  --n_matches 300 \
  --models VLLM/model-a VLLM/model-b VLLM/model-c \
  --judge_model VLLM/Qwen/Qwen3-32B \
  --judge_gpus 4 \
  --judge_mode pairwise \
  --criteria default \
  --output_dir results/annotate/alpacaeval_matchmaker
```

Outputs:
- `annotate_annotations.json`
- `annotate_config.json`

For a fuller explanation of the flow and artifact shape, see
[ANNOTATE_PIPELINE.md](ANNOTATE_PIPELINE.md).

## Quick Start

### Run an arena from a config file (recommended)

```bash
uv run openjury-evaluate arena --config configs/arena/arena_2model.json
```

### Run a quick 2-model battle from CLI

```bash
uv run openjury-evaluate arena \
  --models VLLM/Qwen/Qwen2.5-0.5B-Instruct VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --dataset alpaca-eval \
  --n_instructions 20 \
  --criteria default \
  --output_dir results/arena/demo
```

This command:
- generates completions for the candidate models (if not already provided in config)
- judges them with the specified local/API judge model
- computes arena ratings
- writes results to `output_dir`

To split evaluation into reusable stages, add `--stage annotate` or `--stage analyze`.

## Pipeline Overview

OpenJury is organized around two primary artifact-producing steps:

1. `openjury-generate` (optional)
   - Generate unified completion artifacts for one model on one dataset subset.
2. `openjury-evaluate annotate`
   - Generate unified judge-annotation artifacts from paired completions.

Built on top of those:

3. `openjury-evaluate arena`
   - Preset/orchestration over `annotate(matchmaker)` plus arena-style rating analysis (BT, Elo, win rates, etc.).
4. `openjury-evaluate agreement`
   - Preset/orchestration over `annotate(dataset_pairs)` plus human-vs-judge analysis on datasets with labels (e.g. `lmsys`, `comparia`).

Execution backends are orthogonal to the task:
- local execution (e.g. `VLLM/...`, `SGLang/...`, `LlamaCpp/...`)
- API execution (e.g. `OpenRouter/...`, `ChatOpenAI/...`)
- SLURM script generation via `--slurm` (same command interface)

## Test Your Model (Examples)

### 1) Quick generation smoke test (single model)

Local model (vLLM or SGLang):

```bash
uv run openjury-generate \
  --model VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --dataset alpaca-eval \
  --n_instructions 20 \
  --output /tmp/qwen_smoke.parquet

uv run openjury-generate \
  --model SGLang/Qwen/Qwen3-8B \
  --dataset alpaca-eval \
  --n_instructions 20 \
  --output /tmp/qwen3_sglang_smoke.parquet
```

API model:

```bash
uv run openjury-generate \
  --model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --dataset alpaca-eval \
  --n_instructions 20 \
  --output /tmp/deepseek_smoke.parquet
```

### 2) Compare your model against a baseline (arena)

Fast CLI battle:

```bash
uv run openjury-evaluate arena \
  --models VLLM/Qwen/Qwen2.5-0.5B-Instruct VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model OpenRouter/qwen/qwen3-32b \
  --dataset alpaca-eval \
  --n_instructions 50 \
  --criteria default \
  --output_dir results/arena/my_model_test
```

Replace either model string with your own model spec (local or API).

### 3) Reuse pre-generated completions (staged, reproducible)

Generate completions first:

```bash
uv run openjury-generate \
  --model VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --dataset alpaca-eval \
  --n_instructions 100 \
  --output /tmp/model_a.parquet

uv run openjury-generate \
  --model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --dataset alpaca-eval \
  --n_instructions 100 \
  --output /tmp/model_b.parquet
```

Then evaluate using an arena config (example `my_arena.json`):

```json
{
  "dataset": "alpaca-eval",
  "n_instructions": 100,
  "models": [
    { "name": "Model/A", "completions": "/tmp/model_a.parquet" },
    { "name": "Model/B", "completions": "/tmp/model_b.parquet" }
  ],
  "judge": {
    "model": "OpenRouter/qwen/qwen3-32b",
    "mode": "samplewise",
    "max_tokens": 2048
  },
  "criteria": "default",
  "output_dir": "results/arena/staged_test"
}
```

```bash
uv run openjury-evaluate arena --config my_arena.json
```

### 4) Same commands on a cluster (SLURM)

Generate SLURM scripts from the same public command by adding `--slurm`:

```bash
uv run openjury-evaluate arena \
  --config my_arena.json \
  --slurm \
  --slurm_output_dir slurm_scripts
```

```bash
uv run openjury-generate \
  --model VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --dataset alpaca-eval \
  --output /tmp/ignored_in_slurm_mode.parquet \
  --n_instructions 100 \
  --slurm \
  --slurm_output_dir slurm_scripts
```

### 5) Prepare caches for offline compute nodes

Generated compute-node scripts enable Hugging Face offline mode. Before running on SLURM, pre-download datasets on a login node and make sure model weights are already cached in your Hugging Face cache.

List available dataset downloads:

```bash
uv run python -m openjury.datasets.download --list
```

Examples:

```bash
uv run python -m openjury.datasets.download lmsys
uv run python -m openjury.datasets.download lmsys-140k comparia
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
  "criteria": "default",
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

Use `openjury-evaluate arena` when you want OpenJury to generate completions and run evaluation in one pipeline.

Candidate models and judge models can be mixed across local and API backends (for example, local `VLLM/...` candidates with an API judge such as `OpenRouter/...`).

You can also generate SLURM scripts from the same command by adding `--slurm`:

```bash
uv run openjury-evaluate arena \
  --models VLLM/Qwen/Qwen2.5-0.5B-Instruct VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --dataset alpaca-eval \
  --slurm \
  --slurm_output_dir slurm_scripts
```

### 2) Staged workflow (generate first, evaluate later)

Use `openjury-generate` to create completions separately, then reference the parquet file in an arena config via `completions`.

This is useful for:
- running generation and judging on different machines
- reusing a fixed completion set across multiple judges/criteria
- SLURM/job-based workflows

### 3) Staged evaluation (annotate first, analyze later)

Both `arena` and `agreement` support stage control:
- `--stage annotate`: run judge annotation and save an intermediate artifact
- `--stage analyze`: load the saved annotation artifact and compute final metrics/results
- `--stage all` (default): run both in one call
- This also works with `--slurm` forwarding (`openjury-evaluate arena/agreement --slurm --stage ...`).

Arena example:

```bash
uv run openjury-evaluate arena \
  --config configs/arena/arena_2model.json \
  --stage annotate

uv run openjury-evaluate arena \
  --output_dir results/arena/demo \
  --stage analyze
```

Agreement example:

```bash
uv run openjury-evaluate agreement \
  --dataset lmsys \
  --judge_model OpenRouter/qwen/qwen3-32b \
  --n_instructions 200 \
  --output_dir results/agreement/lmsys_qwen3 \
  --stage annotate

uv run openjury-evaluate agreement \
  --output_dir results/agreement/lmsys_qwen3 \
  --stage analyze
```

Intermediate artifacts:
- `arena_annotations.json`
- `agreement_annotations.json`

## SLURM Script Helper (Optional)

Use `openjury-slurm` to generate job scripts for staged execution on SLURM clusters.
It supports mixed execution targets (GPU jobs via `sbatch`, API models on the login node).
For common generation/arena workflows, `openjury-generate` and `openjury-evaluate arena` also support a `--slurm` forwarding mode.

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

Agreement mode example:

```bash
uv run openjury-slurm \
  --mode agreement \
  --dataset lmsys \
  --judge_model OpenRouter/qwen/qwen3-32b \
  --n_instructions 200 \
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
- `openjury-evaluate agreement --slurm` is supported and forwards to `openjury-slurm --mode agreement`.
- `openjury-generate --slurm` generates/submits SLURM scripts and uses OpenJury-managed SLURM work/cache paths (it does not write directly to the local `--output` parquet path).
- Cluster settings (`ACCOUNT`, `PARTITION`, `USER_WORK_DIR`, `TIME_LIMIT`) are read from environment variables or `.env`.
- Compute-node scripts source `.env` and then enable HF offline mode by default, so datasets and model weights must already be cached.
- `OPENJURY_VENV` is used for host-based compute jobs. Containerized `VLLM/...` jobs use `OPENJURY_CONTAINER_RUNTIME`, `OPENJURY_CONTAINER_IMAGE`, and `OPENJURY_CONTAINER_HOME` instead.
- Recommended Leonardo workflow for containerized vLLM: set the container vars in `.env`, run `openjury-container-prepare` on the login node, pre-cache datasets and model weights into `HF_HOME`, then use the normal `--slurm` commands.

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

- `samplewise` (default): scores each model independently on the criteria
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

## Agreement Step (Human vs Judge)

Use `openjury-evaluate agreement` (or `openjury-agreement`) for datasets that already include completions and human preference labels, such as `lmsys` and `comparia`.

Quick smoke test:

```bash
uv run openjury-evaluate agreement --help
```

Supported human-preference datasets include:
- `lmsys`
- `lmsys-140k`
- `comparia`

Samplewise (default):

```bash
uv run openjury-evaluate agreement \
  --dataset lmsys \
  --judge_model OpenRouter/qwen/qwen3-32b \
  --n_instructions 200 \
  --output_dir results/agreement/lmsys_qwen3
```

Pairwise:

```bash
uv run openjury-evaluate agreement \
  --dataset comparia \
  --judge_model VLLM/Qwen/Qwen3-32B \
  --judge_mode pairwise \
  --language fr \
  --output_dir results/agreement/comparia_pairwise
```

Outputs:
- `agreement.json` (metrics + per-sample results + system prompt)
- `agreement_details.csv` (flat table for analysis)
- `agreement_annotations.json` (intermediate annotation artifact for `--stage analyze`)

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
- `SGLang/...`
- `ChatOpenAI/...`
- `OpenRouter/...`
- `LiteLLM/...`
- `LlamaCpp/...`
- `Dummy/...`

Examples:

```bash
VLLM/Qwen/Qwen2.5-7B-Instruct
SGLang/Qwen/Qwen3-8B
ChatOpenAI/gpt-4o-mini
OpenRouter/deepseek/deepseek-chat-v3.1
LiteLLM/anthropic/claude-3-5-sonnet
LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf
```

## Criteria

Built-in criteria sets:
- `default`
- `coding`
- `translation`
- `overall` (single-criterion set for single-score evaluation)

There is no separate "single-score without criteria" mode in the current pipeline.
Use `overall` when you want a single scalar quality score per completion.

Use a built-in criteria set:

```bash
uv run openjury-evaluate arena --config my_arena.json --criteria coding
```

Use a custom criteria set by setting `criteria` to a JSON file path in the config.

Custom criteria JSON shape:

```json
{
  "name": "custom",
  "description": "My criteria set",
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
- `arena.json` (matches, criteria scores, ratings, metadata)
- `arena_config.json` (config snapshot)

Typical run command:

```bash
uv run openjury-evaluate arena --config my_arena.json
```

Results are written to `output_dir` from the config (or `--output_dir`).

## Standalone Generation Step (Optional)

Generate completions first, then reference them in an arena config:

```bash
uv run openjury-generate \
  --model VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --dataset alpaca-eval \
  --output /tmp/qwen05.parquet \
  --n_instructions 100 \
  --tensor_parallel_size 1
```

Config-file mode is also supported (JSON/YAML):

```bash
uv run openjury-generate --config configs/generate/my_model.json
```

Minimal `generate` config example:

```json
{
  "dataset": {
    "name": "alpaca-eval",
    "n_instructions": 100,
    "seed": 42
  },
  "model": {
    "name": "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
    "gpus": 1,
    "max_tokens": 4096
  },
  "output": "/tmp/qwen05.parquet",
  "runtime": {
    "truncate_input_chars": 8192,
    "gpu_memory_utilization": 0.9,
    "use_tqdm": false,
    "base_model": false,
    "ignore_cache": false
  }
}
```

### Generate completions with an API model

```bash
uv run openjury-generate \
  --model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --dataset alpaca-eval \
  --output /tmp/deepseek.parquet \
  --n_instructions 100
```

### Generate via SLURM script generation (same command + `--slurm`)

```bash
uv run openjury-generate \
  --model VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --dataset alpaca-eval \
  --output /tmp/qwen05.parquet \
  --n_instructions 100 \
  --slurm \
  --slurm_output_dir slurm_scripts
```

In `--slurm` mode, the command forwards to `openjury-slurm --mode generate`, so the local `--output` path is not used as the final completion path.

### vLLM chat template override (optional)

For models without a tokenizer-defined chat template (or when you want to override it), you can provide a template string or file.

CLI (generation):

```bash
uv run openjury-generate \
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
| `lmsys` | LM Arena human-preference 100k dataset |
| `lmsys-140k` | LM Arena human-preference 140k dataset |
| `comparia` | Human-preference style dataset |
### Annotate on SLURM

You can forward annotate runs through the built-in SLURM planner:

```bash
uv run openjury-evaluate annotate \
  --config configs/annotate/lmsys_gpt-oss20b_qwen3-32b.yaml \
  --slurm --submit --detach \
  --slurm_output_dir results/annotate_slurm \
  --slurm_tag gpt-oss20b-qwen3-32b
```

If the challenger completions are not precomputed, the SLURM planner will
generate them first when needed and then run the annotate step where the judge
belongs.
