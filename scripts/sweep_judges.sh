#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Sweep: run agreement on one dataset with multiple judge models.
#
#  Usage:
#    bash scripts/sweep_judges.sh                     # dry-run (no --submit)
#    bash scripts/sweep_judges.sh --submit            # submit all jobs
#    bash scripts/sweep_judges.sh --submit --detach   # submit & don't wait
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

# Load repo .env so container/runtime settings can be forwarded explicitly.
if [ -f ".env" ]; then
    set -a
    source ".env"
    set +a
fi

# ── Configuration ────────────────────────────────────────────────
DATASET="lmsys-140k"
N_INSTRUCTIONS=25000
SEED=42
LANGUAGE="null"
BALANCE_BY="models"
CRITERIA="default"
MODE="pairwise"
ENFORCE_EAGER="true"
INCLUDE_COMPLETIONS="false"
IGNORE_SCORE_CACHE="true"
SLURM_TIME="16:00:00"
SLURM_QOS="normal"
SLURM_OUTPUT_DIR="results/agreement_sweep_lmsys_100k_new"
DEFAULT_JUDGE_MAX_TOKENS=4096

# Judge models and their GPU requirements.
# Format: "HF_PATH:NUM_GPUS:MAX_MODEL_LEN[:JUDGE_MAX_TOKENS]"
# Set MAX_MODEL_LEN to cap VRAM usage. Use 0 for model default.
# Set JUDGE_MAX_TOKENS to cap the judge output length. If omitted, the
# script uses DEFAULT_JUDGE_MAX_TOKENS above.
JUDGES=(
    # Qwen3 series
    # "Qwen/Qwen3-8B:2:0:4096"
    "Qwen/Qwen3-32B:4:0:4096"
    # # Qwen3.5 Series for scale experiments
    # "Qwen/Qwen3.5-4B:2:0:4096"
    # "Qwen/Qwen3.5-9B:2:0:4096"
    # "Qwen/Qwen3.5-27B:4:102400:4096"
    # "Qwen/Qwen3.5-35B-A3B:4:32768:4096"
    # # # QWQ Reasoning
    # # "Qwen/QwQ-32B:4:32768:8192"
    # # # OpenAI GPT-OSS
    # "openai/gpt-oss-20b:4:71680:4096"
    # "openai/gpt-oss-120b:4:71680:4096"
    # # # # Meta LLAMA
    # "meta-llama/Llama-3.3-70B-Instruct:4:102400:4096"
    # "meta-llama/Llama-3.1-8B-Instruct:2:0:4096"
    # # # Google Gemma
    # "google/gemma-4-26B-A4B-it:4:102400:4096"
    # "google/gemma-4-E4B-it:2:0:4096"
)

# ── Flags forwarded to openjury-evaluate ─────────────────────────
EXTRA_FLAGS=("$@")   # e.g. --submit --detach
SLURM_FLAGS=()
if [ -n "${OPENJURY_CONTAINER_RUNTIME:-}" ] && [ "${OPENJURY_CONTAINER_RUNTIME}" != "none" ]; then
    SLURM_FLAGS+=(--slurm_container_runtime "$OPENJURY_CONTAINER_RUNTIME")
fi
if [ -n "${OPENJURY_CONTAINER_IMAGE:-}" ]; then
    SLURM_FLAGS+=(--slurm_container_image "$OPENJURY_CONTAINER_IMAGE")
fi
if [ -n "${OPENJURY_CONTAINER_HOME:-}" ]; then
    SLURM_FLAGS+=(--slurm_container_home "$OPENJURY_CONTAINER_HOME")
fi

# ── Generate & submit ───────────────────────────────────────────
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

for entry in "${JUDGES[@]}"; do
    IFS=':' read -r MODEL_PATH GPUS MAX_LEN JUDGE_MAX_TOKENS <<< "$entry"
    JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-$DEFAULT_JUDGE_MAX_TOKENS}"
    MODEL_SHORT=$(basename "$MODEL_PATH")
    TAG="${DATASET}-${MODEL_SHORT}"

    CFG="$TMPDIR/${TAG}.yaml"
    cat > "$CFG" <<EOF
dataset: ${DATASET}
n_instructions: ${N_INSTRUCTIONS}
seed: ${SEED}
language: ${LANGUAGE}
balance_by: ${BALANCE_BY}
criteria: ${CRITERIA}
output_dir: ${SLURM_OUTPUT_DIR}

generation:
  ignore_score_cache: ${IGNORE_SCORE_CACHE}

judge:
  model: VLLM/${MODEL_PATH}
  gpus: ${GPUS}
  mode: ${MODE}
  enforce_eager: ${ENFORCE_EAGER}
  max_tokens: ${JUDGE_MAX_TOKENS}
  generation_kwargs:
    truncate_prompt_tokens: 40000
$([ "${MAX_LEN:-0}" -gt 0 ] && echo "  max_model_len: ${MAX_LEN}")

include_completions: ${INCLUDE_COMPLETIONS}
EOF

    echo "═══════════════════════════════════════════════════"
    echo "  Judge: ${MODEL_PATH} (${GPUS} GPU(s))"
    echo "  Tag:   ${TAG}"
    echo "  Config: ${CFG}"
    echo "═══════════════════════════════════════════════════"

    uv run openjury-evaluate agreement \
        --config "$CFG" \
        --slurm \
        --slurm_output_dir "$SLURM_OUTPUT_DIR" \
        --slurm_tag "$TAG" \
        --slurm_time "$SLURM_TIME" \
        "${SLURM_FLAGS[@]}" \
        "${EXTRA_FLAGS[@]}"

    echo ""
done

echo "Done. All jobs generated."
echo "Results will be in: ${SLURM_OUTPUT_DIR}/"
